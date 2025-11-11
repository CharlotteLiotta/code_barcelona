import numpy as np # type: ignore
import scipy # type: ignore
import statsmodels.api as sm # type: ignore
from stargazer.stargazer import Stargazer
from scipy.optimize import curve_fit

from model import *
from plotting_tools import *

def calibration_utility_amenity(x, gdf, income_levels, alpha, print_summary=0, export_amenities=0):
    """Calibrate BETA and U_LOW/MED/HIGH by minimizing likelihood with numerical stabilization."""
    
    BETA, U_LOW, U_MED, U_HIGH = x
    U = {"LOW": U_LOW, "MED": U_MED, "HIGH": U_HIGH}
    print(f"x = {x}")

    pop_sum = gdf[["pop_LOW","pop_MED","pop_HIGH"]].sum(axis=1)
    w = {lvl: np.clip(gdf[f"pop_{lvl}"] / pop_sum, 1e-12, 1) for lvl in income_levels}

    wage_minus_tc = {lvl: np.clip(gdf[f"wage_{lvl}"] - gdf[f"transport_cost_{lvl}"], 1e-12, None)
                     for lvl in income_levels}

    factor = ((1-BETA)**(1-BETA)) * (BETA**BETA)

    estimated_A = {}
    for lvl in income_levels:
        estimated_A[lvl] = np.clip(U[lvl] / (factor * (wage_minus_tc[lvl] / gdf["rent_m2"])), 1e-12, None)

    gdf["log_A"] = np.log(sum(w[lvl] * estimated_A[lvl] for lvl in income_levels))

    # --- 3. Estimate amenities regression ---
    gdf_here = gdf.loc[~np.isnan(gdf["log_A"]) & ~np.isinf(gdf["log_A"]), :]
    y = gdf_here["log_A"]
    X = gdf_here.loc[:, ["beach_500m", "parc_500m", "parc_500m_1km", "parc_1km_2km",
                         "parc_500m_b", "parc_500m_1km_b", "parc_1km_2km_b",
                         "station_500m", "station_500m_1km", "station_1km_2km",
                         "airport_500m", "high_tourism", 'mean_activity',
                         'pedestrian_density', 'slope_20', 'fgc_500m', 'rodalies_500m']]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    if print_summary == 1:
        print(model.summary())
        stargazer = Stargazer([model])
        print(stargazer.render_latex())
    residuals = model.resid
    epsilon_A = max(np.nanmean(residuals**2), 1e-12)
    log_L_A = - (len(gdf_here)/2)*np.log(2*np.pi*epsilon_A) - np.nansum(residuals**2)/(2*epsilon_A)

    # --- 4. Compute bid rents per group ---
    weighted_A = sum(w[lvl] * estimated_A[lvl] for lvl in income_levels)
    R = {}
    for lvl in income_levels:
        R[lvl] = compute_rents(BETA, gdf[f"wage_{lvl}"], U[lvl]/weighted_A, gdf[f"transport_cost_{lvl}"])
    
    # Normalize rents for softmax
    R_stack = np.vstack([R[lvl] for lvl in income_levels])
    R_mean, R_std = np.nanmean(R_stack), np.nanstd(R_stack) + 1e-9
    R_n = {lvl: (R[lvl] - R_mean)/R_std for lvl in income_levels}

    # Soft assignment
    R_max = np.maximum.reduce([R_n[lvl] for lvl in income_levels])
    exp_R = {lvl: np.exp(alpha * (R_n[lvl] - R_max)) for lvl in income_levels}
    denom = sum(exp_R.values())
    w_est = {lvl: exp_R[lvl] / denom for lvl in income_levels}

    log_sorting = sum(np.nansum(np.log(w_est[lvl]) * gdf[f"pop_{lvl}"]) for lvl in income_levels)

    # --- 5. Compute city sizes ---
    R_total = sum(w[lvl]*R[lvl] for lvl in income_levels)
    R_total = np.maximum(R_total, 1e-12)
    size_est = sum(w[lvl] * BETA * (gdf[f"wage_{lvl}"] - gdf[f"transport_cost_{lvl}"]) / R_total
                   for lvl in income_levels)
    
    diff_size = gdf["size"] - size_est
    mask = (~np.isnan(diff_size)) & (~np.isinf(diff_size))
    epsilon_size = max(np.nanmean(diff_size[mask]**2), 1e-12)
    log_L_size = - np.sum(mask)/2 * np.log(2*np.pi*epsilon_size) - np.nansum(diff_size[mask]**2)/(2*epsilon_size)
    
    # --- 7. Export amenities if requested ---
    if export_amenities:
        amenities = np.exp(np.nansum(X.iloc[:,1:] * model.params.iloc[1:], 1))
        gdf_here = gdf_here.copy()
        gdf_here["amenities"] = amenities
        return gdf_here[["ID", "amenities"]]
    
    else:
        # Final log-likelihood (maximize sum of log-likelihoods and sorting)
        return - (log_L_A + log_sorting + log_L_size)

def compute_cost_car(gdf, import_trans_mode, PRICE_TIME, WORKING_DAYS, PRICE_FUEL, path_data):
    """ Calibrate the fixed cost of private car to match the transport modes data """

    trans_mode = import_trans_mode(path_data)

    gdf = gdf.merge(trans_mode.loc[:,["code_city", "share_car"]], on = "code_city", how = "left")
    
    def compute_error_transport(x):
        FIXED_COST_CAR = x[0]
        gdf_here = compute_transport_cost(gdf, PRICE_TIME, WORKING_DAYS, FIXED_COST_CAR, PRICE_FUEL, tax = 0)
        error1 = np.nansum(np.abs(((1 - gdf_here["transport_mode"]) * gdf_here["pop"]) - (gdf_here["share_car"] * gdf_here["pop"])))
        print(f"x = {x[0]}, error1 = {error1}") #Error on transport mode by census tract

        gdf_here = gdf_here.loc[~np.isnan(gdf_here.share_car),:]
        error2 = np.nansum(gdf_here["pop"]) * np.abs((np.nansum(gdf_here.share_car * gdf_here["pop"]) / np.nansum(gdf_here["pop"])) - (np.nansum((1 - gdf_here["transport_mode"]) * gdf_here["pop"]) / np.nansum(gdf_here["pop"])))
        print(f"x = {x[0]}, error2 = {error2}") #Error on transport mode at the AMB level
        return error1 + error2

    solving_transport = scipy.optimize.minimize(compute_error_transport, x0=[300], method='Nelder-Mead')
    FIXED_COST_CAR = solving_transport.x
    return gdf, FIXED_COST_CAR

def compute_cost_car_logit(gdf, Y, import_trans_mode, PRICE_TIME, WORKING_DAYS, PRICE_FUEL, path_data):
    """ Calibrate the fixed cost of private car to match the transport modes data """

    trans_mode = import_trans_mode(path_data)

    gdf = gdf.merge(trans_mode.loc[:,["code_city", "share_car"]], on = "code_city", how = "left")
    
    def compute_error_transport(x):
        FIXED_COST_CAR = x[0]
        LAMBDA = x[1]
        gdf_here = compute_transport_cost_logit(gdf, Y, PRICE_TIME, WORKING_DAYS, FIXED_COST_CAR, PRICE_FUEL, LAMBDA, tax = 0)
        error1 = np.nansum(np.abs(((1 - gdf_here["transport_mode"]) * gdf_here["pop"]) - (gdf_here["share_car"] * gdf_here["pop"])))
        print(f"x = {x}, error1 = {error1}") #Error on transport mode by census tract

        gdf_here = gdf_here.loc[~np.isnan(gdf_here.share_car),:]
        error2 = np.nansum(gdf_here["pop"]) * np.abs((np.nansum(gdf_here.share_car * gdf_here["pop"]) / np.nansum(gdf_here["pop"])) - (np.nansum((1 - gdf_here["transport_mode"]) * gdf_here["pop"]) / np.nansum(gdf_here["pop"])))
        print(f"x = {x}, error2 = {error2}") #Error on transport mode at AMB level
        return error1 + error2

    solving_transport = scipy.optimize.minimize(compute_error_transport, x0=[100, 40], method='L-BFGS-B', bounds=[(0, 500), (0, None)])
    FIXED_COST_CAR = solving_transport.x[0]
    LAMBDA = solving_transport.x[1]
    return gdf, FIXED_COST_CAR, LAMBDA

#def compute_cost_car_poly(gdf, Y, import_trans_mode, PRICE_TIME, WORKING_DAYS, PRICE_FUEL, travel_time_matrix_car, travel_time_matrix_transit, employment_centers, path_data, jobs_in_toll_area, houses_in_toll_area):
#    """ Calibrate the fixed cost of private car to match the transport modes data """

#    trans_mode = import_trans_mode(path_data)

#    gdf = gdf.merge(trans_mode.loc[:,["code_city", "share_car"]], on = "code_city", how = "left")
    
#    def compute_error_transport_poly(x):
#        FIXED_COST_CAR = x[0]
#        LAMBDA = x[1]
#        ARRAY_WAGE = x[2:]

#        gdf_here, employed_results, travel_matrix = compute_transport_cost_poly(gdf, travel_time_matrix_car, travel_time_matrix_transit, PRICE_TIME, WORKING_DAYS, FIXED_COST_CAR, PRICE_FUEL, LAMBDA, ARRAY_WAGE, jobs_in_toll_area, houses_in_toll_area, tax = 0)
        
#        error1 = np.nansum(np.abs(((1 - gdf_here["transport_mode"]) * gdf_here["pop"]) - (gdf_here["share_car"] * gdf_here["pop"])))
#        print(f"x = {x}, error_mode_by_tract = {error1}") #Error on transport mode by census tract

#        gdf_here = gdf_here.loc[~np.isnan(gdf_here.share_car),:]
#        error2 = np.nansum(gdf_here["pop"]) * np.abs((np.nansum(gdf_here.share_car * gdf_here["pop"]) / np.nansum(gdf_here["pop"])) - (np.nansum((1 - gdf_here["transport_mode"]) * gdf_here["pop"]) / np.nansum(gdf_here["pop"])))
#        print(f"x = {x}, error_mode_AMB = {error2}") #Error on transport mode by census tract = {error2}") #Error on transport mode at AMB level
        
        #error3:avg wage
#        estimated_wage = np.nansum(gdf_here["pop"] * gdf_here["wage"]) / np.nansum(gdf_here["pop"])
#        error3 = np.abs(Y-estimated_wage)
#        print(f"x = {x}, error_wage = {error3}") #Error on transport mode by census tract = {error2}") #Error on transport mode at AMB level
        
        #error4:ppl per employment center
#        employed_results = employed_results.merge(employment_centers, left_index = True, right_on = "cluster")
#        error4= np.nansum(np.abs(employed_results.employed - employed_results.employment)) / 2
#        print(f"x = {x}, error_employment = {error4}") #Error on transport mode by census tract = {error2}") #Error on transport mode at AMB level
        
#        return error1 + error2 + error3 + error4

#    solving_transport = scipy.optimize.minimize(compute_error_transport_poly, x0=[150, 200] + (np.ones(len(np.unique(employment_centers.cluster))) * Y).tolist(), method='L-BFGS-B', bounds=[(0, 300), (0, 400)]+ [(0, 10000)] * len(np.unique(employment_centers.cluster)))
#    FIXED_COST_CAR = solving_transport.x[0]
#    LAMBDA = solving_transport.x[1]
#    ARRAY_WAGE = solving_transport.x[2:]
#    return gdf, FIXED_COST_CAR, LAMBDA, ARRAY_WAGE

def compute_cost_car_poly_i(gdf, Y_median, import_trans_mode, PRICE_TIME, WORKING_DAYS, PRICE_FUEL, travel_time_matrix_car, travel_time_matrix_transit, employment_centers, path_data, jobs_in_toll_area, houses_in_toll_area, income_levels, wage_factors):
    """ Calibrate the fixed cost of private car to match the transport modes data """

    trans_mode = import_trans_mode(path_data)

    gdf = gdf.merge(trans_mode.loc[:,["code_city", "share_car"]], on = "code_city", how = "left")
    
    def compute_error_transport_poly(x):
        FIXED_COST_CAR = x[0]
        LAMBDA = x[1]
        n_centers = len(employment_centers)
        ARRAY_WAGE_LOW = x[2:2 + n_centers]
        ARRAY_WAGE_MED = x[2 + n_centers:2 + 2 * n_centers]
        ARRAY_WAGE_HIGH = x[2 + 2 * n_centers:]

        gdf_here, employed_results, travel_matrix = compute_transport_cost_poly_i(gdf, travel_time_matrix_car, travel_time_matrix_transit, PRICE_TIME, WORKING_DAYS, FIXED_COST_CAR, PRICE_FUEL, LAMBDA, ARRAY_WAGE_LOW, ARRAY_WAGE_MED, ARRAY_WAGE_HIGH, jobs_in_toll_area, houses_in_toll_area, tax, income_levels, wage_factors)
        
        car_use_data = (gdf_here["share_car"] * gdf_here["pop"])
        car_use_estimated = np.sum((1 - gdf_here[f"transport_mode_{lvl}"]) * gdf_here[f"pop_{lvl}"] for lvl in income_levels)
        error_mode_by_tract = np.nansum(np.abs(car_use_estimated - car_use_data))
        print(f"x = {x}")
        print(f"error_mode_by_tract = {error_mode_by_tract}")

        gdf_valid = gdf_here.dropna(subset=["share_car"])
        error_mode_AMB = np.abs(np.nansum(np.sum((1 - gdf_valid[f"transport_mode_{lvl}"]) * gdf_valid[f"pop_{lvl}"] for lvl in income_levels) - gdf_valid["share_car"] * gdf_valid["pop"]))
        print(f"error_mode_AMB = {error_mode_AMB}") #Error on transport mode by census tract = {error2}") #Error on transport mode at AMB level
        
        employed_results = employed_results.merge(employment_centers, left_index = True, right_on = "cluster")
        error_employment = np.nansum(np.abs(employed_results[[f"employed_{lvl}" for lvl in income_levels]].sum(axis=1) - employed_results["employment"])) / 2
        print(f"error_employment = {error_employment}") #Error on transport mode by census tract = {error2}") #Error on transport mode at AMB level
        
        error_wage = 0
        for lvl in income_levels:
            estimated_wage = np.nansum(gdf_here[f"pop_{lvl}"] * gdf_here[f"wage_{lvl}"]) / np.nansum(gdf_here[f"pop_{lvl}"])
            print(f"estimated_wage_{lvl}", estimated_wage)
            error_wage += np.abs(Y_median * wage_factors[lvl] - estimated_wage) * np.nansum(gdf_here[f"pop_{lvl}"])
        error_wage /= np.nansum([gdf_here[f"pop_{lvl}"] for lvl in income_levels])
        print(f"error_wage = {error_wage}") #Error on transport mode by census tract = {error2}") #Error on transport mode at AMB level
        
        estimated_wage_spatial = sum(gdf_here[f"pop_{lvl}"] * gdf_here[f"wage_{lvl}"] for lvl in income_levels) / \
                                 np.nansum([gdf_here[f"pop_{lvl}"] for lvl in income_levels])
        error_spatial_wage = np.nansum(np.abs(estimated_wage_spatial - gdf["net_income"])) / np.nansum(gdf["pop"])
        print(f"error_spatial_wage = {error_spatial_wage}")
        
        return (error_mode_by_tract + error_mode_AMB + error_employment + error_wage + error_spatial_wage)

    init_wage = np.array([3126, 2905, 3040, 2976, 2903, 2979, 2869, 3005])
    x0 = [200, 250] + (init_wage * 0.6).tolist() + init_wage.tolist() + (init_wage * 1.4).tolist()
    bounds = [(0, 300), (0, 400)] + [(0, 10000)] * 3 * len(np.unique(employment_centers.cluster))
    solving_transport = scipy.optimize.minimize(compute_error_transport_poly, x0=x0, method='L-BFGS-B', bounds=bounds)
    FIXED_COST_CAR = solving_transport.x[0]
    LAMBDA = solving_transport.x[1]
    n_centers = len(employment_centers)
    ARRAY_WAGE_LOW = solving_transport.x[2:2 + n_centers]
    ARRAY_WAGE_MED = solving_transport.x[2 + n_centers:2 + 2 * n_centers]
    ARRAY_WAGE_HIGH = solving_transport.x[2 + 2 * n_centers:]
    
    return gdf, FIXED_COST_CAR, LAMBDA, ARRAY_WAGE_LOW, ARRAY_WAGE_MED, ARRAY_WAGE_HIGH

def calibrate_beta(gdf, Y):
    """ Calibrate BETA as the average share of income net of transport cost used for housing """

    gdf["income_net_of_transport_cost"] = Y - gdf["transport_cost"]
    gdf["rent_share"] = gdf["rent_m2"] * gdf["size"] / (gdf["income_net_of_transport_cost"])
    BETA = np.nansum(gdf["pop"][~np.isnan(gdf["rent_share"])] * gdf["rent_share"][~np.isnan(gdf["rent_share"])]) / np.nansum(gdf["pop"][~np.isnan(gdf["rent_share"])])
    return BETA

def calibrate_b_kappa(gdf, mask, RHO, option_function = "CES", option_calib = "housing"):
    """ Calibrate B and KAPPA using the rent data """
        
    gdf_here = gdf.loc[mask,:].copy()
    gdf_here.loc[:,"log_n"] = np.log(gdf_here["pop"])
    gdf_here.loc[:,"log_R"] = np.log(gdf_here["rent_m2"])
    gdf_here.loc[:,"log_L"] = np.log(gdf_here["land"])
    gdf_here.loc[:,"log_q"] = np.log(gdf_here["size"])
    gdf_here.loc[:,"log_h"] = gdf_here["log_n"] + gdf_here["log_q"] - gdf_here["log_L"]
    

    if option_calib == "population":
        y = (gdf_here["log_n"])
        X = (gdf_here.loc[:,["log_R", "log_L", "log_q"]])
    elif option_calib == "housing":
        X = (gdf_here.loc[:,["log_R",]])
        y = (gdf_here["log_h"])

    if option_function == "Cobb-Douglas":

        X = sm.add_constant(X)  # Adds intercept
        model_statsmodel = sm.OLS(y, X).fit()
        print(model_statsmodel.summary())

        stargazer = Stargazer([model_statsmodel])
        print(stargazer.render_latex())

        B = model_statsmodel.params["log_R"] / (1 + model_statsmodel.params["log_R"])
        KAPPA = np.exp(((1-B) * model_statsmodel.params["const"]) - (B * np.log(B/RHO)))
        print("B: ", B, "KAPPA: ", KAPPA)
        SIGMA = 0

        plot_calib_housing(gdf_here, model_statsmodel.fittedvalues)

    elif option_function == "CES":

        def model(x, kappa, a, sigma):
            return kappa * (a ** (-sigma / (1 - sigma))) * (
            (1 - ((1 - a) ** sigma) * ((kappa * x) ** (sigma - 1))) ** (sigma / (1 - sigma))
            )

        p0 = [0.1, 0.45, 5/7]  # initial guesses for kappa, a, sigma

        params, cov = curve_fit(CES_func, np.exp(gdf_here.log_R), np.exp(gdf_here.log_h), p0=p0, maxfev=10000)
        KAPPA, A, SIGMA = params
        B = 1-A
        print(params)

        fitted_h_ces = KAPPA * (A ** (-SIGMA / (1 - SIGMA)))*(1 - (1 - A) ** (SIGMA) * ((KAPPA * np.exp(gdf_here.log_R)) ** (SIGMA-1))) ** (SIGMA /(1-SIGMA))
        plot_calib_housing(gdf_here, np.log(fitted_h_ces))
    
    return B, KAPPA, SIGMA