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

    w = {lvl: 0 for lvl in income_levels}

    w["LOW"] = ((gdf["pop_LOW"] >= gdf["pop_MED"]) & (gdf["pop_LOW"] >= gdf["pop_HIGH"])) * 1
    w["MED"] = ((gdf["pop_MED"] > gdf["pop_LOW"]) & (gdf["pop_MED"] > gdf["pop_HIGH"])) * 1
    w["HIGH"] = ((gdf["pop_HIGH"] > gdf["pop_LOW"]) & (gdf["pop_HIGH"] > gdf["pop_MED"])) * 1
    w["LOW"][w["LOW"] + w["MED"] + w["HIGH"] == 0] = 1

    wage_minus_tc = {lvl: np.clip(gdf[f"wage_{lvl}"] - gdf[f"transport_cost_{lvl}"], 1e-12, None)
                     for lvl in income_levels}

    factor = ((1-BETA)**(1-BETA)) * (BETA**BETA)

    estimated_A = {}
    for lvl in income_levels:
        estimated_A[lvl] = np.clip(U[lvl] / (factor * (wage_minus_tc[lvl] / (gdf["rent_m2"]** BETA))), 1e-12, None)

    gdf["log_A"] = np.log(sum(w[lvl] * estimated_A[lvl] for lvl in income_levels)) #np.log
    
    # --- 3. Estimate amenities regression ---
    gdf_here = gdf.loc[~np.isnan(gdf["log_A"]) & ~np.isinf(gdf["log_A"]), :]
    y = (gdf_here["log_A"])
    gdf_here["rodalies_500m_1km"] = ((gdf_here["min_distance_rodalies"] > 500) & (gdf_here["min_distance_rodalies"] < 1000)) * 1
    gdf_here["fgc_500m_1km"] = ((gdf_here["min_distance_fgc"] > 500) & (gdf_here["min_distance_fgc"] < 1000)) * 1


    X = gdf_here.loc[:, ["beach_500m", "beach_500m_1km",
                         'parc_2h_500m','parc_2h_500m_1km',
                         #'parc_combined_2h_500m','parc_combined_2h_500m_1km',
                         "station_500m", "station_500m_1km", 
                         'fgc_500m', 'rodalies_500m', 'rodalies_500m_1km', 'fgc_500m_1km',
                         "airport_500m",
                         "high_tourism", "medium_tourism",
                         'mean_activity',
                         'pedestrian_data_density']] #pedestrian_data_density
    
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    if print_summary == 1:
        print(model.summary())
        stargazer = Stargazer([model])
        print(stargazer.render_latex())
    parametersAmenities = model.params
    errorAmenities = model.resid_pearson

    ComputeLogLikelihood = (
        lambda sigma, error:
            np.nansum(- np.log(2 * np.pi * sigma ** 2) / 2
                      - 1 / (2 * sigma ** 2) * (error) ** 2)
            )

    sigmaAmenities = np.sqrt(np.nansum(errorAmenities ** 2) / np.nansum(~np.isnan(errorAmenities)))
    scoreAmenities = ComputeLogLikelihood(sigmaAmenities, errorAmenities)
    
    gdf["log_A"] = gdf["log_A"].fillna(gdf["log_A"].median())  # or interpolate
    R = {lvl: compute_rents(BETA, gdf[f"wage_{lvl}"], U[lvl] / np.exp(gdf["log_A"]), gdf[f"transport_cost_{lvl}"])
         for lvl in income_levels}


    # --- Normalize rents for softmax ---
    R_stack = np.vstack(list(R.values()))
    R_stack[R_stack > 1e100] = 1e100
    R_mean, R_std = np.nanmean(R_stack), np.nanstd(R_stack) + 1e-9
    R_n = {lvl: (R[lvl] - R_mean) / R_std for lvl in income_levels}

    # --- Soft assignment (numerically stable softmax) ---
    R_max = np.maximum.reduce(list(R_n.values()))
    exp_R = {lvl: np.exp(alpha * (R_n[lvl] - R_max)) for lvl in income_levels}
    denom = sum(exp_R.values())
    w_est = {lvl: exp_R[lvl] / denom for lvl in income_levels}
    w_est = {lvl: np.clip(w_est[lvl], 1e-12, None) for lvl in income_levels}
    
    for lvl in income_levels:
        #print(sum(w_est[lvl] == 0))
        w_est["MED"].loc[w_est["MED"]==0] = 0.000000000000000001
        w_est["HIGH"].loc[w_est["HIGH"]==0] = 0.000000000000000001
    pop_sum = gdf[["pop_LOW","pop_MED","pop_HIGH"]].sum(axis=1)
    w_here = {lvl: np.clip(gdf[f"pop_{lvl}"] / pop_sum, 1e-12, 1) for lvl in income_levels}

    
    log_sorting = sum(np.nansum(np.log(w_est[lvl]) * w_here[lvl]) for lvl in income_levels)
       
    # --- 5. Compute city sizes ---
    R_model = sum(w_est[lvl] * R[lvl] for lvl in income_levels)
    
    size_est = sum(w_est[lvl] * BETA * (gdf[f"wage_{lvl}"] - gdf[f"transport_cost_{lvl}"]) / R_model #R[lvl] #Rtotal
                   for lvl in income_levels)

    size_est = sum(w[lvl] * BETA * (gdf[f"wage_{lvl}"] - gdf[f"transport_cost_{lvl}"]) / R[lvl] #R[lvl] #Rtotal
                   for lvl in income_levels)
    
    #diff_size = gdf["size"] - size_est
    errorDwellingSize = np.log(size_est) - np.log(gdf["size"]) #np.log(sum(R[lvl])) - np.log(gdf["rent_m2"]) #
    sigmaDwellingSize = np.sqrt(np.nansum(errorDwellingSize ** 2) / np.nansum(~np.isnan(errorDwellingSize)))
    scoreDwellingSize = ComputeLogLikelihood(sigmaDwellingSize, errorDwellingSize)
    



    # --- 7. Export amenities if requested ---
    if export_amenities:
        amenities = np.exp(np.nansum(X.iloc[:,1:] * model.params.iloc[1:], 1))
        gdf_here = gdf_here.copy()
        gdf_here["amenities"] =  amenities #np.exp(y) #amenities
        
        #estimated_A = {}
        #for lvl in income_levels:
        #    estimated_A[lvl] = np.clip(U[lvl] / (factor * (wage_minus_tc[lvl] / ( R[lvl]** BETA))), 1e-12, None)

        #gdf = gdf.drop(columns = "log_A")
        #gdf["log_A"] = np.log(sum(w[lvl] * estimated_A[lvl] for lvl in income_levels)) #np.log
        #gdf_here = gdf_here.drop(columns = "log_A")
        #gdf_here = gdf_here.merge(gdf.loc[:,["ID", "log_A"]], on = "ID")
        #gdf_here["amenities"] = np.exp(gdf_here["log_A"])

        return gdf_here[["ID", "amenities"]]
    
    else:
        return -(scoreDwellingSize + scoreAmenities + log_sorting)


def compute_error_transport(x, employment_centers, gdf, travel_time_matrix_car, travel_time_matrix_transit, PRICE_TIME, WORKING_DAYS, PRICE_FUEL, jobs_in_toll_area, houses_in_toll_area, income_levels, wage_factors, scenario, Y_median, import_trans_mode, path_data):

    trans_mode = import_trans_mode(path_data)

    gdf = gdf.merge(trans_mode.loc[:,["code_city", "share_car"]], on = "code_city", how = "left")
    
    FIXED_COST_CAR = x[0]
    LAMBDA = x[1]
    n_centers = len(employment_centers)
    ARRAY_WAGE_LOW = x[2:2 + n_centers]
    ARRAY_WAGE_MED = x[2 + n_centers:2 + 2 * n_centers]
    ARRAY_WAGE_HIGH = x[2 + 2 * n_centers:]

    gdf_here, employed_results, _ = compute_transport_cost(gdf, travel_time_matrix_car, travel_time_matrix_transit, PRICE_TIME, WORKING_DAYS, FIXED_COST_CAR, PRICE_FUEL, LAMBDA, ARRAY_WAGE_LOW, ARRAY_WAGE_MED, ARRAY_WAGE_HIGH, jobs_in_toll_area, houses_in_toll_area, 0, income_levels, wage_factors, scenario)
        
    car_use_data = (gdf_here["share_car"] * gdf_here["pop"])
    car_use_estimated = sum((1 - gdf_here[f"transport_mode_{lvl}"]) * gdf_here[f"pop_{lvl}"] for lvl in income_levels)
    error_mode_by_tract = np.nansum(np.abs(car_use_estimated - car_use_data))
    print(f"x = {x}")
    print(f"error_mode_by_tract = {error_mode_by_tract}")

    gdf_valid = gdf_here.dropna(subset=["share_car"])
    error_mode_AMB = np.abs(np.nansum(sum((1 - gdf_valid[f"transport_mode_{lvl}"]) * gdf_valid[f"pop_{lvl}"] for lvl in income_levels)) - np.nansum(gdf_valid["share_car"] * gdf_valid["pop"]))
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
        sum([gdf_here[f"pop_{lvl}"] for lvl in income_levels])
    error_spatial_wage = np.nansum(gdf["pop"] * np.abs(estimated_wage_spatial - gdf["net_income_median"])) / np.nansum(gdf["pop"])
    print(f"error_spatial_wage = {error_spatial_wage}")
        
    return (error_mode_by_tract + error_mode_AMB + error_employment + error_wage + error_spatial_wage)


def compute_cost_car(gdf, Y_median, import_trans_mode, PRICE_TIME, WORKING_DAYS, PRICE_FUEL, travel_time_matrix_car, travel_time_matrix_transit, employment_centers, path_data, jobs_in_toll_area, houses_in_toll_area, income_levels, wage_factors, scenario, compute_error_transport):
    """ Calibrate the fixed cost of private car to match the transport modes data """

    init_wage = np.array([3126, 2905, 3040, 2976, 2903, 2979, 2869, 3005])
    init_wage = init_wage * Y_median / np.nanmean(init_wage)
    init_wage_high = init_wage * [1.4, 1.4, 1.4, 1.42, 1.4, 1.4, 1.42, 1.42]
    init_wage_low = init_wage * [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
    init_wage_med = init_wage * [1, 1, 1, 0.98, 1, 1, 0.98, 0.98]
    x0 = [200, 250] + (init_wage_low).tolist() + (init_wage_med).tolist() + (init_wage_high).tolist()
    bounds = [(0, 300), (0, 400)] + [(0, 10000)] * 3 * len(np.unique(employment_centers.cluster))
    
    def compute_error_transport_here(x):
        return compute_error_transport(x, employment_centers, gdf, travel_time_matrix_car, travel_time_matrix_transit, PRICE_TIME, WORKING_DAYS, PRICE_FUEL, jobs_in_toll_area, houses_in_toll_area, income_levels, wage_factors, scenario, Y_median, import_trans_mode, path_data)
    
    solving_transport = scipy.optimize.minimize(compute_error_transport_here, x0=x0, method='L-BFGS-B', bounds=bounds)
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

        p0 = [0.29, 0.66, 5/7]  # initial guesses for kappa, a, sigma

        params, cov = curve_fit(CES_func, np.exp(gdf_here.log_R), np.exp(gdf_here.log_h), p0=p0, maxfev=10000)
        KAPPA, A, SIGMA = params
        B = 1-A
        print(params)

        fitted_h_ces = KAPPA * (A ** (-SIGMA / (1 - SIGMA)))*(1 - (1 - A) ** (SIGMA) * ((KAPPA * np.exp(gdf_here.log_R)) ** (SIGMA-1))) ** (SIGMA /(1-SIGMA))
        plot_calib_housing(gdf_here, np.log(fitted_h_ces))
    
    return B, KAPPA, SIGMA