import numpy as np # type: ignore
import scipy # type: ignore
import statsmodels.api as sm # type: ignore
from stargazer.stargazer import Stargazer
from scipy.optimize import curve_fit

from model import *
from plotting_tools import *

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

def compute_cost_car_poly(gdf, Y, import_trans_mode, PRICE_TIME, WORKING_DAYS, PRICE_FUEL, travel_time_matrix_car, travel_time_matrix_transit, employment_centers, path_data, jobs_in_toll_area, houses_in_toll_area):
    """ Calibrate the fixed cost of private car to match the transport modes data """

    trans_mode = import_trans_mode(path_data)

    gdf = gdf.merge(trans_mode.loc[:,["code_city", "share_car"]], on = "code_city", how = "left")
    
    def compute_error_transport_poly(x):
        FIXED_COST_CAR = x[0]
        LAMBDA = x[1]
        ARRAY_WAGE = x[2:]

        gdf_here, employed_results, travel_matrix = compute_transport_cost_poly(gdf, travel_time_matrix_car, travel_time_matrix_transit, PRICE_TIME, WORKING_DAYS, FIXED_COST_CAR, PRICE_FUEL, LAMBDA, ARRAY_WAGE, jobs_in_toll_area, houses_in_toll_area, tax = 0)
        
        error1 = np.nansum(np.abs(((1 - gdf_here["transport_mode"]) * gdf_here["pop"]) - (gdf_here["share_car"] * gdf_here["pop"])))
        print(f"x = {x}, error_mode_by_tract = {error1}") #Error on transport mode by census tract

        gdf_here = gdf_here.loc[~np.isnan(gdf_here.share_car),:]
        error2 = np.nansum(gdf_here["pop"]) * np.abs((np.nansum(gdf_here.share_car * gdf_here["pop"]) / np.nansum(gdf_here["pop"])) - (np.nansum((1 - gdf_here["transport_mode"]) * gdf_here["pop"]) / np.nansum(gdf_here["pop"])))
        print(f"x = {x}, error_mode_AMB = {error2}") #Error on transport mode by census tract = {error2}") #Error on transport mode at AMB level
        
        #error3:avg wage
        estimated_wage = np.nansum(gdf_here["pop"] * gdf_here["wage"]) / np.nansum(gdf_here["pop"])
        error3 = np.abs(Y-estimated_wage)
        print(f"x = {x}, error_wage = {error3}") #Error on transport mode by census tract = {error2}") #Error on transport mode at AMB level
        
        #error4:ppl per employment center
        employed_results = employed_results.merge(employment_centers, left_index = True, right_on = "cluster")
        error4= np.nansum(np.abs(employed_results.employed - employed_results.employment)) / 2
        print(f"x = {x}, error_employment = {error4}") #Error on transport mode by census tract = {error2}") #Error on transport mode at AMB level
        
        return error1 + error2 + error3 + error4

    solving_transport = scipy.optimize.minimize(compute_error_transport_poly, x0=[150, 200] + (np.ones(len(np.unique(employment_centers.cluster))) * Y).tolist(), method='L-BFGS-B', bounds=[(0, 300), (0, 400)]+ [(0, 10000)] * len(np.unique(employment_centers.cluster)))
    FIXED_COST_CAR = solving_transport.x[0]
    LAMBDA = solving_transport.x[1]
    ARRAY_WAGE = solving_transport.x[2:]
    return gdf, FIXED_COST_CAR, LAMBDA, ARRAY_WAGE

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