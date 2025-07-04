import numpy as np # type: ignore
import scipy # type: ignore
import statsmodels.api as sm # type: ignore

from model import *

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

def compute_cost_car_logit(gdf, import_trans_mode, PRICE_TIME, WORKING_DAYS, PRICE_FUEL, path_data):
    """ Calibrate the fixed cost of private car to match the transport modes data """

    trans_mode = import_trans_mode(path_data)

    gdf = gdf.merge(trans_mode.loc[:,["code_city", "share_car"]], on = "code_city", how = "left")
    
    def compute_error_transport(x):
        FIXED_COST_CAR = x[0]
        LAMBDA = x[1]
        gdf_here = compute_transport_cost_logit(gdf, PRICE_TIME, WORKING_DAYS, FIXED_COST_CAR, PRICE_FUEL, LAMBDA, tax = 0)
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

def calibrate_beta(gdf, Y):
    """ Calibrate BETA as the average share of income net of transport cost used for housing """

    gdf["income_net_of_transport_cost"] = Y - gdf["transport_cost"]
    gdf["rent_share"] = gdf["rent_m2"] * gdf["size"] / (gdf["income_net_of_transport_cost"])# * gdf["active_per_hh"])
    BETA = np.nansum(gdf["pop"][~np.isnan(gdf["rent_share"])] * gdf["rent_share"][~np.isnan(gdf["rent_share"])]) / np.nansum(gdf["pop"][~np.isnan(gdf["rent_share"])])
    return BETA

def calibrate_b_kappa(gdf, mask, RHO, option_calib = "housing"):
    """ Calibrate B and KAPPA using the rent data """

    gdf["log_n"] = np.log(gdf["pop"])
    gdf["log_R"] = np.log(gdf["rent_m2"])
    gdf["log_L"] = np.log(gdf["land"])
    gdf["log_q"] = np.log(gdf["size"])# / gdf["active_per_hh"])
    gdf["log_h"] = gdf["log_n"] + gdf["log_q"] - gdf["log_L"]
    gdf_here = gdf.loc[mask,:]

    if option_calib == "population":
        y = gdf_here["log_n"]
        X = gdf_here.loc[:,["log_R", "log_L", "log_q"]]
    elif option_calib == "housing":
        X = gdf_here.loc[:,["log_R",]]
        y = gdf_here["log_h"]

    X = sm.add_constant(X)  # Adds intercept
    model_statsmodel = sm.OLS(y, X).fit()
    print(model_statsmodel.summary())

    B = model_statsmodel.params["log_R"] / (1 + model_statsmodel.params["log_R"])
    KAPPA = np.exp(((1-B) * model_statsmodel.params["const"]) - (B * np.log(B/RHO)))
    print("B: ", B, "KAPPA: ", KAPPA)
    return B, KAPPA