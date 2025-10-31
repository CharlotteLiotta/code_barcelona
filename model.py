import numpy as np # type: ignore
import pandas as pd

from model import *

def compute_transport_cost(gdf, PRICE_TIME, WORKING_DAYS, FIXED_COST_CAR, PRICE_FUEL, tax):
    """ Compute the transport cost and modes, assuming that people choose the transport mode that minimize the cost """
    
    gdf["COST_CAR"] = ((gdf["travel_time_car"] / 60) * PRICE_TIME * WORKING_DAYS) + ((gdf.distance_car / 1000) * PRICE_FUEL * WORKING_DAYS) + FIXED_COST_CAR + (tax * WORKING_DAYS)
    gdf["COST_PT"] = ((gdf["travel_time_transit"] / 60) * PRICE_TIME * WORKING_DAYS) + gdf["monthly_cost_transit"]

    stacked = np.vstack([gdf["COST_CAR"], gdf["COST_PT"]])  # Shape (2, N)
    masked = np.where(np.isnan(stacked), np.inf, stacked)
    choice = np.argmin(masked, axis=0)

    gdf["transport_cost"] = np.fmin(gdf["COST_CAR"], gdf["COST_PT"])
    gdf["transport_mode"] = choice
    
    #print("Transport cost: ", sum(np.isnan(gdf["transport_cost"])), "missing values")
    gdf.loc[np.isnan(gdf["transport_cost"]), "transport_cost"] = 120
    gdf.loc[np.isnan(gdf["transport_mode"]), "transport_mode"] = 0

    return gdf

def compute_transport_cost_logit(gdf, Y, PRICE_TIME, WORKING_DAYS, FIXED_COST_CAR, PRICE_FUEL, LAMBDA, tax):
    """ Compute the transport cost and modes, assuming that people choose the transport mode that minimize the cost """
    
    gdf["COST_CAR"] = ((gdf["travel_time_car"] / 60) * PRICE_TIME * WORKING_DAYS) + ((gdf.distance_car / 1000) * PRICE_FUEL * WORKING_DAYS) + FIXED_COST_CAR + (tax * WORKING_DAYS)
    gdf["COST_PT"] = ((gdf["travel_time_transit"] / 60) * PRICE_TIME * WORKING_DAYS) + gdf["monthly_cost_transit"]

    gdf.loc[np.isnan(gdf["COST_CAR"]), "COST_CAR"] = 600
    gdf.loc[np.isnan(gdf["COST_PT"]), "COST_PT"] = 3500
    #stacked = np.vstack([gdf["COST_CAR"], gdf["COST_PT"]])  # Shape (2, N)
    #masked = np.where(np.isnan(stacked), np.inf, stacked)
    #choice = np.argmin(masked, axis=0)

    #gdf["transport_cost"] = np.fmin(gdf["COST_CAR"], gdf["COST_PT"])
    
    
    gdf["transport_mode"] = 1 / (1 + np.exp((gdf["COST_PT"] - gdf["COST_CAR"])/LAMBDA))
    gdf["transport_cost"] = (gdf["transport_mode"] * gdf["COST_PT"]) + ((1 - gdf["transport_mode"]) * gdf["COST_CAR"])
    #print("Transport cost: ", sum(np.isnan(gdf["transport_cost"])), "missing values")
    gdf.loc[np.isnan(gdf["transport_cost"]), "transport_cost"] = gdf["COST_CAR"]
    gdf.loc[np.isnan(gdf["transport_mode"]), "transport_mode"] = 0
    gdf["income_net_of_transport_cost"] = Y - gdf["transport_cost"]

    return gdf

def compute_transport_cost_poly_i(gdf, travel_time_matrix_car, travel_time_matrix_transit, PRICE_TIME, WORKING_DAYS, FIXED_COST_CAR, PRICE_FUEL, LAMBDA, ARRAY_WAGE_LOW, ARRAY_WAGE_MED, ARRAY_WAGE_HIGH, jobs_in_toll_area, houses_in_toll_area, tax):
    """ Compute the transport cost and modes, assuming that people choose the transport mode that minimize the cost """
    
    travel_time_matrix_car["zone_tax"] = 1 * (travel_time_matrix_car.from_id.isin(houses_in_toll_area) | travel_time_matrix_car.to_id.isin(jobs_in_toll_area))
    
    travel_time_matrix_car["COST_CAR_LOW"] = ((travel_time_matrix_car["travel_time"] / 60) * PRICE_TIME * 0.6 * WORKING_DAYS) + ((travel_time_matrix_car.distance_car / 1000) * PRICE_FUEL * WORKING_DAYS) + FIXED_COST_CAR + (tax * WORKING_DAYS * travel_time_matrix_car["zone_tax"])
    travel_time_matrix_transit["COST_PT_LOW"] = ((travel_time_matrix_transit["travel_time"] / 60) * PRICE_TIME* 0.6 * WORKING_DAYS) + travel_time_matrix_transit["monthly_cost_transit"]

    travel_time_matrix_car["COST_CAR_MED"] = ((travel_time_matrix_car["travel_time"] / 60) * PRICE_TIME * WORKING_DAYS) + ((travel_time_matrix_car.distance_car / 1000) * PRICE_FUEL * WORKING_DAYS) + FIXED_COST_CAR + (tax * WORKING_DAYS * travel_time_matrix_car["zone_tax"])
    travel_time_matrix_transit["COST_PT_MED"] = ((travel_time_matrix_transit["travel_time"] / 60) * PRICE_TIME * WORKING_DAYS) + travel_time_matrix_transit["monthly_cost_transit"]

    travel_time_matrix_car["COST_CAR_HIGH"] = ((travel_time_matrix_car["travel_time"] / 60) * PRICE_TIME * 1.4 * WORKING_DAYS) + ((travel_time_matrix_car.distance_car / 1000) * PRICE_FUEL * WORKING_DAYS) + FIXED_COST_CAR + (tax * WORKING_DAYS * travel_time_matrix_car["zone_tax"])
    travel_time_matrix_transit["COST_PT_HIGH"] = ((travel_time_matrix_transit["travel_time"] / 60) * PRICE_TIME * 1.4 * WORKING_DAYS) + travel_time_matrix_transit["monthly_cost_transit"]


    travel_time_matrix_car.loc[np.isnan(travel_time_matrix_car["COST_CAR_LOW"]), "COST_CAR_LOW"] = 600
    travel_time_matrix_transit.loc[np.isnan(travel_time_matrix_transit["COST_PT_LOW"]), "COST_PT_LOW"] = 3500
    
    travel_time_matrix_car.loc[np.isnan(travel_time_matrix_car["COST_CAR_MED"]), "COST_CAR_MED"] = 600
    travel_time_matrix_transit.loc[np.isnan(travel_time_matrix_transit["COST_PT_MED"]), "COST_PT_MED"] = 3500
    
    travel_time_matrix_car.loc[np.isnan(travel_time_matrix_car["COST_CAR_HIGH"]), "COST_CAR_HIGH"] = 600
    travel_time_matrix_transit.loc[np.isnan(travel_time_matrix_transit["COST_PT_HIGH"]), "COST_PT_HIGH"] = 3500
    
    travel_matrix = travel_time_matrix_car.loc[:,['from_id', 'to_id', "COST_CAR_LOW", "COST_CAR_MED", "COST_CAR_HIGH", "distance_car"]].merge(travel_time_matrix_transit.loc[:,['from_id', 'to_id', "COST_PT_LOW", "COST_PT_MED", "COST_PT_HIGH"]], on = ['from_id', 'to_id'])
    
    
    travel_matrix["transport_mode_LOW"] = 1 / (1 + np.exp((travel_matrix["COST_PT_LOW"] - travel_matrix["COST_CAR_LOW"])/LAMBDA))
    travel_matrix["transport_cost_LOW"] = (travel_matrix["transport_mode_LOW"] * travel_matrix["COST_PT_LOW"]) + ((1 - travel_matrix["transport_mode_LOW"]) * travel_matrix["COST_CAR_LOW"])
    
    travel_matrix["transport_mode_MED"] = 1 / (1 + np.exp((travel_matrix["COST_PT_MED"] - travel_matrix["COST_CAR_MED"])/LAMBDA))
    travel_matrix["transport_cost_MED"] = (travel_matrix["transport_mode_MED"] * travel_matrix["COST_PT_MED"]) + ((1 - travel_matrix["transport_mode_MED"]) * travel_matrix["COST_CAR_MED"])
    
    travel_matrix["transport_mode_HIGH"] = 1 / (1 + np.exp((travel_matrix["COST_PT_HIGH"] - travel_matrix["COST_CAR_HIGH"])/LAMBDA))
    travel_matrix["transport_cost_HIGH"] = (travel_matrix["transport_mode_HIGH"] * travel_matrix["COST_PT_HIGH"]) + ((1 - travel_matrix["transport_mode_HIGH"]) * travel_matrix["COST_CAR_HIGH"])
    #gdf.merge(travel_matrix.loc[travel_matrix.to_id == 0.0, ["from_id", "transport_mode"]], left_on = "ID", right_on = "from_id").plot("transport_mode", legend = True)
    
    ARRAY_WAGE_LOW = pd.DataFrame(ARRAY_WAGE_LOW, index = np.sort(np.unique(travel_matrix.to_id)))
    ARRAY_WAGE_LOW.columns = ["income_low"]

    ARRAY_WAGE_LOW = pd.DataFrame(ARRAY_WAGE_LOW, index = np.sort(np.unique(travel_matrix.to_id)))
    ARRAY_WAGE_LOW.columns = ["income_low"]

    ARRAY_WAGE_MED = pd.DataFrame(ARRAY_WAGE_MED, index = np.sort(np.unique(travel_matrix.to_id)))
    ARRAY_WAGE_MED.columns = ["income_med"]

    ARRAY_WAGE_HIGH = pd.DataFrame(ARRAY_WAGE_HIGH, index = np.sort(np.unique(travel_matrix.to_id)))
    ARRAY_WAGE_HIGH.columns = ["income_high"]


    travel_matrix = travel_matrix.merge(ARRAY_WAGE_LOW, left_on = "to_id", right_index = True)
    travel_matrix = travel_matrix.merge(ARRAY_WAGE_MED, left_on = "to_id", right_index = True)
    travel_matrix = travel_matrix.merge(ARRAY_WAGE_HIGH, left_on = "to_id", right_index = True)
    
    #eq2
    travel_matrix["num_proba_center_LOW"] = np.exp((travel_matrix["income_low"] - travel_matrix["transport_cost_LOW"]) / 140)
    travel_matrix["proba_center_LOW"] = travel_matrix['num_proba_center_LOW'] / travel_matrix.groupby('from_id')['num_proba_center_LOW'].transform('sum')
    
    travel_matrix["num_proba_center_MED"] = np.exp((travel_matrix["income_med"] - travel_matrix["transport_cost_MED"]) / 140)
    travel_matrix["proba_center_MED"] = travel_matrix['num_proba_center_MED'] / travel_matrix.groupby('from_id')['num_proba_center_MED'].transform('sum')
    
    travel_matrix["num_proba_center_HIGH"] = np.exp((travel_matrix["income_high"] - travel_matrix["transport_cost_HIGH"]) / 140)
    travel_matrix["proba_center_HIGH"] = travel_matrix['num_proba_center_HIGH'] / travel_matrix.groupby('from_id')['num_proba_center_HIGH'].transform('sum')
    
    #reuls
    travel_matrix["transport_mode_proba_center_LOW"] = travel_matrix["transport_mode_LOW"] * travel_matrix["proba_center_LOW"]
    travel_matrix["transport_cost_proba_center_LOW"] = travel_matrix["transport_cost_LOW"] * travel_matrix["proba_center_LOW"]
    travel_matrix["wage_LOW"] = travel_matrix["income_low"] * travel_matrix["proba_center_LOW"]

    travel_matrix["transport_mode_proba_center_MED"] = travel_matrix["transport_mode_MED"] * travel_matrix["proba_center_MED"]
    travel_matrix["transport_cost_proba_center_MED"] = travel_matrix["transport_cost_MED"] * travel_matrix["proba_center_MED"]
    travel_matrix["wage_MED"] = travel_matrix["income_med"] * travel_matrix["proba_center_MED"]

    travel_matrix["transport_mode_proba_center_HIGH"] = travel_matrix["transport_mode_HIGH"] * travel_matrix["proba_center_HIGH"]
    travel_matrix["transport_cost_proba_center_HIGH"] = travel_matrix["transport_cost_HIGH"] * travel_matrix["proba_center_HIGH"]
    travel_matrix["wage_HIGH"] = travel_matrix["income_high"] * travel_matrix["proba_center_HIGH"]

    travel_results_LOW = travel_matrix.loc[:, ["from_id", "transport_mode_proba_center_LOW", "transport_cost_proba_center_LOW", "wage_LOW"]].groupby("from_id").sum()
    travel_results_LOW.columns = ["transport_mode_LOW", "transport_cost_LOW", "wage_LOW"]
    travel_results_LOW["income_net_of_transport_cost_LOW"] = travel_results_LOW["wage_LOW"] - travel_results_LOW["transport_cost_LOW"]

    travel_results_MED = travel_matrix.loc[:, ["from_id", "transport_mode_proba_center_MED", "transport_cost_proba_center_MED", "wage_MED"]].groupby("from_id").sum()
    travel_results_MED.columns = ["transport_mode_MED", "transport_cost_MED", "wage_MED"]
    travel_results_MED["income_net_of_transport_cost_MED"] = travel_results_MED["wage_MED"] - travel_results_MED["transport_cost_MED"]

    travel_results_HIGH = travel_matrix.loc[:, ["from_id", "transport_mode_proba_center_HIGH", "transport_cost_proba_center_HIGH", "wage_HIGH"]].groupby("from_id").sum()
    travel_results_HIGH.columns = ["transport_mode_HIGH", "transport_cost_HIGH", "wage_HIGH"]
    travel_results_HIGH["income_net_of_transport_cost_HIGH"] = travel_results_HIGH["wage_HIGH"] - travel_results_HIGH["transport_cost_HIGH"]


    #gdf.loc[np.isnan(gdf["transport_cost"]), "transport_cost"] = gdf["COST_CAR"]
    #gdf.loc[np.isnan(gdf["transport_mode"]), "transport_mode"] = 0
    #gdf["income_net_of_transport_cost"] = Y - gdf["transport_cost"]
    gdf = gdf.drop(columns=[col for col in ["from_id", "transport_mode", "transport_cost", "wage", "income_net_of_transport_cost"] if col in gdf.columns])
    gdf = gdf.merge(travel_results_LOW, left_on = "ID", right_index = True)
    gdf = gdf.merge(travel_results_MED, left_on = "ID", right_index = True)
    gdf = gdf.merge(travel_results_HIGH, left_on = "ID", right_index = True)
    
    travel_matrix = travel_matrix.merge(gdf.loc[:,["ID", "pop_LOW", "pop_MED", "pop_HIGH"]], left_on = "from_id", right_on = "ID")
    travel_matrix["employed_LOW"] = travel_matrix["pop_LOW"] * travel_matrix["proba_center_LOW"]
    travel_matrix["employed_MED"] = travel_matrix["pop_MED"] * travel_matrix["proba_center_MED"]
    travel_matrix["employed_HIGH"] = travel_matrix["pop_HIGH"] * travel_matrix["proba_center_HIGH"]

    workers_per_cluster = travel_matrix.loc[:, ["to_id", "employed_LOW", "employed_MED", "employed_HIGH"]].groupby("to_id").sum()

    travel_matrix = travel_matrix.drop(columns = ["transport_mode_proba_center_LOW", "transport_cost_proba_center_LOW", "transport_mode_proba_center_MED", "transport_cost_proba_center_MED", "transport_mode_proba_center_HIGH", "transport_cost_proba_center_HIGH"])
    
    return gdf, workers_per_cluster, travel_matrix

def compute_error_in_population(u, N, BETA, Y, transport_cost, B, KAPPA, SIGMA, RHO, L, option_function = "CES", resid_rent = 0, resid_density = 0, resid_size = 0):
    '''
    Compute the difference between the population estimated by the model if 
    the utility is equal to u and the actual population.

        Parameters:
            N (float): Actual population
            u (float): Utility
                        
        Returns:
            error_population (float): Difference between the estimated and actual population
        '''

    R = compute_rents(BETA, Y, u, transport_cost)
    q = compute_dwelling_size(BETA, Y, transport_cost, R)
    n = compute_population(B, KAPPA, SIGMA, R, RHO, L, q, option_function = option_function)

    R = R * np.exp(resid_rent)
    q = q * np.exp(resid_size)
    n = n * np.exp(resid_density)
    #print("Estimated_population", np.nansum(n))
    #print("Error", N - np.nansum(n))
    error_population = np.abs(N - np.nansum(n))
    return error_population


def compute_rents(beta, Y, u, T):
    '''
    Compute the bid-rent at each location in the city.

            Parameters:
                    beta (float): Parameter of the utility function
                    Y (float): Average income
                    u (float): Average utility
                    T (array): Transportation costs at each location

            Returns:
                    R (array): Bid-rent at each location
    '''
    alpha = 1-beta
    Ro = (((alpha ** alpha) * (beta ** beta) * Y) / u) ** (1/beta)
    R = Ro * ((1 - (T/Y)) ** (1/beta))
    return R


def compute_dwelling_size(beta, Y, T, R):
    '''
    Compute the dwelling size at each location in the city.

            Parameters:
                    beta (float): Parameter of the utility function
                    Y (float): Average income
                    T (array): Transportation costs at each location
                    R (array): Rents at each location

            Returns:
                    q (array): Dwelling size at each location
    '''

    q = beta * (Y - T) / R
    return q

def compute_population(b, kappa, sigma, R, rho, L, q, option_function = "CES"):
    '''
    Compute the population at each location in the city.

            Parameters:
                    b (float): Parameter of the housing supply function
                    kappa (float): Parameter of the housing supply function
                    R (array): Rents at each location
                    rho (array): Interest rate
                    L (array): Land available for housing at each location
                    q (array): Dwelling size at each location

            Returns:
                    q (array): Dwelling size at each location
    '''

    a = 1-b

    if option_function == "Cobb-Douglas":
        n = kappa ** (1/a) * (b * R / rho) ** (b/a) * L / q

    elif option_function == "CES": 
        h = CES_func(R, kappa, a, sigma)
        n = h * L / q

    return n

def CES_func(x, kappa, a, sigma):
    return kappa * (a ** (-sigma / (1 - sigma))) * ((1 - ((1 - a) ** sigma) * ((kappa * x) ** (sigma - 1))) ** (sigma / (1 - sigma)))