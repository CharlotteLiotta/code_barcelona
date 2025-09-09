import numpy as np # type: ignore
import pandas as pd

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

def compute_transport_cost_poly(gdf, travel_time_matrix_car, travel_time_matrix_transit, PRICE_TIME, WORKING_DAYS, FIXED_COST_CAR, PRICE_FUEL, LAMBDA, ARRAY_INCOME, points_in_zone, house_in_zone, tax):
    """ Compute the transport cost and modes, assuming that people choose the transport mode that minimize the cost """
    
    travel_time_matrix_car["zone_tax"] = 1 * (travel_time_matrix_car.from_id.isin(house_in_zone) | travel_time_matrix_car.to_id.isin(points_in_zone))
    travel_time_matrix_car["COST_CAR"] = ((travel_time_matrix_car["travel_time"] / 60) * PRICE_TIME * WORKING_DAYS) + ((travel_time_matrix_car.distance_car / 1000) * PRICE_FUEL * WORKING_DAYS) + FIXED_COST_CAR + (tax * WORKING_DAYS * travel_time_matrix_car["zone_tax"])
    travel_time_matrix_transit["COST_PT"] = ((travel_time_matrix_transit["travel_time"] / 60) * PRICE_TIME * WORKING_DAYS) + travel_time_matrix_transit["monthly_cost_transit"]


    travel_time_matrix_car.loc[np.isnan(travel_time_matrix_car["COST_CAR"]), "COST_CAR"] = 600
    travel_time_matrix_transit.loc[np.isnan(travel_time_matrix_transit["COST_PT"]), "COST_PT"] = 3500
    
    travel_matrix = travel_time_matrix_car.loc[:,['from_id', 'to_id', "COST_CAR", "distance_car"]].merge(travel_time_matrix_transit.loc[:,['from_id', 'to_id', "COST_PT"]], on = ['from_id', 'to_id'])
    travel_matrix["transport_mode"] = 1 / (1 + np.exp((travel_matrix["COST_PT"] - travel_matrix["COST_CAR"])/LAMBDA))
    travel_matrix["transport_cost"] = (travel_matrix["transport_mode"] * travel_matrix["COST_PT"]) + ((1 - travel_matrix["transport_mode"]) * travel_matrix["COST_CAR"])
    #gdf.merge(travel_matrix.loc[travel_matrix.to_id == 0.0, ["from_id", "transport_mode"]], left_on = "ID", right_on = "from_id").plot("transport_mode", legend = True)
    
    ARRAY_INCOME = pd.DataFrame(ARRAY_INCOME, index = np.sort(np.unique(travel_matrix.to_id)))
    ARRAY_INCOME.columns = ["income"]
    travel_matrix = travel_matrix.merge(ARRAY_INCOME, left_on = "to_id", right_index = True)
    
    travel_matrix["num_proba_center"] = np.exp((travel_matrix["income"] - travel_matrix["transport_cost"]) / 140)
    travel_matrix["proba_center"] = travel_matrix['num_proba_center'] / travel_matrix.groupby('from_id')['num_proba_center'].transform('sum')
    travel_matrix["transport_mode_save"] = travel_matrix["transport_mode"]
    travel_matrix["transport_mode"] = travel_matrix["transport_mode"] * travel_matrix["proba_center"]
    travel_matrix["transport_cost"] = travel_matrix["transport_cost"] * travel_matrix["proba_center"]
    travel_matrix["wage"] = travel_matrix["income"] * travel_matrix["proba_center"]

    travel_results = travel_matrix.loc[:, ["from_id", "transport_mode", "transport_cost", "wage"]].groupby("from_id").sum()
    #gdf.loc[np.isnan(gdf["transport_cost"]), "transport_cost"] = gdf["COST_CAR"]
    #gdf.loc[np.isnan(gdf["transport_mode"]), "transport_mode"] = 0
    #gdf["income_net_of_transport_cost"] = Y - gdf["transport_cost"]
    gdf = gdf.drop(columns=[col for col in ["from_id", "transport_mode", "transport_cost", "wage"] if col in gdf.columns])
    gdf = gdf.merge(travel_results, left_on = "ID", right_index = True)
    
    travel_matrix = travel_matrix.merge(gdf.loc[:,["ID", "pop"]], left_on = "from_id", right_on = "ID")
    travel_matrix["employed"] = travel_matrix["pop"] * travel_matrix["proba_center"]

    employed_results = travel_matrix.loc[:, ["to_id", "employed"]].groupby("to_id").sum()
    
    return gdf, employed_results, travel_matrix

def compute_error_in_population(u, N, BETA, Y, transport_cost, B, KAPPA, RHO, L, resid_rent = 0, resid_density = 0, resid_size = 0):
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
    n = compute_population(B, KAPPA, R, RHO, L, q)

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

def compute_population(b, kappa, R, rho, L, q):
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
    n = kappa ** (1/a) * (b * R / rho) ** (b/a) * L / q
    return n