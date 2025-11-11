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

def compute_transport_cost_poly_i(gdf, travel_time_car, travel_time_transit, PRICE_TIME, WORKING_DAYS, FIXED_COST_CAR, PRICE_FUEL, LAMBDA, ARRAY_WAGE_LOW, ARRAY_WAGE_MED, ARRAY_WAGE_HIGH, jobs_in_toll_area, houses_in_toll_area, tax, income_levels, wage_factors):
    """ Compute the transport cost and modes, assuming that people choose the transport mode that minimize the cost """
    
    travel_time_car["zone_tax"] = 1 * (travel_time_car.from_id.isin(houses_in_toll_area) | travel_time_car.to_id.isin(jobs_in_toll_area))
    
    # Compute car and transit costs
    for level in income_levels:
        factor = wage_factors[level]
        travel_time_car[f"COST_CAR_{level}"] = (
            (travel_time_car["travel_time"] / 60) * PRICE_TIME * factor * WORKING_DAYS
            + (travel_time_car["distance_car"] / 1000) * PRICE_FUEL * WORKING_DAYS
            + FIXED_COST_CAR
            + tax * WORKING_DAYS * travel_time_car["zone_tax"]
        )
        travel_time_transit[f"COST_PT_{level}"] = (
            (travel_time_transit["travel_time"] / 60) * PRICE_TIME * factor * WORKING_DAYS
            + travel_time_transit["monthly_cost_transit"]
        )

        # Fill missing costs
        travel_time_car[f"COST_CAR_{level}"].fillna(600, inplace=True)
        travel_time_transit[f"COST_PT_{level}"].fillna(3500, inplace=True)
    
    travel_matrix = travel_time_car.loc[:,['from_id', 'to_id', "COST_CAR_LOW", "COST_CAR_MED", "COST_CAR_HIGH", "distance_car"]].merge(travel_time_transit.loc[:,['from_id', 'to_id', "COST_PT_LOW", "COST_PT_MED", "COST_PT_HIGH"]], on = ['from_id', 'to_id'])
    
    for level in income_levels:
        travel_matrix[f"transport_mode_{level}"] = 1 / (1 + np.exp(
            (travel_matrix[f"COST_PT_{level}"] - travel_matrix[f"COST_CAR_{level}"]) / LAMBDA
        ))
        travel_matrix[f"transport_cost_{level}"] = (
            travel_matrix[f"transport_mode_{level}"] * travel_matrix[f"COST_PT_{level}"]
            + (1 - travel_matrix[f"transport_mode_{level}"]) * travel_matrix[f"COST_CAR_{level}"]
        )

    wages = {
        "LOW": pd.DataFrame(ARRAY_WAGE_LOW, index=np.sort(np.unique(travel_matrix.to_id)), columns=["income_LOW"]),
        "MED": pd.DataFrame(ARRAY_WAGE_MED, index=np.sort(np.unique(travel_matrix.to_id)), columns=["income_MED"]),
        "HIGH": pd.DataFrame(ARRAY_WAGE_HIGH, index=np.sort(np.unique(travel_matrix.to_id)), columns=["income_HIGH"])
    }
    for level in income_levels:
        travel_matrix = travel_matrix.merge(wages[level], left_on="to_id", right_index=True)

    for level in income_levels:
        travel_matrix[f"num_proba_center_{level}"] = np.exp(
            (travel_matrix[f"income_{level}"] - travel_matrix[f"transport_cost_{level}"]) / 140
        )
        travel_matrix[f"proba_center_{level}"] = travel_matrix[f"num_proba_center_{level}"] / \
            travel_matrix.groupby('from_id')[f"num_proba_center_{level}"].transform('sum')
        
        travel_matrix[f"transport_mode_proba_center_{level}"] = travel_matrix[f"transport_mode_{level}"] * travel_matrix[f"proba_center_{level}"]
        travel_matrix[f"transport_cost_proba_center_{level}"] = travel_matrix[f"transport_cost_{level}"] * travel_matrix[f"proba_center_{level}"]
        travel_matrix[f"wage_{level}"] = travel_matrix[f"income_{level}"] * travel_matrix[f"proba_center_{level}"]

    travel_results = {}
    for level in income_levels:
        df = travel_matrix.groupby("from_id")[[f"transport_mode_proba_center_{level}", f"transport_cost_proba_center_{level}", f"wage_{level}"]].sum()
        df.columns = [f"transport_mode_{level}", f"transport_cost_{level}", f"wage_{level}"]
        df[f"income_net_of_transport_cost_{level}"] = df[f"wage_{level}"] - df[f"transport_cost_{level}"]
        travel_results[level] = df

    gdf = gdf.drop(columns=[col for col in ["from_id", "transport_mode_LOW", "transport_cost_LOW", "wage_LOW", "income_net_of_transport_cost_LOW", "transport_mode_MED", "transport_cost_MED", "wage_MED", "income_net_of_transport_cost_MED", "transport_mode_HIGH", "transport_cost_HIGH", "wage_HIGH", "income_net_of_transport_cost_HIGH"] if col in gdf.columns])
    for level in income_levels:
        gdf = gdf.merge(travel_results[level], left_on="ID", right_index=True)

    travel_matrix = travel_matrix.merge(gdf[["ID", "pop_LOW", "pop_MED", "pop_HIGH"]], left_on="from_id", right_on="ID")
    for level in income_levels:
        travel_matrix[f"employed_{level}"] = travel_matrix[f"pop_{level}"] * travel_matrix[f"proba_center_{level}"]

    workers_per_cluster = travel_matrix.groupby("to_id")[[f"employed_{level}" for level in income_levels]].sum()

    # Clean up
    drop_cols = [f"transport_mode_proba_center_{level}" for level in income_levels] + \
                [f"transport_cost_proba_center_{level}" for level in income_levels]
    travel_matrix.drop(columns=drop_cols, inplace=True)

    return gdf, workers_per_cluster, travel_matrix

def compute_error_in_population_old(u, amen, N, BETA, Y_LOW, Y_MED, Y_HIGH, transport_cost_LOW, transport_cost_MED, transport_cost_HIGH, B, KAPPA, SIGMA, RHO, L, option_function = "CES", resid_rent = 0, resid_density = 0, resid_size = 0):
    '''
    Compute the difference between the population estimated by the model if 
    the utility is equal to u and the actual population.

        Parameters:
            N (float): Actual population
            u (float): Utility
                        
        Returns:
            error_population (float): Difference between the estimated and actual population
        '''

    print("u", u)
    R_LOW = compute_rents(BETA, Y_LOW, u[0]/amen, transport_cost_LOW)
    R_MED = compute_rents(BETA, Y_MED, u[1]/amen, transport_cost_MED)
    R_HIGH = compute_rents(BETA, Y_HIGH, u[2]/amen, transport_cost_HIGH)

    LOW_here = (R_LOW >= R_MED) & (R_LOW >= R_HIGH)
    MED_here = (R_MED >= R_LOW) & (R_MED >= R_HIGH)
    HIGH_here = (R_HIGH >= R_MED) & (R_HIGH >= R_LOW)

    R = R_HIGH.copy()
    R.loc[LOW_here] = R_LOW.loc[LOW_here]
    R.loc[MED_here] = R_MED.loc[MED_here]

    avg_wage = Y_HIGH.copy()
    avg_wage.loc[LOW_here] = Y_LOW.loc[LOW_here]
    avg_wage.loc[MED_here] = Y_MED.loc[MED_here]

    avg_t_cost = transport_cost_HIGH.copy()
    avg_t_cost.loc[LOW_here] = transport_cost_LOW.loc[LOW_here]
    avg_t_cost.loc[MED_here] = transport_cost_MED.loc[MED_here]

    q = compute_dwelling_size(BETA, avg_wage, avg_t_cost, R)
    n = compute_population(B, KAPPA, SIGMA, R, RHO, L, q, option_function = option_function)
    
    n = n * np.exp(resid_density)
    
    error_population_LOW = np.abs(N[0] - np.nansum(n[LOW_here]))
    print("LOW", np.nansum(n[LOW_here]))
    error_population_MED = np.abs(N[1] - np.nansum(n[MED_here]))
    print("MED", np.nansum(n[MED_here]))
    error_population_HIGH = np.abs(N[2] - np.nansum(n[HIGH_here]))
    print("HIGH", np.nansum(n[HIGH_here]))
    return error_population_LOW + error_population_MED + error_population_HIGH


import numpy as np

def compute_error_in_population(u, amen, N, BETA, Y_LOW, Y_MED, Y_HIGH,
                                transport_cost_LOW, transport_cost_MED, transport_cost_HIGH,
                                B, KAPPA, SIGMA, RHO, L, alpha,
                                option_function="CES", resid_rent=0, resid_density=0, resid_size=0,):
    """
    Compute smooth squared error between model-estimated and observed populations
    given a trial utility vector u = [u_low, base_rent, u_high].
    Uses softmax weights to smooth spatial class assignment.
    """

    # --- Compute rents ---
    R_LOW = compute_rents(BETA, Y_LOW, u[0] / amen, transport_cost_LOW)
    R_MED = compute_rents(BETA, Y_MED, u[1] / amen, transport_cost_MED)
    R_HIGH = compute_rents(BETA, Y_HIGH, u[2] / amen, transport_cost_HIGH)

    # --- Soft assignment using a differentiable "softmax" on rents ---
    # Higher rent => more likely to dominate
    # --- Normalize rents to avoid overflow ---
    # Bring rents to roughly mean-zero, unit-scale before exponentiation
    R_stack = np.vstack([R_LOW, R_MED, R_HIGH])
    R_mean = np.nanmean(R_stack)
    R_std = np.nanstd(R_stack) + 1e-9  # prevent division by zero

    R_LOW_n = (R_LOW - R_mean) / R_std
    R_MED_n = (R_MED - R_mean) / R_std
    R_HIGH_n = (R_HIGH - R_mean) / R_std

    # --- Soft assignment (numerically stable softmax) ---
    # subtract max to avoid overflow
    R_max = np.maximum.reduce([R_LOW_n, R_MED_n, R_HIGH_n])
    exp_LOW = np.exp(alpha * (R_LOW_n - R_max))
    exp_MED = np.exp(alpha * (R_MED_n - R_max))
    exp_HIGH = np.exp(alpha * (R_HIGH_n - R_max))
    denom = exp_LOW + exp_MED + exp_HIGH

    w_LOW = exp_LOW / denom
    w_MED = exp_MED / denom
    w_HIGH = exp_HIGH / denom

    # --- Weighted averages of rent, wage, and transport cost ---
    R = w_LOW * R_LOW + w_MED * R_MED + w_HIGH * R_HIGH
    
    #avg_wage = w_LOW * Y_LOW + w_MED * Y_MED + w_HIGH * Y_HIGH
    #avg_t_cost = w_LOW * transport_cost_LOW + w_MED * transport_cost_MED + w_HIGH * transport_cost_HIGH

    # --- Compute dwelling size and population ---
    q_LOW = compute_dwelling_size(BETA, Y_LOW, transport_cost_LOW, R)
    q_MED = compute_dwelling_size(BETA, Y_MED, transport_cost_MED, R)
    q_HIGH = compute_dwelling_size(BETA, Y_HIGH, transport_cost_HIGH, R)
    
    n = compute_population(B, KAPPA, SIGMA, R, RHO, L, w_LOW * q_LOW + w_MED * q_MED + w_HIGH * q_HIGH, option_function=option_function)
    
    n = n * np.exp(resid_density)

    # --- Smooth population shares ---
    pop_LOW_model = np.nansum(w_LOW * n)
    pop_MED_model = np.nansum(w_MED * n)
    pop_HIGH_model = np.nansum(w_HIGH * n)

    print(f"u={u}, LOW={pop_LOW_model:.2f}, MED={pop_MED_model:.2f}, HIGH={pop_HIGH_model:.2f}")

    # --- Squared errors ---
    error_population_LOW = (N[0] - pop_LOW_model) ** 2
    error_population_MED = (N[1] - pop_MED_model) ** 2
    error_population_HIGH = (N[2] - pop_HIGH_model) ** 2

    return error_population_LOW + error_population_MED + error_population_HIGH

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