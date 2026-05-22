import numpy as np # type: ignore
import pandas as pd

from model import *

def compute_housing_supply(housing_supply_t1_without_inertia, housing_supply_t0, TIME_LAG, DEPRECIATION_TIME):
    diff_housing = ((housing_supply_t1_without_inertia - housing_supply_t0) / TIME_LAG) - (housing_supply_t0 / DEPRECIATION_TIME)
    for i in range(0, len(housing_supply_t1_without_inertia)):
        if housing_supply_t1_without_inertia[i] <= housing_supply_t0[i]:
            diff_housing[i] = - (housing_supply_t0[i] / DEPRECIATION_TIME)
    housing_supply_t1 = housing_supply_t0 + diff_housing
    return housing_supply_t1

def compute_transport_cost(gdf, travel_time_car, travel_time_transit, PRICE_TIME, WORKING_DAYS, FIXED_COST_CAR, PRICE_FUEL, LAMBDA, ARRAY_WAGE_LOW, ARRAY_WAGE_MED, ARRAY_WAGE_HIGH, jobs_in_toll_area, houses_in_toll_area, tax, income_levels, wage_factors, scenario, discount = 0, time_discount = 0):
    """ Compute the transport cost and modes, assuming that people choose the transport mode that minimize the cost """
    
    if scenario == "exemption_trips_inside_zone":
        travel_time_car["zone_tax"] = 1 * ((travel_time_car.from_id.isin(houses_in_toll_area) & (~travel_time_car.to_id.isin(jobs_in_toll_area))) | ((~travel_time_car.from_id.isin(houses_in_toll_area)) & travel_time_car.to_id.isin(jobs_in_toll_area)))
    elif scenario == "discount_residents":
        travel_time_car["zone_tax"] = 1 * ((~travel_time_car.from_id.isin(houses_in_toll_area)) & travel_time_car.to_id.isin(jobs_in_toll_area))
    else:
        travel_time_car["zone_tax"] = 1 * (travel_time_car.from_id.isin(houses_in_toll_area) | travel_time_car.to_id.isin(jobs_in_toll_area))
    
    # Compute car and transit costs
    for level in income_levels:
        factor = wage_factors[level]
        if ((level == "LOW") & (scenario == "discount_low_income")):
            travel_time_car.loc[:,f"COST_CAR_{level}"] = (
            (travel_time_car["travel_time"] / 60) * PRICE_TIME * factor * WORKING_DAYS
            + (travel_time_car["distance_car"] / 1000) * PRICE_FUEL * WORKING_DAYS
            + FIXED_COST_CAR
            #+ tax * WORKING_DAYS * travel_time_car["zone_tax"]
        )
        else:
            travel_time_car.loc[:,f"COST_CAR_{level}"] = (
                (travel_time_car["travel_time"] / 60) * PRICE_TIME * factor * WORKING_DAYS
                + (travel_time_car["distance_car"] / 1000) * PRICE_FUEL * WORKING_DAYS
                + FIXED_COST_CAR
                + tax * WORKING_DAYS * travel_time_car["zone_tax"]
            )
        travel_time_transit.loc[:,f"COST_PT_{level}"] = (
            (((travel_time_transit["travel_time"]) / 60)+time_discount).clip(lower = 0)  * PRICE_TIME * factor * WORKING_DAYS
            + (travel_time_transit["monthly_cost_transit"] - (discount * travel_time_transit["monthly_cost_transit"])).clip(lower=0)
        )

        print(np.nansum((((travel_time_transit["travel_time"]) / 60)+time_discount) == 0))
        print(np.nanmean((((travel_time_transit["travel_time"]) / 60)+time_discount)/(((travel_time_transit["travel_time"]) / 60))))

        # Fill missing costs
        travel_time_car[f"COST_CAR_{level}"] = travel_time_car[f"COST_CAR_{level}"].fillna(600)
        travel_time_transit[f"COST_PT_{level}"] = travel_time_transit[f"COST_PT_{level}"].fillna(3500)

    travel_matrix = travel_time_car.loc[:,['from_id', 'to_id', "COST_CAR_LOW", "COST_CAR_MED", "COST_CAR_HIGH", "distance_car", "distance_in_zone", "distance_out_zone", "speed"]].merge(travel_time_transit.loc[:,['from_id', 'to_id', "COST_PT_LOW", "COST_PT_MED", "COST_PT_HIGH"]], on = ['from_id', 'to_id'])
    
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

def compute_outcomes(amenity_array, utility, gdf, BETA, B, KAPPA, INTEREST_RATE, income_levels, compute_rents, compute_dwelling_size, option_housing_supply = False, housing_supply = 0, option_resid = False, resid_rent = 0, resid_density = 0, resid_size = 0):
     
    amenity_pref = {"LOW": amenity_array[0], "MED": amenity_array[1], "HIGH": amenity_array[2]}

    R_group = {
        lvl: compute_rents(BETA, gdf[f"wage_{lvl}"], utility[i]/(gdf["amenities"] ** amenity_pref[lvl]), gdf[f"transport_cost_{lvl}"])
        for i, lvl in enumerate(income_levels)
        }

    R = np.amax([R_group["LOW"], R_group["MED"], R_group["HIGH"]], 0)
    
    a = 1-B
    if option_housing_supply == False:
        h = KAPPA ** (1/a) * (B * R / INTEREST_RATE) ** (B/a) * gdf["urb_area"]
    elif option_housing_supply == True:
        h = housing_supply
        
    # --- Compute dwelling sizes ---
    q_group = {
        lvl: compute_dwelling_size(BETA, gdf[f"wage_{lvl}"], gdf[f"transport_cost_{lvl}"], R)
        for lvl in income_levels
        }
    
    q = q_group["LOW"]
    for i in range(len(q)):
        if np.amax([R_group["LOW"], R_group["MED"], R_group["HIGH"]], 0)[i] == R_group["MED"][i]:
            q[i] = q_group["MED"][i]
        if np.amax([R_group["LOW"], R_group["MED"], R_group["HIGH"]], 0)[i] == R_group["HIGH"][i]:
            q[i] = q_group["HIGH"][i]

    n = h / q
    
    w = {lvl: (np.amax([R_group["LOW"], R_group["MED"], R_group["HIGH"]], 0) == R_group[lvl]) * 1 for lvl in income_levels}

    if option_resid == True:
        R = R * np.exp(resid_rent)
        q = q * np.exp(resid_size)
        n_group = {lvl: n * w[lvl] * np.exp(resid_density) for lvl in income_levels}
        q_group = {lvl: q_group[lvl] * np.exp(resid_size) for lvl in income_levels}
        R_group = {lvl: R_group[lvl] * np.exp(resid_rent) for lvl in income_levels}
        n = np.nansum(list(n_group.values()), axis=0)
        n[np.isnan(n)] = 0
        for lvl in income_levels:
            n_group[lvl][np.isnan(n_group[lvl])] = 0
    
    else:
        n_group = {lvl: n * w[lvl] for lvl in income_levels}

    return R, q, n, w, R_group, q_group, n_group, h

def compute_error_in_population(amenity_array, u, income_levels, gdf, N, BETA,
                                B, KAPPA, RHO, option_resid, resid_rent=0, resid_density=0, resid_size=0, option_housing_supply = False, housing_supply = 0):
    """
    Compute smooth squared error between model-estimated and observed populations
    given a trial utility vector u = [u_low, base_rent, u_high].
    Uses softmax weights to smooth spatial class assignment.
    """

    _, _, _, _, _, _, n_group, _ = compute_outcomes(amenity_array, u, gdf, BETA, B, KAPPA, RHO, income_levels, compute_rents, compute_dwelling_size, option_housing_supply, housing_supply, option_resid, resid_rent, resid_density, resid_size)

    return np.array([
        np.nansum(n_group["LOW"]) - N[0],
        np.nansum(n_group["MED"]) - N[1],
        np.nansum(n_group["HIGH"]) - N[2]])
    
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

