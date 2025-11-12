import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from numba import njit, prange # type: ignore

def compute_utility_manually(Y, T, q, R, BETA, OPTION_HEALTH = 0, N = 0, vkm = 0, marginal_cost_pollution = 0):
    """ Compute agents' utility """
    
    if OPTION_HEALTH == 0:
        composite_good = Y - T - q * R
        composite_good[composite_good < 0] = 0
        u = (composite_good) ** (1 - BETA) * q ** BETA
    elif OPTION_HEALTH == 1:
        health = vkm * marginal_cost_pollution / N
        u = (Y - T - q * R - health) ** (1 - BETA) * q ** BETA
    return u

def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array = array + 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1,array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))


def compute_change_in_welfare(utility_without_tax, utility_with_tax):
    """ Compute the impact of the change in utility on welfare"""
    
    relative_change_utility = np.empty(len(utility_with_tax))
    relative_change_utility[utility_without_tax > 0] = ((utility_with_tax[utility_without_tax > 0] - utility_without_tax[utility_without_tax > 0]) / utility_without_tax[utility_without_tax > 0])
    relative_change_utility[((utility_without_tax == 0) & (utility_with_tax > 0))] = 1
    relative_change_utility[((utility_without_tax == 0) & (utility_with_tax == 0))] = 0

    #print("relative_change_utility", 100 * relative_change_utility, "%")
    return (1 / (1 + np.exp(-5 * relative_change_utility)))

def compute_change_in_qol(utility_without_tax, utility_with_tax):
    """ Compute the impact of the change in utility on welfare"""

    relative_change_utility = (utility_with_tax - utility_without_tax) / utility_without_tax
    #print("relative_change_qol", 100 * relative_change_utility, "%")
    return (1 / (1 + np.exp(3 * relative_change_utility)))

def compute_change_in_inequalities(utility_without_tax, utility_with_tax):
    print("change gini", (gini(np.array(utility_with_tax)) - gini(np.array(utility_without_tax))))
    if gini(np.array(utility_without_tax)) != gini(np.array(utility_with_tax)):
        Q = (1 / (1 + np.exp(10 * (gini(np.array(utility_with_tax)) - gini(np.array(utility_without_tax)))))) #/ ))) #gini(np.array(utility_without_tax)))))
    else:
        Q = 0.5

    return Q

def compute_change_in_emissions(gdf, travel_matrix, emissions_init, n_with_tax_LOW, n_with_tax_MED,n_with_tax_HIGH):
    travel_matrix["distance_emi_LOW"] = (travel_matrix["distance_car"] /1000) * travel_matrix["proba_center_LOW"] * (1 - travel_matrix["transport_mode_LOW"])
    distance_emi_LOW = travel_matrix.loc[:,["distance_emi_LOW", "from_id"]].groupby("from_id").sum()
    gdf = gdf.drop(columns = "distance_emi_LOW")
    gdf = gdf.merge(distance_emi_LOW, left_on = "ID", right_index = True)
    
    travel_matrix["distance_emi_MED"] = (travel_matrix["distance_car"] /1000) * travel_matrix["proba_center_MED"] * (1 - travel_matrix["transport_mode_MED"])
    distance_emi_MED = travel_matrix.loc[:,["distance_emi_MED", "from_id"]].groupby("from_id").sum()
    gdf = gdf.drop(columns = "distance_emi_MED")
    gdf = gdf.merge(distance_emi_MED, left_on = "ID", right_index = True)
    
    travel_matrix["distance_emi_HIGH"] = (travel_matrix["distance_car"] /1000) * travel_matrix["proba_center_HIGH"] * (1 - travel_matrix["transport_mode_HIGH"])
    distance_emi_HIGH = travel_matrix.loc[:,["distance_emi_HIGH", "from_id"]].groupby("from_id").sum()
    gdf = gdf.drop(columns = "distance_emi_HIGH")
    gdf = gdf.merge(distance_emi_HIGH, left_on = "ID", right_index = True)
    
    
    emissions = sum(n_with_tax_LOW * (gdf["distance_emi_LOW"])) + sum(n_with_tax_MED * (gdf["distance_emi_MED"])) + sum(n_with_tax_HIGH * (gdf["distance_emi_HIGH"]))
    relative_change_emission = (emissions - emissions_init) / emissions_init
    #print("relative_change_emission", 100 * relative_change_emission, "%")
    return (1 / (1 + np.exp(3 * relative_change_emission))), emissions

def compute_qol(save_population_LOW, save_population_MED, save_population_HIGH, gdf, travel_matrix, clusters_in_zone, house_in_zone):
    population_here_LOW = save_population_LOW
    population_here_MED = save_population_MED
    population_here_HIGH = save_population_HIGH

    travel = travel_matrix
    gdf["population_here_LOW"] = population_here_LOW
    gdf["population_here_MED"] = population_here_MED
    gdf["population_here_HIGH"] = population_here_HIGH

    travel = travel.merge(gdf.loc[:,["population_here_LOW", "population_here_MED","population_here_HIGH","ID"]], left_on = "from_id", right_on = "ID")
    travel["total_commuters_LOW"] = travel["population_here_LOW"] * travel["proba_center_LOW"]
    travel["total_commuters_MED"] = travel["population_here_MED"] * travel["proba_center_MED"]
    travel["total_commuters_HIGH"] = travel["population_here_HIGH"] * travel["proba_center_HIGH"]
    
    travel["car_commuters_LOW"] = travel["total_commuters_LOW"] * (1 - travel["transport_mode_LOW"])
    travel["car_commuters_MED"] = travel["total_commuters_MED"] * (1 - travel["transport_mode_MED"])
    travel["car_commuters_HIGH"] = travel["total_commuters_HIGH"] * (1 - travel["transport_mode_HIGH"])
    travel["car_commuters"] = travel["car_commuters_LOW"] + travel["car_commuters_MED"] + travel["car_commuters_HIGH"]

    #working in zone
    car_users_working_in_zone = np.nansum(travel.loc[travel.to_id.isin(clusters_in_zone),["car_commuters"]])
    
    #living or working in zone
    car_users_living_or_working_in_zone = np.nansum(travel.loc[travel.from_id.isin(house_in_zone) | travel.to_id.isin(clusters_in_zone),["car_commuters"]])

    #proba to commute in the tax zone by car
    ppl_commuting_in_tax_zone_by_car_LOW = travel.loc[travel.to_id.isin(clusters_in_zone),["car_commuters_LOW", "from_id"]].groupby("from_id").sum()
    ppl = travel.loc[:,["total_commuters_LOW", "from_id"]].groupby("from_id").sum()
    proba_commuting_in_tax_zone_by_car_LOW = np.array(ppl_commuting_in_tax_zone_by_car_LOW.car_commuters_LOW / ppl.total_commuters_LOW)

    ppl_commuting_in_tax_zone_by_car_MED = travel.loc[travel.to_id.isin(clusters_in_zone),["car_commuters_MED", "from_id"]].groupby("from_id").sum()
    ppl = travel.loc[:,["total_commuters_MED", "from_id"]].groupby("from_id").sum()
    proba_commuting_in_tax_zone_by_car_MED = np.array(ppl_commuting_in_tax_zone_by_car_MED.car_commuters_MED / ppl.total_commuters_MED)
    

    ppl_commuting_in_tax_zone_by_car_HIGH = travel.loc[travel.to_id.isin(clusters_in_zone),["car_commuters_HIGH", "from_id"]].groupby("from_id").sum()
    ppl = travel.loc[:,["total_commuters_HIGH", "from_id"]].groupby("from_id").sum()
    proba_commuting_in_tax_zone_by_car_HIGH = np.array(ppl_commuting_in_tax_zone_by_car_HIGH.car_commuters_HIGH / ppl.total_commuters_HIGH)
    
    
    return car_users_living_or_working_in_zone, car_users_working_in_zone, proba_commuting_in_tax_zone_by_car_LOW, proba_commuting_in_tax_zone_by_car_MED, proba_commuting_in_tax_zone_by_car_HIGH


