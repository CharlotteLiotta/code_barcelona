import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from numba import njit, prange # type: ignore

def compute_utility_manually(Y, T, q, R, BETA, OPTION_HEALTH = 0, N = 0, vkm = 0, marginal_cost_pollution = 0):
    """ Compute agents' utility """
    
    if OPTION_HEALTH == 0:
        u = (Y - T - q * R) ** (1 - BETA) * q ** BETA
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

    relative_change_utility = (utility_with_tax - utility_without_tax) / utility_without_tax
    print("relative_change_utility", 100 * relative_change_utility, "%")
    return (1 / (1 + np.exp(-15 * relative_change_utility)))

def compute_change_in_qol(utility_without_tax, utility_with_tax):
    """ Compute the impact of the change in utility on welfare"""

    relative_change_utility = (utility_with_tax - utility_without_tax) / utility_without_tax
    print("relative_change_qol", 100 * relative_change_utility, "%")
    return (1 / (1 + np.exp(2 * relative_change_utility)))

def compute_change_in_inequalities(utility_without_tax, utility_with_tax):
    print("change gini", (gini(np.array(utility_with_tax)) - gini(np.array(utility_without_tax))))
    if gini(np.array(utility_without_tax)) != gini(np.array(utility_with_tax)):
        Q = (1 / (1 + np.exp(10 * (gini(np.array(utility_with_tax)) - gini(np.array(utility_without_tax)))))) #/ ))) #gini(np.array(utility_without_tax)))))
    else:
        Q = 0.5

    return Q

def compute_change_in_emissions(gdf, travel_matrix, emissions_init, n_with_tax):
    travel_matrix["distance_emi"] = (travel_matrix["distance_car"] /1000) * travel_matrix["proba_center"] * (1 - travel_matrix["transport_mode"])
    distance_emi = travel_matrix.loc[:,["distance_emi", "from_id"]].groupby("from_id").sum()
    gdf = gdf.drop(columns = "distance_emi")
    gdf = gdf.merge(distance_emi, left_on = "ID", right_index = True)
    emissions = sum(n_with_tax * (gdf["distance_emi"]))
    relative_change_emission = (emissions - emissions_init) / emissions_init
    print("relative_change_emission", 100 * relative_change_emission, "%")
    return (1 / (1 + np.exp(2 * relative_change_emission))), emissions

def compute_qol(save_population, gdf, travel_matrix, clusters_in_zone, house_in_zone):
    population_here = save_population

    travel = travel_matrix
    gdf["population_here"] = population_here

    travel = travel.merge(gdf.loc[:,["population_here", "ID"]], left_on = "from_id", right_on = "ID")
    travel["total_commuters"] = travel["population_here"] * travel["proba_center"]
    travel["car_commuters"] = travel["total_commuters"] * (1 - travel["transport_mode_save"])

    #working in zone
    car_users_working_in_zone = np.nansum(travel.loc[travel.to_id.isin(clusters_in_zone),["car_commuters"]])
    
    #living or working in zine
    car_users_living_or_working_in_zone = np.nansum(travel.loc[travel.from_id.isin(house_in_zone) | travel.to_id.isin(clusters_in_zone),["car_commuters"]])

    #proba to commute in the tax zone by car
    ppl_commuting_in_tax_zone_by_car = travel.loc[travel.to_id.isin(clusters_in_zone),["car_commuters", "from_id"]].groupby("from_id").sum()
    ppl = travel.loc[:,["total_commuters", "from_id"]].groupby("from_id").sum()
    proba_commuting_in_tax_zone_by_car = np.array(ppl_commuting_in_tax_zone_by_car.car_commuters / ppl.total_commuters)
    
    return car_users_living_or_working_in_zone, car_users_working_in_zone, proba_commuting_in_tax_zone_by_car


