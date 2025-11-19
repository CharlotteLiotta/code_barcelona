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
    return relative_change_utility

def compute_score(relative_change, param):
    return (1 / (1 + np.exp(param * relative_change)))

def compute_relative_change(outcome_without_tax, outcome_with_tax):
    """ Compute the impact of the change in utility on welfare"""

    return (outcome_with_tax - outcome_without_tax) / outcome_without_tax

def compute_vkm(gdf, travel_matrix, indiv_loc_matrix, income_levels):
    
    for lvl in income_levels:
        travel_matrix[f"distance_emi_{lvl}"] = (
        (travel_matrix["distance_car"] / 1000)
        * travel_matrix[f"proba_center_{lvl}"]
        * (1 - travel_matrix[f"transport_mode_{lvl}"])
        )
        distance_emi = travel_matrix.groupby("from_id", observed=True)[f"distance_emi_{lvl}"].sum()
        if f"distance_emi_{lvl}" in gdf.columns:
            gdf = gdf.drop(columns = f"distance_emi_{lvl}")
        gdf = gdf.merge(distance_emi, left_on="ID", right_index=True, how="left")

    vkm = np.nansum(sum(np.nansum(indiv_loc_matrix[lvl], 0) * gdf[f"distance_emi_{lvl}"] for lvl in income_levels))
    #avg_speed = np.nansum((((travel_matrix["distance_car"] /1000) * travel_matrix["proba_center_LOW"] * (1 - travel_matrix["transport_mode_LOW"])) * travel_matrix["speed"])) / np.nansum(((travel_matrix["distance_car"] /1000) * travel_matrix["proba_center_LOW"] * (1 - travel_matrix["transport_mode_LOW"])))
    
    return vkm

def compute_nb_trips(indiv_loc_matrix, gdf, travel_matrix, income_levels, clusters_in_zone, house_in_zone):
    
    for inc in income_levels:
        gdf[f"population_here_{inc}"] = np.nansum(indiv_loc_matrix[inc], axis=0)

    travel = travel_matrix.merge(
        gdf[["ID"] + [f"population_here_{inc}" for inc in income_levels]],
        left_on="from_id", right_on="ID"
    )

    # total and car commuters by income
    for inc in income_levels:
        travel[f"total_commuters_{inc}"] = (
            travel[f"population_here_{inc}"] * travel[f"proba_center_{inc}"]
        )
        travel[f"car_commuters_{inc}"] = (
            travel[f"total_commuters_{inc}"] * (1 - travel[f"transport_mode_{inc}"])
        )

    travel["car_commuters"] = travel[[f"car_commuters_{inc}" for inc in income_levels]].sum(axis=1)

    car_users_working_in_zone = travel.loc[
        travel.to_id.isin(clusters_in_zone), "car_commuters"
    ].sum(skipna=True)

    car_users_living_or_working_in_zone = travel.loc[
        travel.from_id.isin(house_in_zone) | travel.to_id.isin(clusters_in_zone),
        "car_commuters"
    ].sum(skipna=True)
    
    return car_users_working_in_zone, car_users_living_or_working_in_zone

