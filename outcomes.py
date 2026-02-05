import numpy as np # type: ignore
import pandas as pd
from openpyxl import load_workbook
import os

def compute_utility_manually(Y_net, R, q, BETA, amenities):
    """ Compute agents' utility """
    
    composite_good = Y_net - q * R
    composite_good[composite_good < 0] = 0
    u = (composite_good) ** (1 - BETA) * q ** BETA * amenities
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
    relative_change_utility = 100 * ((utility_with_tax - utility_without_tax) / utility_without_tax)
    #relative_change_utility[((utility_without_tax == 0) & (utility_with_tax > 0))] = np.nan #1
    #relative_change_utility[((utility_without_tax == 0) & (utility_with_tax == 0))] = np.nan
    #relative_change_utility[((utility_without_tax > 0) & (utility_with_tax == 0))] = np.nan


    #print("relative_change_utility", 100 * relative_change_utility, "%")
    return relative_change_utility

def relative_change_outputs(save_emissions, qol_in_zone, qol_out_zone, save_utility, compute_relative_change, compute_change_in_welfare, MAX_YEAR):
    
    emission_change = np.empty(MAX_YEAR)
    change_qol_in_zone = np.empty(MAX_YEAR)
    change_qol_out_zone = np.empty(MAX_YEAR)
    utility_change_low = np.empty(MAX_YEAR)
    utility_change_med = np.empty(MAX_YEAR)
    utility_change_high = np.empty(MAX_YEAR)

    for i in range(MAX_YEAR):
        emission_change[i] = np.nanmedian(compute_relative_change(save_emissions[0], save_emissions[i]))
        change_qol_in_zone[i] = np.nanmean(compute_relative_change(qol_in_zone[0], qol_in_zone[i]))
        change_qol_out_zone[i] = np.nanmean(compute_relative_change(qol_out_zone[0], qol_out_zone[i]))
        utility_change_low[i] = np.nanmedian(compute_change_in_welfare(save_utility["LOW"][:, 0], save_utility["LOW"][:, i]))
        utility_change_med[i] = np.nanmedian(compute_change_in_welfare(save_utility["MED"][:, 0], save_utility["MED"][:, i]))
        utility_change_high[i] = np.nanmedian(compute_change_in_welfare(save_utility["HIGH"][:, 0], save_utility["HIGH"][:, i]))

    return emission_change, change_qol_in_zone, change_qol_out_zone, utility_change_low, utility_change_med, utility_change_high

def append_scenario_to_excel(filename, scenario_name,
                             tax_level, emission_change,
                             change_qol_in_zone, change_qol_out_zone,
                             utility_change_low, utility_change_med, utility_change_high):

    # Put variables into a dict
    data = {
        "tax_level": tax_level,
        "emission_change": emission_change,
        "change_qol_in_zone": change_qol_in_zone,
        "change_qol_out_zone": change_qol_out_zone,
        "utility_change_low": utility_change_low,
        "utility_change_med": utility_change_med,
        "utility_change_high": utility_change_high,
    }

    # Convert to DataFrame (7 rows, 20 columns)
    df = pd.DataFrame(data).T
    df.columns = [f"t{t}" for t in range(1, len(tax_level)+1)]
    df.insert(0, "scenario", scenario_name)

    # Append blank row after each scenario
    df_blank = pd.DataFrame([[""] + [""]*len(tax_level)], columns=df.columns)

    # Write or append
    if not os.path.exists(filename):
        # First write
        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            df.to_excel(writer, index=True)
            df_blank.to_excel(writer, index=False, header=False, startrow=len(df)+1)
    else:
        # Append
        wb = load_workbook(filename)
        ws = wb.active
        startrow = ws.max_row + 1

        with pd.ExcelWriter(filename, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
            df.to_excel(writer, index=True, header=False, startrow=startrow)
            df_blank.to_excel(writer, index=False, header=False, startrow=startrow + len(df))

    print(f"Scenario '{scenario_name}' appended to {filename}.")


def compute_score(relative_change, param):
    return (1 / (1 + np.exp(param * relative_change)))

def compute_relative_change(outcome_without_tax, outcome_with_tax):
    """ Compute the impact of the change in utility on welfare"""

    return 100 * (outcome_with_tax - outcome_without_tax) / outcome_without_tax

def compute_emissions(gdf, travel_matrix, indiv_loc_matrix, income_levels, jobs_in_toll_area, houses_in_toll_area, tax, WORKING_DAYS, discount, SCALE_ABM):
    
    for lvl in income_levels:

        #distance
        travel_matrix[f"distance_emi_{lvl}"] = (
        (travel_matrix["distance_car"] / 1000)
        * travel_matrix[f"proba_center_{lvl}"]
        * (1 - travel_matrix[f"transport_mode_{lvl}"])
        )

        distance_emi = travel_matrix.groupby("from_id", observed=True)[f"distance_emi_{lvl}"].sum()
        if f"distance_emi_{lvl}" in gdf.columns:
            gdf = gdf.drop(columns = f"distance_emi_{lvl}")
        gdf = gdf.merge(distance_emi, left_on="ID", right_index=True, how="left")

        #tax revenues
        travel_matrix[f"tax_revenues_{lvl}"] = (
        travel_matrix[f"proba_center_{lvl}"]
        * (1 - travel_matrix[f"transport_mode_{lvl}"])
        )
        travel_matrix.loc[((~travel_matrix.from_id.isin(houses_in_toll_area)) & (~travel_matrix.to_id.isin(jobs_in_toll_area))), f"tax_revenues_{lvl}"] = 0
        tax_revenues = travel_matrix.groupby("from_id", observed=True)[f"tax_revenues_{lvl}"].sum()
        if f"tax_revenues_{lvl}" in gdf.columns:
            gdf = gdf.drop(columns = f"tax_revenues_{lvl}")
        gdf = gdf.merge(tax_revenues, left_on="ID", right_index=True, how="left")

        #subventions
        travel_matrix[f"subvention_{lvl}"] = (
        travel_matrix[f"proba_center_{lvl}"]
        * (travel_matrix[f"transport_mode_{lvl}"])
        )
        #travel_matrix.loc[((~travel_matrix.from_id.isin(houses_in_toll_area)) & (~travel_matrix.to_id.isin(jobs_in_toll_area))), f"tax_revenues_{lvl}"] = 0
        subvention = travel_matrix.groupby("from_id", observed=True)[f"subvention_{lvl}"].sum()
        if f"subvention_{lvl}" in gdf.columns:
            gdf = gdf.drop(columns = f"subvention_{lvl}")
        gdf = gdf.merge(subvention, left_on="ID", right_index=True, how="left")


        #transit users
        travel_matrix[f"transit_{lvl}"] = (
        travel_matrix[f"proba_center_{lvl}"]
        * (travel_matrix[f"transport_mode_{lvl}"])
        )
        #travel_matrix.loc[((~travel_matrix.from_id.isin(houses_in_toll_area)) & (~travel_matrix.to_id.isin(jobs_in_toll_area))), f"tax_revenues_{lvl}"] = 0
        transit_users = travel_matrix.groupby("from_id", observed=True)[f"transit_{lvl}"].sum()
        if f"transit_{lvl}" in gdf.columns:
            gdf = gdf.drop(columns = f"transit_{lvl}")
        gdf = gdf.merge(transit_users, left_on="ID", right_index=True, how="left")


        #speed
        travel_matrix[f"avg_speed_{lvl}"] = (
        (travel_matrix["speed"])
        * travel_matrix[f"proba_center_{lvl}"]
        * (1 - travel_matrix[f"transport_mode_{lvl}"])
        )

        travel_matrix[f"speed_weight_{lvl}"] = (
        travel_matrix[f"proba_center_{lvl}"]
        * (1 - travel_matrix[f"transport_mode_{lvl}"])
        )

        avg_speed = travel_matrix.groupby("from_id", observed=True)[f"avg_speed_{lvl}"].sum() / travel_matrix.groupby("from_id", observed=True)[f"speed_weight_{lvl}"].sum()
        avg_speed = avg_speed.to_frame(name=f"avg_speed_{lvl}")
        avg_speed.loc[avg_speed[f"avg_speed_{lvl}"] == 0, f"avg_speed_{lvl}"] = 20

        if f"avg_speed_{lvl}" in gdf.columns:
            gdf = gdf.drop(columns = f"avg_speed_{lvl}")
        gdf = gdf.merge(avg_speed, left_on="ID", right_index=True, how="left")

        gdf[f"emissions_{lvl}"] = (3400.373/gdf[f"avg_speed_{lvl}"] + 116.443) * gdf[f"distance_emi_{lvl}"] #source: https://doi.org/10.1016/j.trip.2025.101513, https://doi.org/10.1038/s41598-025-86119-3
        
    emissions = np.nansum(sum(np.nansum(indiv_loc_matrix[lvl], 0) * gdf[f"emissions_{lvl}"] for lvl in income_levels))
    total_vkm = np.nansum(sum(np.nansum(indiv_loc_matrix[lvl], 0) * gdf[f"distance_emi_{lvl}"] for lvl in income_levels))
    total_vkm_lvl = {inc: np.nansum(np.nansum(indiv_loc_matrix[inc], axis=0) * gdf[f"distance_emi_{inc}"]) for inc in income_levels}
    tax_revenues = np.nansum(sum(np.nansum(indiv_loc_matrix[lvl], 0) * gdf[f"tax_revenues_{lvl}"] for lvl in income_levels)) * tax * WORKING_DAYS * 12 * (1/SCALE_ABM)
    subvention = np.nansum(sum(np.nansum(indiv_loc_matrix[lvl], 0) * gdf[f"subvention_{lvl}"] for lvl in income_levels)) * discount * 12 * (1/SCALE_ABM)
    transit_users = np.nansum(sum(np.nansum(indiv_loc_matrix[lvl], 0) * gdf[f"transit_{lvl}"] for lvl in income_levels)) * (1/SCALE_ABM)

    return emissions, total_vkm, tax_revenues, subvention, transit_users, total_vkm_lvl

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


def compute_qol_congestion(gdf, travel_matrix, indiv_loc_matrix, income_levels):
    
    for lvl in income_levels:

        #vkm in zone
        travel_matrix[f"distance_in_zone_{lvl}"] = (
        (travel_matrix["distance_in_zone"] / 1000)
        * travel_matrix[f"proba_center_{lvl}"]
        * (1 - travel_matrix[f"transport_mode_{lvl}"])
        )
        distance_in_zone = travel_matrix.groupby("from_id", observed=True)[f"distance_in_zone_{lvl}"].sum()
        if f"distance_in_zone_{lvl}" in gdf.columns:
            gdf = gdf.drop(columns = f"distance_in_zone_{lvl}")
        gdf = gdf.merge(distance_in_zone, left_on="ID", right_index=True, how="left")

        #vkm out zone
        travel_matrix[f"distance_out_zone_{lvl}"] = (
        (travel_matrix["distance_out_zone"] / 1000)
        * travel_matrix[f"proba_center_{lvl}"]
        * (1 - travel_matrix[f"transport_mode_{lvl}"])
        )
        distance_out_zone = travel_matrix.groupby("from_id", observed=True)[f"distance_out_zone_{lvl}"].sum()
        if f"distance_out_zone_{lvl}" in gdf.columns:
            gdf = gdf.drop(columns = f"distance_out_zone_{lvl}")
        gdf = gdf.merge(distance_out_zone, left_on="ID", right_index=True, how="left")

        #speed
        travel_matrix[f"avg_speed_{lvl}"] = (
        (travel_matrix["speed"])
        * travel_matrix[f"proba_center_{lvl}"]
        * (1 - travel_matrix[f"transport_mode_{lvl}"])
        )

        travel_matrix[f"speed_weight_{lvl}"] = (
        travel_matrix[f"proba_center_{lvl}"]
        * (1 - travel_matrix[f"transport_mode_{lvl}"])
        )

        avg_speed = travel_matrix.groupby("from_id", observed=True)[f"avg_speed_{lvl}"].sum() / travel_matrix.groupby("from_id", observed=True)[f"speed_weight_{lvl}"].sum()
        avg_speed = avg_speed.to_frame(name=f"avg_speed_{lvl}")
        avg_speed.loc[avg_speed[f"avg_speed_{lvl}"] == 0, f"avg_speed_{lvl}"] = 20

        if f"avg_speed_{lvl}" in gdf.columns:
            gdf = gdf.drop(columns = f"avg_speed_{lvl}")
        gdf = gdf.merge(avg_speed, left_on="ID", right_index=True, how="left")

        gdf[f"qol_in_zone{lvl}"] = ((0.809827 + 0.0214 * gdf[f"avg_speed_{lvl}"] + 0.000268 * gdf[f"avg_speed_{lvl}"]**2 + 1.67e-7 * gdf[f"avg_speed_{lvl}"]**3 + 1.2e-9 * gdf[f"avg_speed_{lvl}"]**4)/gdf[f"avg_speed_{lvl}"]) * gdf[f"distance_in_zone_{lvl}"] #source: HBEFA Traffic Situations  Application guidelines
        gdf[f"qol_out_zone{lvl}"] = ((0.809827 + 0.0214 * gdf[f"avg_speed_{lvl}"] + 0.000268 * gdf[f"avg_speed_{lvl}"]**2 + 1.67e-7 * gdf[f"avg_speed_{lvl}"]**3 + 1.2e-9 * gdf[f"avg_speed_{lvl}"]**4)/gdf[f"avg_speed_{lvl}"]) * gdf[f"distance_out_zone_{lvl}"] #HBEFA Traffic Situations  Application guidelines
        
        #gdf[f"emissions_{lvl}"] = (3400.373/gdf[f"avg_speed_{lvl}"] + 116.443) * gdf[f"distance_emi_{lvl}"] #source: https://doi.org/10.1016/j.trip.2025.101513, https://doi.org/10.1038/s41598-025-86119-3
        
    qol_in_zone = np.nansum(sum(np.nansum(indiv_loc_matrix[lvl], 0) * gdf[f"qol_in_zone{lvl}"] for lvl in income_levels))
    qol_out_zone = np.nansum(sum(np.nansum(indiv_loc_matrix[lvl], 0) * gdf[f"qol_out_zone{lvl}"] for lvl in income_levels))
    vkm_in_zone = np.nansum(sum(np.nansum(indiv_loc_matrix[lvl], 0) * gdf[f"distance_in_zone_{lvl}"] for lvl in income_levels))
    vkm_out_zone = np.nansum(sum(np.nansum(indiv_loc_matrix[lvl], 0) * gdf[f"distance_out_zone_{lvl}"] for lvl in income_levels))
    vkm_in_zone_lvl = {inc: np.nansum(np.nansum(indiv_loc_matrix[inc], axis=0) * gdf[f"distance_in_zone_{inc}"]) for inc in income_levels}
    vkm_out_zone_lvl = {inc: np.nansum(np.nansum(indiv_loc_matrix[inc], axis=0) * gdf[f"distance_out_zone_{inc}"]) for inc in income_levels}
    
    return qol_in_zone, qol_out_zone, vkm_in_zone, vkm_out_zone, vkm_in_zone_lvl, vkm_out_zone_lvl

