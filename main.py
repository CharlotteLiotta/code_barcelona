import numpy as np 
from copy import deepcopy
from scipy.sparse import csr_matrix # type: ignore
import pickle
import datetime

from outcomes import *
from ABM import *
from import_data import * # type: ignore
from calibration import * # type: ignore
from model import * # type: ignore
from plotting_tools import * # type: ignore
from import_transport import *
from policy_support import *

### SCENARIOS

scenario = "baseline"
option_center = "catalunya" #test1, "catalunya", "UEA"

#scenario = "exemption_trips_inside_zone"    #DONE
#scenario = "tax_question_P17"               #DONE
#scenario = "increasing_knowledge"           #DONE
#scenario = "instant_welfare_adjust"         #DONE
#scenario = "discount_low_income"            #DONE
#scenario = "discount_residents"             #DONE

#scenario = "less_expensive_transport"       #DONE
#scenario = "improve_rodalies"               #DONE
#scenario = "reduce_transport_time"          #DONE

### IMPORT PARAMETERS

path_data = "../data_barcelona/"
option_function = "Cobb-Douglas"

LOGISTIC_PARAM_WELFARE = -0.3
LOGISTIC_PARAM_QOL = 0.3

#Time
year = 0
MAX_YEAR = 20

#Policy impact model
INTEREST_RATE = 0.05
PRICE_TIME = 10 #euros/h - later modulated by income level
WORKING_DAYS = 40 #20 days per month, with 2 trips per day
PRICE_FUEL = 0.11 #euros/km
center = "0801901025"
SOFT_RENT = 10 #50
DIFF_SPEED_CONGESTION = 10


#ABM
SCALE_ABM = 1/100 #Nb of agents in the ABM
PROBA_MOVE = 0.2
INERTIA_OPINION = 0.8 #inertia

### IMPORT DATA

#Import data
gdf = import_data(path_data, center, option = "SECTION") #"DISTRICT" or "SECTION" - Active population: 1.4M
gdf = import_jobs(gdf, path_data)
gdf = import_land_use(gdf, path_data)
gdf = import_ppl_per_hh(gdf, path_data)
gdf = import_rent_and_size(gdf, path_data)
Y_median, gdf = import_income_new(gdf, path_data)
gdf = import_amenities(gdf, path_data, 0, 0)
employment_centers = gpd.read_file(path_data + "cluster_employment_" + option_center + ".shp")
employment_centers["cluster"] = employment_centers.index
jobs_in_toll_area, houses_in_toll_area, zone_tax = import_tax_zone(gdf, employment_centers)
plot_base_map(gdf)

#Multiple income groups
income_levels = ["LOW", "MED", "HIGH"]
wage_factors = {"LOW": 0.6, "MED": 1.0, "HIGH": 1.4}
pop = {
    "LOW": np.nansum(gdf["pop"] * gdf["share_low_income"] / 100),
    "MED": np.nansum(gdf["pop"] * (100 - gdf["share_high_income"] - gdf["share_low_income"]) / 100),
    "HIGH": np.nansum(gdf["pop"] * gdf["share_high_income"] / 100),
}

#Prepare variables
gdf["size"] = gdf["size_census"] #gdf["size_census"] #gdf["size_AMB"]
gdf.loc[gdf["pop"] == 0, "pop"] = 1
gdf.loc[gdf["rent_m2"] == 0, "rent_m2"] = np.nanmin(gdf.loc[gdf["rent_m2"]>0, "rent_m2"])
gdf.loc[np.isnan(gdf["rent_m2"]), "rent_m2"] = np.nanmin(gdf.loc[gdf["rent_m2"]>0, "rent_m2"])
gdf.loc[np.isnan(gdf["size"]), "size"] = np.nansum(gdf.loc[~np.isnan(gdf["size"]), "size"] * gdf.loc[~np.isnan(gdf["size"]), "pop"]) / np.nansum(gdf.loc[~np.isnan(gdf["size"]), "pop"])

#Import transport data
#import_transport_times_poly(gdf, datetime.datetime(2025, 7, 15, 8, 0, 0), center, path_data, employment_centers, option_center) #datetime.datetime(2025, 7, 15, 8, 0, 0)
travel_time_matrix_car, travel_time_matrix_transit = load_transport_times_poly(gdf, path_data, center, option_center)
travel_time_matrix_car = load_distance_car_poly(travel_time_matrix_car, gdf, employment_centers, jobs_in_toll_area, houses_in_toll_area, zone_tax)
gdf = import_cost_transit(gdf)
travel_time_matrix_transit = travel_time_matrix_transit.merge(gdf[['ID', 'monthly_cost_transit']].rename(columns={'ID': 'from_id'}), on='from_id', how='left')

travel_time_matrix_car["uncongested_speed"] = (travel_time_matrix_car.distance_car / 1000) / (travel_time_matrix_car.travel_time / 60)
travel_time_matrix_car["speed"] = travel_time_matrix_car["uncongested_speed"] - DIFF_SPEED_CONGESTION
travel_time_matrix_car.loc[travel_time_matrix_car["speed"] < 10, "speed"] = 10
travel_time_matrix_car.travel_time = ((travel_time_matrix_car.distance_car / 1000) / travel_time_matrix_car["speed"]) * 60

### CALIBRATION POLICY SUPPORT

#BETA_OPINION, INITIAL_OPINION = import_opinion_parameters(path_data, scenario)
#BETA_PRICE, INITIAL_PRICE, INITIAL_CC, INITIAL_WELFARE, INITIAL_QOL, INITIAL_KNOWLEDGE = import_price_parameters(path_data, scenario)


### CALIBRATION POLICY IMPACT MODEL

#Calibration of B and KAPPA
gdf["land"] = gdf["urb_area"]

mask = ((gdf["rent_m2"] < 25) &(gdf["rent_m2"] > 7)&
        (gdf["pop"] > 484.0)&
        (~np.isnan(gdf["size"]))
        &(~np.isnan(gdf["pop"]))
        &((gdf["pop"] > 0))
        )

B, KAPPA, SIGMA = calibrate_b_kappa(gdf, mask, INTEREST_RATE, option_function, option_calib = "housing")

#Transport cost calibration
gdf, FIXED_COST_CAR, LAMBDA, ARRAY_WAGE_LOW, ARRAY_WAGE_MED, ARRAY_WAGE_HIGH = compute_cost_car_poly_i(gdf, Y_median, import_trans_mode, PRICE_TIME, WORKING_DAYS, PRICE_FUEL, travel_time_matrix_car, travel_time_matrix_transit, employment_centers, path_data, jobs_in_toll_area, houses_in_toll_area, income_levels, wage_factors, scenario, compute_error_transport_poly)
#with open(path_data + "calib_trans_poly_i" + option_center + ".pkl", "wb") as f: #UEA_v2
#    pickle.dump((FIXED_COST_CAR, LAMBDA, ARRAY_WAGE_LOW, ARRAY_WAGE_MED, ARRAY_WAGE_HIGH), f)
#with open(path_data + "calib_trans_poly_i_" + option_center + ".pkl", "rb") as f:
#    FIXED_COST_CAR, LAMBDA, ARRAY_WAGE_LOW, ARRAY_WAGE_MED, ARRAY_WAGE_HIGH = pickle.load(f)
print("FIXED_COST_CAR: ", FIXED_COST_CAR)
print("LAMBDA: ", LAMBDA)
compute_error_transport_poly([FIXED_COST_CAR, LAMBDA] + list(ARRAY_WAGE_LOW) + list(ARRAY_WAGE_MED) + list(ARRAY_WAGE_HIGH), employment_centers, gdf, travel_time_matrix_car, travel_time_matrix_transit, PRICE_TIME, WORKING_DAYS, PRICE_FUEL, jobs_in_toll_area, houses_in_toll_area, income_levels, wage_factors, scenario, Y_median, import_trans_mode, path_data)
del compute_transport_cost_logit, compute_cost_car_logit

#Compute transport cost
gdf, workers_per_cluster, travel_matrix = compute_transport_cost_poly_i(gdf, travel_time_matrix_car, travel_time_matrix_transit, PRICE_TIME, WORKING_DAYS, FIXED_COST_CAR, PRICE_FUEL, LAMBDA, ARRAY_WAGE_LOW, ARRAY_WAGE_MED, ARRAY_WAGE_HIGH, jobs_in_toll_area, houses_in_toll_area, 0, income_levels, wage_factors, scenario)
print("Modal share of public transport (%):", round(100 * sum(np.nansum(gdf[f"transport_mode_{lvl}"] * gdf[f"pop_{lvl}"]) for lvl in income_levels) / sum(np.nansum(gdf[f"pop_{lvl}"]) for lvl in income_levels)))
print("Modal share of public transport (%):", round(100 * (np.nansum(gdf["transport_mode_HIGH"] * gdf["pop_HIGH"])) / (np.nansum(gdf["pop_HIGH"]))))
print("Modal share of public transport (%):", round(100 * (np.nansum(gdf[f"transport_mode_MED"] * gdf[f"pop_MED"])) / (np.nansum(gdf[f"pop_MED"]))))
print("Modal share of public transport (%):", round(100 * (np.nansum(gdf[f"transport_mode_LOW"] * gdf[f"pop_LOW"])) / (np.nansum(gdf[f"pop_LOW"]))))
plot_transport_cost_i(gdf, "_HIGH")
plot_transport_mode_i(gdf, "_HIGH")
plot_employment(gdf, employment_centers, employment_centers['employment'] * 0.0015)

gdf["rodalies_500m_1km"] = ((gdf["min_distance_rodalies"] > 500) & (gdf["min_distance_rodalies"] < 1000)) * 1
gdf["fgc_500m_1km"] = ((gdf["min_distance_fgc"] > 500) & (gdf["min_distance_fgc"] < 1000)) * 1
gdf["rodalies_1km_2km"] = ((gdf["min_distance_rodalies"] > 1000) & (gdf["min_distance_rodalies"] < 2000)) * 1
gdf["fgc_1km_2km"] = ((gdf["min_distance_fgc"] > 1000) & (gdf["min_distance_fgc"] < 2000)) * 1

#Calibrate beta and amenities
def compute_log_likelihood(x):
    print(x)
    return calibration_utility_amenity3(x, gdf, income_levels, SOFT_RENT, 0, 0)

calib_beta = scipy.optimize.minimize(compute_log_likelihood, [0.44, 183, 256, 434], bounds=[(0.2,0.7), (0,None), (0,None), (0,None)]) #[0.45, 100, 500, 900] #[0.3, 224, 386, 553]
BETA = calib_beta.x[0]
print("BETA:", BETA)
gdf = gdf.drop(columns="amenities", errors="ignore")
amenities = calibration_utility_amenity3(calib_beta.x, gdf, income_levels, SOFT_RENT, 1, 1)
gdf = gdf.merge(amenities, on = "ID", how = "left")
gdf.loc[np.isnan(gdf["amenities"]), "amenities"] = 1

# Solve the model
def compute_error_in_population_from_utility(u):
    """ Compute error in population associated to utility u"""
    #print(u)
    #print(compute_error_in_population(u, gdf["amenities"], [pop[lvl] for lvl in income_levels], BETA, gdf["wage_LOW"], gdf["wage_MED"], gdf["wage_HIGH"], gdf["transport_cost_LOW"], gdf["transport_cost_MED"], gdf["transport_cost_HIGH"], B, KAPPA, SIGMA, INTEREST_RATE, gdf["urb_area"], SOFT_RENT, False, option_function))
    return sum(compute_error_in_population(u, gdf["amenities"], [pop[lvl] for lvl in income_levels], BETA, gdf["wage_LOW"], gdf["wage_MED"], gdf["wage_HIGH"], gdf["transport_cost_LOW"], gdf["transport_cost_MED"], gdf["transport_cost_HIGH"], B, KAPPA, SIGMA, INTEREST_RATE, gdf["urb_area"], SOFT_RENT, False, option_function)**2)

solving_model = scipy.optimize.minimize(compute_error_in_population_from_utility, calib_beta.x[1:4], bounds=[(0,None), (0,None), (0,None)], method = "Powell") #[161, 287, 411] #[219, 329, 471]
solving_model = scipy.optimize.minimize(compute_error_in_population_from_utility, solving_model.x, bounds=[(0,None), (0,None), (0,None)], method = "Nelder-Mead") #np.array([399,690,995]) np.array([370,690,800]) #[180, 420, 600] #150 270 427
print(solving_model)

if solving_model.fun < 20000000:
    R, q, n, w, R_group, q_group = compute_outcomes(solving_model.x, gdf, BETA, B, KAPPA, SIGMA, INTEREST_RATE, SOFT_RENT, income_levels, compute_rents, compute_dwelling_size, compute_population, option_function)
else:
    raise ValueError("Minimization failed!")

#share_col = {"LOW": "share_low_income", "MED": None, "HIGH": "share_high_income"}
#density_residual = {
#    lvl: np.log(
#        (gdf["pop"] * (gdf[share_col[lvl]] / 100 if share_col[lvl] else (100 - gdf["share_low_income"] - gdf["share_high_income"]) / 100))
#        / (w[lvl] * n)
#    )
#    for lvl in income_levels
#}
density_residual = np.log(gdf["pop"] / n)
rent_residual = np.log(gdf["rent_m2"] / R)
size_residual = np.log(gdf["size"] / q)

# Plot the result of the calibration
#print_maps(gdf, n, q, R)
plot_line_charts(gdf, n, q, R)

#initial state
def compute_error_in_population_from_utility(u):
    """ Compute error in population associated to utility u"""

    return sum((compute_error_in_population(u, gdf["amenities"], [pop[lvl] for lvl in income_levels], BETA, gdf["wage_LOW"], gdf["wage_MED"], gdf["wage_HIGH"], gdf["transport_cost_LOW"], gdf["transport_cost_MED"], gdf["transport_cost_HIGH"], B, KAPPA, SIGMA, INTEREST_RATE, gdf["urb_area"], SOFT_RENT, True, option_function, rent_residual, density_residual, size_residual))**2)

solving_model = scipy.optimize.minimize(compute_error_in_population_from_utility, [159.7, 283.9, 404.7], method = "Powell") #[159.7, 283.9, 404.7]

if solving_model.fun < 50000000:

    R, q, n, w, R_group, q_group = compute_outcomes(solving_model.x, gdf, BETA, B, KAPPA, SIGMA, INTEREST_RATE, SOFT_RENT, income_levels, compute_rents, compute_dwelling_size, compute_population, option_function)

    R = R * np.exp(rent_residual)
    q = q * np.exp(size_residual)
    n_group = {lvl: n * w[lvl] * np.exp(density_residual) for lvl in ["LOW", "MED", "HIGH"]}
    n = np.nansum(list(n_group.values()), axis=0)
    n[np.isnan(n)] = 0
    for lvl in income_levels:
        n_group[lvl][np.isnan(n_group[lvl])] = 0

else:
    raise ValueError("Minimization failed!")

plot_line_charts(gdf, n, q, R)

print(100 * np.nansum(n_group["LOW"].loc[gdf.ID.isin(houses_in_toll_area)]) / np.nansum(n_group["LOW"]))
print(100 * np.nansum(gdf.pop_LOW.loc[gdf.ID.isin(houses_in_toll_area)]) / np.nansum(n_group["LOW"]))

print(100 * np.nansum(n_group["MED"].loc[gdf.ID.isin(houses_in_toll_area)]) / np.nansum(n_group["MED"]))
print(100 * np.nansum(gdf.pop_MED.loc[gdf.ID.isin(houses_in_toll_area)]) / np.nansum(n_group["MED"]))

print(100 * np.nansum(n_group["HIGH"].loc[gdf.ID.isin(houses_in_toll_area)]) / np.nansum(n_group["HIGH"]))
print(100 * np.nansum(gdf.pop_HIGH.loc[gdf.ID.isin(houses_in_toll_area)]) / np.nansum(n_group["HIGH"]))




# ABM: translate outputs at the household level
N = {lvl: round(pop[lvl] * SCALE_ABM) for lvl in income_levels}
#support = {lvl: INITIAL_OPINION * np.ones(N[lvl]) for lvl in income_levels}
indiv_loc_matrix = {
    lvl: compute_indiv_loc_matrix2(
        N[lvl], len(gdf), (n_group[lvl] * SCALE_ABM).to_numpy()
    )
    for lvl in income_levels
}

rent_indiv = {lvl: indiv_loc_matrix[lvl] @ R_group[lvl].to_numpy() for lvl in income_levels}
dwelling_size_indiv = {lvl: indiv_loc_matrix[lvl] @ q_group[lvl] for lvl in income_levels}

utility = {}

for lvl in income_levels:
    u = compute_utility_manually(
        indiv_loc_matrix[lvl] @ gdf[f"wage_{lvl}"],
        indiv_loc_matrix[lvl] @ gdf[f"transport_cost_{lvl}"],
        dwelling_size_indiv[lvl],
        rent_indiv[lvl],
        BETA, indiv_loc_matrix[lvl] @ gdf["amenities"]
    )
    u[np.isnan(u)] = 0
    utility[lvl] = u

housing_indiv = {
    lvl: dwelling_size_indiv[lvl] @ csr_matrix(indiv_loc_matrix[lvl])
    for lvl in income_levels
}

# Save outputs
save_housing = {lvl: np.zeros((len(gdf), MAX_YEAR)) for lvl in income_levels}
save_rent = {lvl: np.zeros((N[lvl], MAX_YEAR)) for lvl in income_levels}
save_dwelling_size = {lvl: np.zeros((N[lvl], MAX_YEAR)) for lvl in income_levels}
save_transport_mode = {lvl: np.zeros((N[lvl], MAX_YEAR)) for lvl in income_levels}
save_utility = {lvl: np.zeros((N[lvl], MAX_YEAR)) for lvl in income_levels}

for lvl in income_levels:
    save_housing[lvl][:, 0] = deepcopy(housing_indiv[lvl])
    save_rent[lvl][:, 0] = deepcopy(rent_indiv[lvl])
    save_dwelling_size[lvl][:, 0] = deepcopy(dwelling_size_indiv[lvl])
    save_transport_mode[lvl][:, 0] = deepcopy(indiv_loc_matrix[lvl] @ gdf[f"transport_mode_{lvl}"])
    save_utility[lvl][:, 0] = deepcopy(utility[lvl])

save_population = np.zeros((len(gdf), MAX_YEAR))
save_population[:, 0] = np.sum([np.nansum(indiv_loc_matrix[lvl], 0) for lvl in income_levels], axis=0)
save_population_lvl = {lvl: np.zeros((len(gdf), MAX_YEAR)) for lvl in income_levels}
for lvl in income_levels:
    save_population_lvl[lvl][:, 0] = np.nansum(indiv_loc_matrix[lvl], 0)

save_tax = np.zeros(MAX_YEAR)
save_tax[0] = 0
save_emissions = np.zeros(MAX_YEAR)
save_score_welfare = {lvl: np.zeros((N[lvl], MAX_YEAR)) for lvl in income_levels}
save_score_qol = {lvl: np.zeros((N[lvl], MAX_YEAR)) for lvl in income_levels}
save_score_congestion = {lvl: np.zeros((N[lvl], MAX_YEAR)) for lvl in income_levels}
save_median_support = np.zeros(MAX_YEAR)
qol_in_zone = np.zeros(MAX_YEAR)
qol_out_zone = np.zeros(MAX_YEAR)
vkm_in_zone = np.zeros(MAX_YEAR)
vkm_out_zone = np.zeros(MAX_YEAR)
total_vkm = np.zeros(MAX_YEAR)
total_vkm_lvl = np.empty(MAX_YEAR, dtype=object)
vkm_in_zone_lvl = np.empty(MAX_YEAR, dtype=object)
vkm_out_zone_lvl = np.empty(MAX_YEAR, dtype=object)
mode_shares_lvl = {lvl: np.full(MAX_YEAR, np.nan) for lvl in income_levels}
avg_vkm_lvl = {lvl: np.full(MAX_YEAR, np.nan) for lvl in income_levels}
avg_vkm_in_zone_lvl = {lvl: np.full(MAX_YEAR, np.nan) for lvl in income_levels}
avg_vkm_out_zone_lvl = {lvl: np.full(MAX_YEAR, np.nan) for lvl in income_levels}
live_and_work_in_toll_lvl = {lvl: np.full(MAX_YEAR, np.nan) for lvl in income_levels}
live_out_and_work_in_toll_lvl = {lvl: np.full(MAX_YEAR, np.nan) for lvl in income_levels}
live_in_toll_and_work_out_lvl = {lvl: np.full(MAX_YEAR, np.nan) for lvl in income_levels}
live_out_and_work_out_lvl = {lvl: np.full(MAX_YEAR, np.nan) for lvl in income_levels}
avg_wage_lvl = {lvl: np.full(MAX_YEAR, np.nan) for lvl in income_levels}
avg_tcost_lvl = {lvl: np.full(MAX_YEAR, np.nan) for lvl in income_levels}
avg_rent_lvl = {lvl: np.full(MAX_YEAR, np.nan) for lvl in income_levels}
avg_dsize_lvl = {lvl: np.full(MAX_YEAR, np.nan) for lvl in income_levels}

tax_revenues = np.zeros(MAX_YEAR)

save_emissions[0], total_vkm[0], tax_revenues[0], _, _, total_vkm_lvl[0] = compute_emissions(gdf, travel_matrix, indiv_loc_matrix, income_levels, jobs_in_toll_area, houses_in_toll_area, 0, WORKING_DAYS, 0, SCALE_ABM)
qol_in_zone[0], qol_out_zone[0], vkm_in_zone[0], vkm_out_zone[0], vkm_in_zone_lvl[0], vkm_out_zone_lvl[0] = compute_qol_congestion(gdf, travel_matrix, indiv_loc_matrix, income_levels)

### DECOMPOSITION TRANSPORT

gdf["live_in_toll"] = (gdf.ID.isin(houses_in_toll_area))
for lvl in income_levels:
    s = (travel_matrix.loc[travel_matrix.to_id.isin(jobs_in_toll_area),
                      ["from_id", f"proba_center_{lvl}"]].groupby("from_id")[f"proba_center_{lvl}"].sum())
    gdf = gdf.merge(s, left_on="ID", right_on="from_id", how="left").rename(columns={f"proba_center_{lvl}": f"work_in_toll_{lvl}"}).fillna(0)

for lvl in income_levels:
    #mode share
    mode_shares_lvl[lvl][0] = np.nansum(gdf[f"transport_mode_{lvl}"] * np.nansum(indiv_loc_matrix[lvl], 0)) / np.nansum(np.nansum(indiv_loc_matrix[lvl], 0))
    #distances
    avg_vkm_lvl[lvl][0] = (total_vkm_lvl[0][lvl] / (np.nansum(gdf[f"transport_mode_{lvl}"] * np.nansum(indiv_loc_matrix[lvl], 0))))
    avg_vkm_in_zone_lvl[lvl][0] = vkm_in_zone_lvl[0][lvl] / (np.nansum(gdf[f"transport_mode_{lvl}"] * np.nansum(indiv_loc_matrix[lvl], 0)))
    avg_vkm_out_zone_lvl[lvl][0] = vkm_out_zone_lvl[0][lvl] / (np.nansum(gdf[f"transport_mode_{lvl}"] * np.nansum(indiv_loc_matrix[lvl], 0)))
    #repartition
    gdf[f"live_and_work_in_toll_{lvl}"] = (gdf["live_in_toll"] == True) * gdf[f"work_in_toll_{lvl}"]
    gdf[f"live_out_and_work_in_toll_{lvl}"] = (gdf["live_in_toll"] == False) * gdf[f"work_in_toll_{lvl}"]
    gdf[f"live_in_toll_and_work_out_{lvl}"] = (gdf["live_in_toll"] == True) * (1 - gdf[f"work_in_toll_{lvl}"])
    gdf[f"live_out_and_work_out_toll_{lvl}"] = (gdf["live_in_toll"] == False) * (1 - gdf[f"work_in_toll_{lvl}"])
    live_and_work_in_toll_lvl[lvl][0] = np.nansum(save_population_lvl[lvl][:,0] * gdf[f"live_and_work_in_toll_{lvl}"]) / N[lvl]
    live_out_and_work_in_toll_lvl[lvl][0] = np.nansum(save_population_lvl[lvl][:,0] * gdf[f"live_out_and_work_in_toll_{lvl}"]) / N[lvl]
    live_in_toll_and_work_out_lvl[lvl][0] = np.nansum(save_population_lvl[lvl][:,0] * gdf[f"live_in_toll_and_work_out_{lvl}"]) / N[lvl]
    live_out_and_work_out_lvl[lvl][0] = np.nansum(save_population_lvl[lvl][:,0] * gdf[f"live_out_and_work_out_toll_{lvl}"]) / N[lvl]
    #decompo util
    avg_wage_lvl[lvl][0] = np.nanmean(indiv_loc_matrix[lvl] @ gdf[f"wage_{lvl}"])
    avg_tcost_lvl[lvl][0] = np.nanmean(indiv_loc_matrix[lvl] @ gdf[f"transport_cost_{lvl}"])
    avg_rent_lvl[lvl][0] = np.nanmean(rent_indiv[lvl])
    avg_dsize_lvl[lvl][0] = np.nanmean(dwelling_size_indiv[lvl])
BETA_CONG = DIFF_SPEED_CONGESTION/total_vkm[0]

year = year + 1
#plot_distance_distrib_check(income_levels, travel_matrix, "HIGH")

tax_revenue_here = 137000000
subvention_here = 137000000
efficiency = 0
discount = 12.79346
time_discount = 0

### FIRST SIMULATION TO COMPUTE UTILITY LOSSES

gdf_init, _, _ = compute_transport_cost_poly_i(gdf, travel_time_matrix_car, travel_time_matrix_transit, PRICE_TIME, WORKING_DAYS, FIXED_COST_CAR, PRICE_FUEL, LAMBDA, ARRAY_WAGE_LOW, ARRAY_WAGE_MED, ARRAY_WAGE_HIGH, jobs_in_toll_area, houses_in_toll_area, 0, income_levels, wage_factors, "baseline", 0, 0)
 
utility_init = {lvl: compute_utility_manually(gdf_init[f"wage_{lvl}"],
                                              gdf_init[f"transport_cost_{lvl}"],
                                              q, #q_group[f"{lvl}"],
                                              R, #R_group[f"{lvl}"],
                                              BETA, gdf["amenities"]
                                              ) for lvl in income_levels}

gdf_tax, _, _ = compute_transport_cost_poly_i(gdf, travel_time_matrix_car, travel_time_matrix_transit, PRICE_TIME, WORKING_DAYS, FIXED_COST_CAR, PRICE_FUEL, LAMBDA, ARRAY_WAGE_LOW, ARRAY_WAGE_MED, ARRAY_WAGE_HIGH, jobs_in_toll_area, houses_in_toll_area, 0.5, income_levels, wage_factors, "baseline", 0, 0)
 
utility_tax = {lvl: compute_utility_manually(gdf_tax[f"wage_{lvl}"],
                                              gdf_tax[f"transport_cost_{lvl}"],
                                              q, #q_group[f"{lvl}"],
                                              R, #R_group[f"{lvl}"],
                                              BETA, gdf["amenities"]
                                              ) for lvl in income_levels}

for lvl in income_levels:
    gdf[f"score_welfare_{lvl.lower()}"] = compute_score(compute_change_in_welfare(utility_init[lvl], utility_tax[lvl]), LOGISTIC_PARAM_WELFARE)

expected_welfare_loss = gdf[["ID", "geometry"] + [f"score_welfare_{lvl.lower()}" for lvl in income_levels]]

### CALIBRATION POLICY SUPPORT

BETA_OPINION, BETA_PRICE, INITIAL_OPINION, INITIAL_PRICE, INITIAL_CC, INITIAL_WELFARE, INITIAL_QOL, INITIAL_KNOWLEDGE = import_opinion_parameters2(path_data, scenario, expected_welfare_loss)

tax = INITIAL_PRICE
acceptable_price = {"LOW": INITIAL_PRICE, "MED": INITIAL_PRICE, "HIGH": INITIAL_PRICE}
#support = {"LOW": INITIAL_OPINION, "MED": INITIAL_OPINION, "HIGH": INITIAL_OPINION}
save_score_emissions = np.zeros(MAX_YEAR)
save_score_emissions[0] = INITIAL_CC
save_knowledge = np.zeros(MAX_YEAR)
save_knowledge[0] = INITIAL_KNOWLEDGE
score_qol = {"LOW": INITIAL_QOL, "MED": INITIAL_QOL, "HIGH": INITIAL_QOL}
score_welfare = {"LOW": INITIAL_WELFARE, "MED": INITIAL_WELFARE, "HIGH": INITIAL_WELFARE}
support = {lvl: INITIAL_OPINION * np.ones(N[lvl]) for lvl in income_levels}

### MODELING THE PSC

while year < MAX_YEAR:

    print("YEAR", year)

    if ((scenario == "improved_rodalies") & (year == 6)):
        gdf = gdf.drop(columns = "amenities")

    save_knowledge[year] = save_knowledge[year-1] + 0.02

    # Urban form with the tax, without inertia
    
    TRIP_TO_ZONE_OUTPUT = total_vkm[year - 1]
    TRIP_TO_ZONE_INPUT = 0
    index_q = 0

    condition = True

    while condition == True:

        if scenario == "less_expensive_transport":
            discount = discount - (subvention_here-tax_revenue_here)/10000000
            time_discount = 0
        elif scenario == "reduce_transport_time":
            time_discount = time_discount + ((efficiency - 73) / 5000)
            discount = 0
        else:
            discount = 0
            time_discount = 0
        
        TRIP_TO_ZONE_INPUT = TRIP_TO_ZONE_OUTPUT.copy()
        index_q = index_q + 1
        travel_time_matrix_car["speed"] = travel_time_matrix_car["uncongested_speed"] - BETA_CONG * TRIP_TO_ZONE_INPUT
        travel_time_matrix_car.loc[travel_time_matrix_car["speed"] < 10, "speed"] = 10
        travel_time_matrix_car.travel_time = ((travel_time_matrix_car.distance_car / 1000) / travel_time_matrix_car["speed"]) * 60

        gdf, workers_per_cluster, travel_matrix = compute_transport_cost_poly_i(gdf, travel_time_matrix_car, travel_time_matrix_transit, PRICE_TIME, WORKING_DAYS, FIXED_COST_CAR, PRICE_FUEL, LAMBDA, ARRAY_WAGE_LOW, ARRAY_WAGE_MED, ARRAY_WAGE_HIGH, jobs_in_toll_area, houses_in_toll_area, tax, income_levels, wage_factors, scenario, discount, time_discount)

        def compute_error_in_population_from_utility(u):
            """ Compute error in population associated to utility u"""

            if ((scenario == "improved_rodalies") & (year > 5)):
                error_population = compute_error_in_population(u, gdf["amenities_improved_rodalies"], [pop[lvl] for lvl in income_levels], BETA, gdf["wage_LOW"], gdf["wage_MED"], gdf["wage_HIGH"], gdf["transport_cost_LOW"], gdf["transport_cost_MED"], gdf["transport_cost_HIGH"], B, KAPPA, SIGMA, INTEREST_RATE, gdf["urb_area"], SOFT_RENT, True, option_function, rent_residual, density_residual, size_residual)
            else:
                error_population = sum((compute_error_in_population(u, gdf["amenities"], [pop[lvl] for lvl in income_levels], BETA, gdf["wage_LOW"], gdf["wage_MED"], gdf["wage_HIGH"], gdf["transport_cost_LOW"], gdf["transport_cost_MED"], gdf["transport_cost_HIGH"], B, KAPPA, SIGMA, INTEREST_RATE, gdf["urb_area"], SOFT_RENT, True, option_function, rent_residual, density_residual, size_residual)) ** 2)
            return error_population

        solving_model = scipy.optimize.minimize(compute_error_in_population_from_utility, solving_model.x, method = "Nelder-Mead")

        if solving_model.fun < 50000000:

            R, q, n, w, R_group, q_group = compute_outcomes(solving_model.x, gdf, BETA, B, KAPPA, SIGMA, INTEREST_RATE, SOFT_RENT, income_levels, compute_rents, compute_dwelling_size, compute_population, option_function)

            R = R * np.exp(rent_residual)
            q = q * np.exp(size_residual)
            n_group = {lvl: n * w[lvl] * np.exp(density_residual) for lvl in ["LOW", "MED", "HIGH"]}
            n = np.nansum(list(n_group.values()), axis=0)
            n[np.isnan(n)] = 0
            for lvl in income_levels:
                n_group[lvl][np.isnan(n_group[lvl])] = 0

        else:
            raise ValueError("Minimization failed!")

        housing_without_inertia = n * q

        # AMB
        has_moved = {}
        indiv_loc_matrix_new = {}

        for lvl in income_levels:
            #proba_from, proba_to = compute_proba_of_moving(
            #    save_housing[lvl][:, year - 1],
            #    (housing_without_inertia * SCALE_ABM * w[lvl]).to_numpy()
            #    )

            #proba_from, proba_to = compute_proba_of_moving(
            #    np.nansum(indiv_loc_matrix[lvl], 0), #save_housing[lvl][:, year - 1],
            #    (n_group[lvl] * SCALE_ABM).to_numpy() #(n_group[lvl] * q_group[lvl] * SCALE_ABM).to_numpy()
            #    )
            
            proba_from, proba_to = compute_proba_of_moving(
                np.nansum(indiv_loc_matrix[lvl], 0), #save_housing[lvl][:, year - 1],
                (n_group[lvl] * SCALE_ABM).to_numpy() #(n_group[lvl] * q_group[lvl] * SCALE_ABM).to_numpy()
                )
            
            #proba_from, proba_to = compute_proba_of_moving(
            #    save_housing[lvl][:, year - 1],
            #    (n_group[lvl] * q_group[lvl] * SCALE_ABM).to_numpy()
            #    )
    
            indiv_loc_matrix_new[lvl] = deepcopy(indiv_loc_matrix[lvl])
    
            indiv_loc_matrix_new[lvl], has_moved[lvl] = make_people_move(
                indiv_loc_matrix_new[lvl],
                N[lvl],
                len(gdf),
                indiv_loc_matrix[lvl],
                proba_from,
                proba_to,
                PROBA_MOVE
                )
        
        _, TRIP_TO_ZONE_OUTPUT, tax_revenue_here, subvention_here, commuters_transit, _ = compute_emissions(gdf, travel_matrix, indiv_loc_matrix, income_levels, jobs_in_toll_area, houses_in_toll_area, tax, WORKING_DAYS, discount, SCALE_ABM)

        efficiency = - time_discount * commuters_transit / (tax_revenue_here / 1000000)

        if scenario == "less_expensive_transport":
            condition = (np.abs(TRIP_TO_ZONE_INPUT- TRIP_TO_ZONE_OUTPUT) > 1) | (np.abs(subvention_here - tax_revenue_here)> 1)
        elif scenario == "reduce_transport_time":
            condition = (np.abs(TRIP_TO_ZONE_INPUT- TRIP_TO_ZONE_OUTPUT) > 1) | (np.abs(efficiency - 73)> 0.1)
        else:
            print("(TRIP_TO_ZONE_INPUT- TRIP_TO_ZONE_OUTPUT)", (TRIP_TO_ZONE_INPUT- TRIP_TO_ZONE_OUTPUT))
            condition = (np.abs(TRIP_TO_ZONE_INPUT- TRIP_TO_ZONE_OUTPUT) > 3000)
    

    # AMB
    has_moved = {}
    indiv_loc_matrix_new = {}

    for lvl in income_levels:
        #proba_from, proba_to = compute_proba_of_moving(
        #    save_housing[lvl][:, year - 1],
        #    (housing_without_inertia * SCALE_ABM * w[lvl]).to_numpy()
        #    )
        
        #proba_from, proba_to = compute_proba_of_moving(
        #    save_housing[lvl][:, year - 1],
        #    (n_group[lvl] * q_group[lvl] * SCALE_ABM).to_numpy()
        #    )
        
        proba_from, proba_to = compute_proba_of_moving(
                np.nansum(indiv_loc_matrix[lvl], 0), #save_housing[lvl][:, year - 1],
                (n_group[lvl] * SCALE_ABM).to_numpy() #(n_group[lvl] * q_group[lvl] * SCALE_ABM).to_numpy()
                )
    
        indiv_loc_matrix_new[lvl] = deepcopy(indiv_loc_matrix[lvl])
    
        indiv_loc_matrix[lvl], has_moved[lvl] = make_people_move(
            indiv_loc_matrix_new[lvl],
            N[lvl],
            len(gdf),
            indiv_loc_matrix[lvl],
            proba_from,
            proba_to,
            PROBA_MOVE
            )
    
    rent_indiv_new = {lvl: indiv_loc_matrix[lvl] @ R_group[lvl].to_numpy() for lvl in income_levels}
    dwelling_size_indiv_new = {lvl: indiv_loc_matrix[lvl] @ q_group[lvl] for lvl in income_levels}

    for lvl in income_levels:
        # Update rents and dwelling sizes for movers
        mask_moved = has_moved[lvl] == 1
        rent_indiv[lvl][mask_moved] = rent_indiv_new[lvl][mask_moved]
        dwelling_size_indiv[lvl][mask_moved] = dwelling_size_indiv_new[lvl][mask_moved]

        # Compute utility
        utility[lvl] = compute_utility_manually(
            indiv_loc_matrix[lvl] @ gdf[f"wage_{lvl}"],
            indiv_loc_matrix[lvl] @ gdf[f"transport_cost_{lvl}"],
            dwelling_size_indiv[lvl],
            rent_indiv[lvl],
            BETA, indiv_loc_matrix[lvl] @ gdf["amenities"]
        )
    
        utility[lvl][np.isnan(utility[lvl])] = 0

        # Compute individual housing
        housing_indiv[lvl] = dwelling_size_indiv[lvl] @ csr_matrix(indiv_loc_matrix[lvl])
    
    #Save outputs
    save_population[:, year] = np.sum([np.nansum(indiv_loc_matrix[lvl], 0) for lvl in income_levels], axis=0)
    save_tax[year] = tax
    for lvl in income_levels:
        save_population_lvl[lvl][:, year] = np.nansum(indiv_loc_matrix[lvl], 0)

    for lvl in income_levels:
        save_housing[lvl][:, year] = deepcopy(housing_indiv[lvl])
        save_rent[lvl][:, year] = deepcopy(rent_indiv[lvl])
        save_dwelling_size[lvl][:, year] = deepcopy(dwelling_size_indiv[lvl])
        save_transport_mode[lvl][:, year] = deepcopy(indiv_loc_matrix[lvl] @ gdf[f"transport_mode_{lvl}"])
        save_utility[lvl][:, year] = deepcopy(utility[lvl])

    # Policy support
    score_welfare = {lvl: compute_score(compute_change_in_welfare(save_utility[lvl][:, 0], save_utility[lvl][:, year]), LOGISTIC_PARAM_WELFARE)
                 for lvl in income_levels}
    
    save_emissions[year], total_vkm[year], tax_revenues[year], _, _, total_vkm_lvl[year] = compute_emissions(gdf, travel_matrix, indiv_loc_matrix, income_levels, jobs_in_toll_area, houses_in_toll_area, tax, WORKING_DAYS, discount, SCALE_ABM)

    qol_in_zone[year], qol_out_zone[year], vkm_in_zone[year], vkm_out_zone[year], vkm_in_zone_lvl[year], vkm_out_zone_lvl[year] = compute_qol_congestion(gdf, travel_matrix, indiv_loc_matrix, income_levels)


    ### DECOMPOSITION TRANSPORT

    gdf = gdf.drop(columns = ["work_in_toll_LOW", "work_in_toll_MED", "work_in_toll_HIGH"])
    for lvl in income_levels:
        s = (travel_matrix.loc[travel_matrix.to_id.isin(jobs_in_toll_area),
                      ["from_id", f"proba_center_{lvl}"]].groupby("from_id")[f"proba_center_{lvl}"].sum())
        gdf = gdf.merge(s, left_on="ID", right_on="from_id", how="left").rename(columns={f"proba_center_{lvl}": f"work_in_toll_{lvl}"}).fillna(0)

    for lvl in income_levels:
        #mode share
        mode_shares_lvl[lvl][year] = np.nansum(gdf[f"transport_mode_{lvl}"] * np.nansum(indiv_loc_matrix[lvl], 0)) / np.nansum(np.nansum(indiv_loc_matrix[lvl], 0))
        #distances
        avg_vkm_lvl[lvl][year] = (total_vkm_lvl[year][lvl] / (np.nansum(gdf[f"transport_mode_{lvl}"] * np.nansum(indiv_loc_matrix[lvl], 0))))
        avg_vkm_in_zone_lvl[lvl][year] = vkm_in_zone_lvl[year][lvl] / (np.nansum(gdf[f"transport_mode_{lvl}"] * np.nansum(indiv_loc_matrix[lvl], 0)))
        avg_vkm_out_zone_lvl[lvl][year] = vkm_out_zone_lvl[year][lvl] / (np.nansum(gdf[f"transport_mode_{lvl}"] * np.nansum(indiv_loc_matrix[lvl], 0)))
        #repartition
        gdf[f"live_and_work_in_toll_{lvl}"] = (gdf["live_in_toll"] == True) * gdf[f"work_in_toll_{lvl}"]
        gdf[f"live_out_and_work_in_toll_{lvl}"] = (gdf["live_in_toll"] == False) * gdf[f"work_in_toll_{lvl}"]
        gdf[f"live_in_toll_and_work_out_{lvl}"] = (gdf["live_in_toll"] == True) * (1 - gdf[f"work_in_toll_{lvl}"])
        gdf[f"live_out_and_work_out_toll_{lvl}"] = (gdf["live_in_toll"] == False) * (1 - gdf[f"work_in_toll_{lvl}"])
        live_and_work_in_toll_lvl[lvl][year] = np.nansum(save_population_lvl[lvl][:,year] * gdf[f"live_and_work_in_toll_{lvl}"]) / N[lvl]
        live_out_and_work_in_toll_lvl[lvl][year] = np.nansum(save_population_lvl[lvl][:,year] * gdf[f"live_out_and_work_in_toll_{lvl}"]) / N[lvl]
        live_in_toll_and_work_out_lvl[lvl][year] = np.nansum(save_population_lvl[lvl][:,year] * gdf[f"live_in_toll_and_work_out_{lvl}"]) / N[lvl]
        live_out_and_work_out_lvl[lvl][year] = np.nansum(save_population_lvl[lvl][:,year] * gdf[f"live_out_and_work_out_toll_{lvl}"]) / N[lvl]
        #decompo util
        avg_wage_lvl[lvl][year] = np.nanmean(indiv_loc_matrix[lvl] @ gdf[f"wage_{lvl}"])
        avg_tcost_lvl[lvl][year] = np.nanmean(indiv_loc_matrix[lvl] @ gdf[f"transport_cost_{lvl}"])
        avg_rent_lvl[lvl][year] = np.nanmean(rent_indiv[lvl])
        avg_dsize_lvl[lvl][year] = np.nanmean(dwelling_size_indiv[lvl])



    save_score_emissions[year] = compute_score(compute_relative_change(save_emissions[0], save_emissions[year]), LOGISTIC_PARAM_QOL)
    
    score_qol_zone = compute_score(compute_relative_change(qol_in_zone[0], qol_in_zone[year]), LOGISTIC_PARAM_QOL)
    score_qol_out = compute_score(compute_relative_change(qol_out_zone[0], qol_out_zone[year]), LOGISTIC_PARAM_QOL)
    
    
    political_opinion = {}
    price_here = {}
    acceptable_price_new = {}

    for lvl in income_levels:
        # QOL score
        mask_zone = gdf.ID.isin(houses_in_toll_area).to_numpy()
        mask_out = ~gdf.ID.isin(houses_in_toll_area).to_numpy()
        
        if scenario == "instant_welfare_adjust":
            new_score_qol = (indiv_loc_matrix[lvl] @ mask_zone) * score_qol_zone + (indiv_loc_matrix[lvl] @ mask_out) * score_qol_out
            score_qol[lvl] = (INERTIA_OPINION * score_qol[lvl]) + ((1 - INERTIA_OPINION) * new_score_qol) 
        else:
            score_qol[lvl] = (indiv_loc_matrix[lvl] @ mask_zone) * score_qol_zone + (indiv_loc_matrix[lvl] @ mask_out) * score_qol_out

        # Political opinion and price
        political_opinion[lvl] = compute_opinion2(score_welfare[lvl], score_qol[lvl],
                                                       save_score_emissions[year], BETA_OPINION, scenario, lvl, save_knowledge[year])
        
        price_here[lvl] = compute_opinion2(score_welfare[lvl], score_qol[lvl], save_score_emissions[year], BETA_PRICE, scenario, lvl, save_knowledge[year])
        #political_opinion[lvl] = compute_political_opinion(score_welfare[lvl], score_qol[lvl],
        #                                               save_score_emissions[year], 0, BETA_OPINION, False)
       # 
        #price_here[lvl] = compute_price(score_welfare[lvl], score_qol[lvl],
        #                                save_score_emissions[year], 0, BETA_PRICE, False, scenario, save_knowledge[year], lvl)
        
        if scenario == "instant_welfare_adjust":
            support[lvl] = political_opinion[lvl]
            acceptable_price[lvl] = price_here[lvl]
        else:
            support[lvl] = (INERTIA_OPINION * support[lvl]) + ((1 - INERTIA_OPINION) * political_opinion[lvl])
            acceptable_price[lvl] = (INERTIA_OPINION * acceptable_price[lvl]) + ((1 - INERTIA_OPINION) * price_here[lvl])

        # Save scores
        save_score_welfare[lvl][:, year] = score_welfare[lvl]
        save_score_qol[lvl][:, year] = score_qol[lvl]

    save_median_support[year] = np.nanmedian(np.concatenate([support[lvl] for lvl in income_levels]))

    #Policy update
    tax = np.nanmedian(np.concatenate([acceptable_price[lvl] for lvl in income_levels]))
    #tax = np.fmin(tax * 1.05, np.nanmedian(acceptable_price))

    year = year + 1


emission_change, change_qol_in_zone, change_qol_out_zone, utility_change_low, utility_change_med, utility_change_high = relative_change_outputs(save_emissions, qol_in_zone, qol_out_zone, save_utility, compute_relative_change, compute_change_in_welfare, MAX_YEAR)

#Main results plot
main_plot(save_tax, emission_change, change_qol_in_zone, change_qol_out_zone, utility_change_low, utility_change_med, utility_change_high)

### MODE SHARES
values_0 = [100*mode_shares_lvl[level][0] for level in income_levels]
values_20 = [100*mode_shares_lvl[level][MAX_YEAR - 1] for level in income_levels]
x = np.arange(len(income_levels))
width = 0.35
plt.figure()
plt.bar(x - width/2, values_0, width, label="Year 0", color = "#4C72B0")
plt.bar(x + width/2, values_20, width, label="Year " + str(MAX_YEAR - 1), color = "#DD8452")
plt.xticks(x, ["Low-income", "Middle-income", "High-income"])
plt.ylabel("Share of public transport users (%)")
plt.legend()
plt.show()
plt.close()


### DISTANCES
in_0  = [avg_vkm_in_zone_lvl[lvl][0]  for lvl in income_levels]
in_20 = [avg_vkm_in_zone_lvl[lvl][MAX_YEAR - 1] for lvl in income_levels]
out_0  = [avg_vkm_out_zone_lvl[lvl][0]  for lvl in income_levels]
out_20 = [avg_vkm_out_zone_lvl[lvl][MAX_YEAR - 1] for lvl in income_levels]

x = np.arange(len(income_levels))
width = 0.35

plt.figure()
plt.bar(x - width/2, in_0,  width, label="Inside the tax zone (year 0)",  color="#4C72B0")
plt.bar(x - width/2, out_0, width, bottom=in_0, label="Outside of the tax zone (year 0)", color="#9ecae9")
plt.bar(x + width/2, in_20,  width, label="Inside the tax zone (year "+str(MAX_YEAR-1)+")",  color="#DD8452")
plt.bar(x + width/2, out_20, width, bottom=in_20, label="Outside of the tax zone (year "+str(MAX_YEAR-1)+")", color="#fdd0a2")
plt.xticks(x,  ["Low-income", "Middle-income", "High-income"])
plt.ylabel("Average vehicle-km driven")
plt.legend()
plt.show()
plt.close()


plt.plot(avg_vkm_lvl["LOW"], label = "Low-income")
plt.plot(avg_vkm_in_zone_lvl["LOW"] + avg_vkm_out_zone_lvl["LOW"], label = "Low-income")


##### LIVING

# Extract values for each category and year
lw_in_0  = [100 * live_and_work_in_toll_lvl[l][0]  for l in income_levels]
lw_in_20 = [100 * live_and_work_in_toll_lvl[l][MAX_YEAR - 1] for l in income_levels]

lw_out_0  = [100 * live_out_and_work_out_lvl[l][0]  for l in income_levels]
lw_out_20 = [100 * live_out_and_work_out_lvl[l][MAX_YEAR - 1] for l in income_levels]

in_out_0  = [100 * live_in_toll_and_work_out_lvl[l][0]  for l in income_levels]
in_out_20 = [100 * live_in_toll_and_work_out_lvl[l][MAX_YEAR - 1] for l in income_levels]

out_in_0  = [100 * live_out_and_work_in_toll_lvl[l][0]  for l in income_levels]
out_in_20 = [100 * live_out_and_work_in_toll_lvl[l][MAX_YEAR - 1] for l in income_levels]

x = np.arange(len(income_levels))
width = 0.32
# Muted academic palette
colors = {
    "lw_in":  "#5B7C99",  # slate blue
    "lw_out": "#8C9A5B",  # muted olive
    "in_out": "#C2A878",  # sand
    "out_in": "#B07A8F"   # dusty rose
}


from matplotlib.patches import Patch
plt.figure()

edge_col = "0.3"
edge_lw = 0.6

# --- Year 0 ---
bottom_0 = np.zeros(len(income_levels))
pos_0 = x - width/2

for vals, key in zip(
    [lw_in_0, lw_out_0, in_out_0, out_in_0],
    ["lw_in", "lw_out", "in_out", "out_in"]
):
    plt.bar(pos_0, vals, width,
            bottom=bottom_0,
            color=colors[key],
            edgecolor=edge_col,
            linewidth=edge_lw)
    bottom_0 += vals

# --- Year 20 ---
bottom_20 = np.zeros(len(income_levels))
pos_20 = x + width/2

for vals, key in zip(
    [lw_in_20, lw_out_20, in_out_20, out_in_20],
    ["lw_in", "lw_out", "in_out", "out_in"]
):
    plt.bar(pos_20, vals, width,
            bottom=bottom_20,
            color=colors[key],
            edgecolor=edge_col,
            linewidth=edge_lw)
    bottom_20 += vals

# Year labels above bars
offset = 0.02
for i in range(len(income_levels)):
    plt.text(pos_0[i],  bottom_0[i]  + offset, "Year 0",  ha="center", va="bottom")
    plt.text(pos_20[i], bottom_20[i] + offset, "Year " + str(MAX_YEAR - 1), ha="center", va="bottom")

plt.xticks(x,  ["Low-income", "Middle-income", "High-income"])
plt.ylabel("Share (%)")
plt.ylim(0, max(max(bottom_0), max(bottom_20)) + 0.08)

legend_elements = [
    Patch(facecolor=colors["lw_in"],  edgecolor=edge_col, label="Live and work inside the tax zone"),
    Patch(facecolor=colors["lw_out"], edgecolor=edge_col, label="Live and work outside of the tax zone"),
    Patch(facecolor=colors["in_out"], edgecolor=edge_col, label="Live inside and work outside of the tax zone"),
    Patch(facecolor=colors["out_in"], edgecolor=edge_col, label="Live outside and work inside of the tax zone"),
]

plt.legend(
    handles=legend_elements,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),  # 0.5 = center horizontally, -0.12 = below plot
    ncol=2  # one column per category
)
plt.tight_layout()
plt.show()
plt.close()



############

indicators = ["Average generalized travel cost", "Average rent per m2", "Average dwelling size"]

# Compute % change from year 0 to 20
change_tcost = [
    (avg_tcost_lvl[l][MAX_YEAR-1] - avg_tcost_lvl[l][0]) / avg_tcost_lvl[l][0] * 100
    for l in income_levels
]

change_rent = [
    (avg_rent_lvl[l][MAX_YEAR-1] - avg_rent_lvl[l][0]) / avg_rent_lvl[l][0] * 100
    for l in income_levels
]

change_dsize = [
    (avg_dsize_lvl[l][MAX_YEAR-1] - avg_dsize_lvl[l][0]) / avg_dsize_lvl[l][0] * 100
    for l in income_levels
]

# Combine into a single array
changes = np.array([change_tcost, change_rent, change_dsize])  # shape: 3 indicators x 3 income levels

x = np.arange(len(income_levels))
width = 0.25

# Muted palette for indicators
colors = ["#4C72B0", "#55A868", "#C44E52"]

plt.figure(figsize=(8,5))

# Plot bars for each indicator
for i in range(len(indicators)):
    plt.bar(x + (i-1)*width, changes[i], width, color=colors[i], edgecolor="0.3", label=indicators[i])

plt.xticks(x, income_levels)
plt.ylabel("Change between year 0 and " + str(MAX_YEAR - 1) + " (%)")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()
plt.close()

#Spatial analysis plot: price
weighted_values = {
    lvl: compute_weighted_mean_opinions(acceptable_price[lvl], indiv_loc_matrix[lvl], N[lvl]) * 2
    for lvl in income_levels}

for lvl in income_levels:
    plot_spatial_price(gdf, weighted_values[lvl])

#Spatial analysis plot: population
plot_change_population_custom(gdf, save_population)
plot_change_pop_line(gdf, save_population)
moving = {}
for lvl in income_levels:
    moving[lvl] = 0
    for i in range(19):
        moving[lvl] = moving[lvl] + np.nansum(np.abs(save_population_lvl[lvl][:, i+1] - save_population_lvl[lvl][:, i]))/2
    print(lvl, 100 * (moving[lvl] / N[lvl]),  "% moving relative to the population")



#append_scenario_to_excel(
#    filename="simulation_results.xlsx",
#    scenario_name="discount_public_transport",
#    tax_level=save_tax,
#    emission_change=emission_change,
#    change_qol_in_zone=change_qol_in_zone,
#    change_qol_out_zone=change_qol_out_zone,
#    utility_change_low=utility_change_low,
#    utility_change_med=utility_change_med,
#    utility_change_high=utility_change_high
#)