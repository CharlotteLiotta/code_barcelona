import numpy as np 
from copy import deepcopy
from scipy.sparse import csr_matrix # type: ignore
import pickle
import copy
from stargazer.stargazer import Stargazer
from scipy.optimize import differential_evolution

from outcomes import *
from ABM import *
from import_data import * # type: ignore
from calibration import * # type: ignore
from model import * # type: ignore
from plotting_tools import * # type: ignore
from import_transport import *
from policy_support import *

### IMPORT PARAMETERS

path_data = "../data_barcelona/"
option_function = "Cobb-Douglas"

#Time
year = 0
MAX_YEAR = 20

#Policy impact model
INTEREST_RATE = 0.05 #+ 0.93 #Interest rate + depreciation rate of built capital
PRICE_TIME = 10 #euros/h
WORKING_DAYS = 40 #20 days per month, with 2 trips per day
PRICE_FUEL = 0.11 #euros/km
center = "0801901025"
alpha = 50

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
Y, Y_median, gdf = import_income(gdf, path_data)
gdf = import_amenities(gdf, path_data, 0, 0)
employment_centers = gpd.read_file(path_data + "cluster_employment.shp")
employment_centers["cluster"] = employment_centers.index
jobs_in_toll_area, houses_in_toll_area = import_tax_zone(gdf, employment_centers)

#Multiple income groups
income_levels = ["LOW", "MED", "HIGH"]
wage_factors = {"LOW": 0.6, "MED": 1.0, "HIGH": 1.4}
pop_high_income = np.nansum(gdf["pop"] * gdf["share_high_income"] / 100)
pop_medium_income = np.nansum(gdf["pop"] * (100 - gdf["share_high_income"]- gdf["share_low_income"]) / 100)
pop_low_income = np.nansum(gdf["pop"] * gdf["share_low_income"] / 100)

#Import transport data
#import_transport_times_poly(gdf, datetime.datetime(2025, 7, 15, 8, 0, 0), center, path_data, employment_centers) #datetime.datetime(2025, 7, 15, 8, 0, 0)
travel_time_matrix_car, travel_time_matrix_transit = load_transport_times_poly(gdf, path_data, center)
travel_time_matrix_car = load_distance_car_poly(travel_time_matrix_car, gdf, employment_centers)
gdf = import_cost_transit(gdf)
travel_time_matrix_transit = travel_time_matrix_transit.merge(gdf[['ID', 'monthly_cost_transit']].rename(columns={'ID': 'from_id'}), on='from_id', how='left')

### POLICY SUPPORT

BETA_OPINION, INITIAL_OPINION = import_opinion_parameters(path_data)
BETA_PRICE, INITIAL_PRICE = import_price_parameters(path_data)

tax = INITIAL_PRICE
acceptable_price = {"LOW": INITIAL_PRICE, "MED": INITIAL_PRICE, "HIGH": INITIAL_PRICE}

### INITIAL STATE: YEAR 0

#Transport cost calibration
#gdf, FIXED_COST_CAR, LAMBDA, ARRAY_WAGE_LOW, ARRAY_WAGE_MED, ARRAY_WAGE_HIGH = compute_cost_car_poly_i(gdf, Y_median, import_trans_mode, PRICE_TIME, WORKING_DAYS, PRICE_FUEL, travel_time_matrix_car, travel_time_matrix_transit, employment_centers, path_data, jobs_in_toll_area, houses_in_toll_area)
#with open(path_data + "calib_trans_poly_i.pkl", "wb") as f:
#    pickle.dump((FIXED_COST_CAR, LAMBDA, ARRAY_WAGE_LOW, ARRAY_WAGE_MED, ARRAY_WAGE_HIGH), f)
with open(path_data + "calib_trans_poly_i.pkl", "rb") as f:
    FIXED_COST_CAR, LAMBDA, ARRAY_WAGE_LOW, ARRAY_WAGE_MED, ARRAY_WAGE_HIGH = pickle.load(f)
print("FIXED_COST_CAR: ", FIXED_COST_CAR)
print("LAMBDA: ", LAMBDA)
del Y
del compute_transport_cost_logit, compute_cost_car_logit

#Compute transport cost
gdf, workers_per_cluster, travel_matrix = compute_transport_cost_poly_i(gdf, travel_time_matrix_car, travel_time_matrix_transit, PRICE_TIME, WORKING_DAYS, FIXED_COST_CAR, PRICE_FUEL, LAMBDA, ARRAY_WAGE_LOW, ARRAY_WAGE_MED, ARRAY_WAGE_HIGH, jobs_in_toll_area, houses_in_toll_area, 0, income_levels, wage_factors)
print(round(100 * sum(np.nansum(gdf[f"transport_mode_{lvl}"] * gdf[f"pop_{lvl}"]) for lvl in income_levels) / sum(np.nansum(gdf[f"pop_{lvl}"]) for lvl in income_levels)))
plot_transport_cost_i(gdf, "_HIGH")
plot_transport_mode_i(gdf, "_HIGH")
#plot_employment(gdf, employment_centers, var) var = employment_centers['employment'] * 0.001, employment_centers.merge(employed_results, left_on = "cluster", right_on = "to_id")["weighted_employed"] * 0.001,  # adjust scale_factor, ARRAY_WAGE* 0.5
#gdf.merge(travel_matrix.loc[travel_matrix.to_id == 5,:], left_on = "ID", right_on = "from_id").plot("proba_center", legend = True)

#Prepare variables
gdf["size"] = gdf["size_census"] #gdf["size_census"] #gdf["size_AMB"]
gdf["pop"].loc[gdf["pop"] == 0] = 1
gdf["rent_m2"].loc[gdf["rent_m2"] == 0] = np.nanmin(gdf["rent_m2"].loc[gdf["rent_m2"]>0])
gdf["rent_m2"].loc[np.isnan(gdf["rent_m2"])] = np.nanmin(gdf["rent_m2"].loc[gdf["rent_m2"]>0])
gdf["size"].loc[np.isnan(gdf["size"])] = np.nansum(gdf["size"].loc[~np.isnan(gdf["size"])] * gdf["pop"].loc[~np.isnan(gdf["size"])]) / np.nansum(gdf["pop"].loc[~np.isnan(gdf["size"])])

#Calibrate beta and amenities
def compute_log_likelihood(x):
    return calibration_utility_amenity(x, gdf, income_levels, alpha, 0, 0)

calib_beta = scipy.optimize.minimize(compute_log_likelihood, [0.57, 140, 241, 348], bounds=[(0.1,0.9), (100,2000), (100,2000), (100,2000)])
BETA = calib_beta.x[0]
amenities = calibration_utility_amenity(calib_beta.x, gdf, income_levels, alpha, 1, 1)
gdf = gdf.merge(amenities, on = "ID", how = "left")
gdf.loc[np.isnan(gdf["amenities"]), "amenities"] = 1

#Calibration of B and KAPPA
gdf["land"] = gdf["urb_area"]

mask = ((gdf["rent_m2"] < 25) &(gdf["rent_m2"] > 7)&
        (gdf["pop"] > 484.0)&
        (~np.isnan(gdf["size"]))
        &(~np.isnan(gdf["pop"]))
        &((gdf["pop"] > 0))
        )

B, KAPPA, SIGMA = calibrate_b_kappa(gdf, mask, INTEREST_RATE, option_function, option_calib = "housing")

# Solve the model
def compute_error_in_population_from_utility(u):
    """ Compute error in population associated to utility u"""

    return compute_error_in_population(u, gdf["amenities"], [pop_low_income, pop_medium_income, pop_high_income], BETA, gdf["wage_LOW"], gdf["wage_MED"], gdf["wage_HIGH"], gdf["transport_cost_LOW"], gdf["transport_cost_MED"], gdf["transport_cost_HIGH"], B, KAPPA, SIGMA, INTEREST_RATE, gdf["urb_area"], alpha, option_function)

solving_model = scipy.optimize.minimize(compute_error_in_population_from_utility, [180, 420, 600], bounds=[(0,None), (0,None), (0,None)], method = "Nelder-Mead") #np.array([399,690,995]) np.array([370,690,800])

if solving_model.fun < 1:
    utility = solving_model.x
    R_LOW = compute_rents(BETA, gdf["wage_LOW"], utility[0]/gdf["amenities"], gdf["transport_cost_LOW"])
    R_MED = compute_rents(BETA, gdf["wage_MED"], utility[1]/gdf["amenities"], gdf["transport_cost_MED"])
    R_HIGH = compute_rents(BETA, gdf["wage_HIGH"], utility[2]/gdf["amenities"], gdf["transport_cost_HIGH"])

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

    R = w_LOW * R_LOW + w_MED * R_MED + w_HIGH * R_HIGH
    
    # --- Compute dwelling size and population ---
    q_LOW = compute_dwelling_size(BETA, gdf["wage_LOW"], gdf["transport_cost_LOW"], R)
    q_MED = compute_dwelling_size(BETA, gdf["wage_MED"], gdf["transport_cost_MED"], R)
    q_HIGH = compute_dwelling_size(BETA, gdf["wage_HIGH"], gdf["transport_cost_HIGH"], R)
    
    n = compute_population(B, KAPPA, SIGMA, R, INTEREST_RATE, gdf["urb_area"], w_LOW * q_LOW + w_MED * q_MED + w_HIGH * q_HIGH, option_function = option_function)
    
else:
    print("Minimization failed!")

density_residual = np.log(gdf["pop"] / n)
rent_residual = np.log(gdf["rent_m2"] / R)
size_residual = np.log(gdf["size"] / (w_LOW * q_LOW + w_MED * q_MED + w_HIGH * q_HIGH))

# Plot the result of the calibration
map_calibration(gdf, n, gdf["pop"] , "Population")
map_calibration(gdf, w_LOW * q_LOW + w_MED * q_MED + w_HIGH * q_HIGH, gdf["size"], "Dwelling size per capita")
map_calibration(gdf, R, gdf["rent_m2"], "Rent per m2")
map_calibration(gdf, 1000000 * n / gdf["urb_area"], 1000000 * gdf["pop"] / gdf["urb_area"], "Population density")

scatter_calibration(gdf, n, gdf["pop"] , "Population")
scatter_calibration(gdf, w_LOW * q_LOW + w_MED * q_MED + w_HIGH * q_HIGH, gdf["size"], "Dwelling size per capita")
scatter_calibration(gdf, R, gdf["rent_m2"], "Rent per m2")
scatter_calibration(gdf, n * w_LOW * q_LOW + w_MED * q_MED + w_HIGH * q_HIGH / gdf["urb_area"], gdf["pop"] * gdf["size"] / gdf["urb_area"], "Housing")
scatter_calibration(gdf, 1000000 * n / gdf["urb_area"], 1000000 * gdf["pop"] / gdf["urb_area"], "Population density")

agg = compare_rent_or_size(gdf, "size", w_LOW * q_LOW + w_MED * q_MED + w_HIGH * q_HIGH, 1)
agg = compare_rent_or_size(gdf, "rent_m2", R, 1)
agg = compare_var(gdf, n)

plt.plot(agg["distance_bin"], agg["mean_density_pop"], color='red', linewidth=2, label="Densité moyenne (pop)")
plt.plot(agg["distance_bin"], agg["mean_density_n"], color='blue', linewidth=2, label="Densité moyenne (n)")
plt.xlabel("Distance au centre-ville (km)")
plt.ylabel("Densité de population (hab/km²)")
plt.legend()
plt.tight_layout()
plt.show()

#initial state
def compute_error_in_population_from_utility(u):
    """ Compute error in population associated to utility u"""

    return compute_error_in_population(u, gdf["amenities"], [pop_low_income, pop_medium_income, pop_high_income], BETA, gdf["wage_LOW"], gdf["wage_MED"], gdf["wage_HIGH"], gdf["transport_cost_LOW"], gdf["transport_cost_MED"], gdf["transport_cost_HIGH"], B, KAPPA, SIGMA, INTEREST_RATE, gdf["urb_area"], alpha, option_function, rent_residual, density_residual, size_residual)


solving_model = scipy.optimize.minimize(compute_error_in_population_from_utility, solving_model.x)

if solving_model.fun < 1:

    utility = solving_model.x
    R_LOW = compute_rents(BETA, gdf["wage_LOW"], utility[0]/gdf["amenities"], gdf["transport_cost_LOW"])
    R_MED = compute_rents(BETA, gdf["wage_MED"], utility[1]/gdf["amenities"], gdf["transport_cost_MED"])
    R_HIGH = compute_rents(BETA, gdf["wage_HIGH"], utility[2]/gdf["amenities"], gdf["transport_cost_HIGH"])

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

    R = w_LOW * R_LOW + w_MED * R_MED + w_HIGH * R_HIGH
    
    q_LOW = compute_dwelling_size(BETA, gdf["wage_LOW"], gdf["transport_cost_LOW"], R)
    q_MED = compute_dwelling_size(BETA, gdf["wage_MED"], gdf["transport_cost_MED"], R)
    q_HIGH = compute_dwelling_size(BETA, gdf["wage_HIGH"], gdf["transport_cost_HIGH"], R)

    n = compute_population(B, KAPPA, SIGMA, R, INTEREST_RATE, gdf["urb_area"], w_LOW * q_LOW + w_MED * q_MED + w_HIGH * q_HIGH, option_function = option_function)
    
    R = R * np.exp(rent_residual)
    q = (w_LOW * q_LOW + w_MED * q_MED + w_HIGH * q_HIGH) * np.exp(size_residual)
    n = n * np.exp(density_residual)

    n[np.isnan(n)] = 0
else:
    print("Minimization failed!")

agg = compare_rent_or_size(gdf, "size", q, 1)
agg = compare_rent_or_size(gdf, "rent_m2", R, 1)
agg = compare_var(gdf, n)

# ABM: translate outputs at the household level
N_LOW = round(pop_low_income * SCALE_ABM)
N_MED = round(pop_medium_income * SCALE_ABM)
N_HIGH = round(pop_high_income * SCALE_ABM)

support_low = INITIAL_OPINION * np.ones(N_LOW) #0.417
support_med = INITIAL_OPINION * np.ones(N_LOW) #0.417
support_high = INITIAL_OPINION * np.ones(N_LOW) #0.417


indiv_loc_matrix_LOW = compute_indiv_loc_matrix(N_LOW, len(gdf), (n* SCALE_ABM * w_LOW).to_numpy())
indiv_loc_matrix_MED = compute_indiv_loc_matrix(N_MED, len(gdf), (n* SCALE_ABM * w_MED).to_numpy())
indiv_loc_matrix_HIGH = compute_indiv_loc_matrix(N_HIGH, len(gdf), (n* SCALE_ABM * w_HIGH).to_numpy())

#indiv_loc_matrix_0 = copy.deepcopy(indiv_loc_matrix)
rent_indiv_LOW = indiv_loc_matrix_LOW @ R.to_numpy()
dwelling_size_indiv_LOW = indiv_loc_matrix_LOW @ q

rent_indiv_MED = indiv_loc_matrix_MED @ R.to_numpy()
dwelling_size_indiv_MED = indiv_loc_matrix_MED @ q

rent_indiv_HIGH = indiv_loc_matrix_HIGH @ R.to_numpy()
dwelling_size_indiv_HIGH = indiv_loc_matrix_HIGH @ q

utility_LOW = compute_utility_manually(indiv_loc_matrix_LOW @gdf["wage_LOW"], indiv_loc_matrix_LOW @gdf["transport_cost_LOW"], 
                                                dwelling_size_indiv_LOW, rent_indiv_LOW, BETA)
utility_LOW[np.isnan(utility_LOW)] = 0

utility_MED = compute_utility_manually(indiv_loc_matrix_MED @gdf["wage_MED"], indiv_loc_matrix_MED @gdf["transport_cost_MED"], 
                                                dwelling_size_indiv_MED, rent_indiv_MED, BETA)
utility_MED[np.isnan(utility_MED)] = 0

utility_HIGH = compute_utility_manually(indiv_loc_matrix_HIGH @gdf["wage_HIGH"], indiv_loc_matrix_HIGH @gdf["transport_cost_HIGH"], 
                                                dwelling_size_indiv_HIGH, rent_indiv_HIGH, BETA)
utility_HIGH[np.isnan(utility_HIGH)] = 0

housing_indiv_LOW = dwelling_size_indiv_LOW @ csr_matrix(indiv_loc_matrix_LOW)  # shape: (10,)
housing_indiv_MED = dwelling_size_indiv_MED @ csr_matrix(indiv_loc_matrix_MED)  # shape: (10,)
housing_indiv_HIGH = dwelling_size_indiv_HIGH @ csr_matrix(indiv_loc_matrix_HIGH)  # shape: (10,)

# Save outputs
save_housing_LOW = np.zeros((len(gdf["area"]), MAX_YEAR))
save_housing_LOW[:, 0] = deepcopy(housing_indiv_LOW)

save_housing_MED = np.zeros((len(gdf["area"]), MAX_YEAR))
save_housing_MED[:, 0] = deepcopy(housing_indiv_MED)

save_housing_HIGH = np.zeros((len(gdf["area"]), MAX_YEAR))
save_housing_HIGH[:, 0] = deepcopy(housing_indiv_HIGH)

save_population = np.zeros((len(gdf["area"]), MAX_YEAR))
save_population[:, 0] = np.nansum(indiv_loc_matrix_LOW, 0) + np.nansum(indiv_loc_matrix_MED, 0) + np.nansum(indiv_loc_matrix_HIGH, 0)

save_rent_LOW = np.zeros((N_LOW, MAX_YEAR))
save_rent_LOW[:, 0] = deepcopy(rent_indiv_LOW)
save_rent_MED = np.zeros((N_MED, MAX_YEAR))
save_rent_MED[:, 0] = deepcopy(rent_indiv_MED)
save_rent_HIGH = np.zeros((N_HIGH, MAX_YEAR))
save_rent_HIGH[:, 0] = deepcopy(rent_indiv_HIGH)

save_dwelling_size_LOW = np.zeros((N_LOW, MAX_YEAR))
save_dwelling_size_LOW[:, 0] = deepcopy(dwelling_size_indiv_LOW)
save_dwelling_size_MED = np.zeros((N_MED, MAX_YEAR))
save_dwelling_size_MED[:, 0] = deepcopy(dwelling_size_indiv_MED)
save_dwelling_size_HIGH = np.zeros((N_HIGH, MAX_YEAR))
save_dwelling_size_HIGH[:, 0] = deepcopy(dwelling_size_indiv_HIGH)

save_transport_mode_LOW = np.zeros((N_LOW, MAX_YEAR))
save_transport_mode_LOW[:, 0] = deepcopy(indiv_loc_matrix_LOW @gdf["transport_mode_LOW"])
save_transport_mode_MED = np.zeros((N_MED, MAX_YEAR))
save_transport_mode_MED[:, 0] = deepcopy(indiv_loc_matrix_MED @gdf["transport_mode_MED"])
save_transport_mode_HIGH = np.zeros((N_HIGH, MAX_YEAR))
save_transport_mode_HIGH[:, 0] = deepcopy(indiv_loc_matrix_HIGH @gdf["transport_mode_HIGH"])


save_utility_LOW = np.zeros((N_LOW, MAX_YEAR))
save_utility_LOW[:, 0] = deepcopy(utility_LOW)
save_utility_MED = np.zeros((N_MED, MAX_YEAR))
save_utility_MED[:, 0] = deepcopy(utility_MED)
save_utility_HIGH = np.zeros((N_HIGH, MAX_YEAR))
save_utility_HIGH[:, 0] = deepcopy(utility_HIGH)

save_tax = np.zeros(MAX_YEAR)
save_tax[0] = 0
save_qol = np.zeros((MAX_YEAR))
save_congestion = np.zeros((MAX_YEAR))

#save emissions
travel_matrix["distance_emi_LOW"] = (travel_matrix["distance_car"] /1000) * travel_matrix["proba_center_LOW"] * (1 - travel_matrix["transport_mode_LOW"])
distance_emi_LOW = travel_matrix.loc[:,["distance_emi_LOW", "from_id"]].groupby("from_id").sum()
gdf = gdf.merge(distance_emi_LOW, left_on = "ID", right_index = True)
travel_matrix["distance_emi_MED"] = (travel_matrix["distance_car"] /1000) * travel_matrix["proba_center_MED"] * (1 - travel_matrix["transport_mode_MED"])
distance_emi_MED = travel_matrix.loc[:,["distance_emi_MED", "from_id"]].groupby("from_id").sum()
gdf = gdf.merge(distance_emi_MED, left_on = "ID", right_index = True)
travel_matrix["distance_emi_HIGH"] = (travel_matrix["distance_car"] /1000) * travel_matrix["proba_center_HIGH"] * (1 - travel_matrix["transport_mode_HIGH"])
distance_emi_HIGH = travel_matrix.loc[:,["distance_emi_HIGH", "from_id"]].groupby("from_id").sum()
gdf = gdf.merge(distance_emi_HIGH, left_on = "ID", right_index = True)

emissions_init = sum(np.nansum(indiv_loc_matrix_LOW, 0) * (gdf["distance_emi_LOW"])) + sum(np.nansum(indiv_loc_matrix_MED, 0) * (gdf["distance_emi_MED"])) + sum(np.nansum(indiv_loc_matrix_HIGH, 0) * (gdf["distance_emi_HIGH"]))
save_emissions = np.zeros(MAX_YEAR)
save_emissions[0] = emissions_init


save_score_emissions = np.zeros(MAX_YEAR)
save_score_welfare_LOW = np.zeros((N_LOW, MAX_YEAR))
save_score_qol_LOW = np.zeros((N_LOW, MAX_YEAR))
save_score_congestion_LOW = np.zeros((N_LOW, MAX_YEAR))

save_score_welfare_MED = np.zeros((N_MED, MAX_YEAR))
save_score_qol_MED = np.zeros((N_MED, MAX_YEAR))
save_score_congestion_MED = np.zeros((N_MED, MAX_YEAR))

save_score_welfare_HIGH = np.zeros((N_HIGH, MAX_YEAR))
save_score_qol_HIGH = np.zeros((N_HIGH, MAX_YEAR))
save_score_congestion_HIGH = np.zeros((N_HIGH, MAX_YEAR))

#check distances per capita
travel_matrix["pop_LOW"] = travel_matrix["pop_LOW"] * travel_matrix["proba_center_LOW"]
travel_matrix["pop_MED"] = travel_matrix["pop_MED"] * travel_matrix["proba_center_MED"]
travel_matrix["pop_HIGH"] = travel_matrix["pop_HIGH"] * travel_matrix["proba_center_HIGH"]

bins = np.array([0, 0.5, 2, 5, 10, 50])
labels = [f"{i}km" for i in bins[:-1]]
travel_matrix['distance_bin'] = pd.cut(travel_matrix['distance_car'] / 1000, bins=bins, labels=labels, right=False)
pop_by_bin_LOW = travel_matrix.groupby('distance_bin', observed=True)['pop_LOW'].sum()
pop_by_bin_MED = travel_matrix.groupby('distance_bin', observed=True)['pop_MED'].sum()
pop_by_bin_HIGH = travel_matrix.groupby('distance_bin', observed=True)['pop_HIGH'].sum()

pop_by_bin_HIGH.plot(kind='bar', figsize=(8, 4))
plt.ylabel("Population")
plt.xlabel("Distance to center")
plt.title("Population by distance category")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

save_median_support = np.zeros(MAX_YEAR)

year = year + 1



#### COMPUTE QOL

save_qol[0], save_congestion[0], proba_commuting_in_tax_zone_by_car_LOW, proba_commuting_in_tax_zone_by_car_MED, proba_commuting_in_tax_zone_by_car_HIGH = compute_qol(np.nansum(indiv_loc_matrix_LOW, 0), np.nansum(indiv_loc_matrix_MED, 0), np.nansum(indiv_loc_matrix_HIGH, 0), gdf, travel_matrix, jobs_in_toll_area, houses_in_toll_area)

### MODELING THE PSC

while year < MAX_YEAR:

    print("YEAR", year)

    # Urban form with the tax, without inertia

    #gdf = compute_transport_cost_logit(gdf, Y, PRICE_TIME, WORKING_DAYS, FIXED_COST_CAR, PRICE_FUEL, LAMBDA, tax = tax)
    gdf, workers_per_cluster, travel_matrix = compute_transport_cost_poly_i(gdf, travel_time_matrix_car, travel_time_matrix_transit, PRICE_TIME, WORKING_DAYS, FIXED_COST_CAR, PRICE_FUEL, LAMBDA, ARRAY_WAGE_LOW, ARRAY_WAGE_MED, ARRAY_WAGE_HIGH, jobs_in_toll_area, houses_in_toll_area, tax)

    #if year == 1:
    #    net_income_t1 = gdf["income_net_of_transport_cost"]

    def compute_error_in_population_from_utility(u):
        """ Compute error in population associated to utility u"""

        return compute_error_in_population(u, gdf["amenities"], [pop_low_income, pop_medium_income, pop_high_income], BETA, gdf["wage_LOW"], gdf["wage_MED"], gdf["wage_HIGH"], gdf["transport_cost_LOW"], gdf["transport_cost_MED"], gdf["transport_cost_HIGH"], B, KAPPA, SIGMA, INTEREST_RATE, gdf["urb_area"], alpha, option_function, rent_residual, density_residual, size_residual)

    result_global = differential_evolution(compute_error_in_population_from_utility, bounds=[(300,600), (550,900), (800,1400)])
    solving_model = scipy.optimize.minimize(compute_error_in_population_from_utility, result_global.x)

    if solving_model.fun < 1:

        utility = solving_model.x
        R_LOW = compute_rents(BETA, gdf["wage_LOW"], utility[0]/gdf["amenities"], gdf["transport_cost_LOW"])
        R_MED = compute_rents(BETA, gdf["wage_MED"], utility[1]/gdf["amenities"], gdf["transport_cost_MED"])
        R_HIGH = compute_rents(BETA, gdf["wage_HIGH"], utility[2]/gdf["amenities"], gdf["transport_cost_HIGH"])

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

        R = w_LOW * R_LOW + w_MED * R_MED + w_HIGH * R_HIGH
        avg_wage = w_LOW * gdf["wage_LOW"] + w_MED * gdf["wage_MED"] + w_HIGH * gdf["wage_HIGH"]
        avg_t_cost = w_LOW * gdf["transport_cost_LOW"] + w_MED * gdf["transport_cost_MED"] + w_HIGH * gdf["transport_cost_HIGH"]


        q = compute_dwelling_size(BETA, avg_wage, avg_t_cost, R)
        n = compute_population(B, KAPPA, SIGMA, R, INTEREST_RATE, gdf["urb_area"], q, option_function = option_function)
    
        R = R * np.exp(rent_residual)
        q = q * np.exp(size_residual)
        n = n * np.exp(density_residual)
        n[np.isnan(n)] = 0

    else:
        print("Minimization failed!")

    housing_without_inertia = n * q

    # AMB
    proba_of_moving_from_LOW, proba_of_moving_to_LOW = compute_proba_of_moving(save_housing_LOW[:, year - 1], (housing_without_inertia * SCALE_ABM * w_LOW).to_numpy())
    indiv_loc_matrix_new_LOW = deepcopy(indiv_loc_matrix_LOW)
    indiv_loc_matrix_LOW, has_moved_LOW = make_people_move(indiv_loc_matrix_new_LOW, N_LOW, len(gdf), indiv_loc_matrix_LOW, proba_of_moving_from_LOW, proba_of_moving_to_LOW, PROBA_MOVE)

    proba_of_moving_from_MED, proba_of_moving_to_MED = compute_proba_of_moving(save_housing_MED[:, year - 1], (housing_without_inertia * SCALE_ABM * w_MED).to_numpy())
    indiv_loc_matrix_new_MED = deepcopy(indiv_loc_matrix_MED)
    indiv_loc_matrix_MED, has_moved_MED = make_people_move(indiv_loc_matrix_new_MED, N_MED, len(gdf), indiv_loc_matrix_MED, proba_of_moving_from_MED, proba_of_moving_to_MED, PROBA_MOVE)
    
    proba_of_moving_from_HIGH, proba_of_moving_to_HIGH = compute_proba_of_moving(save_housing_HIGH[:, year - 1], (housing_without_inertia * SCALE_ABM * w_HIGH).to_numpy())
    indiv_loc_matrix_new_HIGH = deepcopy(indiv_loc_matrix_HIGH)
    indiv_loc_matrix_HIGH, has_moved_HIGH = make_people_move(indiv_loc_matrix_new_HIGH, N_HIGH, len(gdf), indiv_loc_matrix_HIGH, proba_of_moving_from_HIGH, proba_of_moving_to_HIGH, PROBA_MOVE)
    
    
    
    rent_indiv_new_LOW = indiv_loc_matrix_LOW @ R
    dwelling_size_indiv_new_LOW = indiv_loc_matrix_LOW @ q
    rent_indiv_LOW[has_moved_LOW == 1] = rent_indiv_new_LOW[has_moved_LOW == 1]
    dwelling_size_indiv_LOW[has_moved_LOW == 1] = dwelling_size_indiv_new_LOW[has_moved_LOW == 1]

    utility_LOW = compute_utility_manually(indiv_loc_matrix_LOW @ gdf["wage_LOW"], indiv_loc_matrix_LOW @gdf["transport_cost_LOW"], 
                                                dwelling_size_indiv_LOW, rent_indiv_LOW, BETA)

    housing_indiv_LOW = dwelling_size_indiv_LOW @ csr_matrix(indiv_loc_matrix_LOW)  # shape: (10,)

    rent_indiv_new_MED = indiv_loc_matrix_MED @ R
    dwelling_size_indiv_new_MED = indiv_loc_matrix_MED @ q
    rent_indiv_MED[has_moved_MED == 1] = rent_indiv_new_MED[has_moved_MED == 1]
    dwelling_size_indiv_MED[has_moved_MED == 1] = dwelling_size_indiv_new_MED[has_moved_MED == 1]

    utility_MED = compute_utility_manually(indiv_loc_matrix_MED @ gdf["wage_MED"], indiv_loc_matrix_MED @gdf["transport_cost_MED"], 
                                                dwelling_size_indiv_MED, rent_indiv_MED, BETA)

    housing_indiv_MED = dwelling_size_indiv_MED @ csr_matrix(indiv_loc_matrix_MED)  # shape: (10,)

    rent_indiv_new_HIGH = indiv_loc_matrix_HIGH @ R
    dwelling_size_indiv_new_HIGH = indiv_loc_matrix_HIGH @ q
    rent_indiv_HIGH[has_moved_HIGH == 1] = rent_indiv_new_HIGH[has_moved_HIGH == 1]
    dwelling_size_indiv_HIGH[has_moved_HIGH == 1] = dwelling_size_indiv_new_HIGH[has_moved_HIGH == 1]

    utility_HIGH = compute_utility_manually(indiv_loc_matrix_HIGH @ gdf["wage_HIGH"], indiv_loc_matrix_HIGH @gdf["transport_cost_HIGH"], 
                                                dwelling_size_indiv_HIGH, rent_indiv_HIGH, BETA)

    housing_indiv_HIGH = dwelling_size_indiv_HIGH @ csr_matrix(indiv_loc_matrix_HIGH)  # shape: (10,)

    
    utility_LOW[np.isnan(utility_LOW)] = 0
    utility_MED[np.isnan(utility_MED)] = 0
    utility_HIGH[np.isnan(utility_HIGH)] = 0
    
    #Save outputs
    save_population[:, year] = np.nansum(indiv_loc_matrix_LOW, 0) + np.nansum(indiv_loc_matrix_MED, 0) + np.nansum(indiv_loc_matrix_HIGH, 0)
    save_tax[year] = tax

    save_housing_LOW[:, year] = deepcopy(housing_indiv_LOW)
    save_rent_LOW[:, year] = deepcopy(rent_indiv_LOW)
    save_dwelling_size_LOW[:, year] = deepcopy(dwelling_size_indiv_LOW)
    save_utility_LOW[:, year] = deepcopy(utility_LOW)
    save_transport_mode_LOW[:, year] = deepcopy(indiv_loc_matrix_LOW@gdf["transport_mode_LOW"])

    save_housing_MED[:, year] = deepcopy(housing_indiv_MED)
    save_rent_MED[:, year] = deepcopy(rent_indiv_MED)
    save_dwelling_size_MED[:, year] = deepcopy(dwelling_size_indiv_MED)
    save_utility_MED[:, year] = deepcopy(utility_MED)
    save_transport_mode_MED[:, year] = deepcopy(indiv_loc_matrix_MED@gdf["transport_mode_MED"])

    save_housing_HIGH[:, year] = deepcopy(housing_indiv_HIGH)
    save_rent_HIGH[:, year] = deepcopy(rent_indiv_HIGH)
    save_dwelling_size_HIGH[:, year] = deepcopy(dwelling_size_indiv_HIGH)
    save_utility_HIGH[:, year] = deepcopy(utility_HIGH)
    save_transport_mode_HIGH[:, year] = deepcopy(indiv_loc_matrix_HIGH@gdf["transport_mode_HIGH"])

    
    
    
    # Policy support
    score_welfare_LOW = compute_change_in_welfare(save_utility_LOW[:,0], save_utility_LOW[:,year])
    score_welfare_MED = compute_change_in_welfare(save_utility_MED[:,0], save_utility_MED[:,year])
    score_welfare_HIGH = compute_change_in_welfare(save_utility_HIGH[:,0], save_utility_HIGH[:,year])
    
    
    score_emissions, emissions = compute_change_in_emissions(gdf, travel_matrix, emissions_init, np.nansum(indiv_loc_matrix_LOW, 0), np.nansum(indiv_loc_matrix_MED, 0), np.nansum(indiv_loc_matrix_HIGH, 0))

    
    #save outcomes
    save_qol[year], save_congestion[year], proba_commuting_in_tax_zone_by_car_LOW, proba_commuting_in_tax_zone_by_car_MED, proba_commuting_in_tax_zone_by_car_HIGH = compute_qol(np.nansum(indiv_loc_matrix_LOW, 0), np.nansum(indiv_loc_matrix_MED, 0), np.nansum(indiv_loc_matrix_HIGH, 0), gdf, travel_matrix, jobs_in_toll_area, houses_in_toll_area)

    score_qol_zone = compute_change_in_qol(save_qol[0], save_qol[year])
    
    score_qol_LOW = (indiv_loc_matrix_LOW @ gdf.ID.isin(houses_in_toll_area)) * score_qol_zone
    score_qol_LOW[score_qol_LOW == 0] = 0.5

    score_qol_MED = (indiv_loc_matrix_MED @ gdf.ID.isin(houses_in_toll_area)) * score_qol_zone
    score_qol_MED[score_qol_MED == 0] = 0.5


    score_qol_HIGH = (indiv_loc_matrix_HIGH @ gdf.ID.isin(houses_in_toll_area)) * score_qol_zone
    score_qol_HIGH[score_qol_HIGH == 0] = 0.5

    score_congestion_zone = compute_change_in_qol(save_congestion[0], save_congestion[year])

    
    proba_commuting_in_tax_zone_by_car_LOW[np.isnan(proba_commuting_in_tax_zone_by_car_LOW)] = 0
    score_congestion_LOW = (indiv_loc_matrix_LOW @ proba_commuting_in_tax_zone_by_car_LOW) * score_congestion_zone + (indiv_loc_matrix_LOW @ (1-proba_commuting_in_tax_zone_by_car_LOW)) * 0.5

    proba_commuting_in_tax_zone_by_car_MED[np.isnan(proba_commuting_in_tax_zone_by_car_MED)] = 0
    score_congestion_MED = (indiv_loc_matrix_MED @ proba_commuting_in_tax_zone_by_car_MED) * score_congestion_zone + (indiv_loc_matrix_MED @ (1-proba_commuting_in_tax_zone_by_car_MED)) * 0.5

    proba_commuting_in_tax_zone_by_car_HIGH[np.isnan(proba_commuting_in_tax_zone_by_car_HIGH)] = 0
    score_congestion_HIGH = (indiv_loc_matrix_HIGH @ proba_commuting_in_tax_zone_by_car_HIGH) * score_congestion_zone + (indiv_loc_matrix_HIGH @ (1-proba_commuting_in_tax_zone_by_car_HIGH)) * 0.5


    
    save_emissions[year] = emissions
    save_score_emissions[year] = score_emissions

    save_score_welfare_LOW[:,year] = score_welfare_LOW
    save_score_qol_LOW[:,year] = score_qol_LOW
    save_score_congestion_LOW[:,year] = score_congestion_LOW

    save_score_welfare_MED[:,year] = score_welfare_MED
    save_score_qol_MED[:,year] = score_qol_MED
    save_score_congestion_MED[:,year] = score_congestion_MED

    save_score_welfare_HIGH[:,year] = score_welfare_HIGH
    save_score_qol_HIGH[:,year] = score_qol_HIGH
    save_score_congestion_HIGH[:,year] = score_congestion_HIGH


    political_opinion_LOW = compute_political_opinion(score_welfare_LOW, score_qol_LOW, score_emissions, score_congestion_LOW, BETA_OPINION)   
    price_here_LOW = compute_price(score_welfare_LOW, score_qol_LOW, score_emissions, score_congestion_LOW, BETA_PRICE)
    support_LOW = political_opinion_LOW
    acceptable_price_LOW = (INERTIA_OPINION * acceptable_price_LOW) + ((1 - INERTIA_OPINION) * price_here_LOW) # type: ignore
   
    political_opinion_MED = compute_political_opinion(score_welfare_MED, score_qol_MED, score_emissions, score_congestion_MED, BETA_OPINION)   
    price_here_MED = compute_price(score_welfare_MED, score_qol_MED, score_emissions, score_congestion_MED, BETA_PRICE)
    support_MED = political_opinion_MED
    acceptable_price_MED = (INERTIA_OPINION * acceptable_price_MED) + ((1 - INERTIA_OPINION) * price_here_MED) # type: ignore
   
    political_opinion_HIGH = compute_political_opinion(score_welfare_HIGH, score_qol_HIGH, score_emissions, score_congestion_HIGH, BETA_OPINION)   
    price_here_HIGH = compute_price(score_welfare_HIGH, score_qol_HIGH, score_emissions, score_congestion_HIGH, BETA_PRICE)
    support_HIGH = political_opinion_HIGH
    acceptable_price_HIGH = (INERTIA_OPINION * acceptable_price_HIGH) + ((1 - INERTIA_OPINION) * price_here_HIGH) # type: ignore
   
    save_median_support[year] = np.nanmedian(np.concatenate((support_LOW, support_MED, support_HIGH)))
    #print("support", support)

    #Policy update
    tax = np.nanmedian(np.concatenate((acceptable_price_LOW, acceptable_price_MED, acceptable_price_HIGH)))
    #tax = np.fmin(tax * 1.05, np.nanmedian(acceptable_price))

    year = year + 1


### PLOT RESULTS
#print(round(100 * sum(gdf["transport_mode"] * gdf["pop"]) / sum(gdf["pop"])), " % commute by public transport")

print(round(100 * ((np.nansum((gdf["transport_mode_LOW"]) * np.nansum(indiv_loc_matrix_LOW, 0))) + (np.nansum((gdf["transport_mode_MED"]) * np.nansum(indiv_loc_matrix_MED, 0))) + (np.nansum((gdf["transport_mode_HIGH"]) * np.nansum(indiv_loc_matrix_HIGH, 0)))) / (np.nansum(np.nansum(indiv_loc_matrix_LOW, 0)) + np.nansum(np.nansum(indiv_loc_matrix_MED, 0)) + np.nansum(np.nansum(indiv_loc_matrix_HIGH, 0)))))


print(round(100 * sum(gdf["transport_mode_LOW"] * np.nansum(indiv_loc_matrix_LOW, 0) ) / sum(np.nansum(indiv_loc_matrix_LOW, 0) )), " % commute by public transport")
print(round(100 * sum(gdf["transport_mode_MED"] * np.nansum(indiv_loc_matrix_MED, 0) ) / sum(np.nansum(indiv_loc_matrix_MED, 0) )), " % commute by public transport")
print(round(100 * sum(gdf["transport_mode_HIGH"] * np.nansum(indiv_loc_matrix_HIGH, 0) ) / sum(np.nansum(indiv_loc_matrix_HIGH, 0) )), " % commute by public transport")


values_LOW = compute_weighted_mean_opinions(support_LOW, indiv_loc_matrix_LOW, N_LOW) * 100
plot_spatial_opinions(gdf, values_LOW)

values_MED = compute_weighted_mean_opinions(support_MED, indiv_loc_matrix_MED, N_MED) * 100
plot_spatial_opinions(gdf, values_MED)

values_HIGH = compute_weighted_mean_opinions(support_HIGH, indiv_loc_matrix_HIGH, N_HIGH) * 100
plot_spatial_opinions(gdf, values)


plot_change_population(gdf, save_population)


gdf = gdf.copy()
gdf["population0"] = save_population[:, 0]
gdf["population1"] = save_population[:, 1]
gdf["population5"] = save_population[:, 5]
gdf["population10"] = save_population[:, 10]
gdf["population15"] = save_population[:, 15]
gdf["population19"] = save_population[:, 19]
bins = np.arange(0, gdf["distance_center"].max() + 2, 2)
gdf["distance_bin"] = pd.cut(gdf["distance_center"], bins=bins)
pop_by_bin0 = gdf.groupby("distance_bin")["population0"].sum()
pop_by_bin1 = gdf.groupby("distance_bin")["population1"].sum()
pop_by_bin5 = gdf.groupby("distance_bin")["population5"].sum()
pop_by_bin10 = gdf.groupby("distance_bin")["population10"].sum()
pop_by_bin15 = gdf.groupby("distance_bin")["population15"].sum()
pop_by_bin19 = gdf.groupby("distance_bin")["population19"].sum()
pop_by_bin0.plot(kind="line", figsize=(10,5), label = "0")
#pop_by_bin1.plot(kind="line", figsize=(10,5), label = "1")
#pop_by_bin5.plot(kind="line", figsize=(10,5), label = "5")
#pop_by_bin10.plot(kind="line", figsize=(10,5), label = "10")
#pop_by_bin15.plot(kind="line", figsize=(10,5), label = "15")
pop_by_bin19.plot(kind="line", figsize=(10,5), label = "19")
plt.legend()
plt.ylabel("Population")
plt.xlabel("Distance to city center (km)")
plt.title("Population by distance bins")
plt.show()

plot_tax_suppport(save_tax, save_median_support)
plot_scores(save_score_emissions, save_score_qol_LOW, save_score_congestion_LOW, save_score_welfare_LOW)
plot_scores(save_score_emissions, save_score_qol_MED, save_score_congestion_MED, save_score_welfare_MED)
plot_scores(save_score_emissions, save_score_qol_HIGH, save_score_congestion_HIGH, save_score_welfare_HIGH)

fig, ax1 = plt.subplots(figsize=(8, 6))  # make figure wider
ax1.set_xlabel('Time (year)', fontsize=14)
ax1.set_ylabel('Toll per entry (€)', fontsize=14)
ax1.plot(save_tax[1:], linewidth=1.5, label='Toll increases by 5% per year')
ax1.plot(save_tax_scenario2[1:], linewidth=1.5, label='Toll increases by 10cts per year')
ax1.plot(save_tax_scenario1[1:], linewidth=1.5, label='Toll = Maximum acceptable toll')
ax1.tick_params(axis='y')
plt.tight_layout()
plt.legend()
plt.show()


moving = 0

for i in range(19):
    print(i)
    moving = moving + np.nansum(np.abs(save_population[:, i+1] - save_population[:, i]))/2


# Example data
x = np.arange(3)  # three categories
width = 0.25      # bar width



scenario_5 = [
   100 * (save_emissions[19] - save_emissions[0]) / save_emissions[0],
   100 * (save_congestion[19] - save_congestion[0]) / save_congestion[0],
   100 * (save_qol[19] - save_qol[0]) / save_qol[0]
]

scenario_10 = [
   100 * (save_emissions_scenario2[19] - save_emissions_scenario2[0]) / save_emissions_scenario2[0],
   100 * (save_congestion_scenario2[19] - save_congestion_scenario2[0]) / save_congestion_scenario2[0],
   100 * (save_qol_scenario2[19] - save_qol_scenario2[0]) / save_qol_scenario2[0]
]

scenario_base = [
   100 * (save_emissions_scenario1[19] - save_emissions_scenario1[0]) / save_emissions_scenario1[0],
   100 * (save_congestion_scenario1[19] - save_congestion_scenario1[0]) / save_congestion_scenario1[0],
   100 * (save_qol_scenario1[19] - save_qol_scenario1[0]) / save_qol_scenario1[0]
]

fig, ax = plt.subplots(figsize=(8, 6))
bars1 = ax.bar(x - width, scenario_5, width, label='Toll increases by 5% per year')
bars2 = ax.bar(x , scenario_10, width, label='Toll increases by 10cts per year')
bars3 = ax.bar(x + width, scenario_base, width, label='Toll = Maximum acceptable toll')

# Move x-axis labels to the top
ax.set_xticks(x)
ax.set_xticklabels(['Emissions', 'Congestion', 'Quality of life'])
ax.xaxis.set_ticks_position('top')       # put ticks on top
ax.xaxis.set_label_position('top')       # put label on top (if needed)
ax.tick_params(axis='x', which='both', top=False)  
# Clean axes
ax.set_ylabel("Changes between year 0 and year 19 (%)")
ax.legend()
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(True)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
plt.rcParams.update({'font.size': 14})
plt.tight_layout()
plt.show()



plt.plot(save_emissions)
plt.plot(save_emissions[1:])

plt.plot(np.nanmean(save_transport_mode, 0))
plt.plot(np.nanmean(save_transport_mode, 0)[1:])

plt.plot(np.nanmedian(save_utility, 0))
plt.plot(np.nanmedian(save_utility, 0)[1:])

var_income = (net_income_t1 - net_income_t0) / net_income_t0
plot_with_missing(gdf, var_income)

gdf["income_loss_absolute"] = (net_income_t1 - net_income_t0)
gdf["income_loss_relative"] = (net_income_t1 - net_income_t0) / net_income_t0

gdf.loc[:, ['ID', 'geometry', 'income_loss_absolute', "income_loss_relative"]].to_file(path_data + "income_loss.geojson", driver="GeoJSON")

indiv_loc_matrix_1 = copy.deepcopy(indiv_loc_matrix)

average_utility_per_tract_0 = np.zeros(2149)
counts = np.zeros(2149)
tract_id = np.argmax(indiv_loc_matrix_0, axis=1)
# Sum utilities and counts per tract
np.add.at(average_utility_per_tract_0, tract_id, save_utility[:,0])
np.add.at(counts, tract_id, 1)
average_utility_per_tract_0 /= counts


average_utility_per_tract_1 = np.zeros(2149)
counts = np.zeros(2149)
tract_id = np.argmax(indiv_loc_matrix_1, axis=1)
# Sum utilities and counts per tract
np.add.at(average_utility_per_tract_1, tract_id, save_utility[:,1])
np.add.at(counts, tract_id, 1)
average_utility_per_tract_1 /= counts

plot_with_missing(gdf, average_utility_per_tract_0)
plot_with_missing(gdf, average_utility_per_tract_1)

plot_with_missing(gdf, (average_utility_per_tract_1 - average_utility_per_tract_0) / average_utility_per_tract_0)

gdf["experienced_impact"] = (average_utility_per_tract_1 - average_utility_per_tract_0) / average_utility_per_tract_0

gdf.loc[:, ['ID', 'geometry', 'transport_mode', "experienced_impact"]].to_file(path_data + "experienced_impact.geojson", driver="GeoJSON")



# Create figure
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Plot with column-based coloring
gdf.plot(
    column='transport_mode',
    cmap='viridis',
    edgecolor='black',
    linewidth=0.5,
    legend=True,
    ax=ax,
    legend_kwds={
        'shrink': 0.5,             # make legend smaller
    }
)

# Access the colorbar to change its label and font size
cbar = ax.get_figure().get_axes()[1]  # the second axes is the colorbar
#cbar.set_xlabel("Population", fontsize=12)
cbar.tick_params(labelsize=12)        # increase tick labels

# Remove axes
ax.set_axis_off()


plt.tight_layout()
plt.show()