import numpy as np 
from copy import deepcopy
from scipy.sparse import csr_matrix # type: ignore
import pickle

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
SOFT_RENT = 50
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
employment_centers = gpd.read_file(path_data + "cluster_employment_UEA.shp")
employment_centers["cluster"] = employment_centers.index
jobs_in_toll_area, houses_in_toll_area, zone_tax = import_tax_zone(gdf, employment_centers)

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
#import_transport_times_poly(gdf, datetime.datetime(2025, 7, 15, 8, 0, 0), center, path_data, employment_centers) #datetime.datetime(2025, 7, 15, 8, 0, 0)
travel_time_matrix_car, travel_time_matrix_transit = load_transport_times_poly(gdf, path_data, center)
travel_time_matrix_car = load_distance_car_poly(travel_time_matrix_car, gdf, employment_centers, jobs_in_toll_area, houses_in_toll_area, zone_tax)
gdf = import_cost_transit(gdf)
travel_time_matrix_transit = travel_time_matrix_transit.merge(gdf[['ID', 'monthly_cost_transit']].rename(columns={'ID': 'from_id'}), on='from_id', how='left')

travel_time_matrix_car["uncongested_speed"] = (travel_time_matrix_car.distance_car / 1000) / (travel_time_matrix_car.travel_time / 60)
travel_time_matrix_car["speed"] = travel_time_matrix_car["uncongested_speed"] - DIFF_SPEED_CONGESTION
travel_time_matrix_car.loc[travel_time_matrix_car["speed"] < 10, "speed"] = 10
travel_time_matrix_car.travel_time = ((travel_time_matrix_car.distance_car / 1000) / travel_time_matrix_car["speed"]) * 60

### CALIBRATION POLICY SUPPORT

BETA_OPINION, INITIAL_OPINION = import_opinion_parameters(path_data, scenario)
BETA_PRICE, INITIAL_PRICE, INITIAL_CC, INITIAL_WELFARE, INITIAL_QOL, INITIAL_KNOWLEDGE = import_price_parameters(path_data, scenario)

tax = INITIAL_PRICE
acceptable_price = {"LOW": INITIAL_PRICE, "MED": INITIAL_PRICE, "HIGH": INITIAL_PRICE}
support = {"LOW": INITIAL_OPINION, "MED": INITIAL_OPINION, "HIGH": INITIAL_OPINION}
save_score_emissions = np.zeros(MAX_YEAR)
save_score_emissions[0] = INITIAL_CC
save_knowledge = np.zeros(MAX_YEAR)
save_knowledge[0] = INITIAL_KNOWLEDGE
score_qol = {"LOW": INITIAL_QOL, "MED": INITIAL_QOL, "HIGH": INITIAL_QOL}
score_welfare = {"LOW": INITIAL_WELFARE, "MED": INITIAL_WELFARE, "HIGH": INITIAL_WELFARE}


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
gdf, FIXED_COST_CAR, LAMBDA, ARRAY_WAGE_LOW, ARRAY_WAGE_MED, ARRAY_WAGE_HIGH = compute_cost_car_poly_i(gdf, Y_median, import_trans_mode, PRICE_TIME, WORKING_DAYS, PRICE_FUEL, travel_time_matrix_car, travel_time_matrix_transit, employment_centers, path_data, jobs_in_toll_area, houses_in_toll_area, income_levels, wage_factors, scenario)
#with open(path_data + "calib_trans_poly_i_UEA.pkl", "wb") as f:
#    pickle.dump((FIXED_COST_CAR, LAMBDA, ARRAY_WAGE_LOW, ARRAY_WAGE_MED, ARRAY_WAGE_HIGH), f)
#with open(path_data + "calib_trans_poly_i_UEA.pkl", "rb") as f:
#    FIXED_COST_CAR, LAMBDA, ARRAY_WAGE_LOW, ARRAY_WAGE_MED, ARRAY_WAGE_HIGH = pickle.load(f)
print("FIXED_COST_CAR: ", FIXED_COST_CAR)
print("LAMBDA: ", LAMBDA)
del compute_transport_cost_logit, compute_cost_car_logit

#Compute transport cost
gdf, workers_per_cluster, travel_matrix = compute_transport_cost_poly_i(gdf, travel_time_matrix_car, travel_time_matrix_transit, PRICE_TIME, WORKING_DAYS, FIXED_COST_CAR, PRICE_FUEL, LAMBDA, ARRAY_WAGE_LOW, ARRAY_WAGE_MED, ARRAY_WAGE_HIGH, jobs_in_toll_area, houses_in_toll_area, 0, income_levels, wage_factors, scenario)
print("Modal share of public transport (%):", round(100 * sum(np.nansum(gdf[f"transport_mode_{lvl}"] * gdf[f"pop_{lvl}"]) for lvl in income_levels) / sum(np.nansum(gdf[f"pop_{lvl}"]) for lvl in income_levels)))
print("Modal share of public transport (%):", round(100 * (np.nansum(gdf["transport_mode_HIGH"] * gdf["pop_HIGH"])) / (np.nansum(gdf["pop_HIGH"]))))
print("Modal share of public transport (%):", round(100 * (np.nansum(gdf[f"transport_mode_MED"] * gdf[f"pop_MED"])) / (np.nansum(gdf[f"pop_MED"]))))
print("Modal share of public transport (%):", round(100 * (np.nansum(gdf[f"transport_mode_LOW"] * gdf[f"pop_LOW"])) / (np.nansum(gdf[f"pop_LOW"]))))
plot_transport_cost_i(gdf, "_HIGH")
plot_transport_mode_i(gdf, "_HIGH")
#plot_employment(gdf, employment_centers, var) var = employment_centers['employment'] * 0.001, employment_centers.merge(employed_results, left_on = "cluster", right_on = "to_id")["weighted_employed"] * 0.001,  # adjust scale_factor, ARRAY_WAGE* 0.5
#gdf.merge(travel_matrix.loc[travel_matrix.to_id == 5,:], left_on = "ID", right_on = "from_id").plot("proba_center", legend = True)

#Calibrate beta and amenities
def compute_log_likelihood(x):
    print(x)
    return calibration_utility_amenity2(x, gdf, income_levels, SOFT_RENT, 0, 0)

calib_beta = scipy.optimize.minimize(compute_log_likelihood, [0.45, 224, 386, 553], bounds=[(0.1,0.9), (0,None), (0,None), (0,None)]) #[0.45, 100, 500, 900]
BETA = calib_beta.x[0]
print("BETA:", BETA)
amenities = calibration_utility_amenity(calib_beta.x, gdf, income_levels, SOFT_RENT, 1, 1)
gdf = gdf.merge(amenities, on = "ID", how = "left")
gdf.loc[np.isnan(gdf["amenities"]), "amenities"] = 1
gdf.loc[np.isnan(gdf["amenities_improved_rodalies"]), "amenities_improved_rodalies"] = 1

# Solve the model
def compute_error_in_population_from_utility(u):
    """ Compute error in population associated to utility u"""

    return compute_error_in_population(u, gdf["amenities"], [pop[lvl] for lvl in income_levels], BETA, gdf["wage_LOW"], gdf["wage_MED"], gdf["wage_HIGH"], gdf["transport_cost_LOW"], gdf["transport_cost_MED"], gdf["transport_cost_HIGH"], B, KAPPA, SIGMA, INTEREST_RATE, gdf["urb_area"], SOFT_RENT, False, option_function)

solving_model = scipy.optimize.minimize(compute_error_in_population_from_utility, [100, 180, 300], bounds=[(0,None), (0,None), (0,None)], method = "Nelder-Mead") #np.array([399,690,995]) np.array([370,690,800]) #[180, 420, 600]

if solving_model.fun < 1:
    R, q, n, w, R_group, q_group = compute_outcomes(solving_model.x, gdf, BETA, B, KAPPA, SIGMA, INTEREST_RATE, SOFT_RENT, income_levels, compute_rents, compute_dwelling_size, compute_population, option_function)
else:
    print("Minimization failed!")

share_col = {"LOW": "share_low_income", "MED": None, "HIGH": "share_high_income"}
density_residual = {
    lvl: np.log(
        (gdf["pop"] * (gdf[share_col[lvl]] / 100 if share_col[lvl] else (100 - gdf["share_low_income"] - gdf["share_high_income"]) / 100))
        / (w[lvl] * n)
    )
    for lvl in income_levels
}
rent_residual = np.log(gdf["rent_m2"] / R)
size_residual = np.log(gdf["size"] / q)

# Plot the result of the calibration
print_maps(gdf, n, q, R)
plot_line_charts(gdf, n, q, R)

#initial state
def compute_error_in_population_from_utility(u):
    """ Compute error in population associated to utility u"""

    return compute_error_in_population(u, gdf["amenities"], [pop[lvl] for lvl in income_levels], BETA, gdf["wage_LOW"], gdf["wage_MED"], gdf["wage_HIGH"], gdf["transport_cost_LOW"], gdf["transport_cost_MED"], gdf["transport_cost_HIGH"], B, KAPPA, SIGMA, INTEREST_RATE, gdf["urb_area"], SOFT_RENT, True, option_function, rent_residual, density_residual, size_residual)

solving_model = scipy.optimize.minimize(compute_error_in_population_from_utility, solving_model.x)

if solving_model.fun < 1:

    R, q, n, w, R_group, q_group = compute_outcomes(solving_model.x, gdf, BETA, B, KAPPA, SIGMA, INTEREST_RATE, SOFT_RENT, income_levels, compute_rents, compute_dwelling_size, compute_population, option_function)

    R = R * np.exp(rent_residual)
    q = q * np.exp(size_residual)
    n_group = {lvl: n * w[lvl] * np.exp(density_residual[lvl]) for lvl in ["LOW", "MED", "HIGH"]}
    n = np.nansum(list(n_group.values()), axis=0)
    n[np.isnan(n)] = 0
    for lvl in income_levels:
        n_group[lvl][np.isnan(n_group[lvl])] = 0

else:
    print("Minimization failed!")

plot_line_charts(gdf, n, q, R)

# ABM: translate outputs at the household level
N = {lvl: round(pop[lvl] * SCALE_ABM) for lvl in income_levels}
support = {lvl: INITIAL_OPINION * np.ones(N[lvl]) for lvl in income_levels}
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
tax_revenues = np.zeros(MAX_YEAR)

save_emissions[0], total_vkm[0], tax_revenues[0], _, _ = compute_emissions(gdf, travel_matrix, indiv_loc_matrix, income_levels, jobs_in_toll_area, houses_in_toll_area, 0, WORKING_DAYS, 0, SCALE_ABM)
qol_in_zone[0], qol_out_zone[0], vkm_in_zone[0], vkm_out_zone[0] = compute_qol_congestion(gdf, travel_matrix, indiv_loc_matrix, income_levels)

BETA_CONG = DIFF_SPEED_CONGESTION/total_vkm[0]

year = year + 1
#plot_distance_distrib_check(income_levels, travel_matrix, "HIGH")

tax_revenue_here = 137000000
subvention_here = 137000000
efficiency = 0
discount = 12.79346
time_discount = 0


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

        print("DISCOUNT", discount)
        print("DIFF SUBVENTION TAX REVENUE", subvention_here-tax_revenue_here)

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
                error_population = compute_error_in_population(u, gdf["amenities"], [pop[lvl] for lvl in income_levels], BETA, gdf["wage_LOW"], gdf["wage_MED"], gdf["wage_HIGH"], gdf["transport_cost_LOW"], gdf["transport_cost_MED"], gdf["transport_cost_HIGH"], B, KAPPA, SIGMA, INTEREST_RATE, gdf["urb_area"], SOFT_RENT, True, option_function, rent_residual, density_residual, size_residual)
            return error_population

        solving_model = scipy.optimize.minimize(compute_error_in_population_from_utility, solving_model.x)

        if solving_model.fun < 1:

            R, q, n, w, R_group, q_group = compute_outcomes(solving_model.x, gdf, BETA, B, KAPPA, SIGMA, INTEREST_RATE, SOFT_RENT, income_levels, compute_rents, compute_dwelling_size, compute_population, option_function)

            R = R * np.exp(rent_residual)
            q = q * np.exp(size_residual)
            n_group = {lvl: n * w[lvl] * np.exp(density_residual[lvl]) for lvl in ["LOW", "MED", "HIGH"]}
            n = np.nansum(list(n_group.values()), axis=0)
            n[np.isnan(n)] = 0
            for lvl in income_levels:
                n_group[lvl][np.isnan(n_group[lvl])] = 0

        else:
            print("Minimization failed!")

        housing_without_inertia = n * q

        # AMB
        has_moved = {}
        indiv_loc_matrix_new = {}

        for lvl in income_levels:
            #proba_from, proba_to = compute_proba_of_moving(
            #    save_housing[lvl][:, year - 1],
            #    (housing_without_inertia * SCALE_ABM * w[lvl]).to_numpy()
            #    )

            proba_from, proba_to = compute_proba_of_moving(
                np.nansum(indiv_loc_matrix[lvl], 0), #save_housing[lvl][:, year - 1],
                (n_group[lvl] * SCALE_ABM).to_numpy() #(n_group[lvl] * q_group[lvl] * SCALE_ABM).to_numpy()
                )
    
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
        
        _, TRIP_TO_ZONE_OUTPUT, tax_revenue_here, subvention_here, commuters_transit = compute_emissions(gdf, travel_matrix, indiv_loc_matrix, income_levels, jobs_in_toll_area, houses_in_toll_area, tax, WORKING_DAYS, discount, SCALE_ABM)

        efficiency = - time_discount * commuters_transit / (tax_revenue_here / 1000000)

        if scenario == "less_expensive_transport":
            condition = (np.abs(TRIP_TO_ZONE_INPUT- TRIP_TO_ZONE_OUTPUT) > 1) | (np.abs(subvention_here - tax_revenue_here)> 1)
        elif scenario == "reduce_transport_time":
            condition = (np.abs(TRIP_TO_ZONE_INPUT- TRIP_TO_ZONE_OUTPUT) > 1) | (np.abs(efficiency - 73)> 0.1)
        else:
            condition = (np.abs(TRIP_TO_ZONE_INPUT- TRIP_TO_ZONE_OUTPUT) > 1)
    

    # AMB
    has_moved = {}
    indiv_loc_matrix_new = {}

    for lvl in income_levels:
        #proba_from, proba_to = compute_proba_of_moving(
        #    save_housing[lvl][:, year - 1],
        #    (housing_without_inertia * SCALE_ABM * w[lvl]).to_numpy()
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
        save_housing[lvl][:, year] = deepcopy(housing_indiv[lvl])
        save_rent[lvl][:, year] = deepcopy(rent_indiv[lvl])
        save_dwelling_size[lvl][:, year] = deepcopy(dwelling_size_indiv[lvl])
        save_transport_mode[lvl][:, year] = deepcopy(indiv_loc_matrix[lvl] @ gdf[f"transport_mode_{lvl}"])
        save_utility[lvl][:, year] = deepcopy(utility[lvl])

    # Policy support
    score_welfare = {lvl: compute_score(compute_change_in_welfare(save_utility[lvl][:, 0], save_utility[lvl][:, year]), LOGISTIC_PARAM_WELFARE)
                 for lvl in income_levels}
    
    save_emissions[year], total_vkm[year], tax_revenues[year], _, _ = compute_emissions(gdf, travel_matrix, indiv_loc_matrix, income_levels, jobs_in_toll_area, houses_in_toll_area, tax, WORKING_DAYS, discount, SCALE_ABM)

    qol_in_zone[year], qol_out_zone[year], vkm_in_zone[year], vkm_out_zone[year] = compute_qol_congestion(gdf, travel_matrix, indiv_loc_matrix, income_levels)

    if scenario == "instant_welfare_adjust":
        new_emissions = compute_score(compute_relative_change(save_emissions[0], save_emissions[year]), LOGISTIC_PARAM_QOL)
        save_score_emissions[year] = (INERTIA_OPINION * save_score_emissions[year-1]) + ((1 - INERTIA_OPINION) * new_emissions) 
    else:
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
        political_opinion[lvl] = compute_political_opinion(score_welfare[lvl], score_qol[lvl],
                                                       save_score_emissions[year], 0, BETA_OPINION, False)
        
        price_here[lvl] = compute_price(score_welfare[lvl], score_qol[lvl],
                                        save_score_emissions[year], 0, BETA_PRICE, False, scenario, save_knowledge[year], lvl)
        
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


### PLOT RESULTS

#Evolution of tax, public support and scores
plot_tax_suppport(save_tax, save_median_support)

for lvl in income_levels:
    plot_scores(
        save_score_emissions,
        save_score_qol[lvl],
        save_score_congestion[lvl],
        save_score_welfare[lvl]
    )



emission_change = np.empty(MAX_YEAR)
for i in range(MAX_YEAR):
    emission_change[i] = np.nanmedian(compute_relative_change(save_emissions[0], save_emissions[i]))

change_qol_in_zone = np.empty(MAX_YEAR)
for i in range(MAX_YEAR):
    change_qol_in_zone[i] = np.nanmean(compute_relative_change(qol_in_zone[0], qol_in_zone[i]))

change_qol_out_zone = np.empty(MAX_YEAR)
for i in range(MAX_YEAR):
    change_qol_out_zone[i] = np.nanmean(compute_relative_change(qol_out_zone[0], qol_out_zone[i]))


utility_change_low = np.empty(MAX_YEAR)
for i in range(MAX_YEAR):
    utility_change_low[i] = np.nanmedian(compute_change_in_welfare(save_utility["LOW"][:, 0], save_utility["LOW"][:, i]))

utility_change_med = np.empty(MAX_YEAR)
for i in range(MAX_YEAR):
    utility_change_med[i] = np.nanmedian(compute_change_in_welfare(save_utility["MED"][:, 0], save_utility["MED"][:, i]))

utility_change_high = np.empty(MAX_YEAR)
for i in range(MAX_YEAR):
    utility_change_high[i] = np.nanmedian(compute_change_in_welfare(save_utility["HIGH"][:, 0], save_utility["HIGH"][:, i]))




import pandas as pd
import numpy as np
from openpyxl import load_workbook
import os

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



append_scenario_to_excel(
    filename="simulation_results.xlsx",
    scenario_name="discount_public_transport",
    tax_level=save_tax,
    emission_change=emission_change,
    change_qol_in_zone=change_qol_in_zone,
    change_qol_out_zone=change_qol_out_zone,
    utility_change_low=utility_change_low,
    utility_change_med=utility_change_med,
    utility_change_high=utility_change_high
)




import matplotlib.pyplot as plt
import matplotlib as mpl

# --- Global formatting for academic figures ---
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 9


# Colorblind-safe palette (Wong 2011)
colors = {
    "blue":  "#0072B2",
    "orange":"#E69F00",
    "green": "#009E73",
    "red":   "#D55E00",
    "purple":"#CC79A7",
    "cyan":  "#56B4E9"
}

fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True, constrained_layout=True)
years = range(len(save_tax))

# --- Panel 1 ---
axes[0].plot(years, 2 * save_tax, color="black", linewidth=1.5)
axes[0].set_ylabel('Toll per day (€)')

# --- Panel 2 ---
axes[1].plot(years, utility_change_low,  label="Low-income",    color="orange", linewidth=1.5)
axes[1].plot(years, utility_change_med,  label="Middle-income", color="orangered",  linewidth=1.5)
axes[1].plot(years, utility_change_high, label="High-income",   color="maroon",    linewidth=1.5)
axes[1].set_ylabel("Median utility variation (%)")
axes[1].legend(frameon=False, loc="upper right", fontsize=9)

# --- Panel 3 ---
axes[2].plot(years, emission_change,        label="Transport emissions", color="green", linewidth=1.5)
axes[2].plot(years, change_qol_in_zone,     label="Pollution inside the tax zone",         color="navy",   linewidth=1.5)
axes[2].plot(years, change_qol_out_zone,    label="Pollution outside of the tax zone",    color="cyan", linewidth=1.5)
axes[2].set_ylabel("Mean variation (%)")
axes[2].legend(frameon=False, loc="upper right", fontsize=9)
axes[-1].set_xlabel("Year")
axes[-1].set_xticks(list(years)[::2])   # every 2 years

#fig.tight_layout()
plt.show()















#Spatial analysis: weighted opinion and population

plot_change_population_custom(gdf, save_population)
plot_change_pop_line(gdf, save_population)
moving = 0
for i in range(19):
    print(i)
    moving = moving + np.nansum(np.abs(save_population[:, i+1] - save_population[:, i]))/2

weighted_values = {
    lvl: compute_weighted_mean_opinions(acceptable_price[lvl], indiv_loc_matrix[lvl], N[lvl]) * 2
    for lvl in income_levels
}

for lvl in income_levels:
    plot_spatial_opinions(gdf, weighted_values[lvl])

for lvl in income_levels:
    plot_spatial_price(gdf, weighted_values[lvl])




gdf_proj = gdf.to_crs(epsg=32632).copy()  
# Dissolve by municipality to get one polygon per municipality
muni_gdf = gdf_proj.dissolve(by='NMUN', as_index=False)
muni_gdf['centroid'] = muni_gdf.geometry.centroid
muni_gdf['x'] = muni_gdf.centroid.x
muni_gdf['y'] = muni_gdf.centroid.y
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from shapely.affinity import rotate
gdf_proj["geometry"] = gdf_proj["geometry"].apply(lambda geom: rotate(geom, 25, origin='centroid'))
#muni_gdf["geometry"] = muni_gdf["geometry"].apply(lambda geom: rotate(geom, 25, origin='centroid'))

fig, ax = plt.subplots(figsize=(8, 8))

# Census tracts
gdf_proj.plot(
    ax=ax, facecolor="white", edgecolor="grey", linewidth=0.6, label="Census tracts"
)

# Municipal boundaries
muni_gdf.boundary.plot(
    ax=ax, edgecolor="black", linewidth=0.8, label="Municipal boundaries"
)

# Congestion-pricing zone
city_border = muni_gdf[muni_gdf.ID.str[:5].isin(["08019", "08101", "08194"])]
city_border.dissolve().plot(
    ax=ax, facecolor="none", edgecolor="red", linewidth=2.2, label="Toll area"
)

# Municipality labels
sel = ["Badalona", "Castelldefels", "Castellbisbal", "Sant Cugat del Vallès"]
for _, row in muni_gdf.loc[muni_gdf["NMUN"].isin(["Badalona", "Barcelona", "Castelldefels", "Castellbisbal", "Sant Cugat del Vallès"]),:].iterrows(): ax.text(row.x, row.y, row['NMUN'], fontsize=9, fontweight='bold', ha='center', va='center', color='black')

# ----- Legend -----
# Handles for each layer
tracts_handle = mpatches.Patch(facecolor="white", edgecolor="grey", label="Census tracts")
muni_handle   = mlines.Line2D([], [], color="black", linewidth=0.8, label="Municipal boundaries")
toll_handle   = mlines.Line2D([], [], color="red", linewidth=1.2, label="Toll area")

ax.legend(handles=[tracts_handle, muni_handle, toll_handle], loc="lower right")

# Aesthetics
ax.set_axis_off()
plt.tight_layout()
plt.show()


def plot_spatial_price(gdf, values):
    # --- prepare values ---
    gdf_proj = gdf.to_crs(epsg=32632).copy()   # keep projection if needed
    gdf_proj["value"] = values

    # Dissolve by municipality to get one polygon per municipality
    muni_gdf = gdf_proj.dissolve(by='NMUN', as_index=False)
    muni_gdf['centroid'] = muni_gdf.geometry.centroid
    muni_gdf['x'] = muni_gdf.centroid.x
    muni_gdf['y'] = muni_gdf.centroid.y

    # plotting
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    cmap_name = "RdYlGn"
    vmin, vmax = gdf_proj["value"].min(), gdf_proj["value"].max()

    missing = gdf_proj[gdf_proj["value"].isna()]
    present = gdf_proj[gdf_proj["value"].notna()]

    # First: plot missing polygons in grey
    missing.plot(color="lightgrey", edgecolor="white",
             linewidth=0.2, ax=ax, label="Missing data")
    present.plot(column="value",
                  cmap=cmap_name,
                  vmin=vmin, vmax=vmax,
                  linewidth=0, edgecolor="grey",
                  ax=ax)

    ax.set_axis_off()

    # improve rendering
    for coll in ax.collections:
        coll.set_antialiased(False)
        coll.set_alpha(0.7)

    for _, row in muni_gdf.loc[muni_gdf["NMUN"].isin(["Badalona", "Castelldefels", "Castellbisbal", "Sant Cugat del Vallès"]),:].iterrows():
        ax.text(row.x, row.y, row['NMUN'], fontsize=9, fontweight='bold', ha='center', va='center', color='black')
    
    
    city_border = muni_gdf[muni_gdf.ID.str[:5].isin(["08019", "08101", "08194"])]
    city_border.boundary.plot(ax=ax, color='black', linewidth=2, label = "Toll area")
    

    # continuous colorbar (no title)
    sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    ticks = np.linspace(vmin, vmax, 5)
    cbar.set_ticks(ticks)
    cbar.ax.set_yticklabels([f"{t:.1f}" for t in ticks], fontsize=14)  # show 1 decimal + %
    import matplotlib.patches as mpatches

    missing_patch = mpatches.Patch(facecolor="lightgrey", edgecolor="white", label="Income group not represented")
    toll_patch = mpatches.Patch(facecolor="none", edgecolor="black", linewidth=2, label="Toll area")

    ax.legend(handles=[missing_patch, toll_patch], loc="lower right")

    # title
    #ax.set_title("Average opinions (year 20)", fontsize=14)


print(round(100 * sum(np.nansum(gdf[f"transport_mode_{lvl}"] * np.nansum(indiv_loc_matrix[lvl], 0)) for lvl in income_levels) / sum(np.nansum(np.nansum(indiv_loc_matrix[lvl], 0)) for lvl in income_levels)), "% commute by public transport (all levels)")

for lvl in income_levels:
    print(round(100 * np.nansum(gdf[f"transport_mode_{lvl}"] * np.nansum(indiv_loc_matrix[lvl], 0)) / np.nansum(np.nansum(indiv_loc_matrix[lvl], 0))), f"% commute by public transport ({lvl})")

plt.plot(np.nanmedian(compute_change_in_welfare(save_utility["LOW"][:, 0], save_utility["LOW"][:, year]), 0))

plt.plot(np.nanmedian(save_score_welfare["LOW"][:,1:], 0))