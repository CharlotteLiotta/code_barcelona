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

### IMPORT PARAMETERS

path_data = "../data_barcelona/"
option_function = "Cobb-Douglas"

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
pop = {
    "LOW": np.nansum(gdf["pop"] * gdf["share_low_income"] / 100),
    "MED": np.nansum(gdf["pop"] * (100 - gdf["share_high_income"] - gdf["share_low_income"]) / 100),
    "HIGH": np.nansum(gdf["pop"] * gdf["share_high_income"] / 100),
}

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
print("Modal share of public transport (%):", round(100 * sum(np.nansum(gdf[f"transport_mode_{lvl}"] * gdf[f"pop_{lvl}"]) for lvl in income_levels) / sum(np.nansum(gdf[f"pop_{lvl}"]) for lvl in income_levels)))
plot_transport_cost_i(gdf, "_HIGH")
plot_transport_mode_i(gdf, "_HIGH")
#plot_employment(gdf, employment_centers, var) var = employment_centers['employment'] * 0.001, employment_centers.merge(employed_results, left_on = "cluster", right_on = "to_id")["weighted_employed"] * 0.001,  # adjust scale_factor, ARRAY_WAGE* 0.5
#gdf.merge(travel_matrix.loc[travel_matrix.to_id == 5,:], left_on = "ID", right_on = "from_id").plot("proba_center", legend = True)

#Prepare variables
gdf["size"] = gdf["size_census"] #gdf["size_census"] #gdf["size_AMB"]
gdf.loc[gdf["pop"] == 0, "pop"] = 1
gdf.loc[gdf["rent_m2"] == 0, "rent_m2"] = np.nanmin(gdf.loc[gdf["rent_m2"]>0, "rent_m2"])
gdf.loc[np.isnan(gdf["rent_m2"]), "rent_m2"] = np.nanmin(gdf.loc[gdf["rent_m2"]>0, "rent_m2"])
gdf.loc[np.isnan(gdf["size"]), "size"] = np.nansum(gdf.loc[~np.isnan(gdf["size"]), "size"] * gdf.loc[~np.isnan(gdf["size"]), "pop"]) / np.nansum(gdf.loc[~np.isnan(gdf["size"]), "pop"])

#Calibrate beta and amenities
def compute_log_likelihood(x):
    return calibration_utility_amenity(x, gdf, income_levels, SOFT_RENT, 0, 0)

calib_beta = scipy.optimize.minimize(compute_log_likelihood, [0.57, 140, 241, 348], bounds=[(0.1,0.9), (100,2000), (100,2000), (100,2000)])
BETA = calib_beta.x[0]
print("BETA:", BETA)
amenities = calibration_utility_amenity(calib_beta.x, gdf, income_levels, SOFT_RENT, 1, 1)
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

    return compute_error_in_population(u, gdf["amenities"], [pop[lvl] for lvl in income_levels], BETA, gdf["wage_LOW"], gdf["wage_MED"], gdf["wage_HIGH"], gdf["transport_cost_LOW"], gdf["transport_cost_MED"], gdf["transport_cost_HIGH"], B, KAPPA, SIGMA, INTEREST_RATE, gdf["urb_area"], SOFT_RENT, option_function)

solving_model = scipy.optimize.minimize(compute_error_in_population_from_utility, [180, 420, 600], bounds=[(0,None), (0,None), (0,None)], method = "Nelder-Mead") #np.array([399,690,995]) np.array([370,690,800])

if solving_model.fun < 1:
    R, q, n, w, R_group, q_group = compute_outcomes(solving_model.x, gdf, BETA, B, KAPPA, SIGMA, INTEREST_RATE, SOFT_RENT, income_levels, compute_rents, compute_dwelling_size, compute_population, option_function)
else:
    print("Minimization failed!")

density_residual = np.log(gdf["pop"] / n)
rent_residual = np.log(gdf["rent_m2"] / R)
size_residual = np.log(gdf["size"] / q)

# Plot the result of the calibration
#print_maps(gdf, n, q, R)
#print_scatterplots(gdf, n, q, R)
plot_line_charts(gdf, n, q, R)

#initial state
def compute_error_in_population_from_utility(u):
    """ Compute error in population associated to utility u"""

    return compute_error_in_population(u, gdf["amenities"], [pop[lvl] for lvl in income_levels], BETA, gdf["wage_LOW"], gdf["wage_MED"], gdf["wage_HIGH"], gdf["transport_cost_LOW"], gdf["transport_cost_MED"], gdf["transport_cost_HIGH"], B, KAPPA, SIGMA, INTEREST_RATE, gdf["urb_area"], SOFT_RENT, option_function, rent_residual, density_residual, size_residual)


solving_model = scipy.optimize.minimize(compute_error_in_population_from_utility, solving_model.x)

if solving_model.fun < 1:

    R, q, n, w, R_group, q_group = compute_outcomes(solving_model.x, gdf, BETA, B, KAPPA, SIGMA, INTEREST_RATE, SOFT_RENT, income_levels, compute_rents, compute_dwelling_size, compute_population, option_function)

    R = R * np.exp(rent_residual)
    q = q * np.exp(size_residual)
    n = n * np.exp(density_residual)
    n[np.isnan(n)] = 0

else:
    print("Minimization failed!")

plot_line_charts(gdf, n, q, R)

# ABM: translate outputs at the household level
N = {lvl: round(pop[lvl] * SCALE_ABM) for lvl in income_levels}
support = {lvl: INITIAL_OPINION * np.ones(N[lvl]) for lvl in income_levels}
indiv_loc_matrix = {
    lvl: compute_indiv_loc_matrix(
        N[lvl], len(gdf), (n * SCALE_ABM * w[lvl]).to_numpy()
    )
    for lvl in income_levels
}

rent_indiv = {lvl: indiv_loc_matrix[lvl] @ R.to_numpy() for lvl in income_levels}
dwelling_size_indiv = {lvl: indiv_loc_matrix[lvl] @ q_group[lvl] for lvl in income_levels}

utility = {}

for lvl in income_levels:
    u = compute_utility_manually(
        indiv_loc_matrix[lvl] @ gdf[f"wage_{lvl}"],
        indiv_loc_matrix[lvl] @ gdf[f"transport_cost_{lvl}"],
        dwelling_size_indiv[lvl],
        rent_indiv[lvl],
        BETA,
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
save_qol = np.zeros((MAX_YEAR))
save_congestion = np.zeros((MAX_YEAR))

for lvl in income_levels:
    travel_matrix[f"distance_emi_{lvl}"] = (
        (travel_matrix["distance_car"] / 1000)
        * travel_matrix[f"proba_center_{lvl}"]
        * (1 - travel_matrix[f"transport_mode_{lvl}"])
    )
    distance_emi = travel_matrix.groupby("from_id", observed=True)[f"distance_emi_{lvl}"].sum()
    gdf = gdf.merge(distance_emi, left_on="ID", right_index=True, how="left")

emissions_init = np.nansum(sum(
    np.nansum(indiv_loc_matrix[lvl], 0) * gdf[f"distance_emi_{lvl}"] for lvl in income_levels))

save_emissions = np.zeros(MAX_YEAR)
save_emissions[0] = emissions_init

save_score_welfare = {lvl: np.zeros((N[lvl], MAX_YEAR)) for lvl in income_levels}
save_score_qol = {lvl: np.zeros((N[lvl], MAX_YEAR)) for lvl in income_levels}
save_score_congestion = {lvl: np.zeros((N[lvl], MAX_YEAR)) for lvl in income_levels}

save_score_emissions = np.zeros(MAX_YEAR)
save_median_support = np.zeros(MAX_YEAR)

save_qol[0], save_congestion[0], *proba_commuting_in_tax_zone_by_car = compute_qol(
    *(np.nansum(indiv_loc_matrix[lvl], 0) for lvl in income_levels),
    gdf, travel_matrix, jobs_in_toll_area, houses_in_toll_area
)

proba_commuting_in_tax_zone_by_car = {lvl: val for lvl, val in zip(income_levels, proba_commuting_in_tax_zone_by_car)}

year = year + 1
#plot_distance_distrib_check(income_levels, travel_matrix, "HIGH")

### MODELING THE PSC

while year < MAX_YEAR:

    print("YEAR", year)

    # Urban form with the tax, without inertia
    gdf, workers_per_cluster, travel_matrix = compute_transport_cost_poly_i(gdf, travel_time_matrix_car, travel_time_matrix_transit, PRICE_TIME, WORKING_DAYS, FIXED_COST_CAR, PRICE_FUEL, LAMBDA, ARRAY_WAGE_LOW, ARRAY_WAGE_MED, ARRAY_WAGE_HIGH, jobs_in_toll_area, houses_in_toll_area, tax, income_levels, wage_factors)

    def compute_error_in_population_from_utility(u):
        """ Compute error in population associated to utility u"""

        return compute_error_in_population(u, gdf["amenities"], [pop[lvl] for lvl in income_levels], BETA, gdf["wage_LOW"], gdf["wage_MED"], gdf["wage_HIGH"], gdf["transport_cost_LOW"], gdf["transport_cost_MED"], gdf["transport_cost_HIGH"], B, KAPPA, SIGMA, INTEREST_RATE, gdf["urb_area"], SOFT_RENT, option_function, rent_residual, density_residual, size_residual)

    solving_model = scipy.optimize.minimize(compute_error_in_population_from_utility, solving_model.x)

    if solving_model.fun < 1:

        R, q, n, w, R_group, q_group = compute_outcomes(solving_model.x, gdf, BETA, B, KAPPA, SIGMA, INTEREST_RATE, SOFT_RENT, income_levels, compute_rents, compute_dwelling_size, compute_population, option_function)

        R = R * np.exp(rent_residual)
        q = q * np.exp(size_residual)
        n = n * np.exp(density_residual)
        n[np.isnan(n)] = 0

    else:
        print("Minimization failed!")

    housing_without_inertia = n * q

    # AMB
    has_moved = {}
    indiv_loc_matrix_new = {}

    for lvl in income_levels:
        proba_from, proba_to = compute_proba_of_moving(
            save_housing[lvl][:, year - 1],
            (housing_without_inertia * SCALE_ABM * w[lvl]).to_numpy()
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
    
    rent_indiv_new = {lvl: indiv_loc_matrix[lvl] @ R.to_numpy() for lvl in income_levels}
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
            BETA
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
    score_welfare = {lvl: compute_change_in_welfare(save_utility[lvl][:, 0], save_utility[lvl][:, year]) 
                 for lvl in income_levels}
    
    score_emissions, emissions = compute_change_in_emissions(gdf, travel_matrix, emissions_init, np.nansum(indiv_loc_matrix["LOW"], 0), np.nansum(indiv_loc_matrix["MED"], 0), np.nansum(indiv_loc_matrix["HIGH"], 0))
    save_emissions[year] = emissions
    save_score_emissions[year] = score_emissions

    #save outcomes
    save_qol[year], save_congestion[year], proba_commuting_in_tax_zone_by_car["LOW"], proba_commuting_in_tax_zone_by_car["MED"], proba_commuting_in_tax_zone_by_car["HIGH"] = compute_qol(*(np.nansum(indiv_loc_matrix[lvl], 0) for lvl in income_levels), gdf, travel_matrix, jobs_in_toll_area, houses_in_toll_area)

    score_qol_zone = compute_change_in_qol(save_qol[0], save_qol[year])
    score_congestion_zone = compute_change_in_qol(save_congestion[0], save_congestion[year])

    score_qol = {}
    score_congestion = {}
    political_opinion = {}
    price_here = {}
    support = {}
    acceptable_price_new = {}

    for lvl in income_levels:
        # QOL score
        mask_zone = gdf.ID.isin(houses_in_toll_area).to_numpy()
        score_qol[lvl] = (indiv_loc_matrix[lvl] @ mask_zone) * score_qol_zone
        score_qol[lvl][score_qol[lvl] == 0] = 0.5

        # Congestion score
        proba_commuting = np.nan_to_num(proba_commuting_in_tax_zone_by_car[lvl])
        score_congestion[lvl] = (indiv_loc_matrix[lvl] @ proba_commuting) * score_congestion_zone + \
                                (indiv_loc_matrix[lvl] @ (1 - proba_commuting)) * 0.5

        # Political opinion and price
        political_opinion[lvl] = compute_political_opinion(score_welfare[lvl], score_qol[lvl],
                                                       score_emissions, score_congestion[lvl], BETA_OPINION)
        price_here[lvl] = compute_price(score_welfare[lvl], score_qol[lvl],
                                        score_emissions, score_congestion[lvl], BETA_PRICE)
        support[lvl] = political_opinion[lvl]
        acceptable_price_new[lvl] = (INERTIA_OPINION * acceptable_price[lvl]) + ((1 - INERTIA_OPINION) * price_here[lvl])

        # Save scores
        save_score_welfare[lvl][:, year] = score_welfare[lvl]
        save_score_qol[lvl][:, year] = score_qol[lvl]
        save_score_congestion[lvl][:, year] = score_congestion[lvl]

    save_median_support[year] = np.nanmedian(np.concatenate([support[lvl] for lvl in income_levels]))

    #Policy update
    tax = np.nanmedian(np.concatenate([acceptable_price_new[lvl] for lvl in income_levels]))
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


#Spatial analysis: weighted opinion and population
plot_change_population(gdf, save_population)
plot_change_pop_line(gdf, save_population)
moving = 0
for i in range(19):
    print(i)
    moving = moving + np.nansum(np.abs(save_population[:, i+1] - save_population[:, i]))/2

weighted_values = {
    lvl: compute_weighted_mean_opinions(support[lvl], indiv_loc_matrix[lvl], N[lvl]) * 100
    for lvl in income_levels
}

for lvl in income_levels:
    plot_spatial_opinions(gdf, weighted_values[lvl])


print(round(100 * sum(np.nansum(gdf[f"transport_mode_{lvl}"] * np.nansum(indiv_loc_matrix[lvl], 0)) for lvl in income_levels) / sum(np.nansum(np.nansum(indiv_loc_matrix[lvl], 0)) for lvl in income_levels)), "% commute by public transport (all levels)")

for lvl in income_levels:
    print(round(100 * np.nansum(gdf[f"transport_mode_{lvl}"] * np.nansum(indiv_loc_matrix[lvl], 0)) / np.nansum(np.nansum(indiv_loc_matrix[lvl], 0))), f"% commute by public transport ({lvl})")