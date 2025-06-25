import numpy as np 
from copy import deepcopy
from scipy.sparse import csr_matrix # type: ignore
import jpype # type: ignore
import os
os.environ["R5_JAR"] = "C:/Users/1738037/AppData/Local/miniforge3/envs/r5py/Lib/site-packages/r5py/data/r5-v6.8-all.jar"
jpype.startJVM(classpath=[os.environ["R5_JAR"]])

from functions import *
from import_data import * # type: ignore
from calibration import * # type: ignore
from model import * # type: ignore
from plotting_tools import * # type: ignore

###TO DO:
#LOGIT TRANSPORT COST
#BETTER TRANSPORT DATA WITH FGC,...
#EXTRACT DISTANCE AS WELL AS TRAVEL TIME FOR CARS
#GO POLYCENTRIC
#ADD DIFFERENT INCOME GROUPS?
#DO MAX LIKELIHOOD INSTEAD OF MEAN
#HANDLE MISSING / WEIRD VALUES FOR CALIBRATION (RENT, SMALL OR LARGE CENSUS TRACTS)
#ADD AMENITIES

### IMPORT PARAMETERS

path_data = "C:/Users/1738037/OneDrive - UAB/1- CLIMGROW Charlotte/1- PSC cities/data_barcelona/"

#Time
year = 0
MAX_YEAR = 20

#Policy impact model
RHO = 0.05 + 0.93 #Interest rate + depreciation rate of built capital
PRICE_TIME = 10 #euros/h
WORKING_DAYS = 40 #20 days per month, with 2 trips per day
PRICE_FUEL = 0.11 #euros/km

#ABM
N = 10000 #Nb of agents in the ABM
PROBA_MOVE = 0.3

#Policy support model
#I = np.random.randint(1,10, N)
#DELTA = 0.5
#GAMMA = 0.25
#BETA_OPINION = [0.37, 0.50, 0.21, 0.01, -0.01, -0.01, 0] #[0.33, 0.33, 0.33, 0.01, -0.01, -0.01, 0] #BETA_OPINION = [0.048, 0.058, 0.033, 0.00, -0.00, -0.00, 0.129]

#Tax
tax = 1 #Initial tax level
RATE_INCREASE_TAX = 0.05 #Rate of increase per year

### IMPORT DATA

#Import data
gdf = import_data(option = "SECTION") #"DISTRICT" or "SECTION" - Active population: 1.4M
gdf = import_jobs(gdf)
Y, gdf = import_income(gdf)
gdf = import_land_use(gdf)
gdf = import_rent_and_size(gdf)
gdf = import_ppl_per_hh(gdf)

#Transport times
#import_transport_times(gdf, datetime.datetime(2025, 7, 7, 8, 0, 0), "0801910130", 1)
travel_time_matrix_car, travel_time_matrix_transit = load_transport_times(gdf)
gdf = add_transport(gdf, travel_time_matrix_car, travel_time_matrix_transit, "0801910130")
gdf = import_cost_transit(gdf)

### INITIAL STATE: YEAR 0

#Transport cost calibration
gdf, FIXED_COST_CAR = compute_cost_car(gdf, import_trans_mode, PRICE_TIME, WORKING_DAYS, PRICE_FUEL)
print("FIXED_COST_CAR: ", FIXED_COST_CAR)

#Compute transport cost
gdf = compute_transport_cost(gdf, PRICE_TIME, WORKING_DAYS, FIXED_COST_CAR, PRICE_FUEL, tax = 0)
print(round(100 * sum(gdf["transport_mode"] * gdf["pop"]) / sum(gdf["pop"])), " % commute by public transport")

#Calibration BETA
BETA = calibrate_beta(gdf, Y)
print("BETA: ", BETA)

#Calibration B and KAPPA
mask = ((gdf["log_R"] < 3.2) &(gdf["log_R"] > 2) &(~np.isnan(gdf["log_n"])) & (~np.isnan(gdf["log_R"]))&(~np.isnan(gdf["log_L"]))&(~np.isnan(gdf["log_q"])))
B, KAPPA = calibrate_b_kappa(gdf, mask, option_calib = "housing")

# Solve the model
def compute_error_in_population_from_utility(u):
    """ Compute error in population associated to utility u"""

    return compute_error_in_population(u, np.nansum(gdf["pop"]), BETA, Y, gdf["transport_cost"], B, KAPPA, RHO, gdf["urb_area"])

solving_model = scipy.optimize.minimize(compute_error_in_population_from_utility, 100)

if solving_model.fun < 1:
    utility = solving_model.x
    R = compute_rents(BETA, Y, utility, gdf["transport_cost"])
    q = compute_dwelling_size(BETA, Y, gdf["transport_cost"], R)
    n = compute_population(B, KAPPA, R, RHO, gdf["urb_area"], q)
else:
    print("Minimization failed!")

# Plot the result of the calibration
map_calibration(gdf, n, gdf["pop"] , "Population")
map_calibration(gdf, q, gdf["size"] / gdf["active_per_hh"], "Dwelling size per capita")
map_calibration(gdf, R, gdf["rent_m2"], "Rent per m2")
map_calibration(gdf, 1000000 * n / gdf["urb_area"], 1000000 * gdf["pop"] / gdf["urb_area"], "Population density")

scatter_calibration(gdf, n, gdf["pop"] , "Population")
scatter_calibration(gdf, q, gdf["size"] / gdf["active_per_hh"], "Dwelling size per capita")
scatter_calibration(gdf, R, gdf["rent_m2"], "Rent per m2")
scatter_calibration(gdf, 1000000 * n / gdf["urb_area"], 1000000 * gdf["pop"] / gdf["urb_area"], "Population density")

# ABM: translate outputs at the household level
#### n = n / 140
indiv_loc_matrix = compute_indiv_loc_matrix(N, len(gdf), n.to_numpy())
missing_ppl = N - sum(sum(indiv_loc_matrix))
if missing_ppl > 0:
    print("missing ppl", missing_ppl)
    indiv_loc_matrix[round(N-missing_ppl):(N),0] = np.ones(round(missing_ppl))

rent_indiv = indiv_loc_matrix @ R.to_numpy()
dwelling_size_indiv = indiv_loc_matrix @ q
utility = compute_utility_manually(Y, indiv_loc_matrix @gdf["transport_cost"], 
                                                dwelling_size_indiv, rent_indiv, BETA)

indiv_distance_sparse = csr_matrix(indiv_loc_matrix)
dwelling_size_indiv = dwelling_size_indiv.astype(np.float64)
housing_indiv = dwelling_size_indiv @ indiv_distance_sparse  # shape: (10,)

# Save outputs

save_housing = np.zeros((len(gdf["area"]), MAX_YEAR))
save_housing[:, 0] = deepcopy(housing_indiv)

save_rent = np.zeros((N, MAX_YEAR))
save_rent[:, 0] = deepcopy(rent_indiv)

save_dwelling_size = np.zeros((N, MAX_YEAR))
save_dwelling_size[:, 0] = deepcopy(dwelling_size_indiv)

save_population = np.zeros((len(gdf["area"]), MAX_YEAR))
save_population[:, 0] = np.nansum(indiv_loc_matrix, 0)

save_transport_mode = np.zeros((N, MAX_YEAR))
save_transport_mode[:, 0] = deepcopy(indiv_loc_matrix @gdf["transport_mode"])

save_utility = np.zeros((N, MAX_YEAR))
save_utility[:, 0] = deepcopy(utility)

save_tax = np.zeros(MAX_YEAR)
save_tax[0] = 1

emissions_init = sum((save_population[:, 0] * gdf["distance_center"])[gdf["transport_mode"] == 0])

save_median_support = np.zeros(MAX_YEAR)

year = year + 1

### MODELING THE PSC

while year < MAX_YEAR:
    print("YEAR", year)

    #Urban form with the tax, without inertia

    gdf = compute_transport_cost(gdf, PRICE_TIME, WORKING_DAYS, FIXED_COST_CAR, FIXED_COST_TRANSPORT, tax)

    def compute_error_in_population_from_utility(u):
        """ Compute error in population associated to utility u"""

        return compute_error_in_population(u, N, BETA, Y, gdf["transport_cost"], B, KAPPA, RHO, gdf["area"])

    solving_model = scipy.optimize.minimize(compute_error_in_population_from_utility, utility[0])

    if solving_model.fun < 1:
        utility = solving_model.x
        R = compute_rents(BETA, Y, utility, gdf["transport_cost"])
        q = compute_dwelling_size(BETA, Y, gdf["transport_cost"], R)
        n = compute_population(B, KAPPA, R, RHO, gdf["area"], q)
    else:
        print("Minimization failed!")

    housing_without_inertia = n * q

    proba_of_moving_from, proba_of_moving_to = compute_proba_of_moving(save_housing[:, year - 1], housing_without_inertia.to_numpy())

    indiv_loc_matrix_new = deepcopy(indiv_loc_matrix)
    indiv_loc_matrix, has_moved = make_people_move(indiv_loc_matrix_new, N, len(gdf), indiv_loc_matrix, proba_of_moving_from, proba_of_moving_to, PROBA_MOVE)

    #Rents and dwelling sizes are updated for the households that have moved only
    rent_indiv_new = indiv_loc_matrix @ R
    dwelling_size_indiv_new = indiv_loc_matrix @ q
    rent_indiv[has_moved == 1] = rent_indiv_new[has_moved == 1]
    dwelling_size_indiv[has_moved == 1] = dwelling_size_indiv_new[has_moved == 1]

    utility = compute_utility_manually(Y, indiv_loc_matrix @gdf["transport_cost"], 
                                                dwelling_size_indiv, rent_indiv, BETA)

    indiv_distance_sparse = csr_matrix(indiv_loc_matrix)
    dwelling_size_indiv = dwelling_size_indiv.astype(np.float64)
    housing_indiv = dwelling_size_indiv @ indiv_distance_sparse  # shape: (10,)
    save_housing[:, year] = deepcopy(housing_indiv)

    save_rent[:, year] = deepcopy(rent_indiv)
    save_dwelling_size[:, year] = deepcopy(dwelling_size_indiv)
    save_population[:, year] = np.nansum(indiv_loc_matrix, 0)
    save_utility[:, year] = deepcopy(utility)
    save_tax[year] = tax
    save_transport_mode[:, year] = deepcopy(indiv_loc_matrix@gdf["transport_mode"])

    score_welfare = compute_change_in_welfare(save_utility[:,0], save_utility[:,year])
    score_ineq = compute_change_in_inequalities(save_utility[:,0], save_utility[:,year])
    score_emissions = compute_change_in_emissions(save_population[:, year], gdf["distance_center"], gdf["transport_mode"], emissions_init)

    #Policy support

    political_opinion = compute_political_opinion(score_welfare, score_ineq, score_emissions, I, BETA_OPINION, indiv_loc_matrix)
    
    if year > 1:
        support = (DELTA * support) + ((1 - DELTA) * compute_social_interactions(political_opinion, I, GAMMA)) # type: ignore
    else:
        support = compute_social_interactions(political_opinion, I, GAMMA)
   
    #support = 0.24 * score_welfare + 0.29 * score_ineq + 0.165 * score_emissions
    print("score_welfare", score_welfare)
    print("support", support)
    #Policy update

    if np.nanmedian(support) > 0.51:
        tax = tax * (1 + RATE_INCREASE_TAX)

    save_median_support[year] = np.nanmedian(support)
    year = year + 1


gdf.plot(compute_weighted_mean_opinions(support, indiv_loc_matrix, N), legend = True)
plot_tax_suppport(save_tax, save_median_support)