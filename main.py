import numpy as np 
from copy import deepcopy
from scipy.sparse import csr_matrix # type: ignore
import jpype # type: ignore
import os
#os.environ["R5_JAR"] = "C:/Users/1738037/AppData/Local/miniforge3/envs/r5py/Lib/site-packages/r5py/data/r5-v6.8-all.jar"
#jpype.startJVM(classpath=[os.environ["R5_JAR"]])
import datetime
import warnings
import pickle
import copy
from stargazer.stargazer import Stargazer

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

#Time
year = 0
MAX_YEAR = 20

#Policy impact model
INTEREST_RATE = 0.05 + 0.93 #Interest rate + depreciation rate of built capital
PRICE_TIME = 10 #euros/h
WORKING_DAYS = 40 #20 days per month, with 2 trips per day
PRICE_FUEL = 0.11 #euros/km
center = "0801901025"

#ABM
SCALE_ABM = 1/100 #Nb of agents in the ABM
PROBA_MOVE = 0.2
INERTIA_OPINION = 0.5 #0.8 #inertia

#Tax
#tax = 3 #Initial tax level
#RATE_INCREASE_TAX = 0.00 #Rate of increase per year
#OPINION_THRESHOLD = 0.00

### IMPORT DATA

#Import data
gdf = import_data(path_data, center, option = "SECTION") #"DISTRICT" or "SECTION" - Active population: 1.4M
gdf = import_jobs(gdf, path_data)
gdf = import_land_use(gdf, path_data)
gdf = import_ppl_per_hh(gdf, path_data)
gdf = import_rent_and_size(gdf, path_data)
Y, gdf = import_income(gdf, path_data)
gdf = import_amenities(gdf, path_data, 0, 0)
employment_centers = gpd.read_file(path_data + "cluster_employment.shp")
jobs_in_toll_area, houses_in_toll_area = import_tax_zone(gdf, employment_centers)

#Import transport data
#import_transport_times_poly(gdf, datetime.datetime(2025, 7, 15, 0, 0, 0), center, path_data, employment_centers) #datetime.datetime(2025, 7, 15, 8, 0, 0)
travel_time_matrix_car, travel_time_matrix_transit = load_transport_times_poly(gdf, path_data, center)
travel_time_matrix_car = load_distance_car_poly(travel_time_matrix_car, gdf, employment_centers)
gdf = import_cost_transit(gdf)
travel_time_matrix_transit = travel_time_matrix_transit.merge(gdf[['ID', 'monthly_cost_transit']].rename(columns={'ID': 'from_id'}), on='from_id', how='left')

#Transport times
#import_transport_times(gdf, datetime.datetime(2025, 7, 15, 8, 0, 0), center, path_data, 1)
#import_car_distance(gdf, datetime.datetime(2025, 7, 15, 8, 0, 0), center, path_data)
#travel_time_matrix_car, travel_time_matrix_transit = load_transport_times(gdf, path_data, center)
#travel_distance_matrix_car = load_transport_distance(gdf, path_data, center)
#gdf = add_transport(gdf, travel_time_matrix_car, travel_time_matrix_transit, center)
#gdf = gdf.merge(travel_distance_matrix_car.loc[travel_distance_matrix_car.to_id == center,:], left_on = "ID", right_on = "from_id", how = "left").drop(columns = ['from_id', 'to_id'])
#center_centroid = gdf.loc[gdf['ID'] == center].geometry.centroid.values[0]
#gdf.loc[np.isnan(gdf.distance_car), 'distance_car'] = gdf.geometry.centroid.distance(center_centroid).loc[np.isnan(gdf.distance_car)]
#gdf = import_cost_transit(gdf)

### POLICY SUPPORT

BETA_OPINION, INITIAL_OPINION = import_opinion_parameters(path_data)
BETA_PRICE, INITIAL_PRICE = import_price_parameters(path_data)

#INITIAL_OPINION = compute_political_opinion(0.5, 0.5, 0.5, 0.5, BETA_OPINION)
#INITIAL_PRICE = compute_price(0.5, 0.5, 0.5, 0.5, BETA_PRICE)


#df_reg[["climate_change", "declared_impacts", "quality_of_life", 'congestion']]
#-0.03, 0.21, -0.08, 0.25, 0.28
#BETA_OPINION = np.array([0, 0.1, -0.7, 0.1, 0.1])

tax = INITIAL_PRICE
acceptable_price = INITIAL_PRICE

### INITIAL STATE: YEAR 0

#Transport cost calibration
#gdf, FIXED_COST_CAR, LAMBDA = compute_cost_car_logit(gdf, Y, import_trans_mode, PRICE_TIME, WORKING_DAYS, PRICE_FUEL, path_data)
#gdf, FIXED_COST_CAR, LAMBDA, ARRAY_WAGE = compute_cost_car_poly(gdf, Y, import_trans_mode, PRICE_TIME, WORKING_DAYS, PRICE_FUEL, travel_time_matrix_car, travel_time_matrix_transit, employment_centers, path_data, jobs_in_toll_area, houses_in_toll_area)
#with open(path_data + "calib_trans_poly.pkl", "wb") as f:
#    pickle.dump((FIXED_COST_CAR, LAMBDA, ARRAY_WAGE), f)
with open(path_data + "calib_trans_poly.pkl", "rb") as f:
    FIXED_COST_CAR, LAMBDA, ARRAY_WAGE = pickle.load(f)
print("FIXED_COST_CAR: ", FIXED_COST_CAR)
print("LAMBDA: ", LAMBDA)
del Y
del compute_transport_cost_logit, compute_cost_car_logit

#Compute transport cost
#gdf = compute_transport_cost_logit(gdf, Y, PRICE_TIME, WORKING_DAYS, FIXED_COST_CAR, PRICE_FUEL, LAMBDA, tax = 0)
gdf, workers_per_cluster, travel_matrix = compute_transport_cost_poly(gdf, travel_time_matrix_car, travel_time_matrix_transit, PRICE_TIME, WORKING_DAYS, FIXED_COST_CAR, PRICE_FUEL, LAMBDA, ARRAY_WAGE, jobs_in_toll_area, houses_in_toll_area, tax = 0)

net_income_t0 = gdf["income_net_of_transport_cost"]

print(round(100 * sum(gdf["transport_mode"] * gdf["pop"]) / sum(gdf["pop"])), " % commute by public transport")
plot_transport_cost(gdf)
plot_transport_mode(gdf)
#plot_employment(gdf, employment_centers, var) var = employment_centers['employment'] * 0.001, employment_centers.merge(employed_results, left_on = "cluster", right_on = "to_id")["weighted_employed"] * 0.001,  # adjust scale_factor, ARRAY_WAGE* 0.5
#gdf.merge(travel_matrix.loc[travel_matrix.to_id == 5,:], left_on = "ID", right_on = "from_id").plot("proba_center", legend = True)

#Calibration BETA with amenities
gdf["size"] = gdf["size_census"] #gdf["size_census"] #gdf["size_AMB"]

def calibration_utility_amenity(x, print_summary, export_amenities):
    """ Do the calibration on BETA and AMENITIES by minimizing likelihood """
    
    BETA, U = x
    print(f"x = {x}")

    #Log-likelihood on dwelling size
    #estimated_size = BETA * gdf["income_net_of_transport_cost"]  / gdf["rent_m2"]
    estimated_size = BETA * (gdf["wage"] - gdf["transport_cost"])  / gdf["rent_m2"]
    diff_size = gdf["size"] - estimated_size
    mask = ((~np.isnan(diff_size)) & (~np.isinf(diff_size)))
    epsilon_size = np.nansum(diff_size.loc[mask] ** 2) / sum(mask)
    log_L = - sum(mask)/2 * np.log(2 * np.pi * epsilon_size) - (1 / 2*epsilon_size) * np.nansum(diff_size.loc[mask] ** 2)
    print("log_L = ", log_L)

    #Log-likelihood on amenities
    #estimated_A = U / (((1-BETA) ** (1-BETA)) * (BETA ** BETA) * (gdf["income_net_of_transport_cost"]  / gdf["rent_m2"]))
    estimated_A = U / (((1-BETA) ** (1-BETA)) * (BETA ** BETA) * ((gdf["wage"] - gdf["transport_cost"])  / gdf["rent_m2"]))
    with np.errstate(divide='ignore', invalid='ignore'):
        gdf["log_A"] = np.log(estimated_A.replace([np.inf, -np.inf], np.nan))
    gdf_here = gdf.loc[~np.isnan(gdf.log_A) & ~np.isinf(gdf.log_A),:]
    y = gdf_here["log_A"]
    X = gdf_here.loc[:,["beach_500m", "parc_500m", "parc_500m_1km", "parc_1km_2km", "parc_500m_b", "parc_500m_1km_b", "parc_1km_2km_b", "station_500m", "station_500m_1km", "station_1km_2km", "airport_500m", "high_tourism", 'mean_activity', 'pedestrian_density', 'slope_20', 'fgc_500m', 'rodalies_500m']]
    X = sm.add_constant(X)  # Adds intercept
    model_statsmodel = sm.OLS(y, X).fit()
    if print_summary == 1:
        print(model_statsmodel.summary())
        stargazer = Stargazer([model_statsmodel])
        print(stargazer.render_latex())
    residuals = model_statsmodel.resid
    epsilon_A = np.nansum(np.exp(residuals) ** 2) / sum((~np.isnan(gdf.log_A) & ~np.isinf(gdf.log_A)))
    log_L_A = - (sum(~np.isnan(estimated_A))/2) * np.log(2 * np.pi * epsilon_A) - (1 / (2 * epsilon_A)) * np.nansum(np.exp(residuals) ** 2)
    print("log_L_A = ", log_L_A)

    #Export results
    if export_amenities == 1:
        amenities = np.exp(np.nansum(X.iloc[:,1:] * model_statsmodel.params.iloc[1:], 1))
        gdf_here = gdf_here.copy()
        gdf_here.loc[:, "amenities"] = amenities
        return gdf_here.loc[:,["ID", "amenities"]]
    else:
        return - (log_L+log_L_A)
    
def compute_log_likelihood(x):
    return calibration_utility_amenity(x, 0, 0)

calib_beta = scipy.optimize.minimize(compute_log_likelihood, [0.35, 700], bounds=[(0,1), (0,None)])
BETA = calib_beta.x[0]
amenities = calibration_utility_amenity(calib_beta.x, 1, 1)
gdf = gdf.merge(amenities, on = "ID", how = "left")
gdf.loc[np.isnan(gdf["amenities"]), "amenities"] = 1

#Calibration B and KAPPA
gdf["land"] = gdf["urb_area"]

mask = ((gdf["rent_m2"] < 22) &(gdf["rent_m2"] > 7)
        #&(gdf["land"] > 10000) &(gdf["land"] <10000000)
        #&(1000000 * gdf["pop"] / gdf["land"] > 200)
        #&(gdf["pop"] > 200)
        #&(gdf["pop"] < 1500)
        #&(gdf["size"] < 90)
        &(~np.isnan(gdf["size"]))
        &(~np.isnan(gdf["pop"]))
        &((gdf["pop"] > 0))
        #&(~np.isnan(gdf["active_per_hh"]))
        )

B, KAPPA = calibrate_b_kappa(gdf, mask, INTEREST_RATE, option_calib = "housing")

# Solve the model
def compute_error_in_population_from_utility(u):
    """ Compute error in population associated to utility u"""

    return compute_error_in_population(u / gdf["amenities"], np.nansum(gdf["pop"]), BETA, gdf["wage"], gdf["transport_cost"], B, KAPPA, INTEREST_RATE, gdf["urb_area"])

solving_model = scipy.optimize.minimize(compute_error_in_population_from_utility, 700)

if solving_model.fun < 1:
    utility = solving_model.x
    R = compute_rents(BETA, gdf["wage"], utility / gdf["amenities"], gdf["transport_cost"])
    q = compute_dwelling_size(BETA, gdf["wage"], gdf["transport_cost"], R)
    n = compute_population(B, KAPPA, R, INTEREST_RATE, gdf["urb_area"], q)
else:
    print("Minimization failed!")

density_residual = np.log(gdf["pop"] / n)
density_residual[gdf["pop"] == 0] = 0
rent_residual = np.log(gdf["rent_m2"] / R)
rent_residual[gdf["rent_m2"] == 0] = 0
rent_residual[np.isnan(gdf["rent_m2"])] = 0
size_residual = np.log(gdf["size"] / q)
size_residual[np.isnan(gdf["size"])] = np.nanmean(size_residual)

# Plot the result of the calibration
map_calibration(gdf, n, gdf["pop"] , "Population")
map_calibration(gdf, q, gdf["size"], "Dwelling size per capita")
map_calibration(gdf, R, gdf["rent_m2"], "Rent per m2")
map_calibration(gdf, 1000000 * n / gdf["urb_area"], 1000000 * gdf["pop"] / gdf["urb_area"], "Population density")

scatter_calibration(gdf, n, gdf["pop"] , "Population")
scatter_calibration(gdf, q, gdf["size"], "Dwelling size per capita")
scatter_calibration(gdf, R, gdf["rent_m2"], "Rent per m2")
scatter_calibration(gdf, n * q / gdf["urb_area"], gdf["pop"] * gdf["size"] / gdf["urb_area"], "Housing")
scatter_calibration(gdf, 1000000 * n / gdf["urb_area"], 1000000 * gdf["pop"] / gdf["urb_area"], "Population density")

agg = compare_rent_or_size(gdf, "size", q, 1)
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

    return compute_error_in_population(u / gdf["amenities"], np.nansum(gdf["pop"]), BETA, gdf["wage"], gdf["transport_cost"], B, KAPPA, INTEREST_RATE, gdf["urb_area"], rent_residual, density_residual, size_residual)

solving_model = scipy.optimize.minimize(compute_error_in_population_from_utility, 700)

if solving_model.fun < 1:
    utility = solving_model.x
    R = compute_rents(BETA, gdf["wage"], utility / gdf["amenities"], gdf["transport_cost"])
    q = compute_dwelling_size(BETA, gdf["wage"], gdf["transport_cost"], R)
    n = compute_population(B, KAPPA, R, INTEREST_RATE, gdf["urb_area"], q)
    R = R * np.exp(rent_residual)
    q = q * np.exp(size_residual)
    n = n * np.exp(density_residual)
    n[np.isnan(n)] = 0
else:
    print("Minimization failed!")

agg = compare_rent_or_size(gdf, "size", q, 1)
agg = compare_rent_or_size(gdf, "rent_m2", R, 1)
agg = compare_var(gdf, n)

# ABM: translate outputs at the household level
N = round(np.nansum(gdf["pop"]) * SCALE_ABM)
support = INITIAL_OPINION * np.ones(N) #0.417


indiv_loc_matrix = compute_indiv_loc_matrix(N, len(gdf), n.to_numpy()* SCALE_ABM)
indiv_loc_matrix_0 = copy.deepcopy(indiv_loc_matrix)
rent_indiv = indiv_loc_matrix @ R.to_numpy()
dwelling_size_indiv = indiv_loc_matrix @ q

utility = compute_utility_manually(indiv_loc_matrix @gdf["wage"], indiv_loc_matrix @gdf["transport_cost"], 
                                                dwelling_size_indiv, rent_indiv, BETA)

housing_indiv = dwelling_size_indiv @ csr_matrix(indiv_loc_matrix)  # shape: (10,)

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
save_tax[0] = 0
save_qol = np.zeros((MAX_YEAR))
save_congestion = np.zeros((MAX_YEAR))

#save emissions
travel_matrix["distance_emi"] = (travel_matrix["distance_car"] /1000) * travel_matrix["proba_center"] * (1 - travel_matrix["transport_mode"])
distance_emi = travel_matrix.loc[:,["distance_emi", "from_id"]].groupby("from_id").sum()
gdf = gdf.merge(distance_emi, left_on = "ID", right_index = True)
emissions_init = sum(save_population[:, 0] * (gdf["distance_emi"]))
save_emissions = np.zeros(MAX_YEAR)
save_emissions[0] = emissions_init


save_score_emissions = np.zeros(MAX_YEAR)
save_score_welfare = np.zeros((N, MAX_YEAR))
save_score_qol = np.zeros((N, MAX_YEAR))
save_score_congestion = np.zeros((N, MAX_YEAR))

#check distances per capita
travel_matrix["pop"] = travel_matrix["pop"] * travel_matrix["proba_center"]
bins = np.array([0, 0.5, 2, 5, 10, 50])
labels = [f"{i}km" for i in bins[:-1]]
travel_matrix['distance_bin'] = pd.cut(travel_matrix['distance_car'] / 1000, bins=bins, labels=labels, right=False)
pop_by_bin = travel_matrix.groupby('distance_bin', observed=True)['pop'].sum()

pop_by_bin.plot(kind='bar', figsize=(8, 4))
plt.ylabel("Population")
plt.xlabel("Distance to center")
plt.title("Population by distance category")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

save_median_support = np.zeros(MAX_YEAR)

year = year + 1



#### COMPUTE QOL

save_qol[0], save_congestion[0], proba_commuting_in_tax_zone_by_car = compute_qol(save_population[:, 0], gdf, travel_matrix, jobs_in_toll_area, houses_in_toll_area)

### MODELING THE PSC

while year < MAX_YEAR:

    print("YEAR", year)

    # Urban form with the tax, without inertia

    #gdf = compute_transport_cost_logit(gdf, Y, PRICE_TIME, WORKING_DAYS, FIXED_COST_CAR, PRICE_FUEL, LAMBDA, tax = tax)
    gdf, workers_per_cluster, travel_matrix = compute_transport_cost_poly(gdf, travel_time_matrix_car, travel_time_matrix_transit, PRICE_TIME, WORKING_DAYS, FIXED_COST_CAR, PRICE_FUEL, LAMBDA, ARRAY_WAGE, jobs_in_toll_area, houses_in_toll_area, tax)

    if year == 1:
        net_income_t1 = gdf["income_net_of_transport_cost"]

    def compute_error_in_population_from_utility(u):
        """ Compute error in population associated to utility u"""

        return compute_error_in_population(u / gdf["amenities"], np.nansum(gdf["pop"]), BETA, gdf["wage"], gdf["transport_cost"], B, KAPPA, INTEREST_RATE, gdf["urb_area"], rent_residual, density_residual, size_residual)

    solving_model = scipy.optimize.minimize(compute_error_in_population_from_utility, np.nanmedian(utility))

    if solving_model.fun < 1:
        utility = solving_model.x
        R = compute_rents(BETA, gdf["wage"], utility / gdf["amenities"], gdf["transport_cost"])
        q = compute_dwelling_size(BETA, gdf["wage"], gdf["transport_cost"], R)
        n = compute_population(B, KAPPA, R, INTEREST_RATE, gdf["urb_area"], q)
        R = R * np.exp(rent_residual)
        q = q * np.exp(size_residual)
        n = n * np.exp(density_residual)
        n[np.isnan(n)] = 0
    else:
        print("Minimization failed!")

    housing_without_inertia = n * q

    # AMB
    proba_of_moving_from, proba_of_moving_to = compute_proba_of_moving(save_housing[:, year - 1], housing_without_inertia.to_numpy()* SCALE_ABM)
    indiv_loc_matrix_new = deepcopy(indiv_loc_matrix)
    indiv_loc_matrix, has_moved = make_people_move(indiv_loc_matrix_new, N, len(gdf), indiv_loc_matrix, proba_of_moving_from, proba_of_moving_to, PROBA_MOVE)
    
    
    rent_indiv_new = indiv_loc_matrix @ R
    dwelling_size_indiv_new = indiv_loc_matrix @ q
    rent_indiv[has_moved == 1] = rent_indiv_new[has_moved == 1]
    dwelling_size_indiv[has_moved == 1] = dwelling_size_indiv_new[has_moved == 1]

    utility = compute_utility_manually(indiv_loc_matrix @ gdf["wage"], indiv_loc_matrix @gdf["transport_cost"], 
                                                dwelling_size_indiv, rent_indiv, BETA)

    housing_indiv = dwelling_size_indiv @ csr_matrix(indiv_loc_matrix)  # shape: (10,)

    #Save outputs
    save_housing[:, year] = deepcopy(housing_indiv)
    save_rent[:, year] = deepcopy(rent_indiv)
    save_dwelling_size[:, year] = deepcopy(dwelling_size_indiv)
    save_population[:, year] = np.nansum(indiv_loc_matrix, 0)
    save_utility[:, year] = deepcopy(utility)
    save_tax[year] = tax
    save_transport_mode[:, year] = deepcopy(indiv_loc_matrix@gdf["transport_mode"])

    # Policy support
    score_welfare = compute_change_in_welfare(save_utility[:,0], save_utility[:,year])
    score_emissions, emissions = compute_change_in_emissions(gdf, travel_matrix, emissions_init, save_population[:, year])

    
    #save outcomes
    save_qol[year], save_congestion[year], proba_commuting_in_tax_zone_by_car = compute_qol(save_population[:, year], gdf, travel_matrix, jobs_in_toll_area, houses_in_toll_area)

    score_qol_zone = compute_change_in_qol(save_qol[0], save_qol[year])
    
    score_qol = (indiv_loc_matrix @ gdf.ID.isin(houses_in_toll_area)) * score_qol_zone
    score_qol[score_qol == 0] = 0.5

    
    score_congestion_zone = compute_change_in_qol(save_congestion[0], save_congestion[year])

    proba_commuting_in_tax_zone_by_car[np.isnan(proba_commuting_in_tax_zone_by_car)] = 0
    score_congestion = (indiv_loc_matrix @ proba_commuting_in_tax_zone_by_car) * score_congestion_zone + (indiv_loc_matrix @ (1-proba_commuting_in_tax_zone_by_car)) * 0.5


    political_opinion = compute_political_opinion(score_welfare, score_qol, score_emissions, score_congestion, BETA_OPINION)
    price_here = compute_price(score_welfare, score_qol, score_emissions, score_congestion, BETA_PRICE)
    
    save_emissions[year] = emissions
    save_score_emissions[year] = score_emissions
    save_score_welfare[:,year] = score_welfare
    save_score_qol[:,year] = score_qol
    save_score_congestion[:,year] = score_congestion


    support = (INERTIA_OPINION * support) + ((1 - INERTIA_OPINION) * political_opinion) # type: ignore
    acceptable_price = (INERTIA_OPINION * acceptable_price) + ((1 - INERTIA_OPINION) * price_here) # type: ignore
   
    save_median_support[year] = np.nanmedian(support)
    print("support", support)

    #Policy update
    tax = np.nanmedian(acceptable_price)
    #tax = np.fmin(tax * 1.05, np.nanmedian(acceptable_price))

    year = year + 1


### PLOT RESULTS
print(round(100 * sum(gdf["transport_mode"] * gdf["pop"]) / sum(gdf["pop"])), " % commute by public transport")

values = compute_weighted_mean_opinions(support, indiv_loc_matrix, N) * 100
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
plot_scores(save_score_emissions, save_score_qol, save_score_congestion, save_score_welfare)

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