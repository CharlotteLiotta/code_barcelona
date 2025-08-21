import numpy as np 
from copy import deepcopy
from scipy.sparse import csr_matrix # type: ignore
import jpype # type: ignore
import os
os.environ["R5_JAR"] = "C:/Users/1738037/AppData/Local/miniforge3/envs/r5py/Lib/site-packages/r5py/data/r5-v6.8-all.jar"
jpype.startJVM(classpath=[os.environ["R5_JAR"]])
import datetime
import warnings
import pickle
import copy

from functions import *
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
RHO = 0.05 + 0.93 #Interest rate + depreciation rate of built capital
PRICE_TIME = 10 #euros/h
WORKING_DAYS = 40 #20 days per month, with 2 trips per day
PRICE_FUEL = 0.11 #euros/km
center = "0801901025"

#ABM
SCALE = 1/100 #Nb of agents in the ABM
PROBA_MOVE = 0.1
DELTA = 0.5 #inertia

#Tax
tax = 5 #Initial tax level
RATE_INCREASE_TAX = 0.05 #Rate of increase per year
THRESHOLD = 0.5

### IMPORT DATA

#Import data
gdf = import_data(path_data, center, option = "SECTION") #"DISTRICT" or "SECTION" - Active population: 1.4M
gdf = import_jobs(gdf, path_data)
gdf = import_land_use(gdf, path_data)
gdf = import_ppl_per_hh(gdf, path_data)
gdf = import_rent_and_size(gdf, path_data)
Y, gdf = import_income(gdf, path_data)
gdf = import_amenities(gdf, path_data, 0, 0)

zone_tax = gdf.loc[gdf.ID.str[:5].isin(["08019", "08101", "08194"]),:]
fig, ax = plt.subplots(figsize=(8, 8))
gdf.plot(ax = ax, color = "lightgrey")
zone_tax.plot(ax = ax)

zone_union = zone_tax.unary_union
employment_centers = gpd.read_file(path_data + "cluster_employment.shp")
points_in_zone = employment_centers[employment_centers.within(zone_union)]
clusters_in_zone = points_in_zone["cluster"].unique().tolist()
points_in_zone = gdf[gdf.centroid.within(zone_union)]
house_in_zone = points_in_zone["ID"].unique().tolist()

#Import transport data
employment_centers = gpd.read_file(path_data + "cluster_employment.shp")
job_in_tax = list(employment_centers.cluster)
#import_transport_times_poly(gdf, datetime.datetime(2025, 7, 15, 8, 0, 0), center, path_data, employment_centers)
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

BETA_OPINION = import_opinion_parameters(path_data)

### INITIAL STATE: YEAR 0

#Transport cost calibration
#gdf, FIXED_COST_CAR, LAMBDA = compute_cost_car_logit(gdf, Y, import_trans_mode, PRICE_TIME, WORKING_DAYS, PRICE_FUEL, path_data)
#gdf, FIXED_COST_CAR, LAMBDA, ARRAY_WAGE = compute_cost_car_poly(gdf, Y, import_trans_mode, PRICE_TIME, WORKING_DAYS, PRICE_FUEL, travel_time_matrix_car, travel_time_matrix_transit, employment_centers, path_data)
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
gdf, employed_results, travel_matrix = compute_transport_cost_poly(gdf, travel_time_matrix_car, travel_time_matrix_transit, PRICE_TIME, WORKING_DAYS, FIXED_COST_CAR, PRICE_FUEL, LAMBDA, ARRAY_WAGE, clusters_in_zone, house_in_zone, tax = 0)

print(round(100 * sum(gdf["transport_mode"] * gdf["pop"]) / sum(gdf["pop"])), " % commute by public transport")
plot_with_missing(gdf, gdf["transport_cost"])
plot_with_missing(gdf, gdf["transport_mode"])
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

B, KAPPA = calibrate_b_kappa(gdf, mask, RHO, option_calib = "housing")

# Solve the model
def compute_error_in_population_from_utility(u):
    """ Compute error in population associated to utility u"""

    return compute_error_in_population(u / gdf["amenities"], np.nansum(gdf["pop"]), BETA, gdf["wage"], gdf["transport_cost"], B, KAPPA, RHO, gdf["urb_area"])

solving_model = scipy.optimize.minimize(compute_error_in_population_from_utility, 700)

if solving_model.fun < 1:
    utility = solving_model.x
    R = compute_rents(BETA, gdf["wage"], utility / gdf["amenities"], gdf["transport_cost"])
    q = compute_dwelling_size(BETA, gdf["wage"], gdf["transport_cost"], R)
    n = compute_population(B, KAPPA, R, RHO, gdf["urb_area"], q)
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

    return compute_error_in_population(u / gdf["amenities"], np.nansum(gdf["pop"]), BETA, gdf["wage"], gdf["transport_cost"], B, KAPPA, RHO, gdf["urb_area"], rent_residual, density_residual, size_residual)

solving_model = scipy.optimize.minimize(compute_error_in_population_from_utility, 700)

if solving_model.fun < 1:
    utility = solving_model.x
    R = compute_rents(BETA, gdf["wage"], utility / gdf["amenities"], gdf["transport_cost"])
    q = compute_dwelling_size(BETA, gdf["wage"], gdf["transport_cost"], R)
    n = compute_population(B, KAPPA, R, RHO, gdf["urb_area"], q)
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
N = round(np.nansum(gdf["pop"]) * SCALE)

indiv_loc_matrix = compute_indiv_loc_matrix(N, len(gdf), n.to_numpy()* SCALE)
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

#save emissions
travel_matrix["distance_emi"] = (travel_matrix["distance_car"] /1000) * travel_matrix["proba_center"] * (1 - travel_matrix["transport_mode"])
distance_emi = travel_matrix.loc[:,["distance_emi", "from_id"]].groupby("from_id").sum()
gdf = gdf.merge(distance_emi, left_on = "ID", right_index = True)
emissions_init = sum(save_population[:, 0] * (gdf["distance_emi"]))
save_emissions = np.zeros(MAX_YEAR)
save_emissions[0] = emissions_init

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

### MODELING THE PSC

while year < MAX_YEAR:

    print("YEAR", year)

    # Urban form with the tax, without inertia

    #gdf = compute_transport_cost_logit(gdf, Y, PRICE_TIME, WORKING_DAYS, FIXED_COST_CAR, PRICE_FUEL, LAMBDA, tax = tax)
    gdf, employed_results, travel_matrix = compute_transport_cost_poly(gdf, travel_time_matrix_car, travel_time_matrix_transit, PRICE_TIME, WORKING_DAYS, FIXED_COST_CAR, PRICE_FUEL, LAMBDA, ARRAY_WAGE, clusters_in_zone, house_in_zone, tax)

    def compute_error_in_population_from_utility(u):
        """ Compute error in population associated to utility u"""

        return compute_error_in_population(u / gdf["amenities"], np.nansum(gdf["pop"]), BETA, gdf["wage"], gdf["transport_cost"], B, KAPPA, RHO, gdf["urb_area"], rent_residual, density_residual, size_residual)

    solving_model = scipy.optimize.minimize(compute_error_in_population_from_utility, 700)

    if solving_model.fun < 1:
        utility = solving_model.x
        R = compute_rents(BETA, gdf["wage"], utility / gdf["amenities"], gdf["transport_cost"])
        q = compute_dwelling_size(BETA, gdf["wage"], gdf["transport_cost"], R)
        n = compute_population(B, KAPPA, R, RHO, gdf["urb_area"], q)
        R = R * np.exp(rent_residual)
        q = q * np.exp(size_residual)
        n = n * np.exp(density_residual)
        n[np.isnan(n)] = 0
    else:
        print("Minimization failed!")

    housing_without_inertia = n * q

    # AMB
    proba_of_moving_from, proba_of_moving_to = compute_proba_of_moving(save_housing[:, year - 1], housing_without_inertia.to_numpy()* SCALE)
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
    score_ineq = compute_change_in_inequalities(save_utility[:,0], save_utility[:,year])
    score_emissions, emissions = compute_change_in_emissions(gdf, travel_matrix, emissions_init, save_population[:, year])
    political_opinion = compute_political_opinion(score_welfare, score_ineq, score_emissions, BETA_OPINION)
    
    save_emissions[year] = emissions

    if year > 1:
        support = (DELTA * support) + ((1 - DELTA) * political_opinion) # type: ignore
    else:
        support = political_opinion
   
    save_median_support[year] = np.nanmedian(support)
    print("support", support)

    #Policy update
    if np.nanmedian(support) > THRESHOLD:
        tax = tax #* (1 + RATE_INCREASE_TAX)

    year = year + 1


### PLOT RESULTS
print(round(100 * sum(gdf["transport_mode"] * gdf["pop"]) / sum(gdf["pop"])), " % commute by public transport")

ax = gdf.plot(compute_weighted_mean_opinions(support, indiv_loc_matrix, N), legend = True, edgecolor="none", linewidth=0)
for c in ax.collections:
    c.set_antialiased(False)
plt.show()

var_pop = (save_population[:,19] - save_population[:,0])
#var_pop = (save_population[:,4] - save_population[:,3])>0
var_pop[np.isinf(var_pop)] = np.nan
ax = gdf.plot(var_pop, legend = True, edgecolor="none", linewidth=0)
for c in ax.collections:
    c.set_antialiased(False)
plt.show()
print(sum(np.abs((save_population[:,19] - save_population[:,0]))) / 2)

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
#pop_by_bin0.plot(kind="line", figsize=(10,5), label = "0")
#pop_by_bin1.plot(kind="line", figsize=(10,5), label = "1")
#pop_by_bin5.plot(kind="line", figsize=(10,5), label = "5")
#pop_by_bin10.plot(kind="line", figsize=(10,5), label = "10")
pop_by_bin15.plot(kind="line", figsize=(10,5), label = "15")
pop_by_bin19.plot(kind="line", figsize=(10,5), label = "19")
plt.legend()
plt.ylabel("Population")
plt.xlabel("Distance to city center (km)")
plt.title("Population by distance bins")
plt.show()

plot_tax_suppport(save_tax, save_median_support)

plt.plot(save_emissions)
plt.plot(save_emissions[1:])

plt.plot(np.nanmean(save_transport_mode, 0))
plt.plot(np.nanmean(save_transport_mode, 0)[1:])

plt.plot(np.nanmedian(save_utility, 0))
plt.plot(np.nanmedian(save_utility, 0)[1:])

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