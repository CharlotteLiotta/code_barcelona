import numpy as np 
from copy import deepcopy
from scipy.sparse import csr_matrix # type: ignore
import jpype # type: ignore
import os
os.environ["R5_JAR"] = "C:/Users/1738037/AppData/Local/miniforge3/envs/r5py/Lib/site-packages/r5py/data/r5-v6.8-all.jar"
jpype.startJVM(classpath=[os.environ["R5_JAR"]])
import datetime

from functions import *
from import_data import * # type: ignore
from calibration import * # type: ignore
from model import * # type: ignore
from plotting_tools import * # type: ignore
from import_transport import *

###TO DO:

#BETTER TRANSPORT DATA WITH FGC,...

#GO POLYCENTRIC
#ADD DIFFERENT INCOME GROUPS?
#DO MAX LIKELIHOOD INSTEAD OF MEAN ###1.

#ADD AMENITIES ###1.

#clean the size / active data import
#clean the distance to city center data

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
gdf = import_data(path_data, center, option = "SECTION") #"DISTRICT" or "SECTION" - Active population: 1.4M
gdf = import_jobs(gdf, path_data)
gdf = import_land_use(gdf, path_data)
gdf = import_rent_and_size(gdf, path_data)
gdf = import_ppl_per_hh(gdf, path_data)
Y, gdf = import_income(gdf, path_data)


#### IMPORT AMENITIES

#Distance to the beach
land_cover = gpd.read_file(path_data + "cobertes-sol-v1r0-2023.gpkg", layer="cobertes_sol", engine="fiona")
land_cover = land_cover.to_crs(gdf.crs)
gdf['min_distance_beach'] = gdf.geometry.apply(lambda geom: land_cover.loc[land_cover.nivell_2 == 233,:].distance(geom).min())

#Distance to park

parcs = gpd.read_file(path_data + "pev_parcs_od.gpkg")
parcs = parcs.to_crs(gdf.crs)

parcs2 = gpd.read_file(path_data + "equipaments_pro.kml")
parcs2 = parcs2.to_crs(gdf.crs)

gdf['min_distance_parc'] = gdf.geometry.apply(lambda geom: parcs.distance(geom).min())
gdf['min_distance_parc2'] = gdf.geometry.apply(lambda geom: parcs2.distance(geom).min())
#CENTROID??

#Distance to train stations
stations = [
    {"name": "Barcelona Sants", "lon": 2.1399, "lat": 41.3809},
    {"name": "Estació de França", "lon": 2.1875, "lat": 41.3836},
    {"name": "Passeig de Gràcia", "lon": 2.1665, "lat": 41.3916},
    {"name": "Plaça de Catalunya", "lon": 2.1701, "lat": 41.3870},
    {"name": "El Clot-Aragó", "lon": 2.1924, "lat": 41.4112},
    {"name": "Arc de Triomf", "lon": 2.1802, "lat": 41.3912},
    {"name": "Sant Andreu", "lon": 2.1897, "lat": 41.4351}
]

# Create GeoDataFrame in EPSG:4326
stations = gpd.GeoDataFrame(
    stations,
    geometry=[Point(s["lon"], s["lat"]) for s in stations],
    crs="EPSG:4326"
)

stations = stations.to_crs(gdf.crs)
gdf['min_distance_stations'] = gdf.geometry.apply(lambda geom: stations.iloc[0,:].distance(geom).min())

# Reproject to EPSG:25830
gdf = gdf.to_crs(epsg=25830)

#Distance to FGC stations / Renfe stations

#Distance to markets

#Slope

#Airport
lon, lat = 2.0783, 41.2969
airport_wgs84 = gpd.GeoDataFrame(
    {'name': ['Barcelona Airport']},
    geometry=[Point(lon, lat)],
    crs='EPSG:4326'  # WGS84
)
airport_25830 = airport_wgs84.to_crs(epsg=25830)
gdf['min_distance_airport'] = gdf.geometry.apply(lambda geom: airport_25830.distance(geom).min())

#Touristic areas
#https://opendata-ajuntament.barcelona.cat/data/en/dataset/habitatges-us-turistic
#There are others to check as well
tourism = gpd.read_file(path_data + "2019_turisme_allotjament.gpkg")
tourism = tourism.to_crs(gdf.crs)
joined = gpd.sjoin(tourism[['DN', 'geometry']], gdf[['geometry']], how='inner', predicate='intersects')
mean_dn = joined.groupby('index_right')['DN'].mean()
gdf['index_tourism'] = gdf.index.map(mean_dn)
gdf.loc[np.isnan(gdf.index_tourism), "index_tourism"] = 0

#Activity
activity = gpd.read_file(path_data + "2018_cohesio_sobreocupacio.gpkg")
activity = activity.to_crs(gdf.crs)
joined = gpd.sjoin(activity[['norm', 'geometry']], gdf[['geometry']], how='inner', predicate='intersects')
mean_dn = joined.groupby('index_right')['norm'].mean()
gdf['mean_activity'] = gdf.index.map(mean_dn)
gdf.loc[np.isnan(gdf.mean_activity), "mean_activity"] = 0

#Pedestrian streets

#Heat



#Transport times
#import_transport_times(gdf, datetime.datetime(2025, 7, 7, 8, 0, 0), center, path_data, 1)
#import_car_distance(gdf, datetime.datetime(2025, 7, 7, 8, 0, 0), center, path_data)
travel_time_matrix_car, travel_time_matrix_transit = load_transport_times(gdf, path_data, center)
travel_distance_matrix_car = load_transport_distance(gdf, path_data, center)
gdf = add_transport(gdf, travel_time_matrix_car, travel_time_matrix_transit, center)
gdf = gdf.merge(travel_distance_matrix_car.loc[travel_distance_matrix_car.to_id == center,:], left_on = "ID", right_on = "from_id", how = "left").drop(columns = ['from_id', 'to_id'])
gdf = import_cost_transit(gdf)

### INITIAL STATE: YEAR 0

#Transport cost calibration
gdf, FIXED_COST_CAR, LAMBDA = compute_cost_car_logit(gdf, import_trans_mode, PRICE_TIME, WORKING_DAYS, PRICE_FUEL, path_data)
print("FIXED_COST_CAR: ", FIXED_COST_CAR)
print("LAMBDA: ", LAMBDA)

#Compute transport cost
gdf = compute_transport_cost_logit(gdf, PRICE_TIME, WORKING_DAYS, FIXED_COST_CAR, PRICE_FUEL, LAMBDA, tax = 0)
print(round(100 * sum(gdf["transport_mode"] * gdf["pop"]) / sum(gdf["pop"])), " % commute by public transport")

plot_with_missing(gdf, gdf["transport_cost"])
plot_with_missing(gdf, gdf["transport_mode"])

#Calibration BETA with amenities

def compute_log_likelihood(x):
    BETA = x[0]
    U = x[1]
    print("x = ", x)

    gdf["income_net_of_transport_cost"] = Y - gdf["transport_cost"]

    estimated_size = BETA * gdf["income_net_of_transport_cost"]  / gdf["rent_m2"]
    #estimated_size[estimated_size > 150] = np.nan

    diff_size = gdf["size"] - estimated_size
    mask = ((~np.isnan(diff_size)) & (~np.isinf(diff_size)))
    epsilon_size = np.nansum(diff_size.loc[mask] ** 2) / sum(mask)

    log_L = - sum(mask)/2 * np.log(2 * np.pi * epsilon_size) - (1 / 2*epsilon_size) * np.nansum(diff_size.loc[mask] ** 2)
    print("log_L = ", log_L)

    estimated_A = U / (((1-BETA) ** (1-BETA)) * (BETA ** BETA) * (gdf["income_net_of_transport_cost"]  / gdf["rent_m2"]))


    print(gdf.shape)
    gdf["log_A"] = np.log(estimated_A)
    gdf_here = gdf.loc[~np.isnan(gdf.log_A) & ~np.isinf(gdf.log_A),:]
    y = gdf_here["log_A"]
    gdf_here["park_500m"] = (gdf_here.min_distance_parc2 < 500) * 1
    gdf_here["high_tourism"] = (gdf_here.index_tourism > 40) * 1
    gdf_here["airport_500m"] = (gdf_here.min_distance_airport < 1000) * 1
    gdf_here["station_500m"] = (gdf_here.min_distance_stations < 1000) * 1
    X = gdf_here.loc[:,["high_tourism", 'mean_activity', "airport_500m", "station_500m"]]
    X = sm.add_constant(X)  # Adds intercept
    model_statsmodel = sm.OLS(y, X).fit()
    print(model_statsmodel.summary())
    residuals = model_statsmodel.resid

    epsilon_A = np.nansum(np.exp(residuals) ** 2) / sum((~np.isnan(gdf.log_A) & ~np.isinf(gdf.log_A)))
    log_L_A = - (sum(~np.isnan(estimated_A))/2) * np.log(2 * np.pi * epsilon_A) - (1 / (2 * epsilon_A)) * np.nansum(np.exp(residuals) ** 2)
    print("log_L_A = ", log_L_A)
    amenities = np.exp(np.nansum(X.iloc[:,1:] * model_statsmodel.params.iloc[1:], 1))
    gdf_here["amenities"] = amenities
    return - (log_L+log_L_A)

def compute_log_likelihood_amenities(x):
    BETA = x[0]
    U = x[1]
    print("x = ", x)

    gdf["income_net_of_transport_cost"] = Y - gdf["transport_cost"]

    estimated_size = BETA * gdf["income_net_of_transport_cost"]  / gdf["rent_m2"]
    #estimated_size[estimated_size > 150] = np.nan

    diff_size = gdf["size"] - estimated_size
    mask = ((~np.isnan(diff_size)) & (~np.isinf(diff_size)))
    epsilon_size = np.nansum(diff_size.loc[mask] ** 2) / sum(mask)

    log_L = - sum(mask)/2 * np.log(2 * np.pi * epsilon_size) - (1 / 2*epsilon_size) * np.nansum(diff_size.loc[mask] ** 2)
    print("log_L = ", log_L)

    estimated_A = U / (((1-BETA) ** (1-BETA)) * (BETA ** BETA) * (gdf["income_net_of_transport_cost"]  / gdf["rent_m2"]))


    print(gdf.shape)
    gdf["log_A"] = np.log(estimated_A)
    gdf_here = gdf.loc[~np.isnan(gdf.log_A) & ~np.isinf(gdf.log_A),:]
    y = gdf_here["log_A"]
    gdf_here["park_500m"] = (gdf_here.min_distance_parc2 < 500) * 1
    gdf_here["high_tourism"] = (gdf_here.index_tourism > 50) * 1
    gdf_here["airport_500m"] = (gdf_here.min_distance_airport < 1000) * 1
    gdf_here["station_500m"] = (gdf_here.min_distance_stations < 1000) * 1
    X = gdf_here.loc[:,["high_tourism", 'mean_activity', "airport_500m", "station_500m"]]
    X = sm.add_constant(X)  # Adds intercept
    model_statsmodel = sm.OLS(y, X).fit()
    print(model_statsmodel.summary())
    residuals = model_statsmodel.resid

    epsilon_A = np.nansum(np.exp(residuals) ** 2) / sum((~np.isnan(gdf.log_A) & ~np.isinf(gdf.log_A)))
    log_L_A = - (sum(~np.isnan(estimated_A))/2) * np.log(2 * np.pi * epsilon_A) - (1 / (2 * epsilon_A)) * np.nansum(np.exp(residuals) ** 2)
    print("log_L_A = ", log_L_A)
    amenities = np.exp(np.nansum(X.iloc[:,1:] * model_statsmodel.params.iloc[1:], 1))
    gdf_here["amenities"] = amenities
    return gdf_here.loc[:,["ID", "amenities"]]

calib_beta = scipy.optimize.minimize(compute_log_likelihood, [0.35, 700], bounds=[(0,1), (0,None)])
BETA = calib_beta.x[0]

amenities = compute_log_likelihood_amenities(calib_beta.x)
gdf = gdf.merge(amenities, on = "ID", how = "left")
gdf.loc[np.isnan(gdf["amenities"]), "amenities"] = 1


#Calibration BETA
#BETA = calibrate_beta(gdf, Y)
#print("BETA: ", BETA)

#Calibration B and KAPPA
gdf["land"] = gdf["urb_area"] ### TRY EXCLUDING FOREST ONLY, SHRUBLAND ONLY,...
#gdf["urb_area"] = gdf["urb_area_alt"]

mask = ((gdf["rent_m2"] < 22) &(gdf["rent_m2"] > 7)
        #&(gdf["land"] > 10000) &(gdf["land"] <10000000)
        #&(1000000 * gdf["pop"] / gdf["land"] > 200)
        #&(gdf["pop"] > 200)
        #&(gdf["pop"] < 1500)
        #&(gdf["size"] < 90)
        &(~np.isnan(gdf["size"]))
        #&(~np.isnan(gdf["active_per_hh"]))
        )

B, KAPPA = calibrate_b_kappa(gdf, mask, RHO, option_calib = "housing")

#plot_with_missing(gdf, 1000000 * gdf["pop"] / gdf["urb_area"] < 100)
#A ETUDIER

# Solve the model
def compute_error_in_population_from_utility(u):
    """ Compute error in population associated to utility u"""

    return compute_error_in_population(u / gdf["amenities"], np.nansum(gdf["pop"]), BETA, Y, gdf["transport_cost"], B, KAPPA, RHO, gdf["urb_area"])

solving_model = scipy.optimize.minimize(compute_error_in_population_from_utility, 700)

if solving_model.fun < 1:
    utility = solving_model.x
    R = compute_rents(BETA, Y, utility / gdf["amenities"], gdf["transport_cost"])
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

agg = compare_var(gdf, n)

plt.plot(agg["distance_bin"], agg["mean_density_pop"], color='red', linewidth=2, label="Densité moyenne (pop)")
plt.plot(agg["distance_bin"], agg["mean_density_n"], color='blue', linewidth=2, label="Densité moyenne (n)")

plt.xlabel("Distance au centre-ville (km)")
plt.ylabel("Densité de population (hab/km²)")
plt.legend()
plt.tight_layout()
plt.show()

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