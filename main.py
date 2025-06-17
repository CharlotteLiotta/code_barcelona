import numpy as np 
import pandas as pd
import scipy as sc
import matplotlib.pyplot as plt
import random
from numba import njit, jit
from copy import deepcopy
import geopandas as gpd
from shapely import wkt

from functions import *

### IMPORT PARAMETERS

#Spatial structure of Barcelona
distance = np.arange(0, 100, 1)
L = (np.pi * ((distance + 1) ** 2)) - (np.pi * (distance ** 2))

survey_data = pd.read_spss('C:/Users/1738037/OneDrive - UAB/1- CLIMGROW Charlotte/1- PSC cities/PSC and cities/existing surveys/oltra_barcelona/Matriz_MDK_Feb21.sav')

district = gpd.read_file('C:/Users/1738037/OneDrive - UAB/1- CLIMGROW Charlotte/1- PSC cities/data_barcelona/BarcelonaCiutat_Districtes.csv')
district['geometry'] = district['geometria_etrs89'].apply(wkt.loads)
district = gpd.GeoDataFrame(
    geometry="geometry", data=district
)

district["area"] = district.area

population = pd.read_csv("C:/Users/1738037/OneDrive - UAB/1- CLIMGROW Charlotte/1- PSC cities/data_barcelona/2021_densitat.csv")
sum_pop = population.groupby("Nom_Districte").sum("Població")
district = district.merge(sum_pop["Població"], left_on = "nom_districte", right_index = True)


#Import parameters on urban form - TO UPDATE FOR BARCELONA
B = 0.64
BETA = 0.3
KAPPA = 2.0140
RHO = 0.05
N = 10000
Y = 56098
#np.random.gamma(5, 10000, N)
#MARGINAL_COST_POLLUTION = 0.02

#Import transport cost parameters  - TO UPDATE FOR BARCELONA
T_COST = (1.8 / (100 /8)) + ((1/30)*10)
COST_CAR = distance * 400 * T_COST
COST_PT = distance * 400 * T_COST * (distance / 50)
tax = 1.15

#Import parameters on opinion dynamic
I = np.random.randint(1,10, N)
DELTA = 0.5
GAMMA = 0.25
BETA_OPINION = [0.37, 0.50, 0.21, 0.01, -0.01, -0.01, 0] #[0.33, 0.33, 0.33, 0.01, -0.01, -0.01, 0] #BETA_OPINION = [0.048, 0.058, 0.033, 0.00, -0.00, -0.00, 0.129]

#Import time parameters
year = 0
MAX_YEAR = 20
RATE_INCREASE_TAX = 0.05

### INITIAL CALIBRATION

#Solve the model for year 0

transport_cost, transport_mode = compute_transport_cost(COST_CAR, COST_PT, 1)

def compute_error_in_population_from_utility(u):
    """ Compute error in population associated to utility u"""

    return compute_error_in_population(u, N, BETA, Y, transport_cost, B, KAPPA, RHO, L)

solving_model = scipy.optimize.minimize(compute_error_in_population_from_utility, 2000)

if solving_model.fun < 1:
    utility = solving_model.x
    R = compute_rents(BETA, Y, utility, transport_cost)
    q = compute_dwelling_size(BETA, Y, transport_cost, R)
    n = compute_population(B, KAPPA, R, RHO, L, q)
else:
    print("Minimization failed!")

# ABM: translate outputs at the household level

indiv_distance_matrix = compute_indiv_distance_matrix(N, distance, n)

rent_indiv = indiv_distance_matrix @ R
dwelling_size_indiv = indiv_distance_matrix @ q
utility = compute_utility_manually(Y, indiv_distance_matrix @transport_cost, 
                                                dwelling_size_indiv, rent_indiv, BETA)

#utility_with_health = compute_utility_manually(Y, transport_cost, 
#                                                q, R, BETA, 1, N, sum((n * distance * 400)[transport_mode == 0]), 0.02)

housing_indiv = dwelling_size_indiv@indiv_distance_matrix

# Save outputs

save_housing = np.zeros((len(distance), MAX_YEAR))
save_housing[:, 0] = deepcopy(housing_indiv)

save_rent = np.zeros((N, MAX_YEAR))
save_rent[:, 0] = deepcopy(rent_indiv)

save_dwelling_size = np.zeros((N, MAX_YEAR))
save_dwelling_size[:, 0] = deepcopy(dwelling_size_indiv)

save_population = np.zeros((len(distance), MAX_YEAR))
save_population[:, 0] = np.nansum(indiv_distance_matrix, 0)

save_transport_mode = np.zeros((N, MAX_YEAR))
save_transport_mode[:, 0] = deepcopy(indiv_distance_matrix @transport_mode)

save_utility = np.zeros((N, MAX_YEAR))
save_utility[:, 0] = deepcopy(utility)

save_tax = np.zeros(MAX_YEAR)
save_tax[0] = 1

emissions_init = sum((save_population[:, 0] * distance)[transport_mode == 0])

save_median_support = np.zeros(MAX_YEAR)

year = year + 1

### MODELING THE PSC

while year < MAX_YEAR:

    #Urban form with the tax, without inertia

    transport_cost, transport_mode = compute_transport_cost(COST_CAR, COST_PT, tax)

    def compute_error_in_population_from_utility(u):
        """ Compute error in population associated to utility u"""

        return compute_error_in_population(u, N, BETA, Y, transport_cost, B, KAPPA, RHO, L)

    solving_model = scipy.optimize.minimize(compute_error_in_population_from_utility, utility[0])

    if solving_model.fun < 1:
        utility = solving_model.x
        R = compute_rents(BETA, Y, utility, transport_cost)
        q = compute_dwelling_size(BETA, Y, transport_cost, R)
        n = compute_population(B, KAPPA, R, RHO, L, q)
    else:
        print("Minimization failed!")

    housing_without_inertia = n * q

    proba_of_moving_from, proba_of_moving_to = compute_proba_of_moving(save_housing[:, year - 1], housing_without_inertia)

    indiv_distance_matrix_new = deepcopy(indiv_distance_matrix)
    indiv_distance_matrix, has_moved = make_people_move(indiv_distance_matrix_new, N, distance, indiv_distance_matrix, proba_of_moving_from, proba_of_moving_to)

    #Rents and dwelling sizes are updated for the households that have moved only
    rent_indiv_new = indiv_distance_matrix @ R
    dwelling_size_indiv_new = indiv_distance_matrix @ q
    rent_indiv[has_moved == 1] = rent_indiv_new[has_moved == 1]
    dwelling_size_indiv[has_moved == 1] = dwelling_size_indiv_new[has_moved == 1]

    utility = compute_utility_manually(Y, indiv_distance_matrix @transport_cost, 
                                                dwelling_size_indiv, rent_indiv, BETA)

    save_housing[:, year] = deepcopy(dwelling_size_indiv@indiv_distance_matrix)
    save_rent[:, year] = deepcopy(rent_indiv)
    save_dwelling_size[:, year] = deepcopy(dwelling_size_indiv)
    save_population[:, year] = np.nansum(indiv_distance_matrix, 0)
    save_utility[:, year] = deepcopy(utility)
    save_tax[year] = tax
    save_transport_mode[:, year] = deepcopy(indiv_distance_matrix@transport_mode)

    score_welfare = compute_change_in_welfare(save_utility[:,0], save_utility[:,year])
    score_ineq = compute_change_in_inequalities(save_utility[:,0], save_utility[:,year])
    score_emissions = compute_change_in_emissions(save_population[:, year], distance, transport_mode, emissions_init)

    #Policy support

    political_opinion = compute_political_opinion(score_welfare, score_ineq, score_emissions, I, BETA_OPINION, indiv_distance_matrix)
    
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


plot_variables(distance[~np.isnan(compute_weighted_mean_opinions(support, indiv_distance_matrix, N))], compute_weighted_mean_opinions(support, indiv_distance_matrix, N)[~np.isnan(compute_weighted_mean_opinions(support, indiv_distance_matrix, N))], "Public Support")
plot_tax_suppport(save_tax, save_median_support)