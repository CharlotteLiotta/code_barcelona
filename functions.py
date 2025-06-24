import numpy as np # type: ignore
import scipy # type: ignore
import matplotlib.pyplot as plt # type: ignore
import random
from copy import deepcopy
from numba import njit, prange
import pandas as pd
import geopandas as gpd
from shapely import wkt
from scipy.sparse import csr_matrix
from r5py import TravelTimeMatrixComputer, TransportMode, TravelTimeMatrix, TransportNetwork
import datetime
import os

### URBAN ECONOMICS

def compute_utility_manually(Y, T, q, R, BETA, OPTION_HEALTH = 0, N = 0, vkm = 0, marginal_cost_pollution = 0):
    if OPTION_HEALTH == 0:
        u = (Y - T - q * R) ** (1 - BETA) * q ** BETA
    elif OPTION_HEALTH == 1:
        health = vkm * marginal_cost_pollution / N
        u = (Y - T - q * R - health) ** (1 - BETA) * q ** BETA
    return u


def compute_error_in_population(u, N, BETA, Y, transport_cost, B, KAPPA, RHO, L, housing_lag = None):
    '''
    Compute the difference between the population estimated by the model if 
    the utility is equal to u and the actual population.

        Parameters:
            N (float): Actual population
            u (float): Utility
                        
        Returns:
            error_population (float): Difference between the estimated and actual population
        '''

    R = compute_rents(BETA, Y, u, transport_cost)
    q = compute_dwelling_size(BETA, Y, transport_cost, R)
    if housing_lag is None:
        n = compute_population(B, KAPPA, R, RHO, L, q)
    else:
        n = housing_lag / q
    #print("Estimated_population", np.nansum(n))
    #print("Error", N - np.nansum(n))
    error_population = np.abs(N - np.nansum(n))
    return error_population


def compute_rents(beta, Y, u, T):
    '''
    Compute the bid-rent at each location in the city.

            Parameters:
                    beta (float): Parameter of the utility function
                    Y (float): Average income
                    u (float): Average utility
                    T (array): Transportation costs at each location

            Returns:
                    R (array): Bid-rent at each location
    '''
    alpha = 1-beta
    Ro = (((alpha ** alpha) * (beta ** beta) * Y) / u) ** (1/beta)
    R = Ro * ((1 - (T/Y)) ** (1/beta))
    return R


def compute_dwelling_size(beta, Y, T, R):
    '''
    Compute the dwelling size at each location in the city.

            Parameters:
                    beta (float): Parameter of the utility function
                    Y (float): Average income
                    T (array): Transportation costs at each location
                    R (array): Rents at each location

            Returns:
                    q (array): Dwelling size at each location
    '''

    q = beta * (Y - T) / R
    return q

def compute_population(b, kappa, R, rho, L, q):
    '''
    Compute the population at each location in the city.

            Parameters:
                    b (float): Parameter of the housing supply function
                    kappa (float): Parameter of the housing supply function
                    R (array): Rents at each location
                    rho (array): Interest rate
                    L (array): Land available for housing at each location
                    q (array): Dwelling size at each location

            Returns:
                    q (array): Dwelling size at each location
    '''

    a = 1-b
    n = kappa ** (1/a) * (b * R / rho) ** (b/a) * L / q
    return n

    
def compute_transport_cost(gdf, PRICE_TIME, WORKING_DAYS, FIXED_COST_CAR, PRICE_FUEL, tax):

    gdf["COST_CAR"] = ((gdf["travel_time_car"] / 60) * PRICE_TIME * WORKING_DAYS) + (gdf.distance_center * PRICE_FUEL * WORKING_DAYS) + FIXED_COST_CAR + (tax * WORKING_DAYS)
    gdf["COST_PT"] = ((gdf["travel_time_transit"] / 60) * PRICE_TIME * WORKING_DAYS) + gdf["monthly_cost_transit"]

    stacked = np.vstack([gdf["COST_CAR"], gdf["COST_PT"]])  # Shape (2, N)
    masked = np.where(np.isnan(stacked), np.inf, stacked)
    choice = np.argmin(masked, axis=0)

    gdf["transport_cost"] = np.fmin(gdf["COST_CAR"], gdf["COST_PT"])
    gdf["transport_mode"] = choice
    
    print("Transport cost: ", sum(np.isnan(gdf["transport_cost"])), "missing values")
    gdf.loc[np.isnan(gdf["transport_cost"]), "transport_cost"] = 120
    gdf.loc[np.isnan(gdf["transport_mode"]), "transport_mode"] = 0
    return gdf

### OUTPUT FROM THE URBAN ECON MODEL

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

    relative_change_utility = (100 * (utility_with_tax - utility_without_tax)) / utility_without_tax
    return (1 / (1 + np.exp(-0.15 * relative_change_utility)))

def compute_change_in_inequalities(utility_without_tax, utility_with_tax):
    if gini(np.array(utility_without_tax)) != gini(np.array(utility_with_tax)):
        Q = (1 / (1 + np.exp(10 * (gini(np.array(utility_with_tax)) - gini(np.array(utility_without_tax)))))) #/ ))) #gini(np.array(utility_without_tax)))))
    else:
        Q = 0.5
    return Q

def compute_change_in_emissions(n_with_tax, distance, transport_mode, emissions_init):
    relative_change_emission = 100 * (sum((n_with_tax * distance)[transport_mode == 0]) - emissions_init) / emissions_init

    return (1 / (1 + np.exp(0.02 * relative_change_emission)))

### POLICY SUPPORT FUNCTIONS

def compute_political_opinion(score_welfare, score_ineq, score_emissions, I, BETA_OPINION, opinion_distance_matrix):
    return BETA_OPINION[6] + (BETA_OPINION[0] + BETA_OPINION[3] * I) * score_welfare + (BETA_OPINION[1] + BETA_OPINION[4] * I) * score_ineq + (BETA_OPINION[2] + BETA_OPINION[5] * I) * score_emissions

@njit(parallel = True)
def compute_social_interactions(political_opinion, I, GAMMA):
    N = len(I)
    result = np.zeros(N)
    
    for i in prange(N):
        #print(100 * i/N, "%")
        abs_diff = np.abs(I - I[i])
        matrix_opinion_row = np.exp(-abs_diff)
        
        weighted_sum = np.sum(political_opinion * matrix_opinion_row)
        sum_matrix_opinion = np.sum(matrix_opinion_row)
        
        result[i] = ((1 - GAMMA) * political_opinion[i]) + (GAMMA * (weighted_sum / sum_matrix_opinion))
    
    return result

### ABM

@njit
def compute_indiv_distance_matrix(N, distance, n_with_tax):
    opinion_distance_matrix = np.zeros((N, len(distance)))

    step = 0
    for k in range(len(distance)):
        nb_pers = round(n_with_tax[k])
        if step+nb_pers < N:
            opinion_distance_matrix[step:step+nb_pers, k] = np.ones(nb_pers)
            step += nb_pers
        else:
            opinion_distance_matrix[step:N, k] = np.ones(N-step)
            break
    return opinion_distance_matrix

@njit
def compute_indiv_loc_matrix(N, len_gdf, n):
    opinion_distance_matrix = np.zeros((N, len_gdf))

    step = 0
    for k in range(len_gdf):
        nb_pers = round(n[k])
        if step+nb_pers < N:
            opinion_distance_matrix[step:step+nb_pers, k] = np.ones(nb_pers)
            step += nb_pers
        else:
            opinion_distance_matrix[step:N, k] = np.ones(N-step)
            break
    return opinion_distance_matrix

@njit
def update_indiv_distance_matrix(indiv_distance_matrix, n):
    #print(np.nansum(indiv_distance_matrix,0) - n)
    indiv_distance_matrix_new = (indiv_distance_matrix)

    ### first iteration
    lag = sum(np.abs(np.nansum(indiv_distance_matrix_new,0) - n))
    #print(sum(np.abs(np.nansum(indiv_distance_matrix_new,0) - n)))
    destination = np.argmin((np.nansum(indiv_distance_matrix_new,0) - n))
    origin = np.argmax((np.nansum(indiv_distance_matrix_new,0) - n))
    #indiv_moving = (np.argwhere(indiv_distance_matrix_new[:, origin] == 1)[np.random.randint(1,len(np.argwhere(indiv_distance_matrix_new[:, origin] == 1)))]).astype(int)[0]
    indiv_moving = np.random.choice(np.argwhere(indiv_distance_matrix_new[:, origin] == 1).flatten())
    if indiv_distance_matrix_new[indiv_moving, origin] != 1:
        raise Exception
    indiv_distance_matrix_new[indiv_moving, origin] = 0
    if indiv_distance_matrix_new[indiv_moving, destination] != 0:
        raise Exception
    indiv_distance_matrix_new[indiv_moving, destination] = 1
    new = sum(np.abs(np.nansum(indiv_distance_matrix_new,0) - n))
    #loop
    while np.abs(new-lag)> 0.01:
        lag = new
        #print(sum(np.abs(np.nansum(indiv_distance_matrix_new,0) - n)))
        destination = np.argmin((np.nansum(indiv_distance_matrix_new,0) - n))
        origin = np.argmax((np.nansum(indiv_distance_matrix_new,0) - n))
        #indiv_moving = (np.argwhere(indiv_distance_matrix_new[:, origin] == 1)[np.random.randint(1,len(np.argwhere(indiv_distance_matrix_new[:, origin] == 1)))]).astype(int)[0]
        indiv_moving = np.random.choice(np.argwhere(indiv_distance_matrix_new[:, origin] == 1).flatten())
        if indiv_distance_matrix_new[indiv_moving, origin] != 1:
            raise Exception
        indiv_distance_matrix_new[indiv_moving, origin] = 0
        if indiv_distance_matrix_new[indiv_moving, destination] != 0:
            raise Exception
        indiv_distance_matrix_new[indiv_moving, destination] = 1
        new = sum(np.abs(np.nansum(indiv_distance_matrix_new,0) - n))

    return indiv_distance_matrix_new

@njit
def compute_proba_of_moving(housing_lag, housing_without_inertia):
    """ Compute the probability of moving from, and moving to, each spatial unit"""

    proba_of_moving_from = np.zeros(len(housing_lag))
    mask = housing_lag > 0
    proba_of_moving_from[mask] = (housing_lag[mask] - housing_without_inertia[mask]) / housing_lag[mask]
    
    proba_of_moving_to = (housing_without_inertia - housing_lag)
    proba_of_moving_to[proba_of_moving_to < 0] = 0
    proba_of_moving_to /= np.nansum(proba_of_moving_to)
    return proba_of_moving_from, proba_of_moving_to


@njit
def make_people_move(indiv_loc_matrix_new, N, len_gdf, indiv_loc_matrix, proba_of_moving_from, proba_of_moving_to, PROBA_MOVE):
    
    has_moved = np.zeros(N)
    indiv_moving = np.random.binomial(1, PROBA_MOVE, N) #Each individual has a 30% chance to be willing to move.

    for i in np.arange(N):
        if indiv_moving[i] == 1:
            if sum(indiv_loc_matrix[i,:]) > 0:
                proba_of_moving_here = sum((indiv_loc_matrix[i,:] * proba_of_moving_from)[indiv_loc_matrix[i,:] > 0])
                if proba_of_moving_here < 0:
                    proba_of_moving_here = 0
                moving = np.random.binomial(1, proba_of_moving_here)
                if moving == 1:
                    has_moved[i] = 1
                    indiv_loc_matrix_new[i,:] = np.zeros(len_gdf)
                    #destination = np.random.choice(np.arange(len(distance)), p=proba_of_moving_to)
                    destination = np.arange(len_gdf)[np.searchsorted(np.cumsum(proba_of_moving_to), np.random.random(), side="right")]
                    indiv_loc_matrix_new[i,destination] = 1

    return indiv_loc_matrix_new, has_moved

### PLOT AND VISUALIZE VARIABLES

def plot_variables(distance, variable, title):
    # Normalize the values to range [0, 1] for color mapping
    norm = plt.Normalize(variable[~np.isnan(variable)].min(), variable[~np.isnan(variable)].max())

    # Create a color map (Red to Green)
    cmap = plt.cm.RdYlGn

    # Plot the circles
    fig, ax = plt.subplots()
    for (xi, vi) in zip(distance[::-1], variable[::-1]):
        color = cmap(norm(vi))
        circle = plt.Circle((0, 0), xi, color=color, alpha=0.7)
        ax.add_patch(circle)

    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, orientation='vertical', label='')

    # Set the aspect of the plot to be equal
    ax.set_aspect('equal')

    # Set limits to ensure all circles are visible
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)

    plt.axis('off')

    plt.title(title)

    plt.show()

def compute_weighted_mean_opinions(var, opinion_distance_matrix, N):
    weighted_mean_opinion = np.zeros(opinion_distance_matrix.shape[1])
    weighted_mean_opinion[np.nansum(opinion_distance_matrix, 0) > 0] = np.nansum(opinion_distance_matrix * var.reshape(N, 1), 0)[np.nansum(opinion_distance_matrix, 0) > 0] / np.nansum(opinion_distance_matrix, 0)[np.nansum(opinion_distance_matrix, 0) > 0]
    weighted_mean_opinion[np.nansum(opinion_distance_matrix, 0) == 0] = np.nan
    return weighted_mean_opinion

def plot_tax_suppport(save_tax, save_median_support):

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('time (year)')
    ax1.set_ylabel('tax', color=color)
    ax1.plot(save_tax[1:], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('median support', color=color)  # we already handled the x-label with ax1
    ax2.plot(save_median_support[1:], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def import_data(option):
    if option == "SECTION":
        #https://www.ine.es/dynt3/inebase/en/index.htm?padre=11676&capsel=11681
        df = pd.read_csv('C:/Users/1738037/OneDrive - UAB/1- CLIMGROW Charlotte/1- PSC cities/data_barcelona/70035.csv', sep = ";", encoding="latin1")
        df = df.loc[:,["Sections", "Total"]]
        df = df.dropna(subset=["Sections"])
        df["CUSEC"] = df["Sections"].str[:10]

        def fix_decimal(s):
            if isinstance(s, str) and ',' in s:
                integer, decimal = s.split(',', 1)
                if len(decimal) == 2:
                    return f"{integer},{decimal}0"
            return s

        gdf = gpd.read_file('C:/Users/1738037/OneDrive - UAB/1- CLIMGROW Charlotte/1- PSC cities/data_barcelona/seccionado_2024/SECC_CE_20240101.shp')
        gdf = gdf.merge(df, on = "CUSEC")

        gdf["Total"] = gdf["Total"].apply(fix_decimal)
        gdf["Total"] = pd.to_numeric(gdf["Total"].str.replace(',', ''), errors='coerce')

        gdf["density"] = gdf["Total"] / (gdf["Shape_Area"]/1000000)

        gdf = gdf.loc[gdf.NMUN.isin(['Badalona', 'Badia del Vallès', 'Barberà del Vallès', 'Barcelona','Begues','Castellbisbal', 'Castelldefels', 'Cerdanyola del Vallès', 'Cervelló','Corbera de Llobregat','Papiol, El', 'Prat de Llobregat, El', 'Esplugues de Llobregat','Gavà', "Hospitalet de Llobregat, L'",'Palma de Cervelló, La','Molins de Rei', 'Montcada i Reixac', 'Montgat', 'Pallejà', 'Ripollet','Sant Adrià de Besòs', 'Sant Andreu de la Barca','Sant Boi de Llobregat', 'Sant Climent de Llobregat', 'Sant Cugat del Vallès', 'Sant Feliu de Llobregat','Sant Joan Despí','Sant Just Desvern','Sant Vicenç dels Horts', 'Santa Coloma de Cervelló', 'Santa Coloma de Gramenet','Tiana', 'Torrelles de Llobregat', 'Viladecans']),["CUSEC", "CUMUN", "Shape_Area", "Total", "density", "geometry", "NMUN"]]
        city_center = gdf.loc[gdf.NMUN == "Barcelona",:].centroid
        gdf["distance_center"] = gdf.centroid.distance(city_center.iloc[0], align = False) / 1000
        gdf = gdf.loc[:,["CUSEC", "geometry", "Shape_Area", "Total", "distance_center"]]
        gdf.columns = ["ID", "geometry", "area", "pop", "distance_center"]

    elif option == "DISTRICT":

        gdf = gpd.read_file('C:/Users/1738037/OneDrive - UAB/1- CLIMGROW Charlotte/1- PSC cities/data_barcelona/BarcelonaCiutat_Districtes.csv')
        gdf['geometry'] = gdf['geometria_etrs89'].apply(wkt.loads)
        gdf = gpd.GeoDataFrame(geometry="geometry", data=gdf, crs="EPSG:25831")

        gdf["area"] = gdf.area

        df = pd.read_csv("C:/Users/1738037/OneDrive - UAB/1- CLIMGROW Charlotte/1- PSC cities/data_barcelona/2021_densitat.csv")
        df = df.groupby("Nom_Districte").sum("Població")
        gdf = gdf.merge(df["Població"], left_on = "nom_districte", right_index = True)

        city_center = gdf.loc[gdf.nom_districte == "Eixample",:].centroid

        gdf["distance_center"] = gdf.centroid.distance(city_center.iloc[0], align = False) / 1000

        gdf = gdf.loc[:,["Codi_Districte", "geometry", "area", "Població", "distance_center"]]
        gdf.columns = ["ID", "geometry", "area", "pop", "distance_center"]

    gdf["area"] = gdf["area"] / 1000000
    gdf["density"] = gdf["pop"] / gdf["area"]
    
    return gdf


def import_transport_times(gdf, date_here, center, OPTION_SAVE):
    #le mieux a l'air d'être ici (existe en simplfifié ou normal): https://t-mobilitat.atm.cat/web/t-mobilitat/datos-abiertos/catalogo-de-datos/informacion-estatica
    
    #ADD https://fgc.opendatasoft.com/explore/dataset/gtfs_zip/table/
    #ADD https://datos.gob.es/en/catalogo/a09002970-red-de-transporte-por-carretera-paradas-lineas-y-horarios-de-los-autobuses-interurbanos-de-catalunya
    #ADD https://www.amb.cat/web/area-metropolitana/dades-obertes/cataleg/detall/-/dataset/serveis-gtfs-de-tmb/1107694/11692 PAS DISPO?
    #ADD https://www.amb.cat/es/web/area-metropolitana/dades-obertes/cataleg/detall/-/dataset/informacion-de-companias--lineas-y-recorridos/1033377/11692?_DatasetSearchListPortlet_WAR_AMBSearchPortletportlet_pageNum=4&_DatasetSearchListPortlet_WAR_AMBSearchPortletportlet_categoria=mobilitat&_DatasetSearchListPortlet_WAR_AMBSearchPortletportlet_detailBackURL=https%3A%2F%2Fwww.amb.cat%2Fes%2Fweb%2Farea-metropolitana%2Fdades-obertes%2Fcataleg%2Fllistat (seulement des bus?)
    os.environ["R5_VERBOSE"] = "true"

    points = gdf.copy()
    points["geometry"] = points.centroid
    points = points.loc[:,["ID", "geometry"]]
    points.columns = ["id", "geometry"]

    path_osm = "C:/Users/1738037/OneDrive - UAB/1- CLIMGROW Charlotte/1- PSC cities/data_barcelona/barcelona.osm.pbf"
    #path_osm = "C:/Users/1738037/OneDrive - UAB/1- CLIMGROW Charlotte/1- PSC cities/data_barcelona/cataluna-latest.osm.pbf"

    list_path_gtfs = os.listdir("C:/Users/1738037/OneDrive - UAB/1- CLIMGROW Charlotte/1- PSC cities/data_barcelona/" + 'gtfs/')
    list_path_gtfs = [i for i in list_path_gtfs if i.endswith('zip')]
    list_path_gtfs = ["C:/Users/1738037/OneDrive - UAB/1- CLIMGROW Charlotte/1- PSC cities/data_barcelona/" + 'gtfs/' + sub for sub in list_path_gtfs] 
    #list_path_gtfs = []
    
    print("Start Computing Network")
    transport_network = TransportNetwork(
        # OSM data
        path_osm,
    
        # A list of GTFS file(s)
        list_path_gtfs
        )
    
    print("Transport Network Computed")

    i = 800
    while i < (len(points) - 100):

        i = i + 100
    
        travel_time_matrix_computer_car = TravelTimeMatrixComputer(
            transport_network,
            origins=points.iloc[i - 100:i, :],
            destinations=points.loc[points.id == center,:],
            departure=date_here,
            transport_modes=[TransportMode.CAR]
            )
        

        travel_time_matrix_car = travel_time_matrix_computer_car.compute_travel_times()
    
        np.save("C:/Users/1738037/OneDrive - UAB/1- CLIMGROW Charlotte/1- PSC cities/data_barcelona/" + "travel_time_matrix_car" + "_" + str(i) + ".npy", travel_time_matrix_car)

        print("Travel time Car: ", round(100 * i/len(points)), "%")
    
    travel_time_matrix_computer_car = TravelTimeMatrixComputer(
        transport_network,
        origins=points.iloc[i:len(points), :],
        destinations=points.loc[points.id == center,:],
        departure=date_here,
        transport_modes=[TransportMode.CAR]
        )
        

    travel_time_matrix_car = travel_time_matrix_computer_car.compute_travel_times()
    
    np.save("C:/Users/1738037/OneDrive - UAB/1- CLIMGROW Charlotte/1- PSC cities/data_barcelona/" + "travel_time_matrix_car" + "_" + str(len(points)) + ".npy", travel_time_matrix_car)

    print("Travel Times Car Computed")

    #if OPTION_SAVE == 1:
        #np.save("C:/Users/1738037/OneDrive - UAB/1- CLIMGROW Charlotte/1- PSC cities/data_barcelona/" + "travel_time_matrix_transit.npy", travel_time_matrix_transit)
        #np.save("C:/Users/1738037/OneDrive - UAB/1- CLIMGROW Charlotte/1- PSC cities/data_barcelona/" + "travel_time_matrix_car.npy", travel_time_matrix_car)

    print("Travel Times Car Saved")

    i = 0
    
    while i < (len(points) - 100):

        i = i + 100

        travel_time_matrix_computer_transit = TravelTimeMatrixComputer(
            transport_network,
            origins=points.iloc[i - 100:i, :],
            destinations=points.loc[points.id == center,:],
            departure=date_here, #2023, 6, 22, 8, 0, 0),
            transport_modes=[TransportMode.TRANSIT, TransportMode.WALK]
            )

        print("Travel Time Matrix Transit Computed")

        travel_time_matrix_transit = travel_time_matrix_computer_transit.compute_travel_times()
    
        np.save("C:/Users/1738037/OneDrive - UAB/1- CLIMGROW Charlotte/1- PSC cities/data_barcelona/" + "travel_time_matrix_transit" + "_" + str(i) + ".npy", travel_time_matrix_transit)
    
        print("Travel time Transit: ", round(100 * i/len(points)), "%")

    travel_time_matrix_computer_transit = TravelTimeMatrixComputer(
    transport_network,
    origins=points.iloc[i:len(points), :],
    destinations=points.loc[points.id == center,:],
    departure=date_here, #2023, 6, 22, 8, 0, 0),
    transport_modes=[TransportMode.TRANSIT, TransportMode.WALK]
    )

    print("Travel Time Matrix Transit Computed")

    travel_time_matrix_transit = travel_time_matrix_computer_transit.compute_travel_times()
    
    np.save("C:/Users/1738037/OneDrive - UAB/1- CLIMGROW Charlotte/1- PSC cities/data_barcelona/" + "travel_time_matrix_transit" + "_" + str(len(points)) + ".npy", travel_time_matrix_transit)
        
    print("Travel Times Transit Computed")

    #if OPTION_SAVE == 1:
    #    np.save("C:/Users/1738037/OneDrive - UAB/1- CLIMGROW Charlotte/1- PSC cities/data_barcelona/" + "travel_time_matrix_transit.npy", travel_time_matrix_transit)
        #np.save("C:/Users/1738037/OneDrive - UAB/1- CLIMGROW Charlotte/1- PSC cities/data_barcelona/" + "travel_time_matrix_car.npy", travel_time_matrix_car)

    print("Travel Times Transit Saved")

    #return travel_time_matrix_car, travel_time_matrix_transit

def load_transport_times(gdf):

    i = 100
    travel_time_matrix_transit = np.load("C:/Users/1738037/OneDrive - UAB/1- CLIMGROW Charlotte/1- PSC cities/data_barcelona/" + "travel_time_matrix_transit" + "_" + str(i) + ".npy", allow_pickle= True)
    travel_time_matrix_car = np.load("C:/Users/1738037/OneDrive - UAB/1- CLIMGROW Charlotte/1- PSC cities/data_barcelona/" + "travel_time_matrix_car" + "_" + str(i) + ".npy", allow_pickle= True)

    while i < len(gdf) - 100:
        i = i + 100
        temp_transit = np.load("C:/Users/1738037/OneDrive - UAB/1- CLIMGROW Charlotte/1- PSC cities/data_barcelona/" + "travel_time_matrix_transit" + "_" + str(i) + ".npy", allow_pickle= True)
        temp_car = np.load("C:/Users/1738037/OneDrive - UAB/1- CLIMGROW Charlotte/1- PSC cities/data_barcelona/" + "travel_time_matrix_car" + "_" + str(i) + ".npy", allow_pickle= True)

        travel_time_matrix_transit = np.concatenate((travel_time_matrix_transit, temp_transit), axis=0)
        travel_time_matrix_car = np.concatenate((travel_time_matrix_car, temp_car), axis=0)

    temp_transit = np.load("C:/Users/1738037/OneDrive - UAB/1- CLIMGROW Charlotte/1- PSC cities/data_barcelona/" + "travel_time_matrix_transit" + "_" + str(len(gdf)) + ".npy", allow_pickle= True)
    temp_car = np.load("C:/Users/1738037/OneDrive - UAB/1- CLIMGROW Charlotte/1- PSC cities/data_barcelona/" + "travel_time_matrix_car" + "_" + str(len(gdf)) + ".npy", allow_pickle= True)

    travel_time_matrix_transit = np.concatenate((travel_time_matrix_transit, temp_transit), axis=0)
    travel_time_matrix_car = np.concatenate((travel_time_matrix_car, temp_car), axis=0)
    
    travel_time_matrix_car = pd.DataFrame(travel_time_matrix_car, columns = ['from_id', 'to_id', 'travel_time'])
    travel_time_matrix_transit = pd.DataFrame(travel_time_matrix_transit, columns = ['from_id', 'to_id', 'travel_time'])

    travel_time_matrix_car['travel_time'] = pd.to_numeric(travel_time_matrix_car['travel_time'], errors='coerce')
    travel_time_matrix_transit['travel_time'] = pd.to_numeric(travel_time_matrix_transit['travel_time'], errors='coerce')

    return travel_time_matrix_car, travel_time_matrix_transit

def merge_transport(gdf, travel_time_matrix, center, name):
    gdf = gdf.merge(travel_time_matrix.loc[travel_time_matrix.to_id == center,:], left_on = "ID", right_on = "from_id", how = "left")
    gdf = gdf.drop(columns = ['from_id', 'to_id'])
    gdf = gdf.rename(columns={"travel_time": name
                          })
    return gdf

def add_transport(gdf, travel_time_matrix_car, travel_time_matrix_transit, center):

    gdf = merge_transport(gdf, travel_time_matrix_transit, center, "travel_time_transit")
    gdf = merge_transport(gdf, travel_time_matrix_car, center, "travel_time_car")
    return gdf

def import_jobs(gdf):
    jobs = pd.read_csv('C:/Users/1738037/OneDrive - UAB/1- CLIMGROW Charlotte/1- PSC cities/data_barcelona/employment_distrib.csv')
    jobs = jobs.loc[:,["SPERSONAS", "ID_LUGAR_TRAB_N3"]]
    jobs["code_city"] = jobs["ID_LUGAR_TRAB_N3"].str[:5]
    gdf["code_city"] = gdf["ID"].str[:5]
    jobs = jobs.loc[:,["SPERSONAS", "code_city"]]
    gdf = gdf.merge(jobs, on = "code_city", how = "left")
    size_city = gdf.loc[:,["code_city", "area"]].groupby("code_city").sum("area")
    size_city.columns = ["area_city"]
    gdf = gdf.merge(size_city, on = "code_city", how = "left")
    gdf["density_employment"] = gdf["SPERSONAS"] / gdf["area_city"]
    gdf["employment"] = gdf["SPERSONAS"] * (gdf["area"] / gdf["area_city"])
    gdf = gdf.drop(columns = ["SPERSONAS", "area_city"])
    return gdf

def import_trans_mode():
    #https://www.ine.es/dynt3/inebase/en/index.htm?padre=8981&capsel=8982
    trans_mode = pd.read_csv('C:/Users/1738037/OneDrive - UAB/1- CLIMGROW Charlotte/1- PSC cities/data_barcelona/55377.csv', sep = ";")
    trans_mode = trans_mode.loc[trans_mode.Municipalities.isin(['Badalona', 'Barcelona', 'Castelldefels', 'Cerdanyola del Vallès', 'Cornellà de Llobregat', 'Granollers', "Hospitalet de Llobregat, L'", 'Manresa', 'Mollet del Vallès', 'Prat de Llobregat, El', 'Sabadell', 'Sant Boi de Llobregat', 'Sant Cugat del Vallès', 'Santa Coloma de Gramenet', 'Terrassa', 'Viladecans', 'Vilanova i la Geltrú']),:] #check manual jusqu'a fuengirola
    trans_mode = trans_mode.loc[trans_mode.Age == "Total",:]
    trans_mode = trans_mode.loc[trans_mode.Sex == "Both sexes",:]
    trans_mode = trans_mode.loc[:,["Municipalities", "Means of transport", "Total"]]
    trans_mode = trans_mode.pivot(index='Municipalities', columns='Means of transport', values='Total')
    trans_mode['Total'] = pd.to_numeric(trans_mode['Total'].str.replace(",", ""), errors="coerce")
    for col in ['Public', 'Particular', 'Walking', 'Company or other media']:
        trans_mode[col] = pd.to_numeric(trans_mode[col].str.replace(",", ""), errors="coerce")
        trans_mode[col] = trans_mode[col] / trans_mode['Total']

    mapping = {
        'Badalona': '08015',
        'Barcelona': '08019',
        'Castelldefels': '08056',
        'Cerdanyola del Vallès': '08266',
        'Cornellà de Llobregat': '08073',
        'Granollers': '08096',
        "Hospitalet de Llobregat, L'": '08101',
        'Manresa': '08113',
        'Mollet del Vallès': '08124',
        'Prat de Llobregat, El': '08169',
        'Sabadell': '08187',
        'Sant Boi de Llobregat': '08200',
        'Sant Cugat del Vallès': '08205',
        'Santa Coloma de Gramenet': '08245',
        'Terrassa': '08279',
        'Viladecans': '08301',
        'Vilanova i la Geltrú': '08307'
        # ... add all 17 mappings here
        }

    trans_mode['code_city'] = trans_mode.index.map(mapping)
    trans_mode["share_car"] = trans_mode["Particular"] + trans_mode["Company or other media"]
    return trans_mode

def import_cost_transit(gdf):
    gdf["monthly_cost_transit"] = np.nan
    gdf.loc[gdf.code_city.isin(['08015', '08019', '08056', '08077','08101', '08089', '08125', '08126','08169', '08194', '08200', '08211', '08217', '08221', '08245', '08282', '08301', ]), "monthly_cost_transit"] = 22
    gdf.loc[gdf.code_city.isin(['08020', '08054', '08068', '08072', '08123','08157', '08158', '08180', '08196', '08204','08205', '08244', '08252','08263', '08289', '08904', '08905', '08266']), "monthly_cost_transit"] = 29.65
    return gdf

def compute_cost_car(gdf, import_trans_mode, PRICE_TIME, WORKING_DAYS, PRICE_FUEL):

    trans_mode = import_trans_mode()

    gdf = gdf.merge(trans_mode.loc[:,["code_city", "share_car"]], on = "code_city", how = "left")
    
    def compute_error_transport(x):
        FIXED_COST_CAR = x[0]
        gdf_here = compute_transport_cost(gdf, PRICE_TIME, WORKING_DAYS, FIXED_COST_CAR, PRICE_FUEL, tax = 0)
        error1 = np.nansum(np.abs(((1 - gdf_here["transport_mode"]) * gdf_here["pop"]) - (gdf_here["share_car"] * gdf_here["pop"])))
        print(f"x = {x[0]}, error1 = {error1}")

        gdf_here = gdf_here.loc[~np.isnan(gdf_here.share_car),:]
        error2 = np.nansum(gdf_here["pop"]) * np.abs((np.nansum(gdf_here.share_car * gdf_here["pop"]) / np.nansum(gdf_here["pop"])) - (np.nansum((1 - gdf_here["transport_mode"]) * gdf_here["pop"]) / np.nansum(gdf_here["pop"])))
        print(f"x = {x[0]}, error2 = {error2}")
        return error1 + error2

    solving_transport = scipy.optimize.minimize(compute_error_transport, x0=[300], method='Nelder-Mead')
    FIXED_COST_CAR = solving_transport.x
    return gdf, FIXED_COST_CAR

def import_income(gdf):
    #https://www.ine.es/dynt3/inebase/en/index.htm?padre=12385&capsel=12384
    income = pd.read_csv('C:/Users/1738037/OneDrive - UAB/1- CLIMGROW Charlotte/1- PSC cities/data_barcelona/30896.csv', sep = ";", encoding="latin1")
    income =income.loc[(income.Periodo == 2022) & (income['Mean and median income indicators'] == 'Average net income per person'),["Sections", "Total"]]
    income = income.dropna(subset=["Sections"])
    income["ID"] = income["Sections"].str[:10]
    income.columns = ['Sections', 'net_income', 'ID']
    income.net_income = pd.to_numeric(income.net_income, errors= "coerce")
    income.net_income = income.net_income * 1000
    gdf = gdf.merge(income.loc[:,['net_income', 'ID']], on = "ID", how = "left")
    Y = (np.nansum(gdf.net_income * gdf["pop"]) / np.nansum(gdf["pop"]))
    return Y / 12, gdf

def plot_with_missing(gdf, var):
    gdf.plot(
    column=var,
    legend=True,
    missing_kwds={
        "color": "lightgrey",
        "label": "Missing data"
    })

def import_rent_and_size(gdf):
    #https://habitatge.gencat.cat/ca/dades/indicadors_estadistiques/estadistiques_de_construccio_i_mercat_immobiliari/mercat_de_lloguer/lloguers-municipis-amb/

    #section level - AMB
    rent = pd.read_excel("C:/Users/1738037/OneDrive - UAB/1- CLIMGROW Charlotte/1- PSC cities/data_barcelona/AMB_lloguer_m2.xlsx", header = 5)
    rent = rent.loc[:,["Codi_àmbit", "IV"]]
    gdf["Codi_àmbit"] = pd.to_numeric(gdf["ID"].str[:7])
    rent["IV"] = pd.to_numeric(rent["IV"], errors = "coerce")
    rent.columns = ['Codi_àmbit', 'rent_AMB_section']

    gdf = gdf.merge(rent, on = "Codi_àmbit", how = "left")

    #city level - AMB
    rent = pd.read_excel("C:/Users/1738037/OneDrive - UAB/1- CLIMGROW Charlotte/1- PSC cities/data_barcelona/AMB_lloguer_m2.xlsx", header = 5)
    rent = rent.loc[np.isnan(rent.Codi_àmbit),["Codi_INE", "IV"]]
    gdf["Codi_INE"] = pd.to_numeric(gdf["code_city"])
    rent["IV"] = pd.to_numeric(rent["IV"], errors = "coerce")
    rent = rent.iloc[0:29]
    rent.columns = ['Codi_INE', 'rent_AMB_city']
    rent.Codi_INE = rent.Codi_INE.astype(int)
    gdf = gdf.merge(rent, on = "Codi_INE", how = "left")

    #Barri lebel - Barcelona
    rent = pd.read_excel("C:/Users/1738037/OneDrive - UAB/1- CLIMGROW Charlotte/1- PSC cities/data_barcelona/trimestral_bcn_lloguer_m2.xlsx", header = 20, sheet_name = "2023")
    rent = rent.iloc[:,[0,5]]
    rent.columns = ["code_barri", "rent_barcelona_barri"]
    admin = pd.read_excel("C:/Users/1738037/OneDrive - UAB/1- CLIMGROW Charlotte/1- PSC cities/data_barcelona/BarcelonaCiutat_SeccionsCensals.xlsx")
    admin = admin.loc[:,["codi_districte", "codi_barri", "codi_seccio_censal"]]
    admin["ID"] = "08019" + admin["codi_districte"].astype(str).str.zfill(2) + admin["codi_seccio_censal"].astype(str).str.zfill(3)
    rent = rent.merge(admin, left_on = "code_barri", right_on = "codi_barri", how = "left")

    gdf = gdf.merge(rent.loc[:,["rent_barcelona_barri", "ID"]], on = "ID", how = "left")

    #District level - Barcelona
    rent = pd.read_excel("C:/Users/1738037/OneDrive - UAB/1- CLIMGROW Charlotte/1- PSC cities/data_barcelona/trimestral_bcn_lloguer_m2.xlsx", header = 8, sheet_name = "2023")
    rent = rent.iloc[:,[0,5]]
    rent.columns = ["codi_districte", "rent_barcelona_district"]
    rent = rent.merge(admin, on = "codi_districte", how = "left")

    gdf = gdf.merge(rent.loc[:,["rent_barcelona_district", "ID"]], on = "ID", how = "left")

    gdf["rent_m2"] = gdf['rent_AMB_section']
    gdf.loc[np.isnan(gdf.rent_m2), "rent_m2"] = gdf.loc[np.isnan(gdf.rent_m2), "rent_barcelona_barri"]
    gdf.loc[gdf.rent_m2 < 0.5, "rent_m2"] = np.nan
    gdf.loc[np.isnan(gdf.rent_m2), "rent_m2"] = gdf.loc[np.isnan(gdf.rent_m2), "rent_AMB_city"]
    gdf.loc[np.isnan(gdf.rent_m2), "rent_m2"] = gdf.loc[np.isnan(gdf.rent_m2), "rent_barcelona_district"]

    ## DWELLING SIZE

    #section level - AMB
    size = pd.read_excel("C:/Users/1738037/OneDrive - UAB/1- CLIMGROW Charlotte/1- PSC cities/data_barcelona/AMB_Superficie.xlsx", header = 5)
    size = size.loc[:,["Codi_àmbit", "IV"]]
    size["IV"] = pd.to_numeric(size["IV"], errors = "coerce")
    size.columns = ['Codi_àmbit', 'size_AMB_section']

    gdf = gdf.merge(size, on = "Codi_àmbit", how = "left")

    #city level - AMB
    size = pd.read_excel("C:/Users/1738037/OneDrive - UAB/1- CLIMGROW Charlotte/1- PSC cities/data_barcelona/AMB_Superficie.xlsx", header = 5)
    size = size.loc[np.isnan(size.Codi_àmbit),["Codi_INE", "IV"]]
    size["IV"] = pd.to_numeric(size["IV"], errors = "coerce")
    size = size.iloc[0:29]
    size.columns = ['Codi_INE', 'size_AMB_city']
    size.Codi_INE = size.Codi_INE.astype(int)
    gdf = gdf.merge(size, on = "Codi_INE", how = "left")

    #Barri lebel - Barcelona
    size = pd.read_excel("C:/Users/1738037/OneDrive - UAB/1- CLIMGROW Charlotte/1- PSC cities/data_barcelona/trimestral_bcn_sup.xlsx", header = 20, sheet_name = "2023")
    size = size.iloc[:,[0,5]]
    size.columns = ["code_barri", "size_barcelona_barri"]
    size = size.merge(admin, left_on = "code_barri", right_on = "codi_barri", how = "left")

    gdf = gdf.merge(size.loc[:,["size_barcelona_barri", "ID"]], on = "ID", how = "left")

    #District level - Barcelona
    size = pd.read_excel("C:/Users/1738037/OneDrive - UAB/1- CLIMGROW Charlotte/1- PSC cities/data_barcelona/trimestral_bcn_sup.xlsx", header = 8, sheet_name = "2023")
    size = size.iloc[:,[0,5]]
    size.columns = ["codi_districte", "size_barcelona_district"]
    size = size.merge(admin, on = "codi_districte", how = "left")

    gdf = gdf.merge(size.loc[:,["size_barcelona_district", "ID"]], on = "ID", how = "left")

    gdf["size"] = gdf['size_AMB_section']
    gdf.loc[np.isnan(gdf["size"]), "size"] = gdf.loc[np.isnan(gdf["size"]), "size_barcelona_barri"]
    gdf.loc[gdf["size"] < 0.5, "size"] = np.nan
    gdf.loc[np.isnan(gdf["size"]), "size"] = gdf.loc[np.isnan(gdf["size"]), "size_AMB_city"]
    gdf.loc[np.isnan(gdf["size"]), "size"] = gdf.loc[np.isnan(gdf["size"]), "size_barcelona_district"]

    gdf = gdf.drop(columns = ['Codi_àmbit', 'rent_AMB_section',
       'Codi_INE', 'rent_AMB_city', 'rent_barcelona_barri',
       'rent_barcelona_district', 'size_AMB_section',
       'size_AMB_city', 'size_barcelona_barri', 'size_barcelona_district'])
    
    return gdf
