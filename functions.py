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

def compute_transport_cost(COST_CAR, COST_PT, tax):
    return np.fmin(COST_CAR * tax, COST_PT), np.argmin([COST_CAR * tax, COST_PT], 0)

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
def make_people_move(indiv_distance_matrix_new, N, distance, indiv_distance_matrix, proba_of_moving_from, proba_of_moving_to):
    
    has_moved = np.zeros(N)
    indiv_moving = np.random.binomial(1, 0.3, N) #Each individual has a 30% chance to be willing to move.

    for i in np.arange(N):
        if indiv_moving[i] == 1:
            if sum(indiv_distance_matrix[i,:]) > 0:
                proba_of_moving_here = sum((indiv_distance_matrix[i,:] * proba_of_moving_from)[indiv_distance_matrix[i,:] > 0])
                if proba_of_moving_here < 0:
                    proba_of_moving_here = 0
                moving = np.random.binomial(1, proba_of_moving_here)
                if moving == 1:
                    has_moved[i] = 1
                    indiv_distance_matrix_new[i,:] = np.zeros(len(distance))
                    #destination = np.random.choice(np.arange(len(distance)), p=proba_of_moving_to)
                    destination = np.arange(len(distance))[np.searchsorted(np.cumsum(proba_of_moving_to), np.random.random(), side="right")]
                    indiv_distance_matrix_new[i,destination] = 1

    return indiv_distance_matrix_new, has_moved

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
