import numpy as np # type: ignore
from r5py import TravelTimeMatrixComputer, TransportMode, TransportNetwork # type: ignore
import os

def import_transport_times(gdf, date_here, center, path_data, OPTION_SAVE):
    """ Import transport times using GTFS data for transit, OSM data for private cars, and the r5py package """

    #le mieux a l'air d'être ici (existe en simplfifié ou normal): https://t-mobilitat.atm.cat/web/t-mobilitat/datos-abiertos/catalogo-de-datos/informacion-estatica
    
    #ADD https://fgc.opendatasoft.com/explore/dataset/gtfs_zip/table/
    #ADD https://datos.gob.es/en/catalogo/a09002970-red-de-transporte-por-carretera-paradas-lineas-y-horarios-de-los-autobuses-interurbanos-de-catalunya
    #ADD https://www.amb.cat/web/area-metropolitana/dades-obertes/cataleg/detall/-/dataset/serveis-gtfs-de-tmb/1107694/11692 PAS DISPO?
    #ADD https://www.amb.cat/es/web/area-metropolitana/dades-obertes/cataleg/detall/-/dataset/informacion-de-companias--lineas-y-recorridos/1033377/11692?_DatasetSearchListPortlet_WAR_AMBSearchPortletportlet_pageNum=4&_DatasetSearchListPortlet_WAR_AMBSearchPortletportlet_categoria=mobilitat&_DatasetSearchListPortlet_WAR_AMBSearchPortletportlet_detailBackURL=https%3A%2F%2Fwww.amb.cat%2Fes%2Fweb%2Farea-metropolitana%2Fdades-obertes%2Fcataleg%2Fllistat (seulement des bus?)
    
    points = gdf.copy()
    points["geometry"] = points.centroid
    points = points.loc[:,["ID", "geometry"]]
    points.columns = ["id", "geometry"]

    path_osm = path_data + "barcelona.osm.pbf"
    #path_osm = path_data + "cataluna-latest.osm.pbf"

    list_path_gtfs = os.listdir(path_data + 'gtfs/')
    list_path_gtfs = [i for i in list_path_gtfs if i.endswith('zip')]
    list_path_gtfs = [path_data + 'gtfs/' + sub for sub in list_path_gtfs] 
    
    print("Start Computing Network")
    transport_network = TransportNetwork(
        path_osm,
        list_path_gtfs
        )
    
    print("Transport Network Computed")

    def compute_travel_times(mode):

        if mode == "car":
            transport_mode = [TransportMode.CAR]
        elif mode == "transit":
            transport_mode = [TransportMode.TRANSIT, TransportMode.WALK]
    
        i = 0
        while i < (len(points) - 100):

            i = i + 100
    
            travel_time_matrix_computer = TravelTimeMatrixComputer(
                transport_network,
                origins=points.iloc[i - 100:i, :],
                destinations=points.loc[points.id == center,:],
                departure=date_here,
                transport_modes=transport_mode
                )
        
            travel_time_matrix = travel_time_matrix_computer.compute_travel_times()
            np.save(path_data + "travel_time_matrix_" + mode + "_" + str(i) + ".npy", travel_time_matrix)
            print("Travel time " + mode + ": ", round(100 * i/len(points)), "%")
    
        travel_time_matrix_computer = TravelTimeMatrixComputer(
            transport_network,
            origins=points.iloc[i:len(points), :],
            destinations=points.loc[points.id == center,:],
            departure=date_here,
            transport_modes=transport_mode
            )
        
        travel_time_matrix = travel_time_matrix_computer.compute_travel_times()
        np.save(path_data + "travel_time_matrix_" + mode + "_" + str(len(points)) + ".npy", travel_time_matrix)
        print("Travel time " + mode + " saved")

    compute_travel_times("car")
    compute_travel_times("transit")