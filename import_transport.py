import numpy as np # type: ignore
from r5py import TravelTimeMatrixComputer, TransportMode, TransportNetwork # type: ignore
import os

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
    
    print("Start Computing Network")
    transport_network = TransportNetwork(
        # OSM data
        path_osm,
    
        # A list of GTFS file(s)
        list_path_gtfs
        )
    
    print("Transport Network Computed")

    i = 0
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

    print("Travel Times Transit Saved")