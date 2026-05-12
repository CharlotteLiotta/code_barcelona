import numpy as np # type: ignore
from r5py import TravelTimeMatrixComputer, TransportMode, TransportNetwork, DetailedItineraries # type: ignore
import os
from shapely.geometry import Point # type: ignore
from shapely.geometry import LineString, MultiLineString
import pandas as pd

def import_transport_times(gdf, date_here, path_data, employment_centers, option_center):
    """ Import transport times using GTFS data for transit, OSM data for private cars, and the r5py package """

    points = gdf.copy()
    points["geometry"] = points.centroid
    points = points.loc[:,["ID", "geometry"]]
    points.columns = ["id", "geometry"]

    points_dest = employment_centers.copy()
    points_dest = points_dest.loc[:,["cluster", "geometry"]]
    points_dest.columns = ["id", "geometry"]

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
            i = 0
            transport_mode = [TransportMode.CAR]
        elif mode == "transit":
            i = 0
            transport_mode = [TransportMode.TRANSIT, TransportMode.WALK]
    
        while i < (len(points) - 100):

            i = i + 100
    
            travel_time_matrix_computer = TravelTimeMatrixComputer(
                transport_network,
                origins=points.iloc[i - 100:i, :],
                destinations= points_dest,
                departure=date_here,
                transport_modes=transport_mode
                )
        
            travel_time_matrix = travel_time_matrix_computer.compute_travel_times()

            np.save(path_data + "travel_time_matrix_poly_" + mode + "_" + str(i) + "_" + option_center, travel_time_matrix)
            print("Travel time " + mode + ": ", round(100 * i/len(points)), "%")
    
        travel_time_matrix_computer = TravelTimeMatrixComputer(
            transport_network,
            origins=points.iloc[i:len(points), :],
            destinations=points_dest,
            departure=date_here,
            transport_modes=transport_mode
            )
        
        travel_time_matrix = travel_time_matrix_computer.compute_travel_times()

        np.save(path_data + "travel_time_matrix_poly_" + mode + "_" + str(len(points))+ "_" + option_center, travel_time_matrix)
        print("Travel time " + mode + " saved")

    compute_travel_times("car")
    compute_travel_times("transit")

def import_car_distance(gdf, date_here, center, path_data):
    """ Import transport times using GTFS data for transit, OSM data for private cars, and the r5py package """

    points = gdf.copy()
    points["geometry"] = points.centroid
    points = points.loc[:,["ID", "geometry"]]
    points.columns = ["id", "geometry"]

    path_osm = path_data + "barcelona.osm.pbf"
    #path_osm = path_data + "cataluna-latest.osm.pbf"

    #list_path_gtfs = os.listdir(path_data + 'gtfs/')
    #list_path_gtfs = [i for i in list_path_gtfs if i.endswith('zip')]
    #list_path_gtfs = [path_data + 'gtfs/' + sub for sub in list_path_gtfs] 
    list_path_gtfs = [] 

    print("Start Computing Network")
    transport_network = TransportNetwork(
        path_osm,
        list_path_gtfs
        )
    
    print("Transport Network Computed")

    def compute_travel_distance(mode):

        if mode == "car":
            transport_mode = [TransportMode.CAR]
        elif mode == "transit":
            transport_mode = [TransportMode.TRANSIT, TransportMode.WALK]
    
        i = 1800
        while i < (len(points) - 100):

            i = i + 100
    
            detailed_itin = DetailedItineraries(
                transport_network,
                origins=points.iloc[i - 100:i, :],
                destinations= points.loc[points.id == center,:],
                departure=date_here,
                transport_modes=transport_mode,
                snap_to_network=True
                )
        
            np.save(path_data + "detailed_itin_" + mode + "_" + str(i) + ".npy", detailed_itin)
            print("Travel distance " + mode + ": ", round(100 * i/len(points)), "%")
    
        detailed_itin = DetailedItineraries(
            transport_network,
            origins=points.iloc[i:len(points), :],
            destinations=points.loc[points.id == center,:],
            departure=date_here,
            transport_modes=transport_mode,
            snap_to_network=True
            )
        
        np.save(path_data + "detailed_itin_" + mode + "_" + str(len(points)) + ".npy", detailed_itin)
        print("Travel distance " + mode + " saved")

    compute_travel_distance("car")


def load_distance_car(travel_time_matrix_car, gdf, employment_centers, zone_tax):
    
    travel_time_matrix_car = travel_time_matrix_car.merge(gdf[['ID', 'geometry']].rename(columns={'ID': 'from_id', 'geometry': 'from_geom'}), on='from_id', how='left')
    travel_time_matrix_car = travel_time_matrix_car.merge(employment_centers[['cluster', 'geometry']].rename(columns={'cluster': 'to_id', 'geometry': 'to_geom'}), on='to_id', how='left')
    
    #travel_time_matrix_car['distance_car'] = travel_time_matrix_car.apply(lambda row: row['from_geom'].centroid.distance(row['to_geom'].centroid) if row['from_geom'] is not None and row['to_geom'] is not None else None, axis=1)

    #travel_time_matrix_car["distance_in_zone"] = ((travel_time_matrix_car.from_id.isin(houses_in_toll_area) * travel_time_matrix_car.to_id.isin(jobs_in_toll_area))) * travel_time_matrix_car['distance_car']
    #travel_time_matrix_car['distance_out_zone'] = ((~travel_time_matrix_car.from_id.isin(houses_in_toll_area) * ~travel_time_matrix_car.to_id.isin(jobs_in_toll_area))) * travel_time_matrix_car['distance_car']

    

    def split_distance(row, zone):
        p1 = row['from_geom']
        p2 = row['to_geom']

        if p1 is None or p2 is None:
            return pd.Series([None, None, None])

        line = LineString([p1.centroid, p2.centroid])

        inside = line.intersection(zone)
        outside = line.difference(zone)

        d_inside  = inside.length if not inside.is_empty else 0
        d_outside = outside.length if not outside.is_empty else 0

        return pd.Series([line.length, d_inside, d_outside])

    
    travel_time_matrix_car[['dist_total',
                        'dist_inside',
                        'dist_outside']] = (travel_time_matrix_car.apply(split_distance,
                                 zone=zone_tax,
                                 axis=1))
    
    travel_time_matrix_car.loc[:, "distance_in_zone"] = travel_time_matrix_car.loc[:, "dist_inside"]
    travel_time_matrix_car.loc[:, "distance_out_zone"] = travel_time_matrix_car.loc[:, "dist_outside"]

    travel_time_matrix_car = travel_time_matrix_car.drop(columns = ['dist_total','dist_inside','dist_outside'])
    travel_time_matrix_car['distance_car'] = travel_time_matrix_car['distance_in_zone'] + travel_time_matrix_car['distance_out_zone']

    return travel_time_matrix_car