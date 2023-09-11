import scipy
import sys
import os
import itertools
import numpy as np
np.set_printoptions(threshold=np.inf, linewidth=300)
import pyproj
import geopandas
import pandas
import time
import math
from typing import Any
from exceptions import ErrorHandler
error_handler = ErrorHandler()

class Node:
    def __init__(self, i, j, lon, lat):
        self.i = i
        self.j = j
        self.lon = lon
        self.lat = lat
        self.neighbor_node_numbers: list[int] = []

    def set_neighbors (self, neighbors):
        self.neighbor_node_numbers = neighbors

    def set_wind(self, wind_lon, wind_lat):
        self.wind_lon = wind_lon
        self.wind_lat = wind_lat

def get_csv_lat_lon():
    lat_lon_file: str = "./lat-lon-data/us-airport-codes.csv"
    lat_lon_df = pandas.read_csv(lat_lon_file)
    iata_codes = list(lat_lon_df["iata_code"])
    coordinates = list(lat_lon_df["coordinates"])
    
    airports_lat_lon_info = {}
    for i in range(len(iata_codes)):
        airports_lat_lon_info[iata_codes[i]] = eval(coordinates[i])
    
    return airports_lat_lon_info
    
def get_tsv_airport_wind():
    airport_wind = pandas.read_csv("./wind-data/wind.tsv", sep=' ')
    iata_codes = list(airport_wind["FT"])
    wind_directions = list(airport_wind["30000"])
    for i in range(len(wind_directions)):
        wind_directions[i] = int(str(wind_directions[i])[:2]) * 10
    wind_speeds = list(airport_wind["30000"])
    for i in range(len(wind_speeds)):
        wind_speeds[i] = int(str(wind_directions[i])[:2])
    
    airport_wind_info = {}
    for i in range(len(iata_codes)):
        if(str(iata_codes[i]) not in airports_lat_lon_info.keys()):
            continue
        # airport_wind_info[str(iata_codes[i])] = (wind_speeds[i] * math.sin(math.radians(wind_directions[i])), wind_speeds[i] * math.cos(math.radians(wind_directions[i])))
        airport_wind_info[str(iata_codes[i])] = (wind_directions[i], wind_speeds[i] * 1.15078) # convert from knots to mph
    
    return airport_wind_info
    
def get_grid_lat_lon():
    usa_shape_file: str = "./usa-shape/usa-states-census-2014.shp"
    usa_shp_df = geopandas.read_file(usa_shape_file)
    print(f"The CRS:\n{usa_shp_df.crs}")
    usa_bbox = usa_shp_df.total_bounds # [-124.725839   24.498131  -66.949895   49.384358]
    print(f"BBOX:\n{usa_bbox}")
    lon_axis = np.linspace(usa_bbox[0], usa_bbox[2], num=ngrid_lon)
    lat_axis = np.linspace(usa_bbox[1], usa_bbox[3], num=ngrid_lat)
    
    return usa_shp_df, usa_bbox, lon_axis, lat_axis


def get_node_number(lon_i, lat_j):
    return lon_i + lat_j * len(lon_axis)

def get_geodesic_distance( lon1, lat1, lon2, lat2 ):
    geodesic_distance = geodesic_projection.line_length([lon1, lon2], [lat1, lat2]) # meters
    return geodesic_distance * 1.0e-3 / 1.60934 # miles

def define_nodes_and_wind():
    nodes: dict[int, Any] = {}
    neighbors = list(range(ngrid_lat * ngrid_lon))
    for i, j in itertools.product(range(len(lon_axis)), range(len(lat_axis))):
        # print("(" + str(i) + ", " + str(j) + ")")
        node_number = get_node_number (i,j)
        #print (f"i:{i} j:{j} node:{node_number} lon:{lon_axis[i]} lat:{lat_axis[j]}")
        nodes[node_number] = Node(i, j, lon_axis[i], lat_axis[j])
        nodes[node_number].set_neighbors(neighbors)
        wind_lon, wind_lat = set_wind(lon_axis[i], lat_axis[j])
        nodes[node_number].set_wind(wind_lon, wind_lat)
    print("Done defining nodes and wind")
    return nodes

def set_wind(lon, lat):
    
    # dictionary in format {iata_code : (wind_lon, wind_lat)}
    airport_wind_info = get_tsv_airport_wind()
    # list of tuples in format [distance from node, wind_lon, wind_lat] for each airport within 500 miles of node
    nearby_airports = []
    total_distance = 0
    
    for iata_code in airport_wind_info.keys():
        # airports_lat_lon_info[iata_code][1] gives LONGITUDE coordinate of airport. tuple in the form of (lat, lon) so opposite
        dist_from_node = get_geodesic_distance(airports_lat_lon_info[iata_code][1], airports_lat_lon_info[iata_code][0], lon, lat)
        # nodes with no nearby airports will default to wind of 0
        if(dist_from_node < 200):
            nearby_airports.append([dist_from_node, airport_wind_info[iata_code][1] * math.cos(math.radians(airport_wind_info[iata_code][0])), airport_wind_info[iata_code][1] * math.sin(math.radians(airport_wind_info[iata_code][0]))])
            total_distance = total_distance + dist_from_node
    
    wind_lon = 0
    wind_lat = 0
    for airport in nearby_airports:
        wind_lon = wind_lon + float(airport[0]) / total_distance * airport[1]
        wind_lat = wind_lat + float(airport[0]) / total_distance * airport[2]

    # print("(" + str(lon) + ", " + str(lat) + ")")
    return wind_lon, wind_lat

def find_wind(lon, lat):
    return interpolate_for_wind(lon, lat)

def interpolate_for_wind(lon, lat):
    i, j = (np.argmin(np.abs(lon_axis - lon)) , np.argmin(np.abs(lat_axis - lat)))

    i = min(i, len(lon_axis) - 2)
    j = min(j, len(lat_axis) - 2)

    left_bot = get_node_number (i,j)
    right_bot = get_node_number (i+1,j)
    left_top = get_node_number (i,j+1)
    right_top = get_node_number (i+1,j+1)

    wind_lon_bot = ( nodes[left_bot].wind_lon * (lon_axis[i+1] - lon) + nodes[right_bot].wind_lon * (lon - lon_axis[i]) ) / (lon_axis[i+1] - lon_axis[i])
    wind_lon_top = ( nodes[left_top].wind_lon * (lon_axis[i+1] - lon) + nodes[right_top].wind_lon * (lon - lon_axis[i]) ) / (lon_axis[i+1] - lon_axis[i])
    wind_lon = ( wind_lon_bot * (lat_axis[j+1] - lat) + wind_lon_top * (lat - lat_axis[j]) ) / (lat_axis[j+1] - lat_axis[j])

    wind_lat_bot = ( nodes[left_bot].wind_lat * (lon_axis[i+1] - lon) + nodes[right_bot].wind_lat * (lon - lon_axis[i]) ) / (lon_axis[i+1] - lon_axis[i])
    wind_lat_top = ( nodes[left_top].wind_lat * (lon_axis[i+1] - lon) + nodes[right_top].wind_lat * (lon - lon_axis[i]) ) / (lon_axis[i+1] - lon_axis[i])
    wind_lat = ( wind_lat_bot * (lat_axis[j+1] - lat) + wind_lat_top * (lat - lat_axis[j]) ) / ( lat_axis[j+1] - lat_axis[j] )

    return wind_lon, wind_lat

#Origin:(-117.1611, 32.7157) Destination:(-74.006, 40.6712)

def get_geodesic_path_coords_in_rectilinear_lon_lat(lon1, lon2, lat1, lat2, npts=10):
    geodesic_path_coords_in_rectilinear_lon_lat = geodesic_projection.npts(lon1=lon1, lon2=lon2, lat1=lat1, lat2=lat2, npts=npts)
    geodesic_lons = [v[0] for v in geodesic_path_coords_in_rectilinear_lon_lat]
    geodesic_lats = [v[1] for v in geodesic_path_coords_in_rectilinear_lon_lat]
    return geodesic_lons, geodesic_lats

def get_travel_time(src_node, target_node):
    travel_time = 0.0
    points_for_line_integral = 30
    
    lons_path, lats_path = get_geodesic_path_coords_in_rectilinear_lon_lat(src_node.lon, target_node.lon, src_node.lat, target_node.lat, points_for_line_integral)

    for i in range(0, len(lons_path) - 1):
        lon_1, lon_2 = lons_path[i], lons_path[i+1]
        lat_1, lat_2 = lats_path[i], lats_path[i+1]
        wind_1_lon, wind_1_lat = find_wind(lon_1, lat_1)
        wind_2_lon, wind_2_lat = find_wind(lon_2, lat_2)
            
        wind_vector = np.array( [ (wind_1_lon + wind_2_lon)/2, (wind_1_lat + wind_2_lat)/2 ] )
        wind_mag = np.linalg.norm(wind_vector)

        dist_mag = get_geodesic_distance (lon_1, lat_1, lon_2, lat_2)

        # and delta based on wind vector because displacement vector should counteract wind
        no_wind_displacement_vector = np.array([lon_2 - lon_1, lat_2 - lat_1])
        no_wind_displacement_mag = np.linalg.norm(no_wind_displacement_vector)
        displacement_vector = no_wind_displacement_vector - (no_wind_displacement_mag / 500) * wind_vector
        displacement_mag = np.linalg.norm(displacement_vector)

        # No wind: wind_penalty = dist_mag
        # Wind: wind_penalty = dist_mag * (1.0 - dot_product)
        
        # try to consider wind when it is coming from the side.
        # try doing dijkstra with only immediate neighbors
        
        # dot_product = np.dot(displacement_vector, wind_vector) / (wind_mag * displacement_mag). significant difference
        # wind_penalty = wind_penalty - dot_product * dist_mag * wind_mag * 0.05
        speed_dot_product = np.dot(displacement_vector, wind_vector) / displacement_mag # wind speed in the direction of plane's movement. also try dividing by displacement_vector_mag
        
        if(speed_dot_product < -500):
            print("wind greater than 500mph, abort")
            exit()
        
        added_time = dist_mag / (500 + speed_dot_product) # distance divided by speed gives time that wind adds to (or subtracts from) total travel time
        travel_time = travel_time + added_time

    if(travel_time < 0):
        print("negative travel time")
        exit()
    
    return travel_time

def build_penalties_array():
 
    penalties = np.zeros((len(nodes), len(nodes)))
    
    for i in range(len(nodes)):
        # neighbors
        for j in range(len(nodes)):
            
            print("Node: " + str(i))
            print("Neighbor Node: " + str(j))
            if(i != j):
                penalty = get_travel_time(nodes[i], nodes[j])
                if(penalty <= 0):
                    print("negative or zero penalty")
                    exit()
                penalties[i,j] = penalty

    print("Done building penalties array")
    print(penalties)
    return penalties

if __name__ == "__main__":
    start_time_0 = time.perf_counter_ns()
    geodesic_projection = pyproj.Geod(ellps='WGS84')
    airports_lat_lon_info = get_csv_lat_lon()
    # make sure to match with make_path.py
    ngrid_lat, ngrid_lon = 20, 20
    usa_shp_df, usa_bbox, lon_axis, lat_axis = get_grid_lat_lon()

    nodes = define_nodes_and_wind()
    # print(nodes[99].wind_lon)
    # print(nodes[99].wind_lat)
    np.save("nodes.npy", nodes, allow_pickle=True)
    start_time = time.perf_counter_ns()
    np.save("penalties_array.npy", build_penalties_array())
    time_taken = round((time.perf_counter_ns() - start_time_0) * 1.0e-9, 4)
    print(f"\nTime taken to build weights:{time_taken} seconds")



