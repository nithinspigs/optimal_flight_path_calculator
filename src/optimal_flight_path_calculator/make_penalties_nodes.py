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
import utilities
import multiprocessing

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

def define_nodes_and_wind():
    nodes: dict[int, Any] = {}
    neighbors = list(range(ngrid_lat * ngrid_lon))
    for i, j in itertools.product(range(len(lon_axis)), range(len(lat_axis))):
        # print("(" + str(i) + ", " + str(j) + ")")
        node_number = utilities.get_node_number (i,j)
        #print (f"i:{i} j:{j} node:{node_number} lon:{lon_axis[i]} lat:{lat_axis[j]}")
        nodes[node_number] = utilities.Node(i, j, lon_axis[i], lat_axis[j])
        nodes[node_number].set_neighbors(neighbors)
        wind_lon, wind_lat = set_wind(lon_axis[i], lat_axis[j])
        nodes[node_number].set_wind(wind_lon, wind_lat)
    print("Done defining nodes and wind")
    return nodes

def set_wind(lon, lat):
    
    # dictionary in format {iata_code : (wind_lon, wind_lat)}
    airport_wind_info = get_tsv_airport_wind()
    print(airport_wind_info)
    sys.exit()
    # list of tuples in format [distance from node, wind_lon, wind_lat] for each airport within 500 miles of node
    nearby_airports = []
    total_distance = 0
    
    for iata_code in airport_wind_info.keys():
        # airports_lat_lon_info[iata_code][1] gives LONGITUDE coordinate of airport. tuple in the form of (lat, lon) so opposite
        dist_from_node = utilities.get_geodesic_distance(airports_lat_lon_info[iata_code][1], airports_lat_lon_info[iata_code][0], lon, lat)
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

#Origin:(-117.1611, 32.7157) Destination:(-74.006, 40.6712)

def build_penalties_array(cpu_num, cpu_count, nodes, return_dict):
    
    import utilities
    this_penalties_dict = {}
    
    for i in range(int(cpu_num / cpu_count * len(nodes)), int((cpu_num+1) / cpu_count * len(nodes))):
        if(i==0):
            i = i + int(cpu_num / cpu_count * len(nodes))
        # neighbors
        for j in range(len(nodes)):
            
            print("Node: " + str(i))
            print("Neighbor Node: " + str(j))
            #print(cpu_num)
            if(i != j):
                penalty = utilities.get_travel_time(nodes[i], nodes[j], nodes)
                if(penalty <= 0):
                    print("negative or zero penalty")
                    exit()
                this_penalties_dict[(i,j)] = penalty

    return_dict[cpu_num] = this_penalties_dict
    print("Done building this_penalties_dict_" + str(cpu_num))
    # print(penalties_array)

if __name__ == "__main__":
    start_time_0 = time.perf_counter_ns()
    airports_lat_lon_info = utilities.get_csv_lat_lon()
    # make sure to match with make_path.py
    ngrid_lat, ngrid_lon = utilities.ngrid_lat, utilities.ngrid_lon
    usa_shp_df, usa_bbox, lon_axis, lat_axis = utilities.usa_shp_df, utilities.usa_bbox, utilities.lon_axis, utilities.lat_axis

    nodes = define_nodes_and_wind()
    # print(nodes[99].wind_lon)
    # print(nodes[99].wind_lat)
    np.save("nodes.npy", nodes, allow_pickle=True)
    
    start_time = time.perf_counter_ns()
    penalties_array = np.zeros((len(nodes), len(nodes)))
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    processes = []
    cpu_count = multiprocessing.cpu_count()
    nodes_length = len(nodes)
    for i in range(cpu_count):
        p = multiprocessing.Process(target=build_penalties_array, args=(i, cpu_count, nodes, return_dict))
        processes.append(p)
        p.start()
    for process in processes:
        process.join()
    print(len(return_dict))
    
    for penalty_dict in list(return_dict.values()):
        for tuple in penalty_dict.keys():
            i = tuple[0]
            j = tuple[1]
            penalties_array[i,j] = penalty_dict[tuple]
    
    
    np.save("penalties_array.npy", penalties_array)
    time_taken = round((time.perf_counter_ns() - start_time_0) * 1.0e-9, 4)
    print(f"\nTime taken to build weights:{time_taken} seconds")
    #print(penalties_array)



