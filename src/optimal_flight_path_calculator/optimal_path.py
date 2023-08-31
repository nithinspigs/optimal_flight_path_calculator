import argparse
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

import matplotlib.pyplot as plt
import sys
import os
import itertools
import numpy as np
np.set_printoptions(threshold=np.inf, linewidth=300)

from matplotlib.animation import FuncAnimation

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

    def set_wind (self, wind_lon, wind_lat):
        self.wind_lon = wind_lon
        self.wind_lat = wind_lat

def plot_wind(ax):
    step = 1
    for j in range(0,len(lat_axis), step):
        for i in range(0,len(lon_axis), step):
            node = nodes[get_node_number(i,j)]
            vx = node.wind_lon
            vy = node.wind_lat
            if (abs(vx) > 0.1):
                vx = vx / abs(vx)
            if (abs(vy) > 0.1):
                vy = vy / abs(vy)
            if (abs(vx) > 0.5) or (abs(vy) > 0.5):
                ax.arrow(node.lon, node.lat, vx, vy, head_length = 0.15, head_width = 0.15)
                #ax.arrow(node.lon, node.lat, node.wind_lon, node.wind_lat, head_length = 0.15, head_width = 0.15)

def get_csv_lat_lon():
    lat_long_file: str = "./lat-long-data/airports.csv"
    lat_long_df = pandas.read_csv(lat_long_file)
    iata_codes = list(lat_long_df["IATA"])
    lats = list(lat_long_df["LATITUDE"])
    lons = list(lat_long_df["LONGITUDE"])
    lat_lon_info = {iata_codes[i] : (lons[i], lats[i]) for i in range(len(iata_codes))}
    #print(lat_lon_info)
    #print(type(lat_lon_info))
    #print(type(lat_lon_info['SAN']))
    return lat_lon_info
    
def get_grid_lat_lon():
    usa_shape_file: str = "./usa-shape/usa-states-census-2014.shp"
    usa_shp_df = geopandas.read_file(usa_shape_file)
    print (f"The CRS:\n{usa_shp_df.crs}")
    usa_bbox = usa_shp_df.total_bounds # [-124.725839   24.498131  -66.949895   49.384358]
    print (f"BBOX:\n{usa_bbox}")
    lon_axis = np.linspace(usa_bbox[0], usa_bbox[2], num=ngrid_lon)
    lat_axis = np.linspace(usa_bbox[1], usa_bbox[3], num=ngrid_lat)
    
    # rounds origin coordinates to closest node on the grid. lon_axis is a list
    (origin_i, origin_j) = (np.argmin(np.abs(lon_axis - lon_origin)) , np.argmin(np.abs(lat_axis - lat_origin)))
    lon_axis[origin_i] = lon_origin
    lat_axis[origin_j] = lat_origin
    
    (destination_i, destination_j) = (np.argmin(np.abs(lon_axis - lon_destination)) , np.argmin(np.abs(lat_axis - lat_destination)))
    lon_axis[destination_i] = lon_destination
    lat_axis[destination_j] = lat_destination
    
    print (f"\nOrigin:{origin} Destination:{destination}")
    return usa_shp_df, usa_bbox, lon_axis, lat_axis, origin_i, origin_j, destination_i, destination_j


def get_node_number (lon_i, lat_j):
    return lon_i + lat_j * len(lon_axis)

def get_euclidean_distance(lat1, lon1, lat2, lon2):
    displacement_vector = np.array([lat1 - lat2, lon1 - lon2])
    euclidean_distance = np.linalg.norm(displacement_vector)
    return euclidean_distance

def get_geodesic_distance( lon1, lat1, lon2, lat2 ):
    geodesic_distance = geodesic_projection.line_length([lon1, lon2], [lat1, lat2]) # meters
    return geodesic_distance * 1.0e-3 / 1.60934 # miles

def process_args ():
    parser = argparse.ArgumentParser(
        description="Find the optimal flight path between an origin/destination"
    )
    parser.add_argument(
        "--euclidean",
        help="Use Euclidan. True/False. Default is geodesic",
        action="store",
        default=0,
        required=True,
        type=int
    )
    parser.add_argument(
        "--ngrid_lon",
        help="Number of grid points for Longitude",
        action="store",
        required=True,
        type=int
    )   
    parser.add_argument(
        "--ngrid_lat",
        help="Number of grid points for Latitude",
        action="store",
        required=True,
        type=int
    )
    parser.add_argument(
        "--origin",
        nargs='+',
        help="IATA code of origin",
        action="store",
        required=True,
        #type=string
    )   
    parser.add_argument(
        "--destination",
        nargs='+',
        help="IATA code of destination",
        action="store",
        required=True,
        #type=string
    )   
    parser.add_argument(
        "--wind_magnitude",
        help="Magnitude of the wind",
        default=1.0,
        action="store",
        required=False,
        type=float
    )   

    args = parser.parse_args()
    euclidean= args.euclidean
    if euclidean == 1:
        euclidean = True
    else:
        euclidean = False
    ngrid_lat = args.ngrid_lat
    ngrid_lon = args.ngrid_lon
    origin = lat_lon_info[args.origin[0]]
    destination = lat_lon_info[args.destination[0]]
    wind_magnitude = args.wind_magnitude
    print (euclidean, ngrid_lat, ngrid_lon, origin, destination, wind_magnitude)
    return euclidean, ngrid_lat, ngrid_lon, origin, destination, wind_magnitude

def define_nodes_and_wind():
    nodes: dict[int, Any] = {}
    neighbors = list(range(ngrid_lat * ngrid_lon))
    for i, j in itertools.product(range(len(lon_axis)), range(len(lat_axis))):
        node_number = get_node_number (i,j)
        #print (f"i:{i} j:{j} node:{node_number} lon:{lon_axis[i]} lat:{lat_axis[j]}")
        nodes[node_number] = Node(i, j, lon_axis[i], lat_axis[j])
        nodes[node_number].set_neighbors(neighbors)
        wind_lon, wind_lat = set_wind(lon_axis[i], lat_axis[j])
        nodes[node_number].set_wind(wind_lon, wind_lat)
    return nodes

def find_wind (lon, lat):
    return set_wind (lon, lat)

def interpolate_for_wind (lon, lat):
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

def set_wind(lon, lat):
    '''
    angle = 0
    wind_mag = 0.0

    if lon >= lon_origin-2 and lon <= lon_origin+2 and lat >= lat_origin+1 and lat <= lat_destination-1:
        wind_mag = wind_magnitude
        angle = math.pi/2.0
    elif lon >= lon_origin+2 and lon <= lon_destination-2 and lat >= lat_destination-1 and lat <= lat_destination+1:
        wind_mag = wind_magnitude
        angle = 0.0
    elif lon >= lon_destination-2 and lon <= lon_destination+2 and lat <= lat_destination-1 and lat >= lat_origin+1:
        wind_mag = wind_magnitude
        angle = -math.pi/2.0
    elif lon >= lon_origin+2 and lon <= lon_destination-2 and lat >= lat_origin-1 and lat <= lat_origin+1:
        wind_mag = wind_magnitude
        angle = math.pi
    wind_lon = wind_mag*math.cos(angle)
    wind_lat = wind_mag*math.sin(angle)
    '''
    
    angle = 0
    wind_mag = 0
    if(lat > 34 and lat < 39):
        wind_mag = wind_magnitude
    wind_lon = wind_mag*math.cos(angle)
    wind_lat = wind_mag*math.sin(angle)

    return wind_lon, wind_lat

def get_travel_time(src_node, target_node):
    # is distance is 3000 miles, we want 30 points
    # overall_dist = get_geodesic_distance(src_node.lon, src_node.lat, target_node.lon, target_node.lat)
    travel_time = 0.0
    points_for_line_integral = 30
    if euclidean:
        lons_path = np.linspace(src_node.lon, target_node.lon, num=points_for_line_integral)
        if abs(target_node.lon - src_node.lon) > 1.0e-6:
            slope = (target_node.lat - src_node.lat) / (target_node.lon - src_node.lon)
            lats_path = [src_node.lat + slope * (lons_path_lon - src_node.lon) for lons_path_lon in lons_path]
        else:
            lats_path = np.linspace(src_node.lat, target_node.lat, num=points_for_line_integral)
    else:
        lons_path, lats_path = get_geodesic_path_coords_in_rectilinear_lon_lat(src_node.lon, target_node.lon, src_node.lat, target_node.lat, points_for_line_integral)

    for i in range(0, len(lons_path) - 1):
        lon_1, lon_2 = lons_path[i], lons_path[i+1]
        lat_1, lat_2 = lats_path[i], lats_path[i+1]
        wind_1_lon, wind_1_lat = find_wind(lon_1, lat_1)
        wind_2_lon, wind_2_lat = find_wind(lon_2, lat_2)
            
        wind_vector = np.array( [ (wind_1_lon + wind_2_lon)/2, (wind_1_lat + wind_2_lat)/2 ] )
        wind_mag = np.linalg.norm(wind_vector)

        if euclidean:
            dist_mag = get_euclidean_distance (lon_1, lat_1, lon_2, lat_2)
        else:
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
        if wind_mag > 0:
            # dot_product = np.dot(displacement_vector, wind_vector) / (wind_mag * displacement_mag). significant difference
            # wind_penalty = wind_penalty - dot_product * dist_mag * wind_mag * 0.05
            speed_dot_product = np.dot(displacement_vector, wind_vector) / displacement_mag # wind speed in the direction of plane's movement. also try dividing by displacement_vector_mag
            added_time = dist_mag / (500 + speed_dot_product) # distance divided by speed gives time that wind adds to (or subtracts from) total travel time
            '''
            if(added_time < 0):
                print(speed_dot_product)
                print("source: (" + str(src_node.lon) + ", " + str(src_node.lat) + ") target: (" + str(target_node.lon) + ", " + str(target_node.lat) + ")")
                exit()
            '''
            travel_time = travel_time + added_time
        else:
            travel_time = travel_time + dist_mag / 500

    return travel_time

def get_distance(src_node, target_node):
    if euclidean:
        distance = get_euclidean_distance(src_node.lon, src_node.lat, target_node.lon, target_node.lat)
    else:
        distance = get_geodesic_distance(src_node.lon, src_node.lat, target_node.lon, target_node.lat)
    return distance

def build_csr_matrix():
 
    penalties = np.zeros((len(nodes), len(nodes)))
    
    for i in range(len(nodes)):
        # neighbors
        for j in range(len(nodes)):
            
            if(i != j):
                penalties[i,j] = get_travel_time(nodes[i], nodes[j])

    weights = csr_matrix(penalties)
    np.save("penalties_array.npy", penalties)
    print (f"\nCSR Weight Matrix:{weights.shape}")
    # print (f"\nPenalty Computation Times. Dist:{dist_time_taken*1.0e-9} Wind:{wind_time_taken*1.0e-9}")
    return weights

def process_results():
    if euclidean:
        min_dist_cost = get_euclidean_distance (lon_origin, lat_origin, lon_destination, lat_destination)
    else:
        #min_dist_cost = get_geodesic_distance (lon_origin, lat_origin, lon_destination, lat_destination)
        geodesic_cost_o_d = get_travel_time(nodes[origin_node_number], nodes[dest_node_number])
        geodesic_cost_d_o = get_travel_time(nodes[dest_node_number], nodes[origin_node_number])

    o_d_cost = dist_matrix[0, dest_node_number]
    d_o_cost = dist_matrix[1, origin_node_number]
    print (f"Geodesic Cost: origin => dest: {geodesic_cost_o_d}")
    print (f"Geodesic Cost: origin => dest: {geodesic_cost_d_o}")
    print (f"Resolution: {ngrid_lat}x{ngrid_lon} Min Distance+Wind Cost: origin => dest: {o_d_cost}")
    print (f"Resolution: {ngrid_lat}x{ngrid_lon} Min Distance+Wind Cost: dest=> origin: {d_o_cost}")

    geodesic_node_string = str(origin_node_number) + '=>' + str(dest_node_number)
    str1 = "Geodesic o-d: " + geodesic_node_string + " : " + str(round(geodesic_cost_o_d, 3))
    ax.text(-124, 30, str1, color='k',fontsize='small')
    
    geodesic_node_string = str(dest_node_number) + '=>' + str(origin_node_number)
    str1 = "Geodesic d-o: " + geodesic_node_string + " : " + str(round(geodesic_cost_d_o, 3))
    ax.text(-124, 29, str1, color='k',fontsize='small')

    o_d_path = [dest_node_number]
    prev_one = predecessors[0,dest_node_number]
    while prev_one != -9999:
        o_d_path.append(prev_one)
        prev_one = predecessors[0,prev_one]
    o_d_path.reverse()
    
    path_node_string = '=>'.join([str(node_number) for node_number in o_d_path])
    str2 = "O=>D: " + path_node_string + " : " + str(round(o_d_cost, 3))
    ax.text(-124, 28, str2, color='g',fontsize='small')

    print (f"O => D Path:{path_node_string}")
    print (path_node_string)
    x_path_o_d = [nodes[i].lon for i in o_d_path]
    y_path_o_d = [nodes[i].lat for i in o_d_path]
    plt.scatter(x_path_o_d, y_path_o_d, color='g', marker='o')

    d_o_path = [origin_node_number]
    prev_one = predecessors[1,origin_node_number]
    str3 = ''
    while prev_one != -9999:
        d_o_path.append(prev_one)
        prev_one = predecessors[1,prev_one]
    if (o_d_path != d_o_path):
        d_o_path.reverse()
        path_node_string = '=>'.join([str(node_number) for node_number in d_o_path])
        print (f"\n{dest_node_number} => {origin_node_number} Takes a DIFFERENT PATH\n")
        print (path_node_string)
        str3 = "D=>O: " + path_node_string + " : " + str(round(d_o_cost, 3))
        ax.text(-124, 27, str3, color='b',fontsize='small')
        x_path_d_o = [nodes[i].lon for i in d_o_path]
        y_path_d_o = [nodes[i].lat for i in d_o_path]
        plt.scatter(x_path_d_o, y_path_d_o, color='b', marker='o')
    else:
        d_o_path.reverse()
        x_path_d_o = [nodes[i].lon for i in d_o_path]
        y_path_d_o = [nodes[i].lat for i in d_o_path]
        print (f"Destination => Origin Takes an identical path")
    plot_flight_paths(x_path_o_d, y_path_o_d, x_path_d_o, y_path_d_o, str1, str2, str3)
    plt.savefig("flight/euclidean-" + str(euclidean) + "-wind-" + str(wind_magnitude) + "-" + str(ngrid_lon) + "-" + str(ngrid_lat) + ".jpeg")

def get_geodesic_path_coords_in_rectilinear_lon_lat(lon1, lon2, lat1, lat2, npts=10):
    geodesic_path_coords_in_rectilinear_lon_lat = geodesic_projection.npts(lon1=lon1, lon2=lon2, lat1=lat1, lat2=lat2, npts=npts)
    geodesic_lons = [v[0] for v in geodesic_path_coords_in_rectilinear_lon_lat]
    geodesic_lats = [v[1] for v in geodesic_path_coords_in_rectilinear_lon_lat]
    return geodesic_lons, geodesic_lats

def plot_flight_paths(x_path_o_d, y_path_o_d, x_path_d_o, y_path_d_o, str1, str2, str3):
    nframes = 10
    lons_path_o_d, lats_path_o_d = [], []
    lons_path_d_o, lats_path_d_o = [], []
    if euclidean:
        for i in range(len(x_path_o_d)-1):
            lons_path_o_d = lons_path_o_d + np.linspace(x_path_o_d[i], x_path_o_d[i+1], num=nframes).tolist()
            lats_path_o_d = lats_path_o_d + np.linspace(y_path_o_d[i], y_path_o_d[i+1], num=nframes).tolist()
        for i in range(len(x_path_d_o)-1):
            lons_path_d_o = lons_path_d_o + np.linspace(x_path_d_o[i], x_path_d_o[i+1], num=nframes).tolist()
            lats_path_d_o = lats_path_d_o + np.linspace(y_path_d_o[i], y_path_d_o[i+1], num=nframes).tolist()
    else:
        for i in range(len(x_path_o_d)-1):
            geodesic_lons, geodesic_lats = get_geodesic_path_coords_in_rectilinear_lon_lat(x_path_o_d[i], x_path_o_d[i+1], y_path_o_d[i], y_path_o_d[i+1], npts=nframes)
            lons_path_o_d = lons_path_o_d + [x_path_o_d[i]] + geodesic_lons + [x_path_o_d[i+1]]
            lats_path_o_d = lats_path_o_d + [y_path_o_d[i]] + geodesic_lats + [y_path_o_d[i+1]]
        for i in range(len(x_path_d_o)-1):
            geodesic_lons, geodesic_lats = get_geodesic_path_coords_in_rectilinear_lon_lat(x_path_d_o[i], x_path_d_o[i+1], y_path_d_o[i], y_path_d_o[i+1], npts=nframes)
            lons_path_d_o = lons_path_d_o + [x_path_d_o[i]] + geodesic_lons + [x_path_d_o[i+1]]
            lats_path_d_o = lats_path_d_o + [y_path_d_o[i]] + geodesic_lats + [y_path_d_o[i+1]]

    plt.plot(lons_path_o_d, lats_path_o_d, color='g')
    plt.plot(lons_path_d_o, lats_path_d_o, color='b')
    # animate(lons_path_o_d, lats_path_o_d, lons_path_d_o, lats_path_d_o, str1, str2, str3)

def initialize_plot():
    ax = plt.axes()
    usa_shp_df.boundary.plot(ax=ax, color='red', linewidth=1)

    ax.text(lon_origin-1, lat_origin-1, "Origin", color='g', fontsize='small',fontweight='bold')
    ax.text(lon_destination-1, lat_destination-1, "Dest", color='b', fontsize='small',fontweight='bold')
    ax.set_xlim(usa_bbox[0], usa_bbox[2])
    ax.set_ylim(usa_bbox[1], usa_bbox[3])

    if euclidean:
        plt.plot([lon_origin, lon_destination], [lat_origin, lat_destination] , color='k', linestyle='--')
    else:
        geodesic_lons, geodesic_lats = get_geodesic_path_coords_in_rectilinear_lon_lat(lon_origin, lon_destination, lat_origin, lat_destination, npts=100)
        geodesic_lons = [lon_origin] + geodesic_lons + [lon_destination]
        geodesic_lats = [lat_origin] + geodesic_lats + [lat_destination]
        plt.plot(geodesic_lons, geodesic_lats, color='k', linestyle='--')
    return ax

if __name__ == "__main__":
    start_time_0 = time.perf_counter_ns()
    geodesic_projection = pyproj.Geod(ellps='WGS84')
    lat_lon_info = get_csv_lat_lon()
    euclidean, ngrid_lat, ngrid_lon, origin, destination, wind_magnitude = process_args()
    lon_origin, lat_origin, lon_destination, lat_destination = origin[0], origin[1], destination[0], destination[1]
    usa_shp_df, usa_bbox, lon_axis, lat_axis, origin_i, origin_j, destination_i, destination_j = get_grid_lat_lon()
    origin_node_number = get_node_number(origin_i, origin_j)
    dest_node_number = get_node_number(destination_i, destination_j)
    print (f"\nOrigin in Grid: {(origin_i, origin_j)} => {origin_node_number}")
    print (f"\nDestination in Grid: {(destination_i, destination_j)} => {dest_node_number}")

    fig = plt.figure(figsize=(12,8),dpi=720)
    fig.set_layout_engine('tight')
    ax = initialize_plot()

    nodes = define_nodes_and_wind()
    plot_wind(ax)

    start_time = time.perf_counter_ns()
    weights = build_csr_matrix ()
    time_taken = round((time.perf_counter_ns() - start_time_0) * 1.0e-9, 4)
    print( f"\nTime taken to build weights:{time_taken} seconds\n")
    
    # just these few lines that the app will run in real time. store csr matrix in memory, update every 6 hours
    dist_matrix, predecessors = shortest_path(csgraph=weights, directed=True, indices=[origin_node_number, dest_node_number], return_predecessors=True)

    #print (dist_matrix)

    process_results()
    time_taken = round((time.perf_counter_ns() - start_time_0) * 1.0e-9, 4)
    print( f"\nTotal Time taken:{time_taken} seconds\n")


