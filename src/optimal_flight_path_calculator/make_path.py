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
    
def get_grid_lat_lon():
    usa_shape_file: str = "./usa-shape/usa-states-census-2014.shp"
    usa_shp_df = geopandas.read_file(usa_shape_file)
    usa_bbox = usa_shp_df.total_bounds # [-124.725839   24.498131  -66.949895   49.384358]
    lon_axis = np.linspace(usa_bbox[0], usa_bbox[2], num=ngrid_lon)
    lat_axis = np.linspace(usa_bbox[1], usa_bbox[3], num=ngrid_lat)
    
    # rounds origin coordinates to closest node on the grid. lon_axis is a list
    (origin_i, origin_j) = (np.argmin(np.abs(lon_axis - lon_origin)) , np.argmin(np.abs(lat_axis - lat_origin)))
    lon_axis[origin_i] = lon_origin
    lat_axis[origin_j] = lat_origin
    
    (destination_i, destination_j) = (np.argmin(np.abs(lon_axis - lon_destination)) , np.argmin(np.abs(lat_axis - lat_destination)))
    lon_axis[destination_i] = lon_destination
    lat_axis[destination_j] = lat_destination
    
    return usa_shp_df, usa_bbox, lon_axis, lat_axis, origin_i, origin_j, destination_i, destination_j

def get_node_number(lon_i, lat_j):
    return lon_i + lat_j * len(lon_axis)

#Origin:(-117.1611, 32.7157) Destination:(-74.006, 40.6712)

def process_args():
    parser = argparse.ArgumentParser(
        description="Find the optimal flight path between an origin/destination"
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

    args = parser.parse_args()
    origin = airports_lat_lon_info[args.origin[0]]
    destination = airports_lat_lon_info[args.destination[0]]
    print (args.origin[0], args.destination[0])
    return origin, destination

def build_csr_matrix():
    penalties = np.load("penalties_array.npy")
    print(penalties[origin_node_number, dest_node_number])
    weights = csr_matrix(penalties)
    return weights


def get_geodesic_distance( lon1, lat1, lon2, lat2 ):
    geodesic_distance = geodesic_projection.line_length([lon1, lon2], [lat1, lat2]) # meters
    return geodesic_distance * 1.0e-3 / 1.60934 # miles
    
def process_results():

    # this is cost of going DIRECTLY from origin to destination, not accounting for the stops made along the way
    o_d_cost = dist_matrix[0, dest_node_number]
    print(o_d_cost)
    d_o_cost = dist_matrix[1, origin_node_number]

    o_d_path = [dest_node_number]
    prev_one = predecessors[0,dest_node_number]
    while prev_one != -9999:
        o_d_path.append(prev_one)
        prev_one = predecessors[0,prev_one]
    o_d_path.reverse()
    
    path_node_string = '=>'.join([str(node_number) for node_number in o_d_path])
    str1 = "O=>D: " + path_node_string + " : " + str(round(o_d_cost, 3))
    ax.text(-124, 30, str1, color='g',fontsize='small')

    x_path_o_d = [nodes[i].lon for i in o_d_path]
    y_path_o_d = [nodes[i].lat for i in o_d_path]
    plt.scatter(x_path_o_d, y_path_o_d, color='g', marker='o')

    d_o_path = [origin_node_number]
    prev_one = predecessors[1,origin_node_number]
    str2 = ''
    while prev_one != -9999:
        d_o_path.append(prev_one)
        prev_one = predecessors[1,prev_one]

    d_o_path.reverse()
    path_node_string = '=>'.join([str(node_number) for node_number in d_o_path])
    str2 = "D=>O: " + path_node_string + " : " + str(round(d_o_cost, 3))
    ax.text(-124, 29, str2, color='b',fontsize='small')
    x_path_d_o = [nodes[i].lon for i in d_o_path]
    y_path_d_o = [nodes[i].lat for i in d_o_path]
    plt.scatter(x_path_d_o, y_path_d_o, color='b', marker='o')
    
    str3 = "O=>D direct (if there was no wind): " + str(origin_node_number) + "=>" + str(dest_node_number) + " : " + str(round(get_geodesic_distance(lon_origin, lat_origin, lon_destination, lat_destination) / 500, 3))
    ax.text(-124, 28, str3, color='gray',fontsize='small')
    
    plot_flight_paths(x_path_o_d, y_path_o_d, x_path_d_o, y_path_d_o)
    plt.savefig("flight/wind-" + str(100.0) + "-" + str(ngrid_lon) + "-" + str(ngrid_lat) + ".jpeg")

def get_geodesic_path_coords_in_rectilinear_lon_lat(lon1, lon2, lat1, lat2, npts=10):
    geodesic_path_coords_in_rectilinear_lon_lat = geodesic_projection.npts(lon1=lon1, lon2=lon2, lat1=lat1, lat2=lat2, npts=npts)
    geodesic_lons = [v[0] for v in geodesic_path_coords_in_rectilinear_lon_lat]
    geodesic_lats = [v[1] for v in geodesic_path_coords_in_rectilinear_lon_lat]
    return geodesic_lons, geodesic_lats

def plot_flight_paths(x_path_o_d, y_path_o_d, x_path_d_o, y_path_d_o):
    nframes = 10
    lons_path_o_d, lats_path_o_d = [], []
    lons_path_d_o, lats_path_d_o = [], []
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

def initialize_plot():
    ax = plt.axes()
    usa_shp_df.boundary.plot(ax=ax, color='red', linewidth=1)

    ax.text(lon_origin-1, lat_origin-1, "Origin", color='g', fontsize='small',fontweight='bold')
    ax.text(lon_destination-1, lat_destination-1, "Dest", color='b', fontsize='small',fontweight='bold')
    ax.set_xlim(usa_bbox[0], usa_bbox[2])
    ax.set_ylim(usa_bbox[1], usa_bbox[3])
    
    geodesic_lons, geodesic_lats = get_geodesic_path_coords_in_rectilinear_lon_lat(lon_origin, lon_destination, lat_origin, lat_destination, npts=100)
    geodesic_lons = [lon_origin] + geodesic_lons + [lon_destination]
    geodesic_lats = [lat_origin] + geodesic_lats + [lat_destination]
    plt.plot(geodesic_lons, geodesic_lats, color='k', linestyle='--')
    return ax

if __name__ == "__main__":
    start_time_0 = time.perf_counter_ns()
    geodesic_projection = pyproj.Geod(ellps='WGS84')
    airports_lat_lon_info = get_csv_lat_lon()
    origin, destination = process_args()
    # make sure to match with make_penalties.py
    ngrid_lat, ngrid_lon = 20, 20
    lon_origin, lat_origin, lon_destination, lat_destination = origin[1], origin[0], destination[1], destination[0]
    usa_shp_df, usa_bbox, lon_axis, lat_axis, origin_i, origin_j, destination_i, destination_j = get_grid_lat_lon()
    origin_node_number = get_node_number(origin_i, origin_j)
    dest_node_number = get_node_number(destination_i, destination_j)
    print(str(lon_origin) + ", " + str(lat_origin))
    print(str(lon_destination) + ", " + str(lat_destination))

    fig = plt.figure(figsize=(12,8),dpi=720)
    fig.set_layout_engine('tight')
    ax = initialize_plot()

    nodes = np.load("nodes.npy", allow_pickle=True).item()
    print(nodes[origin_node_number].lon)
    # print(nodes[99].wind_lon)
    # print(nodes[99].wind_lat)
    plot_wind(ax)

    start_time = time.perf_counter_ns()
    weights = build_csr_matrix()
    
    # just these few lines that the app will run in real time. store csr matrix in memory, update every 6 hours
    dist_matrix, predecessors = shortest_path(csgraph=weights, directed=True, indices=[origin_node_number, dest_node_number], return_predecessors=True)

    #print (dist_matrix)

    process_results()
    time_taken = round((time.perf_counter_ns() - start_time_0) * 1.0e-9, 4)
    print( f"\nTotal Time taken:{time_taken} seconds\n")



