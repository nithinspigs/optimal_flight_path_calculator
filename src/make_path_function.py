import argparse
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import matplotlib
matplotlib.use('agg')
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
import utilities

def plot_wind(ax):
    
    step = 1
    
    max_vx_or_vx = 0
    for j in range(0,len(lat_axis), step):
        for i in range(0,len(lon_axis), step):
            node = nodes[utilities.get_node_number(i,j)]
            if node.wind_lon > max_vx_or_vx:
                max_vx_or_vx = node.wind_lon
            if node.wind_lat > max_vx_or_vx:
                max_vx_or_vx = node.wind_lat
                
    for j in range(0,len(lat_axis), step):
        for i in range(0,len(lon_axis), step):
            node = nodes[utilities.get_node_number(i,j)]
            vx = node.wind_lon / max_vx_or_vx
            vy = node.wind_lat / max_vx_or_vx
            '''
            if (abs(vx) > 0.1):
                vx = vx / abs(vx)
            if (abs(vy) > 0.1):
                vy = vy / abs(vy)
            '''
            if (abs(vx) > 0) or (abs(vy) > 0):
                ax.arrow(node.lon, node.lat, vx, vy, head_length = 0.15, head_width = 0.15)
                #ax.arrow(node.lon, node.lat, node.wind_lon, node.wind_lat, head_length = 0.15, head_width = 0.15)
    
def get_orig_dest():
    # rounds origin coordinates to closest node on the grid. lon_axis is a list
    (origin_i, origin_j) = (np.argmin(np.abs(lon_axis - lon_origin)) , np.argmin(np.abs(lat_axis - lat_origin)))
    lon_axis[origin_i] = lon_origin
    lat_axis[origin_j] = lat_origin
    
    (destination_i, destination_j) = (np.argmin(np.abs(lon_axis - lon_destination)) , np.argmin(np.abs(lat_axis - lat_destination)))
    lon_axis[destination_i] = lon_destination
    lat_axis[destination_j] = lat_destination
    
    return origin_i, origin_j, destination_i, destination_j

def build_csr_matrix():
    penalties = np.load("penalties_array.npy")
    # print(penalties[origin_node_number, dest_node_number])
    weights = csr_matrix(penalties)
    return weights

def plot_flight_paths(x_path_o_d, y_path_o_d, x_path_d_o, y_path_d_o):
    nframes = 10
    lons_path_o_d, lats_path_o_d = [], []
    lons_path_d_o, lats_path_d_o = [], []
    for i in range(len(x_path_o_d)-1):
        geodesic_lons, geodesic_lats = utilities.get_geodesic_path_coords_in_rectilinear_lon_lat(x_path_o_d[i], x_path_o_d[i+1], y_path_o_d[i], y_path_o_d[i+1], npts=nframes)
        lons_path_o_d = lons_path_o_d + [x_path_o_d[i]] + geodesic_lons + [x_path_o_d[i+1]]
        lats_path_o_d = lats_path_o_d + [y_path_o_d[i]] + geodesic_lats + [y_path_o_d[i+1]]
    for i in range(len(x_path_d_o)-1):
        geodesic_lons, geodesic_lats = utilities.get_geodesic_path_coords_in_rectilinear_lon_lat(x_path_d_o[i], x_path_d_o[i+1], y_path_d_o[i], y_path_d_o[i+1], npts=nframes)
        lons_path_d_o = lons_path_d_o + [x_path_d_o[i]] + geodesic_lons + [x_path_d_o[i+1]]
        lats_path_d_o = lats_path_d_o + [y_path_d_o[i]] + geodesic_lats + [y_path_d_o[i+1]]
        
    geodesic_lons, geodesic_lats = utilities.get_geodesic_path_coords_in_rectilinear_lon_lat(x_path_o_d[0], x_path_o_d[-1], y_path_o_d[0], y_path_o_d[-1], npts=100)
    
    plt.plot(geodesic_lons, geodesic_lats, color='k', linestyle='--')
    plt.plot(lons_path_o_d, lats_path_o_d, color='g')
    plt.plot(lons_path_d_o, lats_path_d_o, color='b')

def initialize_plot():
    ax = plt.axes()
    usa_shp_df.boundary.plot(ax=ax, color='red', linewidth=1)

    ax.text(lon_origin-1, lat_origin-1, "Origin", color='g', fontsize='small',fontweight='bold')
    ax.text(lon_destination-1, lat_destination-1, "Dest", color='b', fontsize='small',fontweight='bold')
    ax.set_xlim(usa_bbox[0], usa_bbox[2])
    ax.set_ylim(usa_bbox[1], usa_bbox[3])
    
    return ax
    
def process_results(origin_name, dest_name):

    # declare global variables
    global airports_lat_lon_info, origin, destination, ngrid_lat, ngrid_lon, lon_origin, lat_origin, lon_destination, lat_destination, usa_shp_df, usa_bbox, lon_axis, lat_axis, origin_i, origin_j, destination_i, destination_j, origin_node_number, dest_node_number, ax, nodes, weights

    # taken from main
    airports_lat_lon_info = utilities.get_csv_lat_lon()
    try:
        origin = airports_lat_lon_info[origin_name]
        destination = airports_lat_lon_info[dest_name]
    except:
        return "error"
    
    ngrid_lat, ngrid_lon = utilities.ngrid_lat, utilities.ngrid_lon
    lon_origin, lat_origin, lon_destination, lat_destination = origin[1], origin[0], destination[1], destination[0]
    usa_shp_df, usa_bbox, lon_axis, lat_axis = utilities.usa_shp_df, utilities.usa_bbox, utilities.lon_axis, utilities.lat_axis
    origin_i, origin_j, destination_i, destination_j = get_orig_dest()
    origin_node_number = utilities.get_node_number(origin_i, origin_j)
    dest_node_number = utilities.get_node_number(destination_i, destination_j)
    
    fig = plt.figure(figsize=(12,8),dpi=720)
    fig.set_layout_engine('tight')
    ax = initialize_plot()
    
    nodes = nodes = np.load("nodes.npy", allow_pickle=True).item()
    plot_wind(ax)

    weights = build_csr_matrix()
    
    dist_matrix, predecessors = shortest_path(csgraph=weights, directed=True, indices=[origin_node_number, dest_node_number], return_predecessors=True)
    
    # this is cost of going DIRECTLY from origin to destination, not accounting for the stops made along the way
    o_d_time = dist_matrix[0, dest_node_number]
    # print("direct o_d_cost with wind: " + str(o_d_cost))
    d_o_time = dist_matrix[1, origin_node_number]
    # print("direct d_o_cost with wind: " + str(d_o_cost))

    o_d_path = [dest_node_number]
    prev_one = predecessors[0,dest_node_number]
    while prev_one != -9999:
        o_d_path.append(prev_one)
        prev_one = predecessors[0,prev_one]
    o_d_path.reverse()

    str1 = origin_name + " => " + dest_name + ": " + str(round(o_d_time, 3)) + " hours"
    ax.text(-124, 29, str1, color='g',fontsize='small')

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
    str2 = dest_name + " => " + origin_name + ": " + str(round(d_o_time, 3)) + " hours"
    ax.text(-124, 28, str2, color='b',fontsize='small')
    
    x_path_d_o = [nodes[i].lon for i in d_o_path]
    y_path_d_o = [nodes[i].lat for i in d_o_path]
    plt.scatter(x_path_d_o, y_path_d_o, color='b', marker='o')
    
    str3 = origin_name + " => " + dest_name + ": " + str(round(utilities.get_geodesic_distance(lon_origin, lat_origin, lon_destination, lat_destination) / 500, 3)) + " hours"
    ax.text(-124, 27, str3, color='gray',fontsize='small')
    
    plot_flight_paths(x_path_o_d, y_path_o_d, x_path_d_o, y_path_d_o)
    plt.savefig("static/images/" + str(origin_name) + "-" + str(dest_name) + ".jpeg")
    return None



