# Optimal Flight Path Calculator

This is an application that plots the most optimal flight path between an origin and destination city in the continental United States such that travel time due to distance as well as wind is minimized.

Here is an example flight path where the grid is broken into 40 points along both the longitudinal and latitudinal axes (1600 "nodes" in total).

![SAN-JFK](./flight/ngrid_40-40_SAN-JFK_GOOD.jpeg)

## Summary

**Optimal Flight Path Calculator** is split into two parts: _make\_penalties\_nodes.py_ and _make\_path.py_. _make\_penalties\_nodes.py_ creates the _nodes_ and _penalties\_array_ arrays needed for the path to be created, and _make\_path.py_ then uses those arrays to plot the minimum path.

Each node contains information about its relative (i, j) position in the grid, the longitude and latitude at that point, the wind magnitude in the longitudinal direction, and the wind mangitude in the latitudinal direction.

_penalties\_array_ is a two dimensional array that stores the time it takes to travel from every node to every other node. Thus, if the grid is 40x40 with 1600 nodes, _penalties\_array_ has 2560000 entries. The travel time assumes a plane traveling at 500mph, and takes into account any headwind or tailwind that would alter the travel time.

Computation of the _penalties\_array_ is time intensive, so it can be calculated beforehand so the files can be stored. Then, _make\_path.py_ can be run by itself to display the minimum path between any pair of cities in us-airport-codes.csv.

The algorithm used to calculate the minimum path is a modified version of Dijkstra's algorithm where the neighbor nodes are not limited to those immediately adjacent.

The application uses wind data from the Aviation Weather Center to determine the wind speed and direction at every node in the grid. Wind data at 30000 feet is only collected at certain airports across the country, so in order to define it for every node,a weighted average is taken of all the airports within a 200 mile radius.

## Requirements

    python >= 3.10

##	Dependencies
    argparse
    sys
    os
	scipy
    matplotlib
    pyproj
    itertools
    numpy
    geopandas
    pandas
    time
    math
    typing
    multiprocessing

## How to run the code

    1. Clone this repo

    2. Set up the env:
        python -m venv venv
        source venv/bin/activate
        pip install -r requirements/test-requirements.txt

    3. Run:
        python src/optimal_flight_path_calculator/make_penalties_nodes.py
        python src/optimal_flight_path_calculator/make_path.py

You wil see  output like:

```json
{
  "arms": {
    "sentences": {
      "1": "germany sells arms to saudi arabia",
      "2": "arms bend at the elbow",
      "3": "wave your arms around"
    },
    "dot_product": [
      0.378,
      0.498,
      0.422
    ],
    "euclidean_distance": [
      1.115,
      1.002,
      1.075
    ],
    "manhattan_distance": [
      24.678,
      22.095,
      23.414
    ]
  }
}
```
