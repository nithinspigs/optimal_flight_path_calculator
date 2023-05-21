import shapefile
# module for finding geodesic area of a polygon on earth's surface given coordinates of longitude and latitude
from area import area
from shapely import Polygon
from shapely import Point
import matplotlib.pyplot as plt
import json
from geopy import distance
import pprint
import sys
import math

from dijkstra_module import Vertex
from dijkstra_module import Edge

sf = shapefile.Reader("../Airspace_Boundary/Airspace_Boundary")

vertices = {}
edges = []
increment = 5
#print(distance.distance((30, -110), (40, -70)).miles)

# make vertices dictionary
i = -130
while(i < -60):

    j = 20
    while(j < 75):
    
        print(str(j) + "N " + str(i) + "W")
        coordinate = Point(i, j)
        vertex = Vertex(i, j)
    
        overflight_fee = 0
        for k in range(0, len(sf.shapes())):
        
            fir_polygon = Polygon(sf.shape(k).points)
            rec = sf.record(k)
            
            if(fir_polygon.contains(coordinate)):
            
                if(rec['COUNTRY'] == 'United States'):
                    overflight_fee = 60
                    
                    if(rec['PACIFIC'] == 1):
                        overflight_fee = 30
                    
                    if(rec['ONSHORE'] == 0):
                        overflight_fee = 30
                        
                    if(rec['ONSHORE'] == 1 and i > -125):
                        overflight_fee = 60
                    
                elif(rec['COUNTRY'] == 'Canada'):
                    overflight_fee = 80
                    
                elif(rec['COUNTRY'] == 'Mexico'):
                    overflight_fee = 40
                
                break;
                    
        vertex.setOverflightFee(overflight_fee)
        vertices[(i, j)] = vertex
        
        j = j + increment
    
    i = i + increment
                
# make adjacentEdges lists + do dijkstra
v_origin = (-110, 30)
v_destination = (-70, 40)
unknownVertices = []
for coordinate in vertices.keys():
    
    i = coordinate[0]
    j = coordinate[1]
    vertex = vertices[coordinate]
    
    # all eight neighbors
    neighborsList = []
    n1 = (i - increment, j + increment)
    n2 = (i, j + increment)
    n3 = (i + increment, j + increment)
    n4 = (i - increment, j)
    n5 = (i + increment, j)
    n6 = (i - increment, j - increment)
    n7 = (i, j - increment)
    n8 = (i + increment, j - increment)
    neighborsList.append(n1)
    neighborsList.append(n2)
    neighborsList.append(n3)
    neighborsList.append(n4)
    neighborsList.append(n5)
    neighborsList.append(n6)
    neighborsList.append(n7)
    neighborsList.append(n8)
    
    # if neighbor exists, create an edge and add it to the adjacency list
    for neighbor in neighborsList:
        if(neighbor in vertices.keys()):
            edge = Edge(vertex, vertices[neighbor])
            edge.addWeight(distance.distance((coordinate[1], coordinate[0]), (neighbor[1], neighbor[0])).miles * (vertex.overflightFee + 1))
            vertex.addEdge(edge)
    
    # do dijkstra
    vertex.setDistance(sys.maxsize)
    unknownVertices.append(coordinate)
            
#print(unknownVertices)
#print(len(vertices[v_origin].adjacentEdges))

# do dijkstra
vertices[v_origin].setDistance(0)
while(len(unknownVertices) > 0):

    minDistance = sys.maxsize
    nextVertex = vertices[v_origin]
    nextVertexTuple = v_origin
    nextVertex.setDistance(0)
    
    for vTuple in unknownVertices:
        v = vertices[vTuple]
        if(v.distance < minDistance):
            minDistance = v.distance
            nextVertex = v
            nextVertexTuple = vTuple
            
    #print(nextVertexTuple)
    
    nextVertex.setKnown()
    unknownVertices.remove(nextVertexTuple)
    
    for adjacentEdge in nextVertex.adjacentEdges:
    
        #print(edge.__str__())
        if(adjacentEdge.source.equals(nextVertex)):
            
            targetVertex = adjacentEdge.target
            targetDistance = sys.maxsize
            if(targetVertex.known == False):
                targetDistance = adjacentEdge.weight
                
            if(nextVertex.distance + targetDistance < targetVertex.distance):
                targetVertex.setPrev(nextVertex)
                targetVertex.setDistance(nextVertex.distance + targetDistance)
                
# get min path
v_temp = vertices[v_destination]
path = []

while(v_temp.prev != None):
    
    '''
    for adjacentEdge in v_temp.adjacentEdges:
    
        if(adjacentEdge.source == v_temp.prev):
            path.append(adjacentEdge)
            break
    '''
    path.append(v_temp)
    
    v_temp = v_temp.prev
    
path.append(v_temp)

# plot min path
plt.plot(v_origin[0], v_origin[1], marker="o")
plt.plot(v_destination[0], v_destination[1], marker="o")
print(len(path))
'''
for edge in path:
    plt.axline((edge.source.x, edge.source.y), (edge.target.x, edge.target.y))
    print(edge.__str__())
'''
for i in range (0, len(path) - 1):
    source = path[i]
    target = path[i+1]
    plt.plot((source.x, target.x), (source.y, target.y), marker="o")
    print(source.__str__() + " - " + target.__str__())

plt.show()

'''
try:
    with open("coordinate_value_dict.txt", 'wt') as file:
        file.write("coordinate_value_dict = " + str(coordinate_value_dict))
        file.close()

except:
    print("Unable to save file")
'''

    
    
