# from shapely import Point
# from geopy import distance

class Vertex:

    known = False
    distance = 0 # total distance from origin point
    prev = None
    adjacentEdges = []
    overflightFee = 0

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def equals(self, otherVertex):
        if(otherVertex.x == self.x and otherVertex.y == self.y):
            return True
        else:
            return False
            
    def addEdge(self, edge):
        self.adjacentEdges.append(edge)
        
    def setPrev(self, prev):
        self.prev = prev
        
    def setDistance(self, distance):
        self.distance = distance
        
    def setOverflightFee(self, overflightFee):
        self.overflightFee = overflightFee
    
    def setKnown(self):
        self.known = True
        
    def __str__(self):
        return "(" + str(self.x) + ", " + str(self.y) + ") " + str(self.overflightFee)


class Edge:
    
    weight = 0
    
    def __init__(self, source, target):
        self.source = source
        self.target = target
        # self.distance = float(((source.x - target.x) ** 2 + (source.y - target.y) ** 2) ** (0.5))
        # self.distance = distance.distance(source, target).miles
        # self.weight = self.distance * (source.overflightFee + 1)
        
    def addWeight(self, weight):
        self.weight = weight
    
    def equals(self, otherEdge):
        if((otherEdge.source == self.source and otherEdge.target == self.target) or (otherEdge.source == self.target and otherEdge.target == self.source)):
            return True
        else:
            return False

    def __str__(self):
        return self.source.__str__() + " - " + self.target.__str__()
