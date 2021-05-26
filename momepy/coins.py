"""
This Python script generates strokes from the line type ESRI shapefiles, mainly roads.

Author: Pratyush Tripathy
Date: 29 February 2020
Version: 0.2

The script is a supplementary material to the full length article:

Title:
An open-source tool to extract natural continuity and hierarchy of urban street networks

Journal:
Environment and Planning B: Urban Analytics and City Science

Authors:
Pratyush Tripathy, Pooja Rao, Krishnachandran Balakrishnan, Teja Malladi

Citation:
Tripathy, P., Rao, P., Balakrishnan, K., & Malladi, T. (2020). An open-source tool to extract natural continuity and hierarchy of urban street networks. Environment and Planning B: Urban Analytics and City Science. https://doi.org/10.1177%2F2399808320967680

GitHub repository:
https://github.com/PratyushTripathy/NetworkContinuity

Adapted for momepy by: Andres Morfin
"""

import os, sys, math, time, multiprocessing
from functools import partial
from jinja2.nodes import Mul
import numpy as np
import shapefile as shp
import osmnx as ox
from shapely.geometry import Point, LineString, MultiLineString
from shapely import ops
import geopandas as gpd
import pandas as pd


def main():

    ## get graph from osmnx
    G = ox.load_graphml('processed_graph.graphml')

    ## convert graph to geodataframe
    edges_gdf = ox.graph_to_gdfs(ox.get_undirected(G), nodes=False, edges=True, node_geometry=False, fill_edge_geometry=True) 

    # If you wish to processone file only, change the name in the line below
    t1 = time.time()

    #Read Shapefile
    myStreet = line(edges_gdf)
    #Split lines
    _ = myStreet.splitLines()
    #Create unique ID
    _ = myStreet.uniqueID()
    #Compute connectivity table
    myStreet.getLinks()
    #Find best link at every point for both lines
    myStreet.bestLink()
    #Cross check best links
    #Enter the angle threshold for connectivity here
    myStreet.crossCheckLinks(angleThreshold=0)
    #Merge finalised links
    myStreet.mergeLines()
    
    #Export lines
    #If you wish to export the premerge file,
    #otherwise, feel free to comment the line below (None exports default name)
    edges_premerge = myStreet.create_gdf_premerge()

    #Exporting the strokes (None exports default name)
    edges_strokes = myStreet.create_gdf_strokes()
    
    # export to geopackage for GIS
    edges_premerge = stringify_nonnumeric_cols(edges_premerge)
    edges_premerge.to_file("output.gpkg", driver="GPKG")
    edges_strokes = stringify_nonnumeric_cols(edges_strokes)
    edges_strokes.to_file("output_strokes.gpkg", driver="GPKG")

    
    t2 = time.time()
    
    minutes = math.floor((t2-t1) / 60)
    seconds = (t2 - t1) % 60
    print("Processing complete in %d minutes %.2f seconds." % (minutes, seconds))

########################################################################################
########################################################################################
#################   PLEASE DO NOT EDIT THE BELOW PART OF THE CODE    ###################
#####   SCROLL DOWN TO THE LOWER EXTREME OF THE SCRIPT TO CHANGE INPUT FILE NAME   #####
########################################################################################
########################################################################################

#Set recurrsion depth limit to avoid error at a later stage
sys.setrecursionlimit(10000)

"""
The imported shapefile lines comes as tuple, whereas
the export requires list, this finction converts tuple
inside lines to list
"""
def tupleToList(line):
    for a in range(0,len(line)):
        line[a] = list(line[a])
    return(line)

def listToTuple(line):
    for a in range(0, len(line)):
        line[a] = tuple(line[a])
    return(tuple(line))
"""
The below function takes a line as an input and splits
it at every point.
"""
def listToPairs(inList):
    outList = []
    index = 0
    for index in range(0,len(inList)-1):
        tempList = [list(inList[index]), list(inList[index+1])]
        outList.append(tempList)
    return(outList)

"""
The function below calculates the angle between two points in space.
"""

def computeAngle(point1, point2):
    height = abs(point2[1] - point1[1])
    base = abs(point2[0] - point1[0])
    angle = round(math.degrees(math.atan(height/base)), 3)
    return(angle)

"""
This function calculates the orientation of a line segment.
Point1 is the lower one on the y-axes and vice-cersa for
Point2.
"""
def computeOrientation(line):
    point1 = line[1]
    point2 = line[0]
    """
    If the latutide of a point is less and the longitude is more, or
    If the latitude of a point is more and the longitude is less, then
    the point is oriented leftward and wil have negative orientation.
    """
    if ((point2[0] > point1[0]) and (point2[1] < point1[1])) or ((point2[0] < point1[0]) and (point2[1] > point1[1])):
        return(-computeAngle(point1, point2))
    #If the latitudes are same, the line is horizontal
    elif point2[1] == point1[1]:
        return(0)
    #If the longitudes are same, the line is vertical
    elif point2[0] == point1[0]:
        return(90)
    else:
        return(computeAngle(point1, point2))

"""
This below function calculates the acute joining angle between
two given set of points.
"""
def pointsSetAngle(line1, line2):
    l1orien = computeOrientation(line1)
    l2orien = computeOrientation(line2)
    if ((l1orien>0) and (l2orien<0)) or ((l1orien<0) and (l2orien>0)):
        return(abs(l1orien)+abs(l2orien))
    elif ((l1orien>0) and (l2orien>0)) or ((l1orien<0) and (l2orien<0)):
        theta1 = abs(l1orien) + 180 - abs(l2orien)
        theta2 = abs(l2orien) + 180 - abs(l1orien)
        if theta1 < theta2:
            return(theta1)
        else:
            return(theta2)
    elif (l1orien==0) or (l2orien==0):
        if l1orien<0:
            return(180-abs(l1orien))
        elif l2orien<0:
            return(180-abs(l2orien))
        else:
            return(180 - (abs(computeOrientation(line1)) + abs(computeOrientation(line2))))
    elif (l1orien==l2orien):
        return(180)
        
"""
The below function calculates the joining angle between
two line segments.
"""
def angleBetweenTwoLines(line1, line2):
    l1p1, l1p2 = line1
    l2p1, l2p2 = line2
    l1orien = computeOrientation(line1)
    l2orien = computeOrientation(line2)
    """
    If both lines have same orientation, return 180
    If one of the lines is zero, exception for that
    If both the lines are on same side of the horizontal plane, calculate 180-(sumOfOrientation)
    If both the lines are on same side of the vertical plane, calculate pointSetAngle
    """
    if (l1orien==l2orien): 
        angle = 180
    elif (l1orien==0) or (l2orien==0): 
        angle = pointsSetAngle(line1, line2)
        
    elif l1p1 == l2p1:
        if ((l1p1[1] > l1p2[1]) and (l1p1[1] > l2p2[1])) or ((l1p1[1] < l1p2[1]) and (l1p1[1] < l2p2[1])):
            angle = 180 - (abs(l1orien) + abs(l2orien))
        else:
            angle = pointsSetAngle([l1p1, l1p2], [l2p1,l2p2])
    elif l1p1 == l2p2:
        if ((l1p1[1] > l2p1[1]) and (l1p1[1] > l1p2[1])) or ((l1p1[1] < l2p1[1]) and (l1p1[1] < l1p2[1])):
            angle = 180 - (abs(l1orien) + abs(l2orien))
        else:
            angle = pointsSetAngle([l1p1, l1p2], [l2p2,l2p1])
    elif l1p2 == l2p1:
        if ((l1p2[1] > l1p1[1]) and (l1p2[1] > l2p2[1])) or ((l1p2[1] < l1p1[1]) and (l1p2[1] < l2p2[1])):
            angle = 180 - (abs(l1orien) + abs(l2orien))
        else:
            angle = pointsSetAngle([l1p2, l1p1], [l2p1,l2p2])
    elif l1p2 == l2p2:
        if ((l1p2[1] > l1p1[1]) and (l1p2[1] > l2p1[1])) or ((l1p2[1] < l1p1[1]) and (l1p2[1] < l2p1[1])):
            angle = 180 - (abs(l1orien) + abs(l2orien))
        else:
            angle = pointsSetAngle([l1p2, l1p1], [l2p2,l2p1])
    return(angle)

def getLinksMultiprocessing(n, total, tempArray):
    # Printing the progress bar
    if n%1000==0:
        """
        Dividing by two to have 50 progress steps
        Subtracting from 50, and not hundred to have less progress steps
        """
        currentProgress = math.floor(100*n/total/2)
        remainingProgress = 50 - currentProgress            
        print('>'*currentProgress + '-' * remainingProgress + ' [%d/%d] '%(n,total) + '%d%%'%(currentProgress*2), end='\r')

    # Create mask for adjacent edges as endpoint 1
    m1 = tempArray[:,1]==tempArray[n,1]
    m2 = tempArray[:,2]==tempArray[n,1]
    mask1 = m1 + m2

    # Create mask for adjacent edges as endpoint 2
    m1 = tempArray[:,1]==tempArray[n,2]
    m2 = tempArray[:,2]==tempArray[n,2]
    mask2 = m1 + m2

    # Use the tempArray to extract only the uniqueIDs of the adjacent edges at both ends
    mask1 = tempArray[:,0][~(mask1==0)]
    mask2 = tempArray[:,0][~(mask2==0)]

    # Links (excluding the segment itself) at both the ends are converted to list and added to the 'unique' attribute
    return(n, list(mask1[mask1 != n]), list(mask2[mask2 != n]))

def mergeLinesMultiprocessing(n, total, uniqueDict):
    # Printing the progress bar
    if n%1000==0:
        """
        Dividing by two to have 50 progress steps
        Subtracting from 50, and not hundred to have less progress steps
        """
        currentProgress = math.floor(100*n/total/2)
        remainingProgress = 50 - currentProgress            
        print('>'*currentProgress + '-' * remainingProgress + ' [%d/%d] '%(n,total) + '%d%%'%(currentProgress*2), end='\r')
        
    outlist = set()
    currentEdge1 = n

    outlist.add(currentEdge1)

    while True:
        if type(uniqueDict[currentEdge1][6]) == type(1) and \
           uniqueDict[currentEdge1][6] not in outlist:
            currentEdge1 = uniqueDict[currentEdge1][6]
            outlist.add(currentEdge1)
        elif type(uniqueDict[currentEdge1][7]) == type(1) and \
           uniqueDict[currentEdge1][7] not in outlist:
            currentEdge1 = uniqueDict[currentEdge1][7]
            outlist.add(currentEdge1)
        else:
            break
    currentEdge1 = n
    while True:
        if type(uniqueDict[currentEdge1][7]) == type(1) and \
           uniqueDict[currentEdge1][7] not in outlist:
            currentEdge1 = uniqueDict[currentEdge1][7]
            outlist.add(currentEdge1)
        elif type(uniqueDict[currentEdge1][6]) == type(1) and \
           uniqueDict[currentEdge1][6] not in outlist:
            currentEdge1 = uniqueDict[currentEdge1][6]
            outlist.add(currentEdge1)
        else:
            break

    outlist = list(outlist)
    outlist.sort()
    return(outlist)
        

class line():
    def __init__(self, edges_gdf):
        self.gdfProjection = edges_gdf.crs
        self.lines = [list(value[1].coords) for value in edges_gdf['geometry'].iteritems()]

    def splitLines(self):
        outLine = []
        tempLine = []
        self.tempArray = []
        n = 0
        #Iterate through the lines and split the edges
        for line in self.lines:
            for part in listToPairs(line):
                outLine.append([part, computeOrientation(part), list(), list(), list(), list(), list(), list()])
                # Merge the coordinates as string, this will help in finding adjacent edges in the function below
                self.tempArray.append([n, f'{part[0][0]}_{part[0][1]}', f'{part[1][0]}_{part[1][1]}'])
                n += 1

        self.split = outLine

    def uniqueID(self):
    #Loop through split lines, assign unique ID and
    #store inside a list along with the connectivity dictionary
        self.unique = dict(enumerate(self.split))

    def getLinks(self):
        global result
        print("Finding adjacent segments...")

        self.tempArray = np.array(self.tempArray, dtype=object)
        
        iterations = [n for n in range(0,len(self.unique))]
        
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        constantParameterFunction = partial(getLinksMultiprocessing, total=len(self.unique), tempArray=self.tempArray)
        result = pool.map(constantParameterFunction, iterations)
        pool.close()
        pool.join()
        iterations = None

        for a in result:
            n = a[0]
            self.unique[n][2] = a[1]
            self.unique[n][3] = a[2]
            
        print('>'*50 + ' [%d/%d] '%(len(self.unique),len(self.unique)) + '100%' + '\n', end='\r')
            
    def bestLink(self):
        self.anglePairs = dict()
        for edge in range(0,len(self.unique)):
            p1AngleSet = []
            p2AngleSet = []

            """
            Instead of computing the angle between the two segments twice, the method calculates
            it once and stores in the dictionary for both the keys. So that it does not calculate
            the second time because the key is already present in the dictionary.
            """
            for link1 in self.unique[edge][2]:
                self.anglePairs["%d_%d" % (edge, link1)] = angleBetweenTwoLines(self.unique[edge][0], self.unique[link1][0])
                p1AngleSet.append(self.anglePairs["%d_%d" % (edge, link1)])
                
            for link2 in self.unique[edge][3]:
                self.anglePairs["%d_%d" % (edge, link2)] = angleBetweenTwoLines(self.unique[edge][0], self.unique[link2][0])
                p2AngleSet.append(self.anglePairs["%d_%d" % (edge, link2)])

            """
            Among the adjacent segments deflection angle values, check for the maximum value
            at both the ends. The segment with the maximum angle is stored in the attributes
            to be cross-checked later for before finalising the segments at both the ends.
            """
            if len(p1AngleSet)!=0:
                val1, idx1 = max((val, idx) for (idx, val) in enumerate(p1AngleSet))
                self.unique[edge][4] = self.unique[edge][2][idx1], val1
            else:
                self.unique[edge][4] = 'DeadEnd'
                
            if len(p2AngleSet)!=0:
                val2, idx2 = max((val, idx) for (idx, val) in enumerate(p2AngleSet))
                self.unique[edge][5] = self.unique[edge][3][idx2], val2
            else:
                self.unique[edge][5] = 'DeadEnd'

    def crossCheckLinks(self, angleThreshold=0):
        global edge, bestP1, bestP2
        print("Cross-checking and finalising the links...")
        for edge in range(0,len(self.unique)):
            # Printing the progress bar
            if edge%1000==0:
                """
                Dividing by two to have 50 progress steps
                Subtracting from 50, and not hundred to have less progress steps
                """
                currentProgress = math.floor(100*edge/len(self.unique)/2)
                remainingProgress = 50 - currentProgress            
                print('>'*currentProgress + '-' * remainingProgress + ' [%d/%d] '%(edge,len(self.unique)) + '%d%%'%(currentProgress*2), end='\r')

            bestP1 = self.unique[edge][4][0]
            bestP2 = self.unique[edge][5][0]
            
            if type(bestP1) == type(1) and \
               edge in [self.unique[bestP1][4][0], self.unique[bestP1][5][0]] and \
               self.anglePairs["%d_%d" % (edge, bestP1)] > angleThreshold:
                self.unique[edge][6] = bestP1
            else:
                self.unique[edge][6] = 'LineBreak'
                
            if type(bestP2) == type(1) and \
               edge in [self.unique[bestP2][4][0], self.unique[bestP2][5][0]] and \
               self.anglePairs["%d_%d" % (edge, bestP2)] > angleThreshold:
                self.unique[edge][7] = bestP2
            else:
                self.unique[edge][7] = 'LineBreak'
                
        print('>'*50 + ' [%d/%d] '%(edge+1,len(self.unique)) + '100%' + '\n', end='\r')

    def addLine(self, edge, parent=None, child='Undefined'):
        if child=='Undefined':
            self.mainEdge = len(self.merged)
        if not edge in self.assignedList:
            if parent==None:
                currentid = len(self.merged)
                self.merged[currentid] = set()
            else:
                currentid = self.mainEdge
            self.merged[currentid].add(listToTuple(self.unique[edge][0]))
            self.assignedList.append(edge)
            link1 = self.unique[edge][6]
            link2 = self.unique[edge][7]
            if type(1) == type(link1):
                self.addLine(link1, parent=edge, child=self.mainEdge)
            if type(1) == type(link2):
                self.addLine(link2, parent=edge, child=self.mainEdge)

    def mergeLines(self):
        print('Merging Lines...')
        self.mergingList = list()
        self.merged = list()

        iterations = [n for n in range(0,len(self.unique))]
        
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        constantParameterFunction = partial(mergeLinesMultiprocessing, total=len(self.unique), uniqueDict=self.unique)
        result = pool.map(constantParameterFunction, iterations)
        pool.close()
        pool.join()
        iterations = None

        for tempList in result:
            if not tempList in self.mergingList:
                self.mergingList.append(tempList)
                self.merged.append({listToTuple(self.unique[key][0]) for key in tempList})

        self.merged = dict(enumerate(self.merged))
        print('>'*50 + ' [%d/%d] '%(len(self.unique),len(self.unique)) + '100%' + '\n', end='\r')
        
#Export geodataframes
    def create_gdf_premerge(self):
        # create empty list to fill out
        myList = []
        
        for parts in range(0,len(self.unique)):
            # get all segment points and make line
            lineList = tupleToList(self.unique[parts][0])
            geom_line = LineString([Point(lineList[0]), Point(lineList[1])])
            
            # get other values for premerged
            UniqueID = parts 
            Orientation = self.unique[parts][1]
            linksP1 = self.unique[parts][2]
            linksP2 = self.unique[parts][3]
            bestP1 = self.unique[parts][4]
            bestP2 = self.unique[parts][5]
            P1Final = self.unique[parts][6]
            P2Final = self.unique[parts][7]
            
            # append list
            myList.append([UniqueID, Orientation, linksP1, linksP2, bestP1, bestP2, P1Final, P2Final, geom_line])

        edge_gdf = gpd.GeoDataFrame(myList, columns=['UniqueID','Orientation','linksP1','linksP2','bestP1','bestP2','P1Final','P2Final', 'geometry'], crs=self.gdfProjection)
        
        return edge_gdf

    def create_gdf_strokes(self):

        # create empty list to fill 
        myList = []
        
        # loop through merged geometry
        for a in self.merged:

            # get all segment points and make line strings
            linelist = tupleToList(list(self.merged[a]))
            list_lines_segments = []

            for b in linelist:
                list_lines_segments.append(LineString(b))      
            
            # merge seperate segments
            geom_multi_line = ops.linemerge(MultiLineString(list_lines_segments))

            # get other values for gdf
            ID_value = a 
            nSegments = len(self.merged[a])

            # append list
            myList.append([ID_value, nSegments, geom_multi_line])

        edge_gdf = gpd.GeoDataFrame(myList, columns=['ID_value', 'nSegments', 'geometry'], crs=self.gdfProjection)
        
        return edge_gdf



#######################################################
#######################################################
################   ALGORITHM ENDS HERE    #############
#######################################################
#######################################################
#######################################################

def stringify_nonnumeric_cols(gdf):
    """
    Code taken from osmnx
    
    Make every non-numeric GeoDataFrame column (besides geometry) a string.

    This allows proper serializing via Fiona of GeoDataFrames with mixed types
    such as strings and ints in the same column.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        gdf to stringify non-numeric columns of

    Returns
    -------
    gdf : geopandas.GeoDataFrame
        gdf with non-numeric columns stringified
    """
    # stringify every non-numeric column other than geometry column
    for col in (c for c in gdf.columns if not c == "geometry"):
        if not pd.api.types.is_numeric_dtype(gdf[col]):
            gdf[col] = gdf[col].fillna("").astype(str)

    return gdf


if __name__ == '__main__':
    main()