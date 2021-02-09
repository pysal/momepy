#-------------------------------------------------------------------------------
# Program Name: simple Sky View Factor (SVF) for building morphological
# Purpose: for urban morphological metrics of MOMEPY

# This Python 3 programs depends on: math, shapely, fiona and geopandas

# Version:     0.1 
#              Functionalities:
#              1. SVF
#              2. DEM to be integrated...

# Author:      Momepy
#
# Created:     04/02/2021
# Copyright:   (c) JonWang 2021
# Licence:     <your licence>
#-------------------------------------------------------------------------------


import math
import numpy as np
import fiona
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely import affinity
from shapely.geometry import *


# Create gridded sampling points
# Input: resolution/sampling interval and building map
def ptAll(res, bld, height_col = 'Floor', contain_roof=True):
    # Building map extent
    minx, miny, maxx, maxy = bld.geometry.total_bounds
    
    # Number of points
    num_x = int((maxx-minx)/res) + 1
    num_y = int((maxy-miny)/res) + 1
    
    pt_x = np.linspace(minx,maxx,num_x)
    pt_y = np.linspace(miny,maxy,num_y)
    
    xx, yy = np.meshgrid(pt_x, pt_y)
    
    # Zip list of points
    pt_all = np.vstack([xx.ravel(), yy.ravel()]).T
    pt_all = [Point(pt) for pt in pt_all]
    
    # Convert to geodataframe
    pt_all_gdf = gpd.GeoDataFrame(geometry=pt_all)
    
    # Intersect with building map
    pt_all_gdf = gpd.sjoin(pt_all_gdf, bld, how='left', op='within')
    
    if not contain_roof:
        for ind, row in pt_all_gdf.iterrows():
            if row[height_col] > 0:
                pt_all_gdf.drop(ind, inplace=True)
        
    return pt_all_gdf

# Distance
# Input: shapely points
def dist(p1, p2):
    d = p1.distance(p2)
    return d

# Radians to degree, only for checking the angle in degrees
def rad2deg(rad):
    deg = math.degrees(rad)
    return deg
    
# Get zenith of highest building at observer
def zenith(l1, l2):
    angle = math.atan(l1/l2)  # Radians
    return angle

# Shaded area on the dome: ***Archimedes' Hat-Box Theorem***
# Source: https://brilliant.org/wiki/surface-area-sphere/#archimedes-hat-box-theorem
# Inputs: dome radius, zenith and incremental theta (user defined theta in degrees)
def shaded(R, zen, theta):
    h = R*math.sin(zen)
    S = math.pi*R*h * theta/180
    return S

# Intersect point buffer with building map
# Inputs: point, building geodata frame, dome radius
def bldAtPt(pt, bld, R):
    buff = pt.buffer(R)
    #buff = buff.set_crs(bld.crs)
    bld_clip = gpd.clip(bld, buff)
    return bld_clip

# Draw lines from central point witin each buffer as cutting sections
def drawLines(pt, theta, R):
    #end_points = []
    lines = []
    
    line = LineString([pt, (pt.x + R, pt.y)])

    # Rotate 30 degrees CCW from origin at the center of bbox
    lines = [affinity.rotate(line, i, 'center') for i in range(0, 360, theta)]
    
    '''
    for i in range(0, 360, theta):
        delta_x = R*math.cos(math.radians(i))
        delta_y = R*math.sin(math.radians(i))
        end_points.append(Point(pt.x + delta_x,
                                pt.y + delta_y))
        
        lines.append(LineString([pt, (pt.x + delta_x,
                                      pt.y + delta_y)]))
    '''
    
    lines = gpd.GeoDataFrame(geometry=lines)
    return lines

# SVF at each point
def svf(pt, bld, theta, R, height_col):
    
    if not pt.isnull()[height_col]:
        pt_h = pt[height_col]
    else:
        pt_h = 0
        
    lines = drawLines(pt['geometry'], theta, R)  # Cutting lines of geopandas dataframe
    
    bld_clip = bldAtPt(pt['geometry'], bld, R)
    bld_clip = bld_clip.explode()
    # Convert to linestrings for point intersections
    bld_clip_lines = gpd.GeoDataFrame(bld_clip[height_col], geometry = 
                                          bld_clip.exterior, columns = [height_col])    
    
    s = 0
    # In each direction search high points and compute area
    # Loop over all cutting lines in all direction
    for _, l in lines.itertuples():
        columns_data = []
        geoms = []   
        # Loop over all shapes in a buffer
        for _, h, b in bld_clip_lines.itertuples():
            intersect = l.intersection(b)
            # In each direction, all points sampled from buildings
            if intersect:
                if type(intersect) == Point:
                    columns_data.append(zenith((h-pt_h)*3,dist(intersect, pt['geometry'])))
                    geoms.append(intersect)
                else:
                    for p in intersect:
                        columns_data.append(zenith((h-pt_h)*3,dist(intersect, pt['geometry'])))
                        geoms.append(p)
        all_intersection = gpd.GeoDataFrame(columns_data, geometry=geoms,
                                            columns=['zenith'])
        
        # Pick highest point's zenith and compute area in the direction
        zen_max = all_intersection['zenith'].max()
        if not np.isnan(zen_max):
            s_delta = shaded(R, zen_max, theta)
            s += s_delta
        
    # SVF
    view_fac = 1 - s/(2*math.pi*R**2)
    
    return view_fac
            
# Generate a SVF map
def mapSVF(resolution, bld, theta, R, height_col, contain_roof=True):
    # Generate mesh points
    pt_all = ptAll(resolution, bld, height_col = 'Floor', contain_roof=True)
    
    # Loop over all points in a geodataframe
    coord_all = []
    view_all = []
    for ind, pt in pt_all.iterrows():
        print(ind)
        view = svf(pt, bld, theta, R, height_col)
        view_all.append(view)
        coord_all.append(pt['geometry'])
        
    svf_map = gpd.GeoDataFrame(view_all, geometry=coord_all, columns=['svf'])
    return svf_map
    
    
def main():
    # Buildings polygon layer
    filename = 'wu.shp'
    bld = gpd.GeoDataFrame.from_file(filename)
    print(bld.crs)
    
    height_col = 'Floor'
    theta = 10
    R = 50
    resolution = 10
    
    #cen_p = (239775, 3385668)
    
    svf_map = mapSVF(resolution, bld, theta, R, height_col, contain_roof=True)
    
    plt.scatter(svf_map['geometry'].x.values, svf_map['geometry'].y.values, 
                c=svf_map['svf'].values, cmap='gray')
    
    

if __name__ == '__main__':
    main()










