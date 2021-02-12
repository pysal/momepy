#-------------------------------------------------------------------------------
# Program Name: Simple Building Volume Density (BVD) for building morphological
# Purpose: for urban morphological metrics of MOMEPY

# This Python 3 programs depends on: math, shapely, fiona and geopandas

# Version:     0.1 
#              Functionalities:
#              1. Building volume density
#              2. Volume density computed as the ratio of 
#              building volumes to a virtual box defined by the highest building in an unit area...

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
def ptAll(bld):
    # Find centroid
    # Make copy of the dataframe
    pt_all = bld.copy()
    # Change geometry into centroid
    pt_all['geometry'] = pt_all['geometry'].centroid
    return pt_all

# Intersect point buffer with building map
# Inputs: point, building geodata frame, dome radius
def bldAtPt(pt, bld, win_size):
    buff = pt.buffer(win_size/2)
    enve = buff.envelope  
    #buff = buff.set_crs(bld.crs)
    bld_clip = gpd.clip(bld, enve)
    bld_clip = bld_clip.explode()
    return bld_clip

# SVF at each point
def bvd(bl, win, col):
    # Building footprint area
    bl['area'] = bl['geometry'].area
    # Building volume
    bl['vol'] = bl['area']*bl[col]
    # Building volume total
    v_total = bl['vol'].sum()
    # Highest value of polygons
    h_max = bl[col].max()
    # Bounding cube volume
    v_bound = h_max*win**2
    
    density = v_total/v_bound
    return density
            
# Generate a SVF map
def mapBVD(bld, win, col, contain_roof=True):
    # Generate mesh points
    pt_all = ptAll(bld)
    
    # Loop over all points in a geodataframe
    den_all = []
    bld['density'] = ''
    for ind, pt in pt_all.iterrows():
        print(ind)
        bld_pt = bldAtPt(pt['geometry'], bld, win)  # Buildings within a neighborhood of a point
        den = bvd(bld_pt, win, col)
        bld['density'][ind] = den
        den_all.append(den)
    return bld
    
    
def main():
    # Buildings polygon layer
    filename = 'nmg_all.shp'
    bld = gpd.GeoDataFrame.from_file(filename)
    print(bld.crs)
    
    height_col = 'roof.0.99'
    
    #neighbor_option = ['window', 'plot']  # Specify neighborhood as the unit area
    
    win_size = 100  # Window size if option 'window' is selected
        
    bld_bvd = mapBVD(bld, win=win_size, col=height_col)
    
    bld_bvd.plot(column='density', cmap='Wistia', legend=True)
    
if __name__ == '__main__':
    main()










