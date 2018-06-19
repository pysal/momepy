# elements.py
# generating derived elements (street edge, block)

import geopandas as gpd
from tqdm import tqdm  # progress bar
from osgeo import ogr
from shapely.wkt import loads
import shapely.geometry
from shapely.geometry import MultiPoint, Point, Polygon
import shapely.ops
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt


'''
clean geometry

delete building with zero height (to avoid division by 0)
'''


def clean_buildings(path, height_column):
    print('Loading file.')
    objects = gpd.read_file(path)  # load file into geopandas
    print('Shapefile loaded.')

    objects = objects[objects[height_column] > 0]
    print('Zero height buildings ommited.')

    # save dataframe back to file
    print('Saving file.')
    objects.to_file(path)
    print('File saved.')


# objects.to_file("/Users/martin/Strathcloud/Personal Folders/Test data/Royston/Tess/buildings.shp")
