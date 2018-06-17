# elements.py
# generating derived elements (street edge, block)

import geopandas as gpd
from tqdm import tqdm  # progress bar
from osgeo import ogr
from shapely.wkt import loads
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

'''
Voronoi tesselation as a substitute of morphological plot

input layer: building footprints
'''

path = "/Users/martin/Strathcloud/Personal Folders/Test data/Royston/Tess/bul.shp"
print('Loading file.')
objects = gpd.read_file(path)  # load file into geopandas
print('Shapefile loaded.')

objects['geometry'] = tqdm(objects.buffer(-0.1))  # change original geometry with buffered one


# densify geometry beforo Voronoi tesselation
def segmentize(geom):
    wkt = geom.wkt  # shapely Polygon to wkt
    geom = ogr.CreateGeometryFromWkt(wkt)  # create ogr geometry
    geom.Segmentize(2)  # densify geometry by 2 metres
    wkt2 = geom.ExportToWkt()  # ogr geometry to wkt
    new = loads(wkt2)  # wkt to shapely Polygon
    return new

objects['geometry'] = tqdm(objects['geometry'].map(segmentize))

# Voronoi

voronoi_points = np.empty([1, 2])
voronoi_points

for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
    poly_ext = row['geometry'].exterior
    point_coords = poly_ext.coords
    row_array = np.array(point_coords)
    voronoi_points = np.concatenate((voronoi_points, row_array))

voronoi_diagram = Voronoi(voronoi_points[1:])

voronoi_plot_2d(voronoi_diagram)
# NEXT STEP = SAVE VORONOI TESSELATION TO SHAPEFILE
# https://stackoverflow.com/questions/27548363/from-voronoi-tessellation-to-shapely-polygons


# dissolve


objects.to_file("/Users/martin/Strathcloud/Personal Folders/Test data/Royston/Tess/buildings.shp")
