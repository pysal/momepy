# dimension.py
# definitons of dimension characters

import geopandas as gpd
from tqdm import tqdm  # progress bar

'''
object_area:
    Calculate area of each object in given shapefile. It can be used for any
    suitable element (building footprint, plot, voronoi cell, block).

    Attributes: objects = geoDataFrame with objects
                column_name = name of the column to save the area values
'''


def object_area(objects, column_name):

    # define new column
    objects[column_name] = None
    objects[column_name] = objects[column_name].astype('float')
    print('Column ready. Calculating.')

    # fill new column with the value of area, iterating over rows one by one
    for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
        objects.loc[index, column_name] = row['geometry'].area

    print('Areas calculated.')
