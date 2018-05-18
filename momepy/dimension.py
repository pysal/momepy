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

'''
object_perimeter:
    Calculate perimeter of each object in given shapefile. It can be used for any
    suitable element (building footprint, plot, voronoi cell, block).

    Attributes: objects = geoDataFrame with objects
                column_name = name of the column to save the perimeter values
'''


def object_perimeter(objects, column_name):
    # define new column
    objects[column_name] = None
    objects[column_name] = objects[column_name].astype('float')
    print('Column ready. Calculating.')

    # fill new column with the value of perimeter, iterating over rows one by one
    for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
        objects.loc[index, column_name] = row['geometry'].length

    print('Perimeters calculated.')

'''
object_height_os:
    Function tailored to GB Ordance Survey data of OS Building Heights (Alpha).
    It will copy RelH2 values to new column.
        (relative height from ground level to base of the roof [RelH2])

    Attributes: objects = geoDataFrame with objects
                column_name = name of the column to save the height values
                original_column = name of column where is stored original height
                                  value. Default value 'relh2' (optional)
'''


def object_height_os(objects, column_name, original_column='relh2'):
    # define new column
    objects[column_name] = None
    objects[column_name] = objects[column_name].astype('float')
    print('Column ready. Calculating.')

    # fill new column with the value of perimeter, iterating over rows one by one
    for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
        objects.loc[index, column_name] = row[original_column]

    print('Heights defined.')

'''
object_volume:
    Calculate volume of each object in given shapefile based on its height and area.

    Attributes: objects = geoDataFrame with objects
                column_name = name of the column to save the volume values
                area_column = name of column where is stored area value.
                height_column = name of column where is stored height value.
                area_calculated = boolean value checking whether area has been
                                  previously calculated and stored in separate column.
                                  If set to FALSE, function will calculate areas
                                  during the process without saving them separately.
'''


def object_volume(objects, column_name, area_column, height_column, area_calculated):
    # define new column
    objects[column_name] = None
    objects[column_name] = objects[column_name].astype('float')
    print('Column ready. Calculating.')

    if area_calculated:
        try:
            # fill new column with the value of perimeter, iterating over rows one by one
            for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
                objects.loc[index, column_name] = row[area_column] * row[height_column]
                print('Volumes calculated.')

        except KeyError:
            print('ERROR: Building area column named', area_column, 'not found. Define area_column or set area_calculated to False.')
    else:
        # fill new column with the value of perimeter, iterating over rows one by one
        for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
            objects.loc[index, column_name] = row['geometry'].area * row[height_column]

        print('Volumes calculated.')

'''
object_floor_area:
    Calculate floor area of each object based on height and area. Number of floors
    is simplified into formula height//3 (it is assumed that on average one floor
    is approximately 3 metres)

    Attributes: objects = geoDataFrame with objects
                column_name = name of the column to save the volume values
                area_column = name of column where is stored area value.
                height_column = name of column where is stored height value.
                area_calculated = boolean value checking whether area has been
                                  previously calculated and stored in separate column.
                                  If set to FALSE, function will calculate areas
                                  during the process without saving them separately.
'''


def object_floor_area(objects, column_name, area_column, height_column, area_calculated):
    # define new column
    objects[column_name] = None
    objects[column_name] = objects[column_name].astype('float')
    print('Column ready. Calculating.')

    if area_calculated:
        try:
            # fill new column with the value of perimeter, iterating over rows one by one
            for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
                objects.loc[index, column_name] = row[area_column] * (row[height_column] // 3)

            print('Floor areas calculated.')

        except KeyError:
            print('ERROR: Building area column named', area_column, 'not found. Define area_column or set area_calculated to False.')
    else:
        # fill new column with the value of perimeter, iterating over rows one by one
        for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
            objects.loc[index, column_name] = row['geometry'].area * (row[height_column] // 3)

        print('Floor areas calculated.')
