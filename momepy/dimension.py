# dimension.py
# definitons of dimension characters

import geopandas as gpd  # to remove in the end
from tqdm import tqdm  # progress bar
from shapely.geometry import Polygon
import shapely.ops
from .shape import make_circle

'''
object_area():
    Calculate area of each object in given shapefile. It can be used for any
    suitable element (building footprint, plot, voronoi cell, block).

    Attributes: objects = geoDataFrame with objects
                column_name = name of the column to save the area values
'''


def object_area(objects, column_name):
    # define new column
    objects[column_name] = None
    objects[column_name] = objects[column_name].astype('float')
    print('Calculating areas.')

    # fill new column with the value of area, iterating over rows one by one
    for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
        objects.loc[index, column_name] = row['geometry'].area

    print('Areas calculated.')

'''
object_perimeter():
    Calculate perimeter of each object in given shapefile. It can be used for any
    suitable element (building footprint, plot, voronoi cell, block).

    Attributes: objects = geoDataFrame with objects
                column_name = name of the column to save the perimeter values
'''


def object_perimeter(objects, column_name):
    # define new column
    objects[column_name] = None
    objects[column_name] = objects[column_name].astype('float')
    print('Calculating perimeters.')

    # fill new column with the value of perimeter, iterating over rows one by one
    for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
        objects.loc[index, column_name] = row['geometry'].length

    print('Perimeters calculated.')

'''
object_height_os():
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
    print('Calculating heights.')

    # fill new column with the value of perimeter, iterating over rows one by one
    for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
        objects.loc[index, column_name] = row[original_column]

    print('Heights defined.')

'''
object_volume():
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
    print('Calculating volumes.')

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
object_floor_area():
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
    print('Calculating floor areas.')

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

'''
courtyard_area():
    Calculate area of holes within geometry - area of courtyards.

    Attributes: objects = geoDataFrame with objects
                column_name = name of the column to save the volume values
                area_column = name of column where is stored area value
                area_calculated = boolean value checking whether area has been
                                  previously calculated and stored in separate column.
                                  If set to FALSE, function will calculate areas
                                  during the process without saving them separately.
'''


def courtyard_area(objects, column_name, area_column, area_calculated):
    # define new column
    objects[column_name] = None
    objects[column_name] = objects[column_name].astype('float')
    print('Calculating courtyard areas.')

    if area_calculated:
        try:
            for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
                objects.loc[index, column_name] = Polygon(row['geometry'].exterior).area - row[area_column]

            print('Core area indices calculated.')
        except KeyError:
            print('ERROR: Building area column named', area_column, 'not found. Define area_column or set area_calculated to False.')
    else:
        for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
            objects.loc[index, column_name] = Polygon(row['geometry'].exterior).area - row['geometry'].area

        print('Core area indices calculated.')

'''
longest_axis_length():
    Calculate the length of the longest axis of object.

    Attributes: objects = geoDataFrame with objects
                column_name = name of the column to save the volume values
'''


def longest_axis_length(objects, column_name):
    # define new column
    objects[column_name] = None
    objects[column_name] = objects[column_name].astype('float')
    print('Calculating the longest axis.')

    # calculate the area of circumcircle
    def longest_axis(points):
        circ = make_circle(points)
        return(circ[2] * 2)

    def sort_NoneType(geom):
        if geom is not None:
            if geom.type is not 'Polygon':
                return(0)
            else:
                return(longest_axis(list(geom.exterior.coords)))
        else:
            return(0)

    # fill new column with the value of area, iterating over rows one by one
    for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
        objects.loc[index, column_name] = sort_NoneType(row['geometry'])

    print('The longest axis calculated.')


def longest_axis_length2(objects, column_name):  # based on convex hull for multipart polygons
    # define new column
    objects[column_name] = None
    objects[column_name] = objects[column_name].astype('float')
    print('Calculating the longest axis.')

    # calculate the area of circumcircle
    def longest_axis(points):
        circ = make_circle(points)
        return(circ[2] * 2)

    def sort_NoneType(geom):
        if geom is not None:
            if geom.type is not 'Polygon':
                return(0)
            else:
                return(longest_axis(list(geom.boundary.coords)))
        else:
            return(0)

    # fill new column with the value of area, iterating over rows one by one
    for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
        objects.loc[index, column_name] = longest_axis(row['geometry'].convex_hull.exterior.coords)

    print('The longest axis calculated.')


# to be deleted, keep at the end

# path = "/Users/martin/Dropbox/StrathUni/PhD/Papers/Voronoi tesselation/Data/Zurich/Final data/Voronoi/test/voronoi_10.shp"
# objects = gpd.read_file(path)

# longest_axis_length2(objects, column_name='longest_axis')

# objects.head
# objects.to_file(path)
