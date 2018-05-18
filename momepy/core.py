'''
Core of the momepy. From here everything will be called.

Open/close calls on files always in core.py. Calculations always in helpers.
'''
import geopandas as gpd

from .dimension import *


# to be removed
def gethead(path):
    file = gpd.read_file(path)
    show = file.head(5)
    print(show)


# dimension characters
'''
building_area:
    character id: pdbAre

    Return areas of buildings (or different elements, but then it is recommended
    to change attribute column_name).

    Attributes: path = path to file (tested on shapefile)
                column_name = name of the column, default 'pdbAre' (optional)
'''


def building_area(path, column_name='pdbAre'):
    print('Loading file.')
    objects = gpd.read_file(path)  # load file into geopandas
    print('Shapefile loaded.')

    object_area(objects, column_name)  # call function from dimension

    # save dataframe back to file
    print('Saving file.')
    objects.to_file(path)
    print('File saved.')


'''
building_perimeter:
    character id: pdbPer

    Return perimeters of buildings (or different elements, but then it is recommended
    to change attribute column_name).

    Attributes: path = path to file (tested on shapefile)
                column_name = name of the column, default 'pdbPer' (optional)
'''


def building_perimeter(path, column_name='pdbPer'):
    print('Loading file.')
    objects = gpd.read_file(path)  # load file into geopandas
    print('Shapefile loaded.')

    object_perimeter(objects, column_name)  # call function from dimension

    # save dataframe back to file
    print('Saving file.')
    objects.to_file(path)
    print('File saved.')

'''
building_height_os:
    character id: pdbHei

    Return heights of buildings (based on Ordnance Survey data).

    Attributes: path = path to file (tested on shapefile)
                column_name = name of the column, default 'pdbHei' (optional)
'''


def building_height_os(path, column_name='pdbHei'):
    print('Loading file.')
    objects = gpd.read_file(path)  # load file into geopandas
    print('Shapefile loaded.')

    object_height_os(objects, column_name)  # call function from dimension

    # save dataframe back to file
    print('Saving file.')
    objects.to_file(path)
    print('File saved.')

'''
building_volume:
    character id: pdbVol

    Return volumes of buildings.

    Attributes: path = path to file (tested on shapefile)
                column_name = name of the column, default 'pdbVol' (optional)
                area_column = name of column where is stored area value. Default
                              value 'pdbAre' (optional)
                height_column = name of column where is stored height value.
                                Default value 'pdbHei' (optional)
                area_calculated = boolean value checking whether area has been
                                  previously calculated and stored in separate column.
                                  Default value 'True'. If set to 'False', function
                                  will calculate areas during the process without
                                  saving them separately.
'''


def building_volume(path, column_name='pdbVol', area_column='pdbAre', height_column='pdbHei', area_calculated=True):
    print('Loading file.')
    objects = gpd.read_file(path)  # load file into geopandas
    print('Shapefile loaded.')

    object_volume(objects, column_name, area_column, height_column, area_calculated)  # call function from dimension

    # save dataframe back to file
    print('Saving file.')
    objects.to_file(path)
    print('File saved.')

'''
building_floor_area:
    character id: pdbFlA

    Return floor areas of buildings. In case of OS data, number of buildings is
    estimated as height//3 (every 3 metres make one floor).

    Attributes: path = path to file (tested on shapefile)
                column_name = name of the column, default 'pdbFlA' (optional)
                area_column = name of column where is stored area value. Default
                              value 'pdbAre' (optional)
                height_column = name of column where is stored height value.
                                Default value 'pdbHei' (optional)
                area_calculated = boolean value checking whether area has been
                                  previously calculated and stored in separate column.
                                  Default value 'True'. If set to 'False', function
                                  will calculate areas during the process without
                                  saving them separately.
'''


def building_floor_area(path, column_name='pdbFlA', area_column='pdbAre', height_column='pdbHei', area_calculated=True):
    print('Loading file.')
    objects = gpd.read_file(path)  # load file into geopandas
    print('Shapefile loaded.')

    object_floor_area(objects, column_name, area_column, height_column, area_calculated)  # call function from dimension

    # save dataframe back to file
    print('Saving file.')
    objects.to_file(path)
    print('File saved.')
