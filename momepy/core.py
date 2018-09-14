'''
Core of the momepy. From here everything will be called.

Open/close calls on files always in core.py. Calculations always in helpers.
'''
import geopandas as gpd

from .dimension import *
from .shape import *
from .intensity import *
from .diversity import *


# to be removed
def gethead(path):
    file = gpd.read_file(path)
    show = file.head(5)
    print(show)


# dimension characters
'''
Dimension characters:

    plot scale:

        building_area
        building_perimeter
        building_height_os
        building_volume
        building_floor_area
        building_courtyard_area
        cell_longest_axis_length
'''

'''
building_area():
    character id: pdbAre

    Return areas of buildings (or different elements, but then it is recommended
    to change attribute column_name).

    Attributes: path = path to file (tested on shapefile)
                column_name = name of the column, default 'pdbAre' (optional)
'''


def building_area(path, column_name='pdbAre'):
    print('Loading file...')
    objects = gpd.read_file(path)  # load file into geopandas
    print('Shapefile loaded.')

    object_area(objects, column_name)  # call function from dimension

    # save dataframe back to file
    print('Saving file...')
    objects.to_file(path)
    print('File saved.')


'''
building_perimeter():
    character id: pdbPer

    Return perimeters of buildings (or different elements, but then it is recommended
    to change attribute column_name).

    Attributes: path = path to file (tested on shapefile)
                column_name = name of the column, default 'pdbPer' (optional)
'''


def building_perimeter(path, column_name='pdbPer'):
    print('Loading file...')
    objects = gpd.read_file(path)  # load file into geopandas
    print('Shapefile loaded.')

    object_perimeter(objects, column_name)  # call function from dimension

    # save dataframe back to file
    print('Saving file...')
    objects.to_file(path)
    print('File saved.')

'''
building_height_os():
    character id: pdbHei

    Return heights of buildings (based on Ordnance Survey data).

    Attributes: path = path to file (tested on shapefile)
                column_name = name of the column, default 'pdbHei' (optional)
'''


def building_height_os(path, column_name='pdbHei'):
    print('Loading file...')
    objects = gpd.read_file(path)  # load file into geopandas
    print('Shapefile loaded.')

    object_height_os(objects, column_name)  # call function from dimension

    # save dataframe back to file
    print('Saving file...')
    objects.to_file(path)
    print('File saved.')

'''
building_volume():
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
                                  saving them separately. (optional)
'''


def building_volume(path, column_name='pdbVol', area_column='pdbAre', height_column='pdbHei', area_calculated=True):
    print('Loading file...')
    objects = gpd.read_file(path)  # load file into geopandas
    print('Shapefile loaded.')

    object_volume(objects, column_name, area_column, height_column, area_calculated)  # call function from dimension

    # save dataframe back to file
    print('Saving file...')
    objects.to_file(path)
    print('File saved.')

'''
building_floor_area():
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
                                  saving them separately. (optional)
'''


def building_floor_area(path, column_name='pdbFlA', area_column='pdbAre', height_column='pdbHei', area_calculated=True):
    print('Loading file...')
    objects = gpd.read_file(path)  # load file into geopandas
    print('Shapefile loaded.')

    object_floor_area(objects, column_name, area_column, height_column, area_calculated)  # call function from dimension

    # save dataframe back to file
    print('Saving file...')
    objects.to_file(path)
    print('File saved.')

'''
building_courtyard_area():
    character id: pdbCoA

    Return areas of building courtyards (sum).

    Attributes: path = path to file (tested on shapefile)
                column_name = name of the column, default 'pdbCoA' (optional)
                area_column = name of column where is stored area value. Default
                              value 'pdbAre' (optional)
                area_calculated = boolean value checking whether area has been
                                  previously calculated and stored in separate column.
                                  Default value 'True'. If set to 'False', function
                                  will calculate areas during the process without
                                  saving them separately. (optional)
'''


def building_courtyard_area(path, column_name='pdbCoA', area_column='pdbAre', area_calculated=True):
    print('Loading file...')
    objects = gpd.read_file(path)  # load file into geopandas
    print('Shapefile loaded.')

    courtyard_area(objects, column_name, area_column, area_calculated)  # call function from dimension

    # save dataframe back to file
    print('Saving file...')
    objects.to_file(path)
    print('File saved.')

'''
cell_longest_axis_length():
    character id: pdcLAL

    Return the length of the longest axis of each cell.

    Attributes: path = path to file (tested on shapefile)
                column_name = name of the column, default 'pdcLAL' (optional)
'''


def cell_longest_axis_length(path, column_name='pdcLAL'):
    print('Loading file...')
    objects = gpd.read_file(path)  # load file into geopandas
    print('Shapefile loaded.')

    longest_axis_length(objects, column_name)  # call function from dimension

    # save dataframe back to file
    print('Saving file...')
    objects.to_file(path)
    print('File saved.')

'''
building_dimensions():
    Calculate characters building_area, building_perimeter, building_height_os,
    building_volume, building_floor_area and building_courtyard_area.

    Uses default values.
'''


def building_dimensions(path):
    print('Loading file...')
    objects = gpd.read_file(path)  # load file into geopandas
    print('Shapefile loaded.')

    object_area(objects, 'pdbAre')
    object_perimeter(objects, 'pdbPer')
    object_height_os(objects, 'pdbHei')
    object_volume(objects, column_name='pdbVol', area_column='pdbAre', height_column='pdbHei', area_calculated=True)
    object_floor_area(objects, column_name='pdbFlA', area_column='pdbAre', height_column='pdbHei', area_calculated=True)
    courtyard_area(objects, column_name='pdbCoA', area_column='pdbAre', area_calculated=True)

    # save dataframe back to file
    print('Saving file...')
    objects.to_file(path)
    print('File saved.')


# shape characters
'''
Shape characters:

    plot scale:

        building_form_factor
        building_fractal_dimension
        building_volume_facade_ratio
        building_compactness_index
        building_convexeity
        building_courtyard_index
'''

'''
building_form_factor():
    character id: psbFoF

    Return form factor of building.

    Attributes: path = path to file (tested on shapefile)
                column_name = name of the column, default 'psbFoF' (optional)
                area_column = name of column where is stored area value. Default
                              value 'pdbAre' (optional)
                volume_column = name of the column where is stored volume value.
                                Default value 'pdbVol' (optional)

    Missing: Option to calculate without area and volume being calculated beforehand.
'''


def building_form_factor(path, column_name='psbFoF', area_column='pdbAre', volume_column='pdbVol'):
    print('Loading file...')
    objects = gpd.read_file(path)  # load file into geopandas
    print('Shapefile loaded.')

    form_factor(objects, column_name, area_column, volume_column)  # call function from dimension

    # save dataframe back to file
    print('Saving file...')
    objects.to_file(path)
    print('File saved.')

'''
building_fractal_dimension():
    character id: psbFra

    Return fractal dimension of building.

    Attributes: path = path to file (tested on shapefile)
                column_name = name of the column, default 'psbFrA' (optional)
                area_column = name of column where is stored area value. Default
                              value 'pdbAre' (optional)
                perimeter_column = name of the column where is stored volume value.
                                Default value 'pdbPer' (optional)

    Missing: Option to calculate without area and perimeter being calculated beforehand.
'''


def building_fractal_dimension(path, column_name='psbFra', area_column='pdbAre', perimeter_column='pdbPer'):
    print('Loading file...')
    objects = gpd.read_file(path)  # load file into geopandas
    print('Shapefile loaded.')

    fractal_dimension(objects, column_name, area_column, perimeter_column)  # call function from dimension

    # save dataframe back to file
    print('Saving file...')
    objects.to_file(path)
    print('File saved.')

'''
building_volume_facade_ratio():
    character id: psbVFR

    Return volume / facade ratio of building.

    Attributes: path = path to file (tested on shapefile)
                column_name = name of the column, default 'pdbVFR' (optional)
                volume_column = name of column where is stored volume value. Default
                              value 'pdbVol' (optional)
                perimeter_column = name of the column where is stored volume value.
                                Default value 'pdbPer' (optional)
                height_column = name of the column where is stored heigth value.
                                Default value 'pdbHei' (optional)

    Missing: Option to calculate without values being calculated beforehand.

'''


def building_volume_facade_ratio(path, column_name='psbVFR', volume_column='pdbVol', perimeter_column='pdbPer', height_column='pdbHei'):
    print('Loading file...')
    objects = gpd.read_file(path)  # load file into geopandas
    print('Shapefile loaded.')

    volume_facade_ratio(objects, column_name, volume_column, perimeter_column, height_column)  # call function from dimension

    # save dataframe back to file
    print('Saving file...')
    objects.to_file(path)
    print('File saved.')

'''
building_compactness_index():
    character id: psbCom

    Return compactness index of building.

    Attributes: path = path to file (tested on shapefile)
                column_name = name of the column, default 'psbCom' (optional)
                area_column = name of column where is stored area value. Default
                              value 'pdbAre' (optional)

    Missing: Option to calculate without values being calculated beforehand.
'''


def building_compactness_index(path, column_name='psbCom', area_column='pdbAre'):
    print('Loading file...')
    objects = gpd.read_file(path)  # load file into geopandas
    print('Shapefile loaded.')

    compactness_index(objects, column_name, area_column)  # call function from dimension

    # save dataframe back to file
    print('Saving file...')
    objects.to_file(path)
    print('File saved.')

'''
building_convexeity():
    character id: psbCom

    Return convexeity of building.

    Attributes: path = path to file (tested on shapefile)
                column_name = name of the column, default 'psbCon' (optional)
                area_column = name of column where is stored area value. Default
                              value 'pdbAre' (optional)

    Missing: Option to calculate without values being calculated beforehand.
'''


def building_convexeity(path, column_name='psbCon', area_column='pdbAre'):
    print('Loading file...')
    objects = gpd.read_file(path)  # load file into geopandas
    print('Shapefile loaded.')

    convexeity(objects, column_name, area_column)  # call function from dimension

    # save dataframe back to file
    print('Saving file...')
    objects.to_file(path)
    print('File saved.')

'''
building_courtyard_index():
    character id: psbCoI

    Return courtyard index of building.

    Attributes: path = path to file (tested on shapefile)
                column_name = name of the column, default 'psbCon' (optional)
                area_column = name of column where is stored area value. Default
                              value 'pdbAre' (optional)
                courtyard_column = name of the column where is stored cortyard area
                                   Default value 'psbCoI'

    Missing: Option to calculate without values being calculated beforehand.
'''


def building_courtyard_index(path, column_name='psbCoI', area_column='pdbAre', courtyard_column='pdbCoA'):
    print('Loading file...')
    objects = gpd.read_file(path)  # load file into geopandas
    print('Shapefile loaded.')

    courtyard_index(objects, column_name, area_column, courtyard_column)  # call function from dimension

    # save dataframe back to file
    print('Saving file...')
    objects.to_file(path)
    print('File saved.')

'''
building_rectangularity():
    character id: not used as taxonomic character

    Return number of corners in a building shape.

    Attributes: path = path to file (tested on shapefile)
                column_name = name of the column
                area_column = name of the column with area values

    Missing: Option to calculate without values being calculated beforehand.
'''


def building_rectangularity(path, column_name, area_column):
    print('Loading file...')
    objects = gpd.read_file(path)  # load file into geopandas
    print('Shapefile loaded.')

    corners(objects, column_name, area_column)  # call function from dimension

    # save dataframe back to file
    print('Saving file...')
    objects.to_file(path)
    print('File saved.')


'''
building_shape_index():
    character id: psbShI

    Return number of corners in a building shape.

    Attributes: path = path to file (tested on shapefile)
                column_name = name of the column
                area_column = name of the column with area values
                longest_axis_column = name of the column with longest axis values

    Missing: Option to calculate without values being calculated beforehand.
'''


def building_shape_index(path, column_name='psbShI', area_column='pdbAre', longest_axis_column='pdcLAL'):
    print('Loading file...')
    objects = gpd.read_file(path)  # load file into geopandas
    print('Shapefile loaded.')

    shape_index(objects, column_name, area_column, longest_axis_column)  # call function from dimension

    # save dataframe back to file
    print('Saving file...')
    objects.to_file(path)
    print('File saved.')

'''
building_corners():
    character id: psbCor

    Return number of corners in a building shape.

    Attributes: path = path to file (tested on shapefile)
                column_name = name of the column, default 'psbCor' (optional)
'''


def building_corners(path, column_name='psbCor'):
    print('Loading file...')
    objects = gpd.read_file(path)  # load file into geopandas
    print('Shapefile loaded.')

    corners(objects, column_name)  # call function from dimension

    # save dataframe back to file
    print('Saving file...')
    objects.to_file(path)
    print('File saved.')

'''
building_squareness():
    character id: psbSqu

    Return squareness of a building shape.

    Attributes: path = path to file (tested on shapefile)
                column_name = name of the column, default 'psbSqu' (optional)

'''


def building_squareness(path, column_name='psbSqu'):
    print('Loading file...')
    objects = gpd.read_file(path)  # load file into geopandas
    print('Shapefile loaded.')

    squareness(objects, column_name)  # call function from dimension

    # save dataframe back to file
    print('Saving file...')
    objects.to_file(path)
    print('File saved.')

'''
building_equivalent_rectangular_index():
    character id: psbERI

    Return squareness of a building shape.

    Attributes: path = path to file (tested on shapefile)
                column_name = name of the column, default 'psbERI' (optional)

'''


def building_equivalent_rectangular_index(path, column_name='psbERI', area_column='pdbAre', perimeter_column='pdbPer'):
    print('Loading file...')
    objects = gpd.read_file(path)  # load file into geopandas
    print('Shapefile loaded.')

    equivalent_rectangular_index(objects, column_name, area_column, perimeter_column)  # call function from dimension

    # save dataframe back to file
    print('Saving file...')
    objects.to_file(path)
    print('File saved.')


'''
building_elongation():
    character id: psbElo

    Return Elongation of a building shape.

    Attributes: path = path to file (tested on shapefile)
                column_name = name of the column, default 'psbElo' (optional)

'''


def building_elongation(path, column_name='psbElo'):
    print('Loading file...')
    objects = gpd.read_file(path)  # load file into geopandas
    print('Shapefile loaded.')

    elongation(objects, column_name)  # call function from dimension

    # save dataframe back to file
    print('Saving file...')
    objects.to_file(path)
    print('File saved.')
# intensity characters

'''
Intensity characters:

    vicinity scale:

        cell_frequency
'''

'''
cell_frequency():
    character id: vicFre

    Return frequency of voronoi cells within the vicinity.

    Attributes: path = path to file
                column_name = name of the column, default 'vicFre' (optional)
                look_for = geoDataFrame with measured objects (could be the same)
                column_name = name of the column to save calculated values
                id_column = name of column where is stored unique id of each object.
                            If there is none, it could be generated by unique_id().
                            Default is 'uID'.
                radius = radius of buffer zone for calculation. Default is set to 400 (vicinity).

    Missing: Option to calculate without values being calculated beforehand.
'''


def cell_frequency(path, column_name='vicFre'):
    print('Loading file...')
    objects = gpd.read_file(path)  # load file into geopandas
    print('Shapefile loaded.')

    frequency(objects, objects, column_name)  # call function from dimension

    # save dataframe back to file
    print('Saving file...')
    objects.to_file(path)
    print('File saved.')

'''
cell_covered_area_ratio():
    character id: pivCAR

    TODO
    (objects, look_for, column_name, area_column, look_for_area_column, id_column='uID')
    Missing: Option to calculate without values being calculated beforehand.
'''


def cell_covered_area_ratio(path, look_for, column_name='pivCAR', area_column='ptcAre', look_for_area_column="pdbAre", id_column="uID"):
    print('Loading files...')
    objects = gpd.read_file(path)  # load file into geopandas
    print('1/2 loaded...')
    look_for = gpd.read_file(look_for)
    print('Shapefiles loaded.')

    covered_area_ratio(objects, look_for, column_name, area_column, look_for_area_column, id_column)  # call function from dimension

    # save dataframe back to file
    print('Saving file...')
    objects.to_file(path)
    print('File saved.')

# diversity characters
'''
Diversity characters:

    vicinity scale:

        gini_index
'''

'''
cell_area_gini():
    character id: vvcGAr


'''


def cell_area_gini(path, column_name='vvcGAr', source='ptcAre'):
    print('Loading file...')
    objects = gpd.read_file(path)  # load file into geopandas
    print('Shapefile loaded.')

    gini_index(objects, objects, source, column_name)  # call function from dimension

    # save dataframe back to file
    print('Saving file...')
    objects.to_file(path)
    print('File saved.')


def cell_area_power(path, source='ptcAre'):
    print('Loading file...')
    objects = gpd.read_file(path)  # load file into geopandas
    print('Shapefile loaded.')

    power_law(objects, objects, source)  # call function from dimension

    # save dataframe back to file
    print('Saving file...')
    objects.to_file(path)
    print('File saved.')
