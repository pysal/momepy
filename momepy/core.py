'''
Core of the momepy. From here everything will be called.

Open/close calls on files always in core.py. Calculations always in helpers.
'''
import geopandas as gpd

from .dimension import object_area


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

    object_area(objects, column_name)  # call function object_area from dimension

    # save dataframe back to file
    print('Saving file.')
    objects.to_file(path)
    print('File saved.')
