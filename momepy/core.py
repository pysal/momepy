#!/usr/bin/env python
# -*- coding: utf-8 -*-

import geopandas as gpd

from .dimension import *
from .shape import *
from .distribution import *
from .intensity import *
from .diversity import *


# to be removed
def gethead(path):
    file = gpd.read_file(path)
    show = file.head(5)
    print(show)


# simple reusing of gpd.read_file. Returns GeoDataFrame.
def load(path):
    print('Loading file', path)
    objects = gpd.read_file(path)  # load file into geopandas
    print('Shapefile loaded.')
    return objects


# simple reusing of gpd.to_file. Saves GeoDataFrame to file.
def save(objects, path):
    print('Saving GeoDataFrame to', path)
    objects.to_file(path)  # load file into geopandas
    print('Shapefile saved.')

# dimension characters
'''
building_dimensions():
    Calculate characters all based on buildings.

    objects = GeoDataFrame with building shapes

    Uses default values of momepy.

    ADD MISSING
'''


def building_dimensions(objects):

    area(objects, 'pdbAre')
    perimeter(objects, 'pdbPer')
    volume(objects, column_name='pdbVol', area_column='pdbAre', height_column='pdbHei', area_calculated=True)
    floor_area(objects, column_name='pdbFlA', area_column='pdbAre', height_column='pdbHei', area_calculated=True)
    courtyard_area(objects, column_name='pdbCoA', area_column='pdbAre', area_calculated=True)

    # return objects
