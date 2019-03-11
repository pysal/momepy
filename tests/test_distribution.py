import sys
sys.path.insert(0, "/Users/martin/Strathcloud/Personal Folders/momepy/momepy")

import pytest
import momepy as mm
import geopandas as gpd
from shapely.geometry import Polygon, Point
import numpy as np
import math

from momepy.shape import _make_circle, _circle_area
from momepy import Queen_higher


class TestShape:

    def setup_method(self):

        test_file_path = mm.datasets.get_path('bubenec')
        self.df_buildings = gpd.read_file(test_file_path, layer='buildings')
        self.df_streets = gpd.read_file(test_file_path, layer='streets')
        self.df_tessellation = gpd.read_file(test_file_path, layer='tessellation')
        self.df_buildings['height'] = np.linspace(10., 30., 144)
        self.df_buildings['volume'] = mm.volume(self.df_buildings, 'height')

    def test_orientation(self):
        self.df_buildings['orient'] = mm.orientation(self.df_buildings)
        check = 41.05146788287027
        assert self.df_buildings['orient'][0] == check

    def test_shared_walls_ratio(self):
        self.df_buildings['swr'] = mm.shared_walls_ratio(self.df_buildings, 'uID')
        check = 0.3424804411228673
        assert self.df_buildings['orient'][10] == check
