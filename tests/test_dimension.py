import pytest
import momepy as mm
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np

from momepy.shape import _make_circle
from momepy import Queen_higher


class TestDimensions:

    def setup_method(self):

        test_file_path = mm.datasets.get_path('bubenec')
        self.df_buildings = gpd.read_file(test_file_path, layer='buildings')
        self.df_streets = gpd.read_file(test_file_path, layer='streets')
        self.df_tessellation = gpd.read_file(test_file_path, layer='tessellation')
        self.df_buildings['height'] = np.linspace(10., 30., 144)

    def test_area(self):
        self.df_buildings['area'] = mm.area(self.df_buildings)
        check = self.df_buildings.geometry[0].area
        assert self.df_buildings['area'][0] == check

    def test_perimeter(self):
        self.df_buildings['perimeter'] = mm.perimeter(self.df_buildings)
        check = self.df_buildings.geometry[0].length
        assert self.df_buildings['perimeter'][0] == check

    def test_volume(self):
        self.df_buildings['area'] = self.df_buildings.geometry.area
        self.df_buildings['volume'] = mm.volume(self.df_buildings, 'height', 'area')
        check = self.df_buildings.geometry[0].area * self.df_buildings.height[0]
        assert self.df_buildings['volume'][0] == check

    def test_volume_no_area(self):
        self.df_buildings['volume'] = mm.volume(self.df_buildings, 'height')
        check = self.df_buildings.geometry[0].area * self.df_buildings.height[0]
        assert self.df_buildings['volume'][0] == check

    def test_volume_missing_col(self):
        with pytest.raises(KeyError):
            self.df_buildings['volume'] = mm.volume(self.df_buildings, 'height', 'area')

    def test_floor_area(self):
        self.df_buildings['area'] = self.df_buildings.geometry.area
        self.df_buildings['floor_area'] = mm.floor_area(self.df_buildings, 'height', 'area')
        check = self.df_buildings.geometry[0].area * (self.df_buildings.height[0] // 3)
        assert self.df_buildings['floor_area'][0] == check

    def test_floor_area_no_area(self):
        self.df_buildings['floor_area'] = mm.floor_area(self.df_buildings, 'height')
        check = self.df_buildings.geometry[0].area * (self.df_buildings.height[0] // 3)
        assert self.df_buildings['floor_area'][0] == check

    def test_floor_area_missing_col(self):
        with pytest.raises(KeyError):
            self.df_buildings['floor_area'] = mm.floor_area(self.df_buildings, 'height', 'area')

    def test_courtyard_area(self):
        self.df_buildings['area'] = self.df_buildings.geometry.area
        self.df_buildings['courtyard_area'] = mm.courtyard_area(self.df_buildings, 'area')
        check = Polygon(self.df_buildings.geometry[80].exterior).area - self.df_buildings.geometry[80].area
        assert self.df_buildings['courtyard_area'][80] == check

    def test_courtyard_no_area(self):
        self.df_buildings['courtyard_area'] = mm.courtyard_area(self.df_buildings)
        check = Polygon(self.df_buildings.geometry[80].exterior).area - self.df_buildings.geometry[80].area
        assert self.df_buildings['courtyard_area'][80] == check

    def test_courtyard_missing_col(self):
        with pytest.raises(KeyError):
            self.df_buildings['courtyard_area'] = mm.floor_area(self.df_buildings, 'area')

    def test_longest_axis_length(self):
        self.df_buildings['long_axis'] = mm.longest_axis_length(self.df_buildings)
        check = _make_circle(self.df_buildings.geometry[0].convex_hull.exterior.coords)[2] * 2
        assert self.df_buildings['long_axis'][0] == check

    def test_effective_mesh(self):
        self.df_tessellation['mesh'] = mm.effective_mesh(self.df_tessellation)
        spatial_weights = Queen_higher(k=3, geodataframe=self.df_tessellation)
        neighbours = spatial_weights.neighbors[38]
        total_area = sum(self.df_tessellation.iloc[neighbours].geometry.area) + self.df_tessellation.geometry.area[38]
        check = total_area / (len(neighbours) + 1)
        assert self.df_tessellation['mesh'][38] == check

    def test_effective_mesh_sw(self):
        sw = Queen_higher(k=3, geodataframe=self.df_tessellation)
        self.df_tessellation['mesh'] = mm.effective_mesh(self.df_tessellation, sw)
        neighbours = sw.neighbors[38]
        total_area = sum(self.df_tessellation.iloc[neighbours].geometry.area) + self.df_tessellation.geometry.area[38]
        check = total_area / (len(neighbours) + 1)
        assert self.df_tessellation['mesh'][38] == check

    def test_effective_mesh_area(self):
        self.df_tessellation['area'] = self.df_tessellation.geometry.area
        sw = Queen_higher(k=3, geodataframe=self.df_tessellation)
        self.df_tessellation['mesh'] = mm.effective_mesh(self.df_tessellation, sw, areas='area')
        neighbours = sw.neighbors[38]
        total_area = sum(self.df_tessellation.iloc[neighbours].geometry.area) + self.df_tessellation.geometry.area[38]
        check = total_area / (len(neighbours) + 1)
        assert self.df_tessellation['mesh'][38] == check

    def test_street_profile(self):
        widths, heights, profile = mm.street_profile(self.df_streets, self.df_buildings, heights='height')
        assert widths[16] == 34.722744851010795
        assert heights[16] == 16.13286713286713
        assert profile[16] == 0.46461958010780635

    def test_street_profile_wrongtuple(self):
        with pytest.raises(ValueError):
            widths, heights, profile = mm.street_profile(self.df_streets, self.df_buildings)

    def test_weighted_character(self):
        weighted = mm.weighted_character(self.df_buildings, self.df_tessellation, 'height', 'uID')
        assert weighted[38] == 18.301521351817303

    def test_weighted_character_sw(self):
        sw = Queen_higher(k=3, geodataframe=self.df_tessellation)
        weighted = mm.weighted_character(self.df_buildings, self.df_tessellation, 'height', 'uID', sw)
        assert weighted[38] == 18.301521351817303

    def test_weighted_character_area(self):
        self.df_buildings['area'] = self.df_buildings.geometry.area
        sw = Queen_higher(k=3, geodataframe=self.df_tessellation)
        weighted = mm.weighted_character(self.df_buildings, self.df_tessellation, 'height', 'uID', sw, 'area')
        assert weighted[38] == 18.301521351817303
