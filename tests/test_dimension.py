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

    def test_volume_aray(self):
        area = self.df_buildings.geometry.area
        height = np.linspace(10., 30., 144)
        self.df_buildings['volume'] = mm.volume(self.df_buildings, height, area)
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

    def test_floor_area_array(self):
        area = self.df_buildings.geometry.area
        height = np.linspace(10., 30., 144)
        self.df_buildings['floor_area'] = mm.floor_area(self.df_buildings, height, area)
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

    def test_courtyard_area_array(self):
        area = self.df_buildings.geometry.area
        self.df_buildings['courtyard_area'] = mm.courtyard_area(self.df_buildings, area)
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

    def test_average_character(self):
        spatial_weights = Queen_higher(k=3, geodataframe=self.df_tessellation, ids='uID')
        self.df_tessellation['area'] = area = self.df_tessellation.geometry.area
        self.df_tessellation['mesh_ar'] = mm.average_character(self.df_tessellation, values='area', spatial_weights=spatial_weights,
                                                               unique_id='uID', mode='mode')
        self.df_tessellation['mesh_array'] = mm.average_character(self.df_tessellation, values=area, spatial_weights=spatial_weights,
                                                                  unique_id='uID', mode='median')
        self.df_tessellation['mesh_id'] = mm.average_character(self.df_tessellation, spatial_weights=spatial_weights,
                                                               values='area', rng=(10, 90), unique_id='uID')
        self.df_tessellation['mesh_iq'] = mm.average_character(self.df_tessellation, spatial_weights=spatial_weights,
                                                               values='area', rng=(25, 75), unique_id='uID')
        with pytest.raises(ValueError):
            self.df_tessellation['mesh_ar'] = mm.average_character(self.df_tessellation, values='area', spatial_weights=spatial_weights,
                                                                   unique_id='uID', mode='nonexistent')
        assert self.df_tessellation['mesh_ar'][0] == 249.50382416977067
        assert self.df_tessellation['mesh_array'][0] == 2623.996266097268
        assert self.df_tessellation['mesh_id'][38] == 2250.2241176070806
        assert self.df_tessellation['mesh_iq'][38] == 2118.6091427330666

    def test_street_profile(self):
        results = mm.street_profile(self.df_streets, self.df_buildings, heights='height')
        assert results['widths'][0] == 46.7290758769204
        assert results['width_deviations'][0] == 0.5690458650581023
        assert results['heights'][0] == 19.454545454545457
        assert results['profile'][0] == 0.4163263469148574
        assert results['openness'][0] == 0.9107142857142857
        assert results['heights_deviations'][0] == 7.720786443287433

    def test_street_profile_array(self):
        height = np.linspace(10., 30., 144)
        results = mm.street_profile(self.df_streets, self.df_buildings, heights=height, tick_length=100)
        assert results['widths'][0] == 67.563796073073
        assert results['width_deviations'][0] == 8.791875291865827
        assert results['heights'][0] == 23.756643356643362
        assert results['profile'][0] == 0.3516179483306353
        assert results['openness'][0] == 0.5535714285714286
        assert results['heights_deviations'][0] == 5.526848034418866

    def test_weighted_character_sw(self):
        sw = Queen_higher(k=3, geodataframe=self.df_tessellation, ids='uID')
        weighted = mm.weighted_character(self.df_buildings, 'height', sw, 'uID')
        assert weighted[38] == 18.301521351817303

    def test_weighted_character_area(self):
        self.df_buildings['area'] = self.df_buildings.geometry.area
        sw = Queen_higher(k=3, geodataframe=self.df_tessellation, ids='uID')
        weighted = mm.weighted_character(self.df_buildings, 'height', sw, 'uID', 'area')
        assert weighted[38] == 18.301521351817303

    def test_weighted_character_array(self):
        area = self.df_buildings.geometry.area
        sw = Queen_higher(k=3, geodataframe=self.df_tessellation, ids='uID')
        weighted = mm.weighted_character(self.df_buildings, self.df_buildings.height, sw, 'uID', area)
        assert weighted[38] == 18.301521351817303

    def test_covered_area(self):
        sw = Queen_higher(geodataframe=self.df_tessellation, k=1, ids='uID')
        covered_sw = mm.covered_area(self.df_tessellation, sw, 'uID')
        assert covered_sw[0] == 24115.667218339422

    def test_wall(self):
        sw = Queen_higher(geodataframe=self.df_buildings, k=1)
        wall = mm.wall(self.df_buildings)
        wall_sw = mm.wall(self.df_buildings, sw)
        assert wall[0] == wall_sw[0]
        assert wall[0] == 137.2106961418436

    def test_segments_length(self):
        absol = mm.segments_length(self.df_streets)
        mean = mm.segments_length(self.df_streets, mean=True)
        assert max(absol) == 1907.502238338006
        assert max(mean) == 249.5698434867373
