import momepy as mm
import geopandas as gpd
import numpy as np

from momepy import Queen_higher

import pytest

class TestDiversity:

    def setup_method(self):

        test_file_path = mm.datasets.get_path('bubenec')
        self.df_buildings = gpd.read_file(test_file_path, layer='buildings')
        self.df_streets = gpd.read_file(test_file_path, layer='streets')
        self.df_tessellation = gpd.read_file(test_file_path, layer='tessellation')
        self.df_buildings['height'] = np.linspace(10., 30., 144)
        self.df_tessellation['area'] = mm.area(self.df_tessellation)
        self.sw = Queen_higher(k=3, geodataframe=self.df_tessellation, ids='uID')

    def test_rng(self):
        full_sw = mm.rng(self.df_tessellation, 'area', self.sw, 'uID')
        assert full_sw[0] == 8255.372874447059
        area = self.df_tessellation['area']
        full2 = mm.rng(self.df_tessellation, area, self.sw, 'uID')
        assert full2[0] == 8255.372874447059
        limit = mm.rng(self.df_tessellation, 'area', self.sw, 'uID', rng=(10, 90))
        assert limit[0] == 4122.139212736442

    def test_theil(self):
        full_sw = mm.theil(self.df_tessellation, 'area', self.sw, 'uID')
        assert full_sw[0] == 0.25744684318865324
        limit = mm.theil(self.df_tessellation, 'area', self.sw, 'uID', rng=(10, 90))
        assert limit[0] == 0.13302952097969373

    def test_simpson(self):
        ht_sw = mm.simpson(self.df_tessellation, 'area', self.sw, 'uID')
        assert ht_sw[0] == 0.385
        quan_sw = mm.simpson(self.df_tessellation, 'area', self.sw, 'uID', binning='quantiles', k=3)
        assert quan_sw[0] == 0.395

    def test_gini(self):
        full_sw = mm.gini(self.df_tessellation, 'area', self.sw, 'uID')
        assert full_sw[0] == 0.39453880039926703
        limit = mm.gini(self.df_tessellation, 'area', self.sw, 'uID', rng=(10, 90))
        assert limit[0] == 0.28532814172859305
        self.df_tessellation['negative'] = self.df_tessellation.area - self.df_tessellation.area.mean()
        with pytest.raises(ValueError):
            mm.gini(self.df_tessellation, 'negative', self.sw, 'uID')
