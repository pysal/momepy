import pytest
import momepy as mm
import geopandas as gpd
import numpy as np

from momepy import Queen_higher


class TestDiversity:

    def setup_method(self):

        test_file_path = mm.datasets.get_path('bubenec')
        self.df_buildings = gpd.read_file(test_file_path, layer='buildings')
        self.df_streets = gpd.read_file(test_file_path, layer='streets')
        self.df_tessellation = gpd.read_file(test_file_path, layer='tessellation')
        self.df_buildings['height'] = np.linspace(10., 30., 144)
        self.df_tessellation['area'] = mm.area(self.df_tessellation)

    def test_rng(self):
        full = mm.rng(self.df_tessellation, 'area', order=3)
        assert full[0] == 8255.372874447059
        sw = Queen_higher(k=3, geodataframe=self.df_tessellation)
        full_sw = mm.rng(self.df_tessellation, 'area', spatial_weights=sw)
        assert full_sw[0] == 8255.372874447059
        area = self.df_tessellation['area']
        full2 = mm.rng(self.df_tessellation, area, order=3)
        assert full2[0] == 8255.372874447059
        limit = mm.rng(self.df_tessellation, 'area', spatial_weights=sw, rng=(10, 90))
        assert limit[0] == 4122.139212736442

    def test_theil(self):
        full = mm.theil(self.df_tessellation, 'area', order=3)
        assert full[0] == 0.2574468431886534
        sw = Queen_higher(k=3, geodataframe=self.df_tessellation)
        full_sw = mm.theil(self.df_tessellation, 'area', spatial_weights=sw)
        assert full_sw[0] == 0.2574468431886534
        limit = mm.theil(self.df_tessellation, 'area', spatial_weights=sw, rng=(10, 90))
        assert limit[0] == 0.13302952097969373

    def test_simpson(self):
        ht = mm.simpson(self.df_tessellation, 'area', order=3)
        assert ht[0] == 0.385
        sw = Queen_higher(k=3, geodataframe=self.df_tessellation)
        ht_sw = mm.simpson(self.df_tessellation, 'area', spatial_weights=sw)
        assert ht_sw[0] == 0.385
        quan_sw = mm.simpson(self.df_tessellation, 'area', spatial_weights=sw, binning='quantiles', k=3)
        assert quan_sw[0] == 0.395
