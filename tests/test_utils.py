import momepy as mm
import geopandas as gpd
import numpy as np
import libpysal

from shapely.geometry import Polygon, MultiPolygon

import pytest


class TestUtils:

    def setup_method(self):

        test_file_path = mm.datasets.get_path('bubenec')
        self.df_buildings = gpd.read_file(test_file_path, layer='buildings')
        self.df_tessellation = gpd.read_file(test_file_path, layer='tessellation')
        self.df_buildings['height'] = np.linspace(10., 30., 144)

    def test_dataset_missing(self):
        with pytest.raises(ValueError):
            mm.datasets.get_path('sffgkt')

    def test_Queen_higher(self):
        first_order = libpysal.weights.Queen.from_dataframe(self.df_tessellation)
        from_sw = mm.Queen_higher(2, geodataframe=None, weights=first_order)
        from_df = mm.Queen_higher(2, geodataframe=self.df_tessellation)
        check = [133, 134, 111, 112, 113, 114, 115, 121, 125]
        assert from_sw.neighbors[0] == check
        assert from_df.neighbors[0] == check

        with pytest.raises(Warning):
            mm.Queen_higher(2, geodataframe=None, weights=None)

    def test_multi2single(self):
        polygon = Polygon([(0, 0), (1, 1), (1, 0)])
        polygon2 = Polygon([(2, 3), (1, 2), (2, 4)])
        polygons = MultiPolygon([polygon, polygon2])
        gdf = gpd.GeoDataFrame(geometry=[polygon, polygon2, polygons])
        single = mm._multi2single(gdf)
        assert len(single) == 4
