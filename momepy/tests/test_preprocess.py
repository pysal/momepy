import geopandas as gpd
import numpy as np
from shapely import affinity
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import polygonize

import momepy as mm


class TestPreprocessing:
    def setup_method(self):
        test_file_path = mm.datasets.get_path("bubenec")
        self.df_buildings = gpd.read_file(test_file_path, layer="buildings")
        self.df_tessellation = gpd.read_file(test_file_path, layer="tessellation")
        self.df_streets = gpd.read_file(test_file_path, layer="streets")
        self.df_buildings["height"] = np.linspace(10.0, 30.0, 144)
        test_file_path2 = mm.datasets.get_path("tests")
        self.df_streets_rabs = gpd.read_file(test_file_path2, layer="test_rabs")
        plgns = polygonize(self.df_streets_rabs.geometry)
        self.df_rab_polys = gpd.GeoDataFrame(
            geometry=list(plgns), crs=self.df_streets_rabs.crs
        )
        self.test_file_path3 = mm.datasets.get_path("nyc_graph", extension="graphml")

    def test_CheckTessellationInput(self):
        df = self.df_buildings
        df.loc[144, "geometry"] = Polygon([(0, 0), (0, 1), (1, 0)])
        df.loc[145, "geometry"] = MultiPoint([(0, 0), (1, 0)]).buffer(0.55)
        df.loc[146, "geometry"] = affinity.rotate(df.geometry.iloc[0], 12)
        check = mm.CheckTessellationInput(self.df_buildings)
        assert len(check.collapse) == 1
        assert len(check.split) == 1
        assert len(check.overlap) == 2

        check = mm.CheckTessellationInput(self.df_buildings, collapse=False)
        assert len(check.split) == 1
        assert len(check.overlap) == 2

        check = mm.CheckTessellationInput(self.df_buildings, split=False)
        assert len(check.collapse) == 1
        assert len(check.overlap) == 2

        check = mm.CheckTessellationInput(self.df_buildings, overlap=False)
        assert len(check.collapse) == 1
        assert len(check.split) == 1

        check = mm.CheckTessellationInput(self.df_buildings, shrink=0)
        assert len(check.collapse) == 0
        assert len(check.split) == 0
        assert len(check.overlap) == 4
