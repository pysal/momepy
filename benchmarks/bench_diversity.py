import geopandas as gpd
import numpy as np

import momepy as mm


class TimeDiversity:
    def setup(self):
        test_file_path = mm.datasets.get_path("bubenec")
        self.df_buildings = gpd.read_file(test_file_path, layer="buildings")
        self.df_streets = gpd.read_file(test_file_path, layer="streets")
        self.df_tessellation = gpd.read_file(test_file_path, layer="tessellation")
        self.df_buildings["height"] = np.linspace(10.0, 30.0, 144)
        self.df_tessellation["area"] = mm.Area(self.df_tessellation).series
        self.sw = mm.sw_high(k=3, gdf=self.df_tessellation, ids="uID")
        self.df_tessellation["cat"] = list(range(8)) * 18

    def time_Range(self):
        mm.Range(self.df_tessellation, "area", self.sw, "uID", (25, 75))

    def time_Theil(self):
        mm.Theil(self.df_tessellation, "area", self.sw, "uID")

    def time_Gini(self):
        mm.Gini(self.df_tessellation, "area", self.sw, "uID")

    def time_Unique(self):
        mm.Unique(self.df_tessellation, "cat", self.sw, "uID")


class TimeDiversityBinning:
    param_names = ["binning"]
    params = [("HeadTailBreaks", "Quantiles", "EqualInterval")]

    def setup(self, *args):
        test_file_path = mm.datasets.get_path("bubenec")
        self.df_buildings = gpd.read_file(test_file_path, layer="buildings")
        self.df_streets = gpd.read_file(test_file_path, layer="streets")
        self.df_tessellation = gpd.read_file(test_file_path, layer="tessellation")
        self.df_buildings["height"] = np.linspace(10.0, 30.0, 144)
        self.df_tessellation["area"] = mm.Area(self.df_tessellation).series
        self.sw = mm.sw_high(k=3, gdf=self.df_tessellation, ids="uID")

    def time_Simpson(self, binning):
        mm.Simpson(self.df_tessellation, "area", self.sw, "uID", binning)

    def time_Shannon(self, binning):
        mm.Shannon(self.df_tessellation, "area", self.sw, "uID", binning)
