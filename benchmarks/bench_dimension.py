import geopandas as gpd
import numpy as np

import momepy as mm


class TimeDimension:
    def setup(self):
        test_file_path = mm.datasets.get_path("bubenec")
        self.df_buildings = gpd.read_file(test_file_path, layer="buildings")
        self.df_streets = gpd.read_file(test_file_path, layer="streets")
        self.df_tessellation = gpd.read_file(test_file_path, layer="tessellation")
        self.df_tessellation["area"] = self.df_tessellation.geometry.area
        self.df_buildings["height"] = np.linspace(10.0, 30.0, 144)
        self.df_buildings["area"] = self.df_buildings.geometry.area
        self.spatial_weights = mm.sw_high(k=3, gdf=self.df_tessellation, ids="uID")
        self.sw = mm.sw_high(gdf=self.df_buildings, k=1)

    def time_Area(self):
        mm.Area(self.df_buildings)

    def time_Perimeter(self):
        mm.Perimeter(self.df_buildings)

    def time_Volume(self):
        mm.Volume(self.df_buildings, "height", "area")

    def time_FloorArea(self):
        mm.FloorArea(self.df_buildings, "height", "area")

    def time_CourtyardArea(self):
        mm.CourtyardArea(self.df_buildings, "area")

    def time_LongestAxisLength(self):
        mm.LongestAxisLength(self.df_buildings)

    def time_AverageCharacter(self):
        mm.AverageCharacter(
            self.df_tessellation,
            values="area",
            spatial_weights=self.spatial_weights,
            unique_id="uID",
        )

    def time_StreetProfile(self):
        mm.StreetProfile(self.df_streets, self.df_buildings, heights="height")

    def time_WeightedCharacter(self):
        mm.WeightedCharacter(self.df_buildings, "height", self.spatial_weights, "uID")

    def time_CoveredArea(self):
        mm.CoveredArea(self.df_tessellation, self.spatial_weights, "uID")

    def time_PeimerimeterWall(self):
        mm.PerimeterWall(self.df_buildings, self.sw)

    def time_SegmentsLength(self):
        mm.SegmentsLength(self.df_streets)
