import geopandas as gpd
import numpy as np

import momepy as mm


class TimeShape:
    def setup(self):
        test_file_path = mm.datasets.get_path("bubenec")
        self.df_buildings = gpd.read_file(test_file_path, layer="buildings")
        self.df_streets = gpd.read_file(test_file_path, layer="streets")
        self.df_tessellation = gpd.read_file(test_file_path, layer="tessellation")
        self.df_buildings["height"] = np.linspace(10.0, 30.0, 144)
        self.df_buildings["volume"] = mm.Volume(self.df_buildings, "height").series
        self.df_buildings["peri"] = self.df_buildings.geometry.length
        self.df_buildings["area"] = self.df_buildings.geometry.area
        self.df_buildings["cas"] = mm.CourtyardArea(self.df_buildings).series
        self.df_buildings["la"] = mm.LongestAxisLength(self.df_buildings).series

    def time_FormFactor(self):
        mm.FormFactor(self.df_buildings, "volume")

    def time_FractalDimension(self):
        mm.FractalDimension(self.df_buildings)

    def time_VolumeFacadeRatio(self):
        mm.VolumeFacadeRatio(self.df_buildings, "height", "volume", "peri")

    def time_CircularCompactness(self):
        mm.CircularCompactness(self.df_buildings, "area")

    def time_SquareCompactness(self):
        mm.SquareCompactness(self.df_buildings)

    def time_Convexity(self):
        mm.Convexity(self.df_buildings)

    def time_CourtyardIndex(self):
        mm.CourtyardIndex(self.df_buildings, "cas")

    def time_Rectangularity(self):
        mm.Rectangularity(self.df_buildings)

    def time_ShapeIndex(self):
        mm.ShapeIndex(self.df_buildings, "la")

    def time_Corners(self):
        mm.Corners(self.df_buildings)

    def time_Squareness(self):
        mm.Squareness(self.df_buildings)

    def time_EquivalentRectangularIndex(self):
        mm.EquivalentRectangularIndex(self.df_buildings)

    def time_Elongation(self):
        mm.Elongation(self.df_buildings)

    def time_CentroidCorners(self):
        mm.CentroidCorners(self.df_buildings)

    def time_Linearity(self):
        mm.Linearity(self.df_streets)

    def time_CompactnessWeightedAxis(self):
        mm.CompactnessWeightedAxis(self.df_buildings, "area", "peri", "la")
