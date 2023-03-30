import geopandas as gpd
import numpy as np
from libpysal.weights import Queen

import momepy as mm


class TimeDistribution:
    def setup(self):
        test_file_path = mm.datasets.get_path("bubenec")
        self.df_buildings = gpd.read_file(test_file_path, layer="buildings")
        self.df_streets = gpd.read_file(test_file_path, layer="streets")
        self.df_tessellation = gpd.read_file(test_file_path, layer="tessellation")
        self.df_buildings["height"] = np.linspace(10.0, 30.0, 144)
        self.df_buildings["volume"] = mm.Volume(self.df_buildings, "height").series
        self.df_streets["nID"] = mm.unique_id(self.df_streets)
        self.df_buildings["nID"] = mm.get_network_id(
            self.df_buildings, self.df_streets, "nID"
        )
        self.df_buildings["orient"] = mm.Orientation(self.df_buildings).series
        self.df_tessellation["orient"] = mm.Orientation(self.df_tessellation).series
        self.sw = Queen.from_dataframe(self.df_tessellation, ids="uID")
        self.swh = mm.sw_high(k=3, gdf=self.df_tessellation, ids="uID")
        self.swb = Queen.from_dataframe(self.df_buildings, ids="uID")

    def time_Orientation(self):
        mm.Orientation(self.df_buildings)

    def time_SharedWallsRatio(self):
        mm.SharedWallsRatio(self.df_buildings)

    def time_StreetAlignment(self):
        mm.StreetAlignment(
            self.df_buildings, self.df_streets, "orient", network_id="nID"
        )

    def time_CellAlignment(self):
        mm.CellAlignment(
            self.df_buildings, self.df_tessellation, "orient", "orient", "uID", "uID"
        )

    def time_Alignment(self):
        mm.Alignment(self.df_buildings, self.sw, "uID", self.df_buildings["orient"])

    def time_NeighborDistance(self):
        mm.NeighborDistance(self.df_buildings, self.sw, "uID")

    def time_MeanInterbuildingDistance(self):
        mm.MeanInterbuildingDistance(self.df_buildings, self.sw, "uID", self.swh)

    def time_NeighboringStreetOrientationDeviation(self):
        mm.NeighboringStreetOrientationDeviation(self.df_streets)

    def time_BuildingAdjacency(self):
        mm.BuildingAdjacency(
            self.df_buildings,
            spatial_weights=self.swb,
            unique_id="uID",
            spatial_weights_higher=self.swh,
        )

    def time_Neighbors(self):
        mm.Neighbors(self.df_tessellation, self.sw, "uID")
