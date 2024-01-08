import geopandas as gpd
import numpy as np
from libpysal.weights import Queen

import momepy as mm


class TimeIntensity:
    def setup(self):
        test_file_path = mm.datasets.get_path("bubenec")
        self.df_buildings = gpd.read_file(test_file_path, layer="buildings")
        self.df_streets = gpd.read_file(test_file_path, layer="streets")
        self.df_tessellation = gpd.read_file(test_file_path, layer="tessellation")
        self.df_streets["nID"] = mm.unique_id(self.df_streets)
        self.df_buildings["height"] = np.linspace(10.0, 30.0, 144)
        self.df_tessellation["area"] = self.df_tessellation.geometry.area
        self.df_buildings["area"] = self.df_buildings.geometry.area
        self.df_buildings["fl_area"] = mm.FloorArea(self.df_buildings, "height").series
        self.df_buildings["nID"] = mm.get_network_id(
            self.df_buildings, self.df_streets, "nID"
        )
        blocks = mm.Blocks(
            self.df_tessellation, self.df_streets, self.df_buildings, "bID", "uID"
        )
        self.blocks = blocks.blocks
        self.df_buildings["bID"] = blocks.buildings_id
        self.df_tessellation["bID"] = blocks.tessellation_id
        self.swb = Queen.from_dataframe(self.df_buildings)
        self.sw5 = mm.sw_high(k=5, gdf=self.df_tessellation, ids="uID")
        self.sw3 = mm.sw_high(k=3, gdf=self.df_tessellation, ids="uID")
        self.sws = mm.sw_high(k=2, gdf=self.df_streets)
        nx = mm.gdf_to_nx(self.df_streets)
        nx = mm.node_degree(nx)
        self.nodes, self.edges, W = mm.nx_to_gdf(nx, spatial_weights=True)
        self.swn = mm.sw_high(k=3, weights=W)

    def time_AreaRatio(self):
        mm.AreaRatio(self.df_tessellation, self.df_buildings, "area", "area", "uID")

    def time_Count(self):
        mm.Count(self.blocks, self.df_buildings, "bID", "bID")

    def time_Count_weighted(self):
        mm.Count(self.blocks, self.df_buildings, "bID", "bID", weighted=True)

    def time_Courtyards(self):
        mm.Courtyards(self.df_buildings, "bID", self.swb)

    def time_BlocksCount(self):
        mm.BlocksCount(self.df_tessellation, "bID", self.sw5, "uID")

    def time_Reached(self):
        mm.Reached(self.df_streets, self.df_buildings, "nID", "nID")

    def time_Reached_sw(self):
        mm.Reached(self.df_streets, self.df_buildings, "nID", "nID", self.sws)

    def time_NodeDensity(self):
        mm.NodeDensity(self.nodes, self.edges, self.swn)

    def time_Density(self):
        mm.Density(
            self.df_tessellation,
            self.df_buildings["fl_area"],
            self.sw3,
            "uID",
        )
