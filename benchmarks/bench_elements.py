import geopandas as gpd

import momepy as mm


class TimeElements:
    def setup(self):
        test_file_path = mm.datasets.get_path("bubenec")
        self.df_buildings = gpd.read_file(test_file_path, layer="buildings")
        self.df_tessellation = gpd.read_file(test_file_path, layer="tessellation")
        self.df_streets = gpd.read_file(test_file_path, layer="streets")
        self.df_streets["nID"] = range(len(self.df_streets))
        self.limit = mm.buffered_limit(self.df_buildings, 50)
        nx = mm.gdf_to_nx(self.df_streets)
        self.nodes, self.edges = mm.nx_to_gdf(nx)
        self.df_buildings["nID"] = mm.get_network_id(
            self.df_buildings, self.df_streets, "nID"
        )

    def time_Tessellation(self):
        mm.Tessellation(self.df_buildings, "uID", self.limit, segment=2)

    def time_Blocks(self):
        mm.Blocks(
            self.df_tessellation, self.df_streets, self.df_buildings, "bID", "uID"
        )

    def time_get_network_id(self):
        mm.get_network_id(self.df_buildings, self.df_streets, "nID")

    def time_get_node_id(self):
        mm.get_node_id(self.df_buildings, self.nodes, self.edges, "nodeID", "nID")
