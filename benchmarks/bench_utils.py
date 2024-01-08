import geopandas as gpd
import libpysal

import momepy as mm


class TimeUtils:
    def setup(self):
        test_file_path = mm.datasets.get_path("bubenec")
        self.df_buildings = gpd.read_file(test_file_path, layer="buildings")
        self.df_tessellation = gpd.read_file(test_file_path, layer="tessellation")
        self.df_streets = gpd.read_file(test_file_path, layer="streets")
        self.first_order = libpysal.weights.Queen.from_dataframe(self.df_tessellation)
        self.nx = mm.gdf_to_nx(self.df_streets)
        self.dual = mm.gdf_to_nx(self.df_streets, approach="dual")

        test_file_path2 = mm.datasets.get_path("tests")
        self.os_buildings = gpd.read_file(test_file_path2, layer="os")
        self.false_network = gpd.read_file(test_file_path2, layer="network")

    def time_sw_high(self):
        mm.sw_high(5, gdf=None, weights=self.first_order)

    def time_gdf_to_nx_primal(self):
        mm.gdf_to_nx(self.df_streets)

    def time_gdf_to_nx_dual(self):
        mm.gdf_to_nx(self.df_streets, "dual")

    def time_nx_to_gdf_primal(self):
        mm.nx_to_gdf(self.nx, spatial_weights=True)

    def time_nx_to_gdf_dual(self):
        mm.nx_to_gdf(self.dual)

    def time_preprocess(self):
        mm.preprocess(self.os_buildings)

    def time_network_false_nodes(self):
        mm.network_false_nodes(self.false_network)

    def time_remove_false_nodes(self):
        mm.remove_false_nodes(self.false_network)

    def time_CheckTessellationInput(self):
        mm.CheckTessellationInput(self.df_buildings)
