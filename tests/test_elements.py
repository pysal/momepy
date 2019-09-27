import geopandas as gpd
import libpysal
import momepy as mm


class TestElements:
    def setup_method(self):

        test_file_path = mm.datasets.get_path("bubenec")
        self.df_buildings = gpd.read_file(test_file_path, layer="buildings")
        self.df_tessellation = gpd.read_file(test_file_path, layer="tessellation")
        self.df_streets = gpd.read_file(test_file_path, layer="streets")
        self.df_streets["nID"] = range(len(self.df_streets))
        self.limit = mm.buffered_limit(self.df_buildings, 50)

    def test_Tessellation(self):
        tes = mm.Tessellation(self.df_buildings, "uID", self.limit, segment=2)
        tessellation = tes.tessellation
        assert len(tessellation) == len(self.df_tessellation)
        bands = mm.Tessellation(
            self.df_streets, "nID", mm.buffered_limit(self.df_streets, 50), segment=5
        ).tessellation
        assert len(bands) == len(self.df_streets)
        queen_corners = tes.queen_corners(2)
        w = libpysal.weights.Queen.from_dataframe(queen_corners)
        assert w.neighbors[14] == [35, 36, 13, 15, 26, 27, 28, 30, 31]

    def test_Blocks(self):
        blocks = mm.Blocks(
            self.df_tessellation, self.df_streets, self.df_buildings, "bID", "uID"
        )
        assert not blocks.tessellation_id.isna().any()
        assert not blocks.buildings_id.isna().any()
        assert len(blocks.blocks) == 8

    def test_get_network_id(self):
        buildings_id = mm.get_network_id(
            self.df_buildings, self.df_streets, "nID"
        )
        assert not buildings_id.isna().any()

    def test_get_network_id_duplicate(self):
        self.df_buildings["nID"] = range(len(self.df_buildings))
        buildings_id = mm.get_network_id(
            self.df_buildings, self.df_streets, "nID"
        )
        assert not buildings_id.isna().any()

    def test_get_node_id(self):
        nx = mm.gdf_to_nx(self.df_streets)
        nodes, edges = mm.nx_to_gdf(nx)
        self.df_buildings["nID"] = mm.get_network_id(
            self.df_buildings, self.df_streets, "nID"
        )
        ids = mm.get_node_id(self.df_buildings, nodes, edges, "nodeID", "nID")
        assert not ids.isna().any()

    def test__split_lines(self):
        large = mm.buffered_limit(self.df_buildings, 100)
        dense = mm.elements._split_lines(large, 100)
        small = mm.buffered_limit(self.df_buildings, 30)
        dense2 = mm.elements._split_lines(small, 100)
        assert len(dense) == 53
        assert len(dense2) == 51
