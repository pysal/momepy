import momepy as mm
import geopandas as gpd
import libpysal


class TestElements:

    def setup_method(self):

        test_file_path = mm.datasets.get_path('bubenec')
        self.df_buildings = gpd.read_file(test_file_path, layer='buildings')
        self.df_tessellation = gpd.read_file(test_file_path, layer='tessellation')
        self.df_streets = gpd.read_file(test_file_path, layer='streets')
        self.df_streets['nID'] = range(len(self.df_streets))
        self.limit = mm.buffered_limit(self.df_buildings, 50)

    def test_tessellation(self):
        tessellation = mm.tessellation(self.df_buildings, 'uID', self.limit, segment=2)
        assert len(tessellation) == len(self.df_tessellation)
        queen_corners = mm.tessellation(self.df_buildings, 'uID', self.limit, segment=2, queen_corners=True)
        w = libpysal.weights.Queen.from_dataframe(queen_corners)
        assert w.neighbors[14] == [35, 36, 13, 15, 26, 27, 28, 30, 31]

    def test_snap_street_network_edge(self):
        snapped = mm.snap_street_network_edge(self.df_streets, self.df_buildings, self.df_tessellation, 20, 70)
        assert sum(snapped.geometry.length) == 5980.041004739525

    def test_blocks(self):
        buildings, cells, blocks = mm.blocks(self.df_tessellation, self.df_streets, self.df_buildings, 'bID', 'uID')
        assert not buildings.bID.isna().any()
        assert not cells.bID.isna().any()
        assert len(blocks) == 8

    def test_get_network_id(self):
        buildings_id = mm.get_network_id(self.df_buildings, self.df_streets, 'uID', 'nID')
        assert not buildings_id.isna().any()

    def test_get_network_id_duplicate(self):
        self.df_buildings['nID'] = range(len(self.df_buildings))
        buildings_id = mm.get_network_id(self.df_buildings, self.df_streets, 'uID', 'nID')
        assert not buildings_id.isna().any()

    def test_get_node_id(self):
        nx = mm.gdf_to_nx(self.df_streets)
        nodes, edges = mm.nx_to_gdf(nx)
        self.df_buildings['nID'] = mm.get_network_id(self.df_buildings, self.df_streets, 'uID', 'nID')
        ids = mm.get_node_id(self.df_buildings, nodes, edges, 'nodeID', 'nID')
        assert not ids.isna().any()

    def test__split_lines(self):
        large = mm.buffered_limit(self.df_buildings, 100)
        dense = mm.elements._split_lines(large, 100, self.df_buildings.crs)
        small = mm.buffered_limit(self.df_buildings, 30)
        dense2 = mm.elements._split_lines(small, 100, self.df_buildings.crs)
        assert len(dense) == 237
        assert len(dense2) == 573
