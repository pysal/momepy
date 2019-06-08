import momepy as mm
import geopandas as gpd
import libpysal


class TestElements:

    def setup_method(self):

        test_file_path = mm.datasets.get_path('bubenec')
        self.df_buildings = gpd.read_file(test_file_path, layer='buildings')
        self.df_tessellation = gpd.read_file(test_file_path, layer='tessellation')
        self.df_streets = gpd.read_file(test_file_path, layer='streets')

    def test_tessellation(self):
        tessellation = mm.tessellation(self.df_buildings)
        assert len(tessellation) == len(self.df_tessellation)
        queen_corners = mm.tessellation(self.df_buildings, queen_corners=True)
        w = libpysal.weights.Queen.from_dataframe(queen_corners)
        assert w.neighbors[14] == [35, 36, 13, 15, 26, 27, 28, 30]

    def test_snap_street_network_edge(self):
        snapped = mm.snap_street_network_edge(self.df_streets, self.df_buildings, self.df_tessellation, 20, 70)
        assert sum(snapped.geometry.length) == 5980.041004739525

    def test_blocks(self):
        buildings, cells, blocks = mm.blocks(self.df_tessellation, self.df_streets, self.df_buildings, 'bID', 'uID')
        assert not buildings.bID.isna().any()
        assert not cells.bID.isna().any()
        assert len(blocks) == 8

    def test_get_network_id(self):
        self.df_streets['nID'] = range(len(self.df_streets))
        buildings_id, tessellation_id = mm.get_network_id(self.df_buildings, self.df_streets, 'uID', 'nID', self.df_tessellation, 'uID')
        assert not buildings_id.isna().any()
        assert not tessellation_id.isna().any()

    def test_get_node_id(self):
        nx = mm.gdf_to_nx(self.df_streets)
        nodes = mm.nx_to_gdf(nx, edges=False)
        nodes['nodeid'] = range(len(nodes))
        ids = mm.get_node_id(self.df_buildings, nodes, 'nodeid')
        assert not ids.isna().any()
