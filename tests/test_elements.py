import momepy as mm
import geopandas as gpd
import libpysal


class TestUtils:

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
        assert w.neighbors[14] == [35, 36, 13, 15, 26, 27, 28, 30, 31]

    def test_snap_street_network_edge(self):
        snapped = mm.snap_street_network_edge(self.df_streets, self.df_buildings, self.df_tessellation, 20, 70)
        assert sum(snapped.geometry.length) == 5980.041004739526
