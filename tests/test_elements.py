import momepy as mm
import geopandas as gpd
from geopandas.testing import assert_geodataframe_equal


class TestUtils:

    def setup_method(self):

        test_file_path = mm.datasets.get_path('bubenec')
        self.df_buildings = gpd.read_file(test_file_path, layer='buildings')
        self.df_tessellation = gpd.read_file(test_file_path, layer='tessellation')
        self.df_streets = gpd.read_file(test_file_path, layer='streets')

    def test_tessellation(self):
        tessellation = mm.tesselation(self.df_buildings)
        assert_geodataframe_equal(tessellation, self.df_tessellation)
