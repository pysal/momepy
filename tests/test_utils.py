import momepy as mm
import geopandas as gpd
import numpy as np
import libpysal

from shapely.geometry import Polygon, MultiPolygon

import pytest


class TestUtils:

    def setup_method(self):

        test_file_path = mm.datasets.get_path('bubenec')
        self.df_buildings = gpd.read_file(test_file_path, layer='buildings')
        self.df_tessellation = gpd.read_file(test_file_path, layer='tessellation')
        self.df_streets = gpd.read_file(test_file_path, layer='streets')
        self.df_buildings['height'] = np.linspace(10., 30., 144)

    def test_dataset_missing(self):
        with pytest.raises(ValueError):
            mm.datasets.get_path('sffgkt')

    def test_Queen_higher(self):
        first_order = libpysal.weights.Queen.from_dataframe(self.df_tessellation)
        from_sw = mm.Queen_higher(2, geodataframe=None, weights=first_order)
        from_df = mm.Queen_higher(2, geodataframe=self.df_tessellation)
        check = [133, 134, 111, 112, 113, 114, 115, 121, 125]
        assert from_sw.neighbors[0] == check
        assert from_df.neighbors[0] == check

        with pytest.raises(Warning):
            mm.Queen_higher(2, geodataframe=None, weights=None)

        with pytest.raises(ValueError):
            gdf = self.df_tessellation
            gdf.index = gdf.index + 20
            mm.Queen_higher(2, geodataframe=gdf, weights=None)

    def test_multi2single(self):
        polygon = Polygon([(0, 0), (1, 1), (1, 0)])
        polygon2 = Polygon([(2, 3), (1, 2), (2, 4)])
        polygons = MultiPolygon([polygon, polygon2])
        gdf = gpd.GeoDataFrame(geometry=[polygon, polygon2, polygons])
        single = mm.multi2single(gdf)
        assert len(single) == 4

    def test_gdf_to_nx(self):
        nx = mm.gdf_to_nx(self.df_streets)
        assert nx.number_of_nodes() == 32
        assert nx.number_of_edges() == 33

    def test_nx_to_gdf(self):
        nx = mm.gdf_to_nx(self.df_streets)
        nodes, edges, W = mm.nx_to_gdf(nx, spatial_weights=True)
        assert len(nodes) == 32
        assert len(edges) == 33
        assert W.n == 32
        nodes, edges = mm.nx_to_gdf(nx)
        assert len(nodes) == 32
        assert len(edges) == 33
        edges = mm.nx_to_gdf(nx, nodes=False)
        assert len(edges) == 33
        nodes, W = mm.nx_to_gdf(nx, edges=False, spatial_weights=True)
        assert len(nodes) == 32
        assert W.n == 32
        nodes = mm.nx_to_gdf(nx, edges=False, spatial_weights=False)
        assert len(nodes) == 32

    def test_limit_range(self):
        assert mm.limit_range(range(10), rng=(25, 75)) == [3, 4, 5, 6]
        assert mm.limit_range(range(10), rng=(10, 90)) == [1, 2, 3, 4, 5, 6, 7, 8]

    def test_preprocess(self):
        test_file_path2 = mm.datasets.get_path('tests')
        self.os_buildings = gpd.read_file(test_file_path2, layer='os')
        processed = mm.preprocess(self.os_buildings)
        assert len(processed) == 5
