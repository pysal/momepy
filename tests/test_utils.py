import momepy as mm
import geopandas as gpd
import numpy as np
import libpysal
import networkx

import pytest


class TestUtils:
    def setup_method(self):

        test_file_path = mm.datasets.get_path("bubenec")
        self.df_buildings = gpd.read_file(test_file_path, layer="buildings")
        self.df_tessellation = gpd.read_file(test_file_path, layer="tessellation")
        self.df_streets = gpd.read_file(test_file_path, layer="streets")
        self.df_buildings["height"] = np.linspace(10.0, 30.0, 144)

    def test_dataset_missing(self):
        with pytest.raises(ValueError):
            mm.datasets.get_path("sffgkt")

    def test_sw_high(self):
        first_order = libpysal.weights.Queen.from_dataframe(self.df_tessellation)
        from_sw = mm.sw_high(2, gdf=None, weights=first_order)
        from_df = mm.sw_high(2, gdf=self.df_tessellation)
        rook = mm.sw_high(2, gdf=self.df_tessellation, contiguity="rook")
        check = [133, 134, 111, 112, 113, 114, 115, 121, 125]
        assert from_sw.neighbors[0] == check
        assert from_df.neighbors[0] == check
        assert rook.neighbors[0] == check

        with pytest.raises(AttributeError):
            mm.sw_high(2, gdf=None, weights=None)

        with pytest.raises(ValueError):
            mm.sw_high(2, gdf=self.df_tessellation, contiguity="nonexistent")

    def test_gdf_to_nx(self):
        nx = mm.gdf_to_nx(self.df_streets)
        assert nx.number_of_nodes() == 29
        assert nx.number_of_edges() == 35
        dual = mm.gdf_to_nx(self.df_streets, approach="dual")
        assert dual.number_of_nodes() == 35
        assert dual.number_of_edges() == 148
        with pytest.raises(ValueError):
            mm.gdf_to_nx(self.df_streets, approach="nonexistent")

    def test_nx_to_gdf(self):
        nx = mm.gdf_to_nx(self.df_streets)
        nodes, edges, W = mm.nx_to_gdf(nx, spatial_weights=True)
        assert len(nodes) == 29
        assert len(edges) == 35
        assert W.n == 29
        nodes, edges = mm.nx_to_gdf(nx)
        assert len(nodes) == 29
        assert len(edges) == 35
        edges = mm.nx_to_gdf(nx, points=False)
        assert len(edges) == 35
        nodes, W = mm.nx_to_gdf(nx, lines=False, spatial_weights=True)
        assert len(nodes) == 29
        assert W.n == 29
        nodes = mm.nx_to_gdf(nx, lines=False, spatial_weights=False)
        assert len(nodes) == 29
        dual = mm.gdf_to_nx(self.df_streets, approach="dual")
        edges = mm.nx_to_gdf(dual)
        assert len(edges) == 35
        dual.graph["approach"] = "nonexistent"
        with pytest.raises(ValueError):
            mm.nx_to_gdf(dual)
        G = networkx.Graph()
        with pytest.raises(KeyError):
            mm.nx_to_gdf(G)

    def test_limit_range(self):
        assert mm.limit_range(range(10), rng=(25, 75)) == [2, 3, 4, 5, 6, 7]
        assert mm.limit_range(range(10), rng=(10, 90)) == [1, 2, 3, 4, 5, 6, 7, 8]
        assert mm.limit_range([0, 1], rng=(25, 75)) == [0, 1]

    def test_preprocess(self):
        test_file_path2 = mm.datasets.get_path("tests")
        self.os_buildings = gpd.read_file(test_file_path2, layer="os")
        processed = mm.preprocess(self.os_buildings)
        assert len(processed) == 5

    def test_network_false_nodes(self):
        test_file_path2 = mm.datasets.get_path("tests")
        self.false_network = gpd.read_file(test_file_path2, layer="network")
        fixed = mm.network_false_nodes(self.false_network)
        assert len(fixed) == 55

    def test_snap_street_network_edge(self):
        snapped = mm.snap_street_network_edge(
            self.df_streets, self.df_buildings, 20, self.df_tessellation, 70
        )
        snapped_nonedge = mm.snap_street_network_edge(
            self.df_streets, self.df_buildings, 20
        )
        snapped_edge = mm.snap_street_network_edge(
            self.df_streets,
            self.df_buildings,
            20,
            tolerance_edge=70,
            edge=mm.buffered_limit(self.df_buildings, buffer=50),
        )
        assert sum(snapped.geometry.length) == 5980.041004739525
        assert sum(snapped_edge.geometry.length) == 5980.718889937014
        assert sum(snapped_nonedge.geometry.length) < 5980.041004739525
