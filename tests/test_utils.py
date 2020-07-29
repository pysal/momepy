import geopandas as gpd
import libpysal
import momepy as mm
import networkx
import numpy as np
import pytest
import osmnx as ox

from shapely.geometry import Polygon, MultiPoint, LineString
from shapely import affinity


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

    def test_gdf_to_nx(self):
        nx = mm.gdf_to_nx(self.df_streets)
        assert nx.number_of_nodes() == 29
        assert nx.number_of_edges() == 35
        dual = mm.gdf_to_nx(self.df_streets, approach="dual")
        assert dual.number_of_nodes() == 35
        assert dual.number_of_edges() == 74
        self.df_streets["ix"] = np.arange(0, len(self.df_streets) * 2, 2)
        self.df_streets.set_index("ix", inplace=True)
        dual2 = mm.gdf_to_nx(self.df_streets, approach="dual")
        assert dual2.number_of_nodes() == 35
        assert dual2.number_of_edges() == 74
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

        # check graph without attributes
        G = networkx.MultiGraph()
        key = 0
        for index, row in self.df_streets.iterrows():
            first = row.geometry.coords[0]
            last = row.geometry.coords[-1]

            data = [row[f] for f in list(self.df_streets.columns)]
            attributes = dict(zip(list(self.df_streets.columns), data))
            G.add_edge(first, last, key=key, **attributes)
            key += 1
        nodes, edges = mm.nx_to_gdf(G)
        assert len(nodes) == 29
        assert len(edges) == 35

        # osmnx compatibility
        G = ox.graph_from_place("Preborov, Czechia", network_type="drive")
        pts, lines = mm.nx_to_gdf(G)
        assert len(pts) == 7
        assert len(lines) == 16

        # LineString Z
        line1 = LineString([(0, 0, 0), (1, 1, 1)])
        line2 = LineString([(0, 0, 0), (-1, -1, -1)])
        gdf = gpd.GeoDataFrame(geometry=[line1, line2])
        G = mm.gdf_to_nx(gdf)
        pts, lines = mm.nx_to_gdf(G)
        assert pts.iloc[0].geometry.wkt == "POINT Z (0 0 0)"
        assert lines.iloc[0].geometry.wkt == "LINESTRING Z (0 0 0, 1 1 1)"

    def test_limit_range(self):
        assert list(mm.limit_range(range(10), rng=(25, 75))) == [2, 3, 4, 5, 6, 7]
        assert list(mm.limit_range(range(10), rng=(10, 90))) == [1, 2, 3, 4, 5, 6, 7, 8]
        assert list(mm.limit_range([0, 1], rng=(25, 75))) == [0, 1]

    def test_preprocess(self):
        test_file_path2 = mm.datasets.get_path("tests")
        self.os_buildings = gpd.read_file(test_file_path2, layer="os")
        processed = mm.preprocess(self.os_buildings)
        assert len(processed) == 5

    def test_network_false_nodes(self):
        test_file_path2 = mm.datasets.get_path("tests")
        self.false_network = gpd.read_file(test_file_path2, layer="network")
        self.false_network["vals"] = range(len(self.false_network))
        fixed = mm.network_false_nodes(self.false_network)
        assert len(fixed) == 56
        assert isinstance(fixed, gpd.GeoDataFrame)
        assert self.false_network.crs.equals(fixed.crs)
        assert sorted(self.false_network.columns) == sorted(fixed.columns)
        fixed_series = mm.network_false_nodes(self.false_network.geometry)
        assert len(fixed_series) == 56
        assert isinstance(fixed_series, gpd.GeoSeries)
        assert self.false_network.crs.equals(fixed_series.crs)
        with pytest.raises(TypeError):
            mm.network_false_nodes(list())
        multiindex = self.false_network.explode()
        fixed_multiindex = mm.network_false_nodes(multiindex)
        assert len(fixed_multiindex) == 56
        assert isinstance(fixed, gpd.GeoDataFrame)
        assert sorted(self.false_network.columns) == sorted(fixed.columns)

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

    def test_CheckTessellationInput(self):
        df = self.df_buildings
        df.loc[144, "geometry"] = Polygon([(0, 0), (0, 1), (1, 0)])
        df.loc[145, "geometry"] = MultiPoint([(0, 0), (1, 0)]).buffer(0.55)
        df.loc[146, "geometry"] = affinity.rotate(df.geometry.iloc[0], 12)
        check = mm.CheckTessellationInput(self.df_buildings)
        assert len(check.collapse) == 1
        assert len(check.split) == 1
        assert len(check.overlap) == 2

        check = mm.CheckTessellationInput(self.df_buildings, collapse=False)
        assert len(check.split) == 1
        assert len(check.overlap) == 2

        check = mm.CheckTessellationInput(self.df_buildings, split=False)
        assert len(check.collapse) == 1
        assert len(check.overlap) == 2

        check = mm.CheckTessellationInput(self.df_buildings, overlap=False)
        assert len(check.collapse) == 1
        assert len(check.split) == 1

        check = mm.CheckTessellationInput(self.df_buildings, shrink=0)
        assert len(check.collapse) == 0
        assert len(check.split) == 0
        assert len(check.overlap) == 4
