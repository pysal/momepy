import geopandas as gpd
import networkx
import numpy as np
import pytest
from packaging.version import Version
from shapely.geometry import LineString, Point

import momepy as mm

# https://github.com/geopandas/geopandas/issues/2282
GPD_REGR = Version("0.10.2") < Version(gpd.__version__) < Version("0.11")


class TestUtils:
    def setup_method(self):
        test_file_path = mm.datasets.get_path("bubenec")
        self.df_buildings = gpd.read_file(test_file_path, layer="buildings")
        self.df_tessellation = gpd.read_file(test_file_path, layer="tessellation")
        self.df_streets = gpd.read_file(test_file_path, layer="streets")
        self.df_buildings["height"] = np.linspace(10.0, 30.0, 144)
        self.df_points = gpd.GeoDataFrame(
            data=None, geometry=[Point(0, 0), Point(1, 1)]
        )
        self.df_points_and_linestring = gpd.GeoDataFrame(
            data=None,
            geometry=[Point(0, 0), Point(1, 1), LineString([(1, 0), (0, 0), (1, 1)])],
        )

    def test_dataset_missing(self):
        with pytest.raises(ValueError, match="The dataset 'sffgkt' is not available."):
            mm.datasets.get_path("sffgkt")

    def test_gdf_to_nx(self):
        with pytest.warns(
            RuntimeWarning, match="The given network does not contain any LineString."
        ):
            mm.gdf_to_nx(self.df_points)

        with pytest.warns(
            RuntimeWarning,
            match="The given network consists of multiple geometry types.",
        ):
            mm.gdf_to_nx(self.df_points_and_linestring)

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
        with pytest.raises(
            ValueError, match="Approach 'nonexistent' is not supported."
        ):
            mm.gdf_to_nx(self.df_streets, approach="nonexistent")

        nx = mm.gdf_to_nx(self.df_streets, multigraph=False)
        assert isinstance(nx, networkx.Graph)
        assert nx.number_of_nodes() == 29
        assert nx.number_of_edges() == 35

        nx = mm.gdf_to_nx(self.df_streets, multigraph=False, directed=True)
        assert isinstance(nx, networkx.DiGraph)
        assert nx.number_of_nodes() == 29
        assert nx.number_of_edges() == 35

        nx = mm.gdf_to_nx(self.df_streets, directed=True)
        assert isinstance(nx, networkx.MultiDiGraph)
        assert nx.number_of_nodes() == 29
        assert nx.number_of_edges() == 35

        self.df_streets["oneway"] = True
        self.df_streets.loc[0, "oneway"] = False  # first road section is bidirectional
        nx = mm.gdf_to_nx(self.df_streets, directed=True, oneway_column="oneway")
        assert nx.number_of_edges() == 36

        with pytest.raises(ValueError, match="Bidirectional lines"):
            mm.gdf_to_nx(self.df_streets, directed=False, oneway_column="oneway")

        dual = mm.gdf_to_nx(self.df_streets, approach="dual", angles=False)
        assert (
            dual.edges[
                (1603499.42326969, 6464328.7520580515),
                (1603510.1061735682, 6464204.555117119),
                0,
            ]
            == {}
        )

        dual = mm.gdf_to_nx(self.df_streets, approach="dual", angle="ang")
        assert dual.edges[
            (1603499.42326969, 6464328.7520580515),
            (1603510.1061735682, 6464204.555117119),
            0,
        ] == {"ang": 117.18288698243317}

        dual = mm.gdf_to_nx(
            self.df_streets, approach="dual", angles=False, multigraph=False
        )
        assert isinstance(nx, networkx.Graph)
        assert (
            dual.edges[
                (1603499.42326969, 6464328.7520580515),
                (1603510.1061735682, 6464204.555117119),
            ]
            == {}
        )

        dual = mm.gdf_to_nx(self.df_streets, approach="dual", multigraph=False)
        assert isinstance(nx, networkx.Graph)
        assert dual.edges[
            (1603499.42326969, 6464328.7520580515),
            (1603510.1061735682, 6464204.555117119),
        ] == {"angle": 117.18288698243317}

        with pytest.raises(ValueError, match="Directed graphs are not supported"):
            mm.gdf_to_nx(self.df_streets, approach="dual", directed=True)

    @pytest.mark.skipif(GPD_REGR, reason="regression in geopandas")
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
        with pytest.raises(
            ValueError, match="Approach 'nonexistent' is not supported."
        ):
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

        with pytest.warns(UserWarning, match="Approach is not set"):
            nodes, edges = mm.nx_to_gdf(G)
        assert len(nodes) == 29
        assert len(edges) == 35

        # LineString Z
        line1 = LineString([(0, 0, 0), (1, 1, 1)])
        line2 = LineString([(0, 0, 0), (-1, -1, -1)])
        gdf = gpd.GeoDataFrame(geometry=[line1, line2])
        G = mm.gdf_to_nx(gdf)
        pts, lines = mm.nx_to_gdf(G)
        assert pts.iloc[0].geometry.wkt == "POINT Z (0 0 0)"
        assert lines.iloc[0].geometry.wkt == "LINESTRING Z (0 0 0, 1 1 1)"

    @pytest.mark.xfail(reason="nominatim connection error")
    def test_nx_to_gdf_osmnx(self):
        osmnx = pytest.importorskip("osmnx")
        # osmnx compatibility
        G = osmnx.graph_from_place("Preborov, Czechia", network_type="drive")
        with pytest.warns(UserWarning, match="Approach is not set"):
            pts, lines = mm.nx_to_gdf(G)
        assert len(pts) == 7
        assert len(lines) == 16

    def test_limit_range(self):
        assert list(mm.limit_range(range(10), rng=(25, 75))) == [2, 3, 4, 5, 6, 7]
        assert list(mm.limit_range(range(10), rng=(10, 90))) == [1, 2, 3, 4, 5, 6, 7, 8]
        assert list(mm.limit_range([0, 1], rng=(25, 75))) == [0, 1]
        assert list(
            mm.limit_range(np.array([0, 1, 2, 3, 4, np.nan]), rng=(25, 75))
        ) == [1, 2, 3]
        np.testing.assert_array_equal(
            mm.limit_range(np.array([np.nan, np.nan, np.nan]), rng=(25, 75)),
            np.array([np.nan, np.nan, np.nan]),
        )
