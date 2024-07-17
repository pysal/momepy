import geopandas as gpd
import numpy as np
import pytest
from geopandas.testing import assert_geodataframe_equal
from shapely import affinity
from shapely.geometry import LineString, MultiPoint, Point, Polygon
from shapely.ops import polygonize

import momepy as mm


class TestPreprocessing:
    def setup_method(self):
        test_file_path = mm.datasets.get_path("bubenec")
        self.df_buildings = gpd.read_file(test_file_path, layer="buildings")
        self.df_tessellation = gpd.read_file(test_file_path, layer="tessellation")
        self.df_streets = gpd.read_file(test_file_path, layer="streets")
        self.df_buildings["height"] = np.linspace(10.0, 30.0, 144)
        test_file_path2 = mm.datasets.get_path("tests")
        self.df_streets_rabs = gpd.read_file(test_file_path2, layer="test_rabs")
        plgns = polygonize(self.df_streets_rabs.geometry)
        self.df_rab_polys = gpd.GeoDataFrame(
            geometry=list(plgns), crs=self.df_streets_rabs.crs
        )
        self.test_file_path3 = mm.datasets.get_path("nyc_graph", extension="graphml")

    def test_preprocess(self):
        test_file_path2 = mm.datasets.get_path("tests")
        self.os_buildings = gpd.read_file(test_file_path2, layer="os")
        processed = mm.preprocess(self.os_buildings)
        assert len(processed) == 5

    def test_remove_false_nodes(self):
        test_file_path2 = mm.datasets.get_path("tests")
        self.false_network = gpd.read_file(test_file_path2, layer="network")
        self.false_network["vals"] = range(len(self.false_network))
        fixed = mm.remove_false_nodes(self.false_network)
        assert len(fixed) == 56
        assert isinstance(fixed, gpd.GeoDataFrame)
        assert self.false_network.crs.equals(fixed.crs)
        assert sorted(self.false_network.columns) == sorted(fixed.columns)

        # check loop order
        expected = np.array(
            [
                [-727238.49292668, -1052817.28071986],
                [-727253.1752498, -1052827.47329062],
                [-727223.93217677, -1052829.47624082],
                [-727238.49292668, -1052817.28071986],
            ]
        )
        np.testing.assert_almost_equal(
            np.array(fixed.loc[53].geometry.coords), expected
        )

        fixed_series = mm.remove_false_nodes(self.false_network.geometry)
        assert len(fixed_series) == 56
        assert isinstance(fixed_series, gpd.GeoSeries)
        assert self.false_network.crs.equals(fixed_series.crs)
        multiindex = self.false_network.explode(index_parts=True)
        fixed_multiindex = mm.remove_false_nodes(multiindex)
        assert len(fixed_multiindex) == 56
        assert isinstance(fixed, gpd.GeoDataFrame)
        assert sorted(self.false_network.columns) == sorted(fixed.columns)

        # no node of a degree 2
        df = self.df_streets.drop([4, 7, 17, 22])
        assert_geodataframe_equal(df, mm.remove_false_nodes(df))

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

    def test_close_gaps(self):
        l1 = LineString([(1, 0), (2, 1)])
        l2 = LineString([(2.1, 1), (3, 2)])
        l3 = LineString([(3.1, 2), (4, 0)])
        l4 = LineString([(4.1, 0), (5, 0)])
        l5 = LineString([(5.1, 0), (6, 0)])
        df = gpd.GeoDataFrame(geometry=[l1, l2, l3, l4, l5])

        closed = mm.close_gaps(df, 0.25)
        assert len(closed) == len(df)

        merged = mm.remove_false_nodes(closed)
        assert len(merged) == 1
        assert merged.length[0] == pytest.approx(7.0502, rel=1e-3)

    def test_extend_lines(self):
        l1 = LineString([(1, 0), (1.9, 0)])
        l2 = LineString([(2.1, -1), (2.1, 1)])
        l3 = LineString([(2, 1.1), (3, 1.1)])
        gdf = gpd.GeoDataFrame([1, 2, 3], geometry=[l1, l2, l3])

        ext1 = mm.extend_lines(gdf, 2)
        assert ext1.length.sum() > gdf.length.sum()
        assert ext1.length.sum() == pytest.approx(4.2, rel=1e-3)

        target = gpd.GeoSeries([l2.centroid.buffer(3)])
        ext2 = mm.extend_lines(gdf, 3, target)

        assert ext2.length.sum() > gdf.length.sum()
        assert ext2.length.sum() == pytest.approx(17.3776, rel=1e-3)

        barrier = LineString([(2, -1), (2, 1)])
        ext3 = mm.extend_lines(gdf, 2, barrier=gpd.GeoSeries([barrier]))

        assert ext3.length.sum() > gdf.length.sum()
        assert ext3.length.sum() == pytest.approx(4, rel=1e-3)

        ext4 = mm.extend_lines(gdf, 2, extension=1)
        assert ext4.length.sum() > gdf.length.sum()
        assert ext4.length.sum() == pytest.approx(10.2, rel=1e-3)

        gdf = gpd.GeoDataFrame([1, 2, 3, 4], geometry=[l1, l2, l3, barrier])
        ext5 = mm.extend_lines(gdf, 2)
        assert ext5.length.sum() > gdf.length.sum()
        assert ext5.length.sum() == pytest.approx(6.2, rel=1e-3)

    def test_roundabout_simplification_point_error(self):
        point_df = gpd.GeoDataFrame({"nID": [0]}, geometry=[Point(0, 0)])
        with pytest.raises(TypeError, match="Only LineString geometries are allowed."):
            mm.roundabout_simplification(point_df)

    def test_roundabout_simplification_default(self):
        check = mm.roundabout_simplification(self.df_streets_rabs)
        assert len(check) == 65
        assert len(self.df_streets_rabs) == 88  # checking that nothing has changed

    def test_roundabout_simplification_high_circom_threshold(self):
        check = mm.roundabout_simplification(
            self.df_streets_rabs, self.df_rab_polys, circom_threshold=0.97
        )
        assert len(check) == 77
        assert len(self.df_streets_rabs) == 88

    def test_roundabout_simplification_low_area_threshold(self):
        check = mm.roundabout_simplification(
            self.df_streets_rabs, self.df_rab_polys, area_threshold=0.8
        )
        assert len(check) == 67
        assert len(self.df_streets_rabs) == 88

    def test_roundabout_simplification_exclude_adjacent(self):
        check = mm.roundabout_simplification(
            self.df_streets_rabs, self.df_rab_polys, include_adjacent=False
        )
        assert len(check) == 88
        assert len(self.df_streets_rabs) == 88

    def test_roundabout_simplification_center_type_mean(self):
        check = mm.roundabout_simplification(
            self.df_streets_rabs, self.df_rab_polys, center_type="mean"
        )
        assert len(check) == 65
        assert len(self.df_streets_rabs) == 88

    @pytest.mark.parametrize("method", ["spider", "euclidean", "extend"])
    def test_consolidate_intersections(self, method):
        ox = pytest.importorskip("osmnx")
        graph = ox.convert.to_undirected(ox.load_graphml(self.test_file_path3))

        tol = 30
        graph_simplified = mm.consolidate_intersections(
            graph,
            tolerance=tol,
            rebuild_graph=True,
            rebuild_edges_method=method,
        )
        nodes_simplified, edges_simplified = mm.nx_to_gdf(graph_simplified)

        assert len(nodes_simplified) == 39
        assert len(edges_simplified) == 66

        if method != "euclidean":
            assert edges_simplified.length.min() >= tol

    def test_consolidate_intersections_unsupported(self):
        ox = pytest.importorskip("osmnx")
        graph = ox.convert.to_undirected(ox.load_graphml(self.test_file_path3))
        with pytest.raises(ValueError, match="Simplification 'banana' not recognized"):
            mm.consolidate_intersections(
                graph,
                tolerance=30,
                rebuild_graph=True,
                rebuild_edges_method="banana",
            )


def test_FaceArtifacts():
    pytest.importorskip("esda")
    osmnx = pytest.importorskip("osmnx")
    type_filter = (
        '["highway"~"living_street|motorway|motorway_link|pedestrian|primary'
        "|primary_link|residential|secondary|secondary_link|service|tertiary"
        '|tertiary_link|trunk|trunk_link|unclassified|service"]'
    )
    streets_graph = osmnx.graph_from_point(
        (35.7798, -78.6421),
        dist=1000,
        network_type="all_private",
        custom_filter=type_filter,
        retain_all=True,
        simplify=False,
    )
    streets_graph = osmnx.projection.project_graph(streets_graph)
    gdf = osmnx.graph_to_gdfs(
        osmnx.convert.to_undirected(streets_graph),
        nodes=False,
        edges=True,
        node_geometry=False,
        fill_edge_geometry=True,
    )
    fa = mm.FaceArtifacts(gdf)
    assert 6 < fa.threshold < 9
    assert isinstance(fa.face_artifacts, gpd.GeoDataFrame)
    assert fa.face_artifacts.shape[0] > 200
    assert fa.face_artifacts.shape[1] == 2

    with pytest.warns(UserWarning, match="No threshold found"):
        mm.FaceArtifacts(gdf.cx[712104:713000, 3961073:3961500])

    fa_ipq = mm.FaceArtifacts(gdf, index="isoperimetric_quotient")
    assert 6 < fa_ipq.threshold < 9
    assert fa_ipq.threshold != fa.threshold

    fa_dia = mm.FaceArtifacts(gdf, index="diameter_ratio")
    assert 6 < fa_dia.threshold < 9
    assert fa_dia.threshold != fa.threshold

    fa = mm.FaceArtifacts(gdf, index="isoperimetric_quotient")
    assert 6 < fa.threshold < 9

    with pytest.raises(ValueError, match="'banana' is not supported"):
        mm.FaceArtifacts(gdf, index="banana")
