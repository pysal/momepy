import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from libpysal.graph import Graph
from packaging.version import Version
from pandas.testing import assert_frame_equal, assert_series_equal

import momepy as mm

from .conftest import assert_result

GPD_013 = Version(gpd.__version__) >= Version("0.13")
PD_210 = Version(pd.__version__) >= Version("2.1.0")


class TestDescribe:
    def setup_method(self):
        test_file_path = mm.datasets.get_path("bubenec")
        self.df_buildings = gpd.read_file(test_file_path, layer="buildings")
        self.df_streets = gpd.read_file(test_file_path, layer="streets")
        self.df_tessellation = gpd.read_file(test_file_path, layer="tessellation")
        self.df_streets["nID"] = mm.unique_id(self.df_streets)
        self.df_buildings["height"] = np.linspace(10.0, 30.0, 144)
        self.df_tessellation["area"] = self.df_tessellation.geometry.area
        self.df_buildings["area"] = self.df_buildings.geometry.area
        self.df_buildings["fl_area"] = mm.FloorArea(self.df_buildings, "height").series
        self.df_buildings["nID"] = mm.get_network_id(
            self.df_buildings, self.df_streets, "nID"
        )
        blocks = mm.Blocks(
            self.df_tessellation, self.df_streets, self.df_buildings, "bID", "uID"
        )
        self.blocks = blocks.blocks
        self.df_buildings["bID"] = blocks.buildings_id
        self.df_tessellation["bID"] = blocks.tessellation_id
        self.graph_sw = (
            Graph.build_contiguity(self.df_streets.set_index("nID"), rook=False)
            .higher_order(k=2, lower_order=True)
            .assign_self_weight()
        )
        self.graph = Graph.build_knn(self.df_buildings.centroid, k=3)

    def test_describe(self):
        area = self.df_buildings.area
        r = mm.describe(area, self.graph)

        expected_mean = {
            "mean": 587.3761020554495,
            "sum": 84582.15869598472,
            "min": 50.44045729583316,
            "max": 1187.2662413659234,
        }
        assert_result(r["mean"], expected_mean, self.df_buildings, exact=False)

        expected_median = {
            "mean": 577.4640489818667,
            "sum": 83154.8230533888,
            "min": 50.43336175017242,
            "max": 1225.8094201694726,
        }
        assert_result(r["median"], expected_median, self.df_buildings, exact=False)

        expected_std = {
            "mean": 255.59307136480083,
            "sum": 36805.40227653132,
            "min": 0.05050450812944085,
            "max": 1092.484902679786,
        }
        assert_result(r["std"], expected_std, self.df_buildings, exact=False)

        expected_min = {
            "mean": 349.53354434499295,
            "sum": 50332.830385678986,
            "min": 50.39387578315866,
            "max": 761.0313042971973,
        }
        assert_result(r["min"], expected_min, self.df_buildings, exact=False)

        expected_max = {
            "mean": 835.1307128394886,
            "sum": 120258.82264888636,
            "min": 50.49413435416841,
            "max": 2127.7522277389035,
        }
        assert_result(r["max"], expected_max, self.df_buildings, exact=False)

        expected_sum = {
            "mean": 1762.128306166348,
            "sum": 253746.47608795413,
            "min": 151.32137188749948,
            "max": 3561.79872409777,
        }
        assert_result(r["sum"], expected_sum, self.df_buildings, exact=False)

    def test_describe_quantile(self):
        graph = Graph.build_knn(self.df_buildings.centroid, k=15)
        area = self.df_buildings.area
        r = mm.describe(area, graph, q=(25, 75))

        expected_mean = {
            "mean": 601.6960154385389,
            "sum": 86644.2262231496,
            "min": 250.25984637364323,
            "max": 901.0028506943196,
        }
        assert_result(r["mean"], expected_mean, self.df_buildings, exact=False)

    @pytest.mark.skipif(not GPD_013, reason="get_coordinates() not available")
    def test_describe_mode(self):
        corners = mm.corners(self.df_buildings)
        r = mm.describe(corners, self.graph, include_mode=True)

        expected = {
            "mean": 6.152777777777778,
            "sum": 886,
            "min": 4,
            "max": 17,
        }
        assert_result(r["mode"], expected, self.df_buildings, exact=False)

    @pytest.mark.skipif(not GPD_013, reason="get_coordinates() not available")
    def test_describe_quantile_mode(self):
        graph = Graph.build_knn(self.df_buildings.centroid, k=15)
        corners = mm.corners(self.df_buildings)
        r = mm.describe(corners, graph, q=(25, 75), include_mode=True)

        expected = {
            "mean": 6.958333333333333,
            "sum": 1002.0,
            "min": 4.0,
            "max": 12,
        }
        assert_result(r["mode"], expected, self.df_buildings, exact=False)

    def test_describe_array(self):
        area = self.df_buildings.area
        r = mm.describe(area, self.graph)
        r2 = mm.describe(area.values, self.graph)

        assert_frame_equal(r, r2)

    @pytest.mark.skipif(
        not PD_210, reason="aggregation is different in previous pandas versions"
    )
    def test_describe_reached_input(self):
        with pytest.raises(
            ValueError,
            match=("One of result_index or graph has to be specified, but not both."),
        ):
            mm.describe_reached(self.df_buildings[["area"]], self.df_buildings["nID"])

        with pytest.raises(
            ValueError,
            match=("One of result_index or graph has to be specified, but not both."),
        ):
            mm.describe_reached(
                self.df_buildings[["area"]],
                self.df_buildings["nID"],
                result_index=self.df_streets.index,
                graph=self.graph_sw,
            )

    @pytest.mark.skipif(
        not PD_210, reason="aggregation is different in previous pandas versions"
    )
    def test_describe_reached(self):
        df = mm.describe_reached(
            self.df_buildings[["area", "fl_area"]],
            self.df_buildings["nID"],
            self.df_streets.index,
        )

        # not testing std, there are different implementations:
        # OO momepy uses ddof=0, functional momepy - ddof=1
        expected_area_sum = {
            "min": 618.4470363735187,
            "max": 18085.458977113314,
            "count": 22,
            "mean": 4765.589763157915,
        }
        expected_area_mean = {
            "min": 218.962248810988,
            "max": 1808.5458977113315,
            "count": 22,
            "mean": 746.7028417890866,
        }
        expected_area_count = {
            "min": 0,
            "max": 18,
            "count": 35,
            "mean": 4.114285714285714,
        }
        assert_result(df["area"]["count"], expected_area_count, self.df_streets)
        assert_result(df["area"]["sum"], expected_area_sum, self.df_streets)
        assert_result(df["area"]["mean"], expected_area_mean, self.df_streets)

        expected_fl_area_sum = {
            "min": 1894.492021221426,
            "max": 79169.31385861782,
            "count": 22,
            "mean": 26494.88163951223,
        }
        expected_fl_area_mean = {
            "min": 939.9069666320963,
            "max": 7916.931385861784,
            "count": 22,
            "mean": 3995.8307750062318,
        }
        expected_fl_area_count = {
            "min": 0,
            "max": 18,
            "count": 35,
            "mean": 4.114285714285714,
        }
        assert_result(df["fl_area"]["count"], expected_fl_area_count, self.df_streets)
        assert_result(df["fl_area"]["sum"], expected_fl_area_sum, self.df_streets)
        assert_result(df["fl_area"]["mean"], expected_fl_area_mean, self.df_streets)

    @pytest.mark.skipif(
        not PD_210, reason="aggregation is different in previous pandas versions"
    )
    def test_describe_reached_sw(self):
        df_sw = mm.describe_reached(
            self.df_buildings["fl_area"], self.df_buildings["nID"], graph=self.graph_sw
        )

        # not using assert_result since the method
        # is returning an aggregation, indexed based on nID
        assert max(df_sw["fl_area"]["count"]) == 138
        expected = {"min": 6, "max": 138, "count": 35, "mean": 67.8}
        assert_result(
            df_sw["fl_area"]["count"], expected, self.df_streets, check_names=False
        )

    @pytest.mark.skipif(
        not PD_210, reason="aggregation is different in previous pandas versions"
    )
    def test_describe_reached_input_equality(self):
        island_result_df = mm.describe_reached(
            self.df_buildings[["area"]], self.df_buildings["nID"], self.df_streets.index
        )
        island_result_series = mm.describe_reached(
            self.df_buildings["area"], self.df_buildings["nID"], self.df_streets.index
        )
        island_result_ndarray = mm.describe_reached(
            self.df_buildings["area"].values,
            self.df_buildings["nID"].values,
            self.df_streets.index,
        )

        assert np.allclose(
            island_result_df.values, island_result_series.values, equal_nan=True
        )
        assert np.allclose(
            island_result_df.values, island_result_ndarray.values, equal_nan=True
        )


class TestDescribeEquality:
    def setup_method(self):
        test_file_path = mm.datasets.get_path("bubenec")
        self.df_buildings = gpd.read_file(test_file_path, layer="buildings")
        self.df_streets = gpd.read_file(test_file_path, layer="streets")
        self.df_tessellation = gpd.read_file(test_file_path, layer="tessellation")
        self.df_streets["nID"] = mm.unique_id(self.df_streets)
        self.df_buildings["height"] = np.linspace(10.0, 30.0, 144)
        self.df_tessellation["area"] = self.df_tessellation.geometry.area
        self.df_buildings["area"] = self.df_buildings.geometry.area
        self.df_buildings["fl_area"] = mm.FloorArea(self.df_buildings, "height").series
        self.df_buildings["nID"] = mm.get_network_id(
            self.df_buildings, self.df_streets, "nID"
        )
        blocks = mm.Blocks(
            self.df_tessellation, self.df_streets, self.df_buildings, "bID", "uID"
        )
        self.blocks = blocks.blocks
        self.df_buildings["bID"] = blocks.buildings_id
        self.df_tessellation["bID"] = blocks.tessellation_id
        self.graph_sw = (
            Graph.build_contiguity(self.df_streets.set_index("nID"), rook=False)
            .higher_order(k=2, lower_order=True)
            .assign_self_weight()
        )
        self.graph = Graph.build_knn(self.df_buildings.centroid, k=3)

    @pytest.mark.skipif(
        not PD_210, reason="aggregation is different in previous pandas versions"
    )
    def test_describe_reached_equality(self):
        new_df = mm.describe_reached(
            self.df_buildings["area"], self.df_buildings["nID"], self.df_streets.index
        )

        new_count = new_df["area"]["count"]
        old_count = mm.Reached(self.df_streets, self.df_buildings, "nID", "nID").series
        assert_series_equal(new_count, old_count, check_names=False, check_dtype=False)

        new_area = new_df["area"]["sum"]
        old_area = mm.Reached(
            self.df_streets, self.df_buildings, "nID", "nID", mode="sum"
        ).series
        assert_series_equal(new_area, old_area, check_names=False, check_dtype=False)

        new_area_mean = new_df["area"]["mean"]
        old_area_mean = mm.Reached(
            self.df_streets, self.df_buildings, "nID", "nID", mode="mean"
        ).series
        assert_series_equal(
            new_area_mean, old_area_mean, check_names=False, check_dtype=False
        )

    @pytest.mark.skipif(
        not PD_210, reason="aggregation is different in previous pandas versions"
    )
    def test_describe_reached_equality_sw(self):
        new_df = mm.describe_reached(
            self.df_buildings["fl_area"], self.df_buildings["nID"], graph=self.graph_sw
        )

        new_fl_area = new_df["fl_area"]["sum"]

        sw = mm.sw_high(k=2, gdf=self.df_streets)
        old_fl_area = mm.Reached(
            self.df_streets,
            self.df_buildings,
            "nID",
            "nID",
            spatial_weights=sw,
            mode="sum",
            values="fl_area",
        ).series
        assert_series_equal(
            new_fl_area, old_fl_area, check_names=False, check_dtype=False
        )
