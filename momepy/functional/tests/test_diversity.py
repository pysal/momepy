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


class TestDistribution:
    def setup_method(self):
        test_file_path = mm.datasets.get_path("bubenec")
        self.df_buildings = gpd.read_file(test_file_path, layer="buildings")
        self.df_tessellation = gpd.read_file(test_file_path, layer="tessellation")
        self.df_tessellation["area"] = self.df_tessellation.geometry.area
        self.graph = Graph.build_knn(self.df_buildings.centroid, k=3)

        self.diversity_graph = (
            Graph.build_contiguity(self.df_tessellation)
            .higher_order(k=3, lower_order=True)
            .assign_self_weight()
        )

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

    def test_values_range(self):
        full_sw = mm.values_range(self.df_tessellation["area"], self.diversity_graph)
        full_sw_expected = {
            "count": 144,
            "mean": 13575.258680748986,
            "min": 3789.0228732928035,
            "max": 34510.77694161156,
        }
        print(np.mean(full_sw))
        assert_result(
            full_sw, full_sw_expected, self.df_tessellation, check_names=False
        )

        limit = mm.values_range(
            self.df_tessellation["area"], self.diversity_graph, rng=(10, 90)
        )
        limit_expected = {
            "count": 144,
            "mean": 3358.45027554266,
            "min": 2080.351522584218,
            "max": 5115.169656715312,
        }
        assert_result(limit, limit_expected, self.df_tessellation, check_names=False)

    def test_theil(self):
        full_sw = mm.theil(self.df_tessellation["area"], self.diversity_graph)
        full_sw_expected = {
            "count": 144,
            "mean": 0.3367193709036915,
            "min": 0.0935437083870931,
            "max": 1.0063687846141105,
        }
        assert_result(
            full_sw, full_sw_expected, self.df_tessellation, check_names=False
        )

        limit = mm.theil(
            self.df_tessellation["area"], self.diversity_graph, rng=(10, 90)
        )
        limit_expected = {
            "count": 144,
            "mean": 0.10575479289690606,
            "min": 0.04633949101071495,
            "max": 0.26582672704556626,
        }

        assert_result(limit, limit_expected, self.df_tessellation, check_names=False)

        zeros = mm.theil(
            pd.Series(np.zeros(len(self.df_tessellation)), self.df_tessellation.index),
            self.graph,
        )
        zeros_expected = {"count": 144, "mean": 0, "min": 0, "max": 0.0}
        assert_result(zeros, zeros_expected, self.df_tessellation, check_names=False)

    def test_simpson(self):
        ht_sw = mm.simpson(self.df_tessellation["area"], self.diversity_graph)
        ht_sw_expected = {
            "count": 144,
            "mean": 0.5106343598245804,
            "min": 0.3504,
            "max": 0.7159183673469389,
        }
        assert_result(ht_sw, ht_sw_expected, self.df_tessellation, check_names=False)

        quan_sw = mm.simpson(
            self.df_tessellation.area, self.diversity_graph, binning="quantiles", k=3
        )
        quan_sw_expected = {
            "count": 144,
            "mean": 0.36125200075406005,
            "min": 0.3333333333333333,
            "max": 0.4609375,
        }
        assert_result(
            quan_sw, quan_sw_expected, self.df_tessellation, check_names=False
        )

        with pytest.raises(ValueError):
            mm.simpson(self.df_tessellation.area, self.graph, binning="nonexistent")

        gs = mm.simpson(
            self.df_tessellation.area, self.diversity_graph, gini_simpson=True
        )
        gs_expected = {
            "count": 144,
            "mean": 0.4893656401754196,
            "min": 0.2840816326530611,
            "max": 0.6496,
        }
        assert_result(gs, gs_expected, self.df_tessellation, check_names=False)

        gs_inv = mm.simpson(
            self.df_tessellation.area, self.diversity_graph, inverse=True
        )
        gs_inv_expected = {
            "count": 144,
            "mean": 1.994951794685094,
            "min": 1.3968072976054728,
            "max": 2.853881278538813,
        }
        assert_result(gs_inv, gs_inv_expected, self.df_tessellation, check_names=False)

        self.df_tessellation["cat"] = list(range(8)) * 18
        cat = mm.simpson(
            self.df_tessellation.cat, self.diversity_graph, categorical=True
        )
        cat_expected = {
            "count": 144,
            "mean": 0.13227361237314683,
            "min": 0.1255205234979179,
            "max": 0.15625,
        }
        assert_result(cat, cat_expected, self.df_tessellation, check_names=False)

    def test_gini(self):
        with pytest.raises(ValueError):
            mm.gini(pd.Series(-1, self.df_tessellation.index), self.diversity_graph)

        full_sw = mm.gini(self.df_tessellation["area"], self.diversity_graph)
        full_sw_expected = {
            "count": 144,
            "mean": 0.38686076469743697,
            "min": 0.24235274498955336,
            "max": 0.6400687910616315,
        }
        assert_result(
            full_sw, full_sw_expected, self.df_tessellation, check_names=False
        )

        limit = mm.gini(
            self.df_tessellation["area"], self.diversity_graph, rng=(10, 90)
        )
        limit_expected = {
            "count": 144,
            "mean": 0.2525181248879755,
            "min": 0.17049602697583713,
            "max": 0.39018140635767645,
        }
        assert_result(limit, limit_expected, self.df_tessellation, check_names=False)

    def test_shannon(self):
        with pytest.raises(ValueError):
            mm.shannon(
                self.df_tessellation.area, self.diversity_graph, binning="nonexistent"
            )

        ht_sw = mm.shannon(self.df_tessellation["area"], self.diversity_graph)
        ht_sw_expected = {
            "count": 144,
            "mean": 0.8290031127861055,
            "min": 0.4581441790615257,
            "max": 1.1626998334975678,
        }
        assert_result(ht_sw, ht_sw_expected, self.df_tessellation, check_names=False)

        quan_sw = mm.shannon(
            self.df_tessellation["area"], self.diversity_graph, binning="quantiles", k=3
        )
        quan_sw_expected = {
            "count": 144,
            "mean": 1.0543108593712356,
            "min": 0.8647400965276372,
            "max": 1.0986122886681096,
        }
        assert_result(
            quan_sw, quan_sw_expected, self.df_tessellation, check_names=False
        )

        self.df_tessellation["cat"] = list(range(8)) * 18
        cat = mm.shannon(
            self.df_tessellation.cat, self.diversity_graph, categorical=True
        )
        cat_expected = {
            "count": 144,
            "mean": 2.0493812749063793,
            "min": 1.9561874676604514,
            "max": 2.0774529508369457,
        }
        assert_result(cat, cat_expected, self.df_tessellation, check_names=False)

    def test_unique(self):
        self.df_tessellation["cat"] = list(range(8)) * 18
        un = mm.unique(self.df_tessellation["cat"], self.diversity_graph)
        un_expected = {"count": 144, "mean": 8.0, "min": 8, "max": 8}
        assert_result(un, un_expected, self.df_tessellation, check_names=False)

        self.df_tessellation.loc[0, "cat"] = np.nan
        un_nan = mm.unique(
            self.df_tessellation["cat"], self.diversity_graph, dropna=False
        )
        un_nan_expected = {"count": 144, "mean": 8.13888888888889, "min": 8, "max": 9}
        assert_result(un_nan, un_nan_expected, self.df_tessellation, check_names=False)

        un_nan_drop = mm.unique(
            self.df_tessellation["cat"], self.diversity_graph, dropna=True
        )
        un_nan_drop_expected = {"count": 144, "mean": 8.0, "min": 8, "max": 8}
        assert_result(
            un_nan_drop, un_nan_drop_expected, self.df_tessellation, check_names=False
        )


class TestDiversityEquivalence:
    def setup_method(self):
        test_file_path = mm.datasets.get_path("bubenec")
        self.df_tessellation = gpd.read_file(test_file_path, layer="tessellation")
        self.df_tessellation["area"] = self.df_tessellation.geometry.area
        self.sw = mm.sw_high(k=3, gdf=self.df_tessellation, ids="uID")
        self.graph = (
            Graph.build_contiguity(self.df_tessellation)
            .higher_order(k=3, lower_order=True)
            .assign_self_weight()
        )

    def test_values_range(self):
        full_sw_new = mm.values_range(self.df_tessellation["area"], self.graph)
        full_sw_old = mm.Range(self.df_tessellation, "area", self.sw, "uID").series
        assert_series_equal(
            full_sw_new, full_sw_old, check_dtype=False, check_names=False
        )

        limit_new = mm.values_range(
            self.df_tessellation["area"], self.graph, rng=(10, 90)
        )
        limit_old = mm.Range(
            self.df_tessellation, "area", self.sw, "uID", rng=(10, 90)
        ).series
        assert_series_equal(limit_new, limit_old, check_dtype=False, check_names=False)

    def test_theil(self):
        full_sw_new = mm.theil(self.df_tessellation["area"], self.graph)
        full_sw_old = mm.Theil(self.df_tessellation, "area", self.sw, "uID").series
        assert_series_equal(
            full_sw_new, full_sw_old, check_dtype=False, check_names=False
        )

        limit_new = mm.theil(self.df_tessellation["area"], self.graph, rng=(10, 90))
        limit_old = mm.Theil(
            self.df_tessellation,
            self.df_tessellation.area,
            self.sw,
            "uID",
            rng=(10, 90),
        ).series
        assert_series_equal(limit_new, limit_old, check_dtype=False, check_names=False)

        zeros_new = mm.theil(
            pd.Series(np.zeros(len(self.df_tessellation)), self.df_tessellation.index),
            self.graph,
        )
        zeros_old = mm.Theil(
            self.df_tessellation, np.zeros(len(self.df_tessellation)), self.sw, "uID"
        ).series
        assert_series_equal(zeros_new, zeros_old, check_dtype=False, check_names=False)

    def test_simpson(self):
        ht_sw_new = mm.simpson(self.df_tessellation["area"], self.graph)
        ht_sw_old = mm.Simpson(self.df_tessellation, "area", self.sw, "uID").series
        assert_series_equal(ht_sw_new, ht_sw_old, check_dtype=False, check_names=False)

        quan_sw_new = mm.simpson(
            self.df_tessellation.area, self.graph, binning="quantiles", k=3
        )
        quan_sw_old = mm.Simpson(
            self.df_tessellation,
            self.df_tessellation.area,
            self.sw,
            "uID",
            binning="quantiles",
            k=3,
        ).series
        assert_series_equal(
            quan_sw_new, quan_sw_old, check_dtype=False, check_names=False
        )

        gs_new = mm.simpson(self.df_tessellation.area, self.graph, gini_simpson=True)
        gs_old = mm.Simpson(
            self.df_tessellation, "area", self.sw, "uID", gini_simpson=True
        ).series
        assert_series_equal(gs_new, gs_old, check_dtype=False, check_names=False)

        gs_new = mm.simpson(self.df_tessellation.area, self.graph, inverse=True)
        gs_old = mm.Simpson(
            self.df_tessellation, "area", self.sw, "uID", inverse=True
        ).series
        assert_series_equal(gs_new, gs_old, check_dtype=False, check_names=False)

        self.df_tessellation["cat"] = list(range(8)) * 18
        cat_new = mm.simpson(self.df_tessellation.cat, self.graph, categorical=True)
        cat_old = mm.Simpson(
            self.df_tessellation, "cat", self.sw, "uID", categorical=True
        ).series
        assert_series_equal(cat_new, cat_old, check_dtype=False, check_names=False)

    def test_gini(self):
        full_sw_new = mm.gini(self.df_tessellation["area"], self.graph)
        full_sw_old = mm.Gini(self.df_tessellation, "area", self.sw, "uID").series
        assert_series_equal(
            full_sw_new, full_sw_old, check_dtype=False, check_names=False
        )

        limit_new = mm.gini(self.df_tessellation["area"], self.graph, rng=(10, 90))
        limit_old = mm.Gini(
            self.df_tessellation, "area", self.sw, "uID", rng=(10, 90)
        ).series
        assert_series_equal(limit_new, limit_old, check_dtype=False, check_names=False)

    def test_shannon(self):
        ht_sw_new = mm.shannon(self.df_tessellation["area"], self.graph)
        ht_sw_old = mm.Shannon(self.df_tessellation, "area", self.sw, "uID").series
        assert_series_equal(ht_sw_new, ht_sw_old, check_dtype=False, check_names=False)

        quan_sw_new = mm.shannon(
            self.df_tessellation["area"], self.graph, binning="quantiles", k=3
        )
        quan_sw_old = mm.Shannon(
            self.df_tessellation,
            self.df_tessellation.area,
            self.sw,
            "uID",
            binning="quantiles",
            k=3,
        ).series
        assert_series_equal(
            quan_sw_new, quan_sw_old, check_dtype=False, check_names=False
        )

        self.df_tessellation["cat"] = list(range(8)) * 18
        cat_new = mm.shannon(self.df_tessellation.cat, self.graph, categorical=True)
        cat_old = mm.Shannon(
            self.df_tessellation, "cat", self.sw, "uID", categorical=True
        ).series
        assert_series_equal(cat_new, cat_old, check_dtype=False, check_names=False)

    def test_unique(self):
        self.df_tessellation["cat"] = list(range(8)) * 18
        un_new = mm.unique(self.df_tessellation["cat"], self.graph)
        un_old = mm.Unique(self.df_tessellation, "cat", self.sw, "uID").series
        assert_series_equal(un_new, un_old, check_dtype=False, check_names=False)

        self.df_tessellation.loc[0, "cat"] = np.nan
        un_new = mm.unique(self.df_tessellation["cat"], self.graph, dropna=False)
        un_old = mm.Unique(
            self.df_tessellation, "cat", self.sw, "uID", dropna=False
        ).series
        assert_series_equal(un_new, un_old, check_dtype=False, check_names=False)

        un_new = mm.unique(self.df_tessellation["cat"], self.graph, dropna=True)
        un_old = mm.Unique(
            self.df_tessellation, "cat", self.sw, "uID", dropna=True
        ).series
        assert_series_equal(un_new, un_old, check_dtype=False, check_names=False)
