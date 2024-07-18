import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import scipy
from libpysal.graph import Graph
from packaging.version import Version
from pandas.testing import assert_frame_equal, assert_series_equal

import momepy as mm

from .conftest import assert_frame_result, assert_result

GPD_013 = Version(gpd.__version__) >= Version("0.13")
PD_210 = Version(pd.__version__) >= Version("2.1.0")
SP_112 = Version(scipy.__version__) >= Version("1.12.0")


class TestDescribe:
    def setup_method(self):
        test_file_path = mm.datasets.get_path("bubenec")
        self.df_buildings = gpd.read_file(test_file_path, layer="buildings")
        self.df_tessellation = gpd.read_file(test_file_path, layer="tessellation")
        self.df_tessellation["area"] = self.df_tessellation.geometry.area
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
        self.diversity_graph = (
            Graph.build_contiguity(self.df_tessellation)
            .higher_order(k=3, lower_order=True)
            .assign_self_weight()
        )

        graph = Graph.build_contiguity(self.df_tessellation, rook=False).higher_order(
            k=3, lower_order=True
        )
        from shapely import distance

        centroids = self.df_tessellation.centroid

        def _distance_decay_weights(group):
            focal = group.index[0][0]
            neighbours = group.index.get_level_values(1)
            distances = distance(centroids.loc[focal], centroids.loc[neighbours])
            distance_decay = 1 / distances
            return distance_decay.values

        self.decay_graph = graph.transform(_distance_decay_weights)

        self.diversity_graph = (
            Graph.build_contiguity(self.df_tessellation)
            .higher_order(k=3, lower_order=True)
            .assign_self_weight()
        )

    def test_describe(self):
        area = self.df_buildings.area
        r = self.graph.describe(area)

        expected_mean = {
            "mean": 587.3761020554495,
            "sum": 84582.15869598472,
            "min": 50.44045729583316,
            "max": 1187.2662413659234,
        }
        assert_result(
            r["mean"], expected_mean, self.df_buildings, exact=False, check_names=False
        )

        expected_median = {
            "mean": 577.4640489818667,
            "sum": 83154.8230533888,
            "min": 50.43336175017242,
            "max": 1225.8094201694726,
        }
        assert_result(
            r["median"],
            expected_median,
            self.df_buildings,
            exact=False,
            check_names=False,
        )

        expected_std = {
            "mean": 255.59307136480083,
            "sum": 36805.40227653132,
            "min": 0.05050450812944085,
            "max": 1092.484902679786,
        }
        assert_result(
            r["std"], expected_std, self.df_buildings, exact=False, check_names=False
        )

        expected_min = {
            "mean": 349.53354434499295,
            "sum": 50332.830385678986,
            "min": 50.39387578315866,
            "max": 761.0313042971973,
        }
        assert_result(
            r["min"], expected_min, self.df_buildings, exact=False, check_names=False
        )

        expected_max = {
            "mean": 835.1307128394886,
            "sum": 120258.82264888636,
            "min": 50.49413435416841,
            "max": 2127.7522277389035,
        }
        assert_result(
            r["max"], expected_max, self.df_buildings, exact=False, check_names=False
        )

        expected_sum = {
            "mean": 1762.128306166348,
            "sum": 253746.47608795413,
            "min": 151.32137188749948,
            "max": 3561.79872409777,
        }
        assert_result(
            r["sum"], expected_sum, self.df_buildings, exact=False, check_names=False
        )

    def test_describe_quantile(self):
        graph = Graph.build_knn(self.df_buildings.centroid, k=15)
        area = self.df_buildings.area
        r = graph.describe(area, q=(25, 75))

        expected_mean = {
            "mean": 601.6960154385389,
            "sum": 86644.2262231496,
            "min": 250.25984637364323,
            "max": 901.0028506943196,
        }
        assert_result(
            r["mean"], expected_mean, self.df_buildings, exact=False, check_names=False
        )

    @pytest.mark.skipif(not GPD_013, reason="get_coordinates() not available")
    def test_describe_mode(self):
        corners = mm.corners(self.df_buildings)
        r = self.graph.describe(corners)

        expected = {
            "mean": 6.152777777777778,
            "sum": 886,
            "min": 4,
            "max": 17,
        }
        assert_result(
            r["mode"], expected, self.df_buildings, exact=False, check_names=False
        )

    @pytest.mark.skipif(not GPD_013, reason="get_coordinates() not available")
    def test_describe_quantile_mode(self):
        graph = Graph.build_knn(self.df_buildings.centroid, k=15)
        corners = mm.corners(self.df_buildings)
        r = graph.describe(corners, q=(25, 75))

        expected = {
            "mean": 6.958333333333333,
            "sum": 1002.0,
            "min": 4.0,
            "max": 12,
        }
        assert_result(
            r["mode"], expected, self.df_buildings, exact=False, check_names=False
        )

    def test_describe_array(self):
        area = self.df_buildings.area
        r = self.graph.describe(area)
        r2 = self.graph.describe(area.values)

        assert_frame_equal(r, r2, check_names=False)
        assert_frame_equal(r, r2)

    def test_values_range(self):
        full_sw = mm.values_range(self.df_tessellation["area"], self.diversity_graph)
        full_sw_expected = {
            "count": 144,
            "mean": 13575.258680748986,
            "min": 3789.0228732928035,
            "max": 34510.77694161156,
        }
        assert_result(
            full_sw, full_sw_expected, self.df_tessellation, check_names=False
        )

        limit = mm.values_range(
            self.df_tessellation["area"], self.diversity_graph, q=(10, 90)
        )
        limit_expected = {
            "mean": 3551.9379326637954,
            "max": 6194.978308458511,
            "min": 2113.282481158694,
            "count": 144,
        }

        assert_result(limit, limit_expected, self.df_tessellation, check_names=False)

    def test_theil(self):
        full_sw = mm.theil(self.df_tessellation["area"], self.diversity_graph)
        full_sw2 = mm.theil(
            self.df_tessellation["area"], self.diversity_graph, q=(0, 100)
        )
        full_sw_expected = {
            "count": 144,
            "mean": 0.3367193709036915,
            "min": 0.0935437083870931,
            "max": 1.0063687846141105,
        }
        assert_result(
            full_sw, full_sw_expected, self.df_tessellation, check_names=False
        )
        assert_result(
            full_sw2, full_sw_expected, self.df_tessellation, check_names=False
        )

        # mismatch between percentile interpolation methods
        limit = mm.theil(self.df_tessellation["area"], self.diversity_graph, q=(10, 90))
        limit_expected = {
            "count": 144,
            "mean": 0.09689345872019642,
            "min": 0.03089398223055910,
            "max": 0.2726670141461655,
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

        # mismatch between interpolation methods
        limit = mm.gini(self.df_tessellation["area"], self.diversity_graph, q=(10, 90))
        limit_expected = {
            "count": 144,
            "mean": 0.2417437064941186,
            "min": 0.14098983070917345,
            "max": 0.3978182288393458,
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
        un = self.diversity_graph.describe(
            self.df_tessellation["cat"], statistics=["nunique"]
        )["nunique"]
        un_expected = {"count": 144, "mean": 8.0, "min": 8, "max": 8}
        assert_result(un, un_expected, self.df_tessellation, check_names=False)

        self.df_tessellation.loc[0, "cat"] = np.nan

        un_nan_drop = self.diversity_graph.describe(
            self.df_tessellation["cat"], statistics=["nunique"]
        )["nunique"]
        un_nan_drop_expected = {"count": 144, "mean": 8.0, "min": 8, "max": 8}
        assert_result(
            un_nan_drop, un_nan_drop_expected, self.df_tessellation, check_names=False
        )

    @pytest.mark.skipif(
        not PD_210, reason="aggregation is different in previous pandas versions"
    )
    def test_describe_agg(self):
        df = mm.describe_agg(
            self.df_buildings["area"],
            self.df_buildings["nID"],
        )

        result_index = self.df_buildings["nID"].value_counts().sort_index()
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
            "min": 1,
            "max": 18,
            "count": 22,
            "mean": 6.545454545454546,
        }
        assert_result(df["count"], expected_area_count, result_index, check_names=False)
        assert_result(df["sum"], expected_area_sum, result_index, check_names=False)
        assert_result(df["mean"], expected_area_mean, result_index, check_names=False)

        filtered_counts = mm.describe_agg(
            self.df_buildings["area"],
            self.df_buildings["nID"],
            q=(10, 90),
            statistics=["count"],
        )["count"]
        expected_filtered_area_count = {
            "min": 1,
            "max": 14,
            "count": 22,
            "mean": 4.727272,
        }
        assert_result(
            filtered_counts,
            expected_filtered_area_count,
            result_index,
            check_names=False,
        )

        df = mm.describe_agg(
            self.df_buildings["fl_area"].values,
            self.df_buildings["nID"],
        )

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

        assert_result(df["count"], expected_area_count, result_index)
        assert_result(df["sum"], expected_fl_area_sum, result_index)
        assert_result(df["mean"], expected_fl_area_mean, result_index)

    @pytest.mark.skipif(
        not PD_210, reason="aggregation is different in previous pandas versions"
    )
    def test_describe_cols(self):
        df = mm.describe_agg(
            self.df_buildings["area"],
            self.df_buildings["nID"],
            statistics=["min", "max"],
        )
        assert list(df.columns) == ["min", "max"]

    @pytest.mark.skipif(
        not PD_210, reason="aggregation is different in previous pandas versions"
    )
    def test_describe_reached_agg(self):
        df_sw = mm.describe_reached_agg(
            self.df_buildings["fl_area"], self.df_buildings["nID"], graph=self.graph_sw
        )
        expected = {"min": 6, "max": 138, "count": 35, "mean": 67.82857}
        assert_result(df_sw["count"], expected, self.df_streets, check_names=False)

        df_sw_dummy_filtration = mm.describe_reached_agg(
            self.df_buildings["fl_area"],
            self.df_buildings["nID"],
            graph=self.graph_sw,
            q=(0, 100),
        )
        assert_frame_equal(
            df_sw, df_sw_dummy_filtration, check_names=False, check_index_type=False
        )

        filtered_df = mm.describe_reached_agg(
            self.df_buildings["fl_area"],
            self.df_buildings["nID"],
            graph=self.graph_sw,
            q=(10, 90),
            statistics=["count"],
        )
        filtered_expected = {"min": 4, "max": 110, "count": 35, "mean": 53.48571}
        assert_result(
            filtered_df["count"], filtered_expected, self.df_streets, check_names=False
        )

    @pytest.mark.skipif(
        not PD_210, reason="aggregation is different in previous pandas versions"
    )
    def test_describe_reached_input_equality(self):
        island_result_df = mm.describe_agg(
            self.df_buildings["area"], self.df_buildings["nID"]
        )

        island_result_ndarray = mm.describe_agg(
            self.df_buildings["area"].values,
            self.df_buildings["nID"].values,
        )

        assert np.allclose(
            island_result_df.values, island_result_ndarray.values, equal_nan=True
        )

    @pytest.mark.skipif(
        not PD_210, reason="aggregation is different in previous pandas versions"
    )
    def test_describe_reached_cols(self):
        df = mm.describe_reached_agg(
            self.df_buildings["fl_area"],
            self.df_buildings["nID"],
            graph=self.graph_sw,
            q=(10, 90),
            statistics=["min", "max"],
        )
        assert list(df.columns) == ["min", "max"]

    @pytest.mark.skipif(
        not PD_210, reason="aggregation is different in previous pandas versions"
    )
    def test_na_results(self):
        nan_areas = self.df_buildings["area"]
        nan_areas.iloc[range(0, len(self.df_buildings), 3),] = np.nan

        pandas_agg_vals = mm.describe_agg(
            nan_areas,
            self.df_buildings["nID"],
        )

        numba_agg_vals = mm.describe_agg(
            nan_areas, self.df_buildings["nID"], q=(0, 100)
        )

        assert_frame_equal(pandas_agg_vals, numba_agg_vals)

    def test_covered_area(self):
        graph = (
            Graph.build_contiguity(self.df_tessellation)
            .higher_order(k=3, lower_order=True)
            .assign_self_weight()
        )
        covered_sw = graph.describe(self.df_tessellation.area, statistics=["sum"])[
            "sum"
        ]

        covered_sw2 = graph.describe(
            self.df_tessellation.area.values, statistics=["sum"]
        )["sum"]

        expected_covered_sw = {
            "sum": 11526021.19027327,
            "mean": 80041.81382134215,
            "min": 26013.0106743472,
            "max": 131679.18084183024,
        }
        assert_result(
            covered_sw,
            expected_covered_sw,
            self.df_tessellation,
            check_names=False,
            exact=False,
        )
        assert_series_equal(
            covered_sw, covered_sw2, check_names=False, check_index_type=False
        )

    def test_density(self):
        graph = (
            Graph.build_contiguity(self.df_tessellation, rook=False)
            .higher_order(k=3, lower_order=True)
            .assign_self_weight()
        )
        fl_area = graph.describe(self.df_buildings["fl_area"])["sum"]
        tess_area = graph.describe(self.df_tessellation["area"])["sum"]
        dens_new = fl_area / tess_area
        dens_expected = {
            "count": 144,
            "mean": 1.6615871155383324,
            "max": 2.450536855278486,
            "min": 0.9746481727569978,
        }
        assert_result(
            dens_new,
            dens_expected,
            self.df_tessellation,
            exact=False,
            check_names=False,
        )

    @pytest.mark.skipif(
        not PD_210, reason="aggregation is different in previous pandas versions"
    )
    def test_unweighted_percentile(self):
        perc = mm.percentile(self.df_tessellation["area"], self.diversity_graph)
        perc_expected = {
            "count": 144,
            "mean": 2109.739467856585,
            "min": 314.3067794345771,
            "max": 4258.008903612521,
        }

        assert_frame_result(
            perc, perc_expected, self.df_tessellation, check_names=False
        )

        perc = mm.percentile(
            pd.Series(list(range(8)) * 18, index=self.df_tessellation.index),
            self.diversity_graph,
        )
        perc_expected = {
            "count": 144,
            "mean": 3.5283564814814814,
            "min": 0.75,
            "max": 6.0,
        }
        assert_frame_result(
            perc, perc_expected, self.df_tessellation, check_names=False
        )

        perc = mm.percentile(
            self.df_tessellation["area"], self.diversity_graph, q=[30, 70]
        )
        perc_expected = {
            "count": 144,
            "mean": 2096.4500111386724,
            "min": 484.37546961694574,
            "max": 4160.642824113784,
        }
        assert_frame_result(
            perc, perc_expected, self.df_tessellation, check_names=False
        )

        # test isolates
        graph = Graph.build_contiguity(self.df_tessellation.iloc[:100])
        perc = mm.percentile(self.df_tessellation["area"].iloc[:100], graph)
        assert perc.loc[0].isna().all()

    @pytest.mark.skipif(
        not PD_210, reason="aggregation is different in previous pandas versions"
    )
    def test_distance_decay_linearly_weighted_percentiles(self):
        # setup weight decay graph

        perc = mm.percentile(
            self.df_tessellation["area"],
            self.decay_graph,
        )
        perc_expected = {
            "count": 144,
            "mean": 1956.0672714756156,
            "min": 110.43272692016959,
            "max": 4331.418546462096,
        }
        assert_frame_result(
            perc, perc_expected, self.df_tessellation, check_names=False
        )

        perc = mm.percentile(
            self.df_tessellation["area"],
            self.decay_graph,
            q=[30, 70],
        )
        perc_expected = {
            "count": 144,
            "mean": 1931.8544987242813,
            "min": 122.04102848302165,
            "max": 4148.563252265954,
        }
        assert_frame_result(
            perc, perc_expected, self.df_tessellation, check_names=False
        )

    def test_describe_nunique(self):
        graph = (
            Graph.build_contiguity(self.df_tessellation, rook=False)
            .higher_order(k=5, lower_order=True)
            .assign_self_weight()
        )

        unweighted_expected = {
            "count": 144,
            "min": 3,
            "max": 8,
            "mean": 5.222222222222222,
        }

        unweighted = graph.describe(
            self.df_tessellation["bID"], statistics=["nunique"]
        )["nunique"]

        unweighted2 = graph.describe(
            self.df_tessellation["bID"], q=(0, 100), statistics=["nunique"]
        )["nunique"]

        assert_result(
            unweighted,
            unweighted_expected,
            self.df_tessellation,
            exact=False,
            check_names=False,
        )
        assert_result(
            unweighted2,
            unweighted_expected,
            self.df_tessellation,
            exact=False,
            check_names=False,
        )
        assert_series_equal(unweighted2, unweighted, check_dtype=False)

    def test_count_unique_using_describe(self):
        graph = (
            Graph.build_contiguity(self.df_tessellation, rook=False)
            .higher_order(k=5, lower_order=True)
            .assign_self_weight()
        )

        count = graph.describe(self.df_tessellation["bID"])["nunique"]
        agg_areas = graph.describe(self.df_tessellation["area"])["sum"]
        weighted_count = count / agg_areas
        weighted_count_expected = {
            "count": 144,
            "min": 2.0989616504225266e-05,
            "max": 4.2502425045664464e-05,
            "mean": 3.142437439120778e-05,
        }
        assert_result(
            weighted_count,
            weighted_count_expected,
            self.df_tessellation,
            exact=False,
            check_names=False,
        )

    def test_mean_deviation(self):
        street_graph = Graph.build_contiguity(self.df_streets, rook=False)
        y = mm.orientation(self.df_streets)
        deviations = mm.mean_deviation(y, street_graph)
        expected = {
            "count": 35,
            "mean": 7.527840590385933,
            "min": 0.00798704765839,
            "max": 20.9076846002,
        }
        assert_result(deviations, expected, self.df_streets)


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
        graph = Graph.build_contiguity(self.df_tessellation, rook=False).higher_order(
            k=3, lower_order=True
        )
        from shapely import distance

        centroids = self.df_tessellation.centroid

        def _distance_decay_weights(group):
            focal = group.index[0][0]
            neighbours = group.index.get_level_values(1)
            distances = distance(centroids.loc[focal], centroids.loc[neighbours])
            distance_decay = 1 / distances
            return distance_decay.values

        self.decay_graph = graph.transform(_distance_decay_weights)

        # for diversity tests
        self.sw = mm.sw_high(k=3, gdf=self.df_tessellation, ids="uID")
        self.graph_diversity = (
            Graph.build_contiguity(self.df_tessellation)
            .higher_order(k=3, lower_order=True)
            .assign_self_weight()
        )

    @pytest.mark.skipif(
        not PD_210, reason="aggregation is different in previous pandas versions"
    )
    def test_describe_reached_equality(self):
        new_df = mm.describe_agg(self.df_buildings["area"], self.df_buildings["nID"])

        new_count = new_df["count"]
        old_count = mm.Reached(self.df_streets, self.df_buildings, "nID", "nID").series
        old_count = old_count[old_count > 0]
        assert_series_equal(new_count, old_count, check_names=False, check_dtype=False)

        new_area = new_df["sum"]
        old_area = mm.Reached(
            self.df_streets, self.df_buildings, "nID", "nID", mode="sum"
        ).series
        old_area = old_area[old_area.notna()]
        assert_series_equal(new_area, old_area, check_names=False, check_dtype=False)

        new_area_mean = new_df["mean"]
        old_area_mean = mm.Reached(
            self.df_streets, self.df_buildings, "nID", "nID", mode="mean"
        ).series
        old_area_mean = old_area_mean[old_area_mean.notna()]
        assert_series_equal(
            new_area_mean, old_area_mean, check_names=False, check_dtype=False
        )

    @pytest.mark.skipif(
        not PD_210, reason="aggregation is different in previous pandas versions"
    )
    def test_describe_reached_equality_sw(self):
        new_df = mm.describe_reached_agg(
            self.df_buildings["fl_area"], self.df_buildings["nID"], graph=self.graph_sw
        )

        new_fl_area = new_df["sum"]

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

        old_mean_fl_area = mm.Reached(
            self.df_streets,
            self.df_buildings,
            "nID",
            "nID",
            spatial_weights=sw,
            mode="mean",
            values="fl_area",
        ).series
        assert_series_equal(
            new_df["mean"], old_mean_fl_area, check_names=False, check_dtype=False
        )

    def test_blocks_counts(self):
        graph = (
            Graph.build_contiguity(self.df_tessellation, rook=False)
            .higher_order(k=5, lower_order=True)
            .assign_self_weight()
        )
        sw = mm.sw_high(k=5, gdf=self.df_tessellation, ids="uID")

        unweighted_new = graph.describe(self.df_tessellation["bID"])["nunique"]
        unweighted_old = mm.BlocksCount(
            self.df_tessellation, "bID", sw, "uID", weighted=False
        ).series
        assert_series_equal(
            unweighted_new,
            unweighted_old,
            check_index_type=False,
            check_names=False,
            check_dtype=False,
        )

        agg_areas = graph.describe(self.df_tessellation["area"])["sum"]
        count_new = unweighted_new / agg_areas
        count_old = mm.BlocksCount(self.df_tessellation, "bID", sw, "uID").series
        assert_series_equal(
            count_new,
            count_old,
            check_names=False,
            check_dtype=False,
            check_index_type=False,
        )

    def test_values_range(self):
        full_sw_new = mm.values_range(
            self.df_tessellation["area"], self.graph_diversity
        )
        full_sw_old = mm.Range(self.df_tessellation, "area", self.sw, "uID").series
        assert_series_equal(
            full_sw_new, full_sw_old, check_dtype=False, check_names=False
        )

        limit_new = mm.values_range(
            self.df_tessellation["area"], self.graph_diversity, q=(10, 90)
        )
        limit_old = mm.Range(
            self.df_tessellation,
            "area",
            self.sw,
            "uID",
            interpolation="hazen",
            rng=(10, 90),
        ).series
        assert_series_equal(limit_new, limit_old, check_dtype=False, check_names=False)

    def test_theil(self):
        full_sw_new = mm.theil(self.df_tessellation["area"], self.graph_diversity)
        full_sw_old = mm.Theil(self.df_tessellation, "area", self.sw, "uID").series
        assert_series_equal(
            full_sw_new, full_sw_old, check_dtype=False, check_names=False
        )

        # old and new have different percentile interpolation methods
        # therefore the comparison needs a higher rtol
        limit_new = mm.theil(
            self.df_tessellation["area"], self.graph_diversity, q=(10, 90)
        )
        limit_old = mm.Theil(
            self.df_tessellation,
            self.df_tessellation.area,
            self.sw,
            "uID",
            rng=(10, 90),
        ).series
        assert_series_equal(
            limit_new, limit_old, rtol=0.5, check_dtype=False, check_names=False
        )

        zeros_new = mm.theil(
            pd.Series(np.zeros(len(self.df_tessellation)), self.df_tessellation.index),
            self.graph_diversity,
        )
        zeros_old = mm.Theil(
            self.df_tessellation, np.zeros(len(self.df_tessellation)), self.sw, "uID"
        ).series
        assert_series_equal(zeros_new, zeros_old, check_dtype=False, check_names=False)

    def test_simpson(self):
        ht_sw_new = mm.simpson(self.df_tessellation["area"], self.graph_diversity)
        ht_sw_old = mm.Simpson(self.df_tessellation, "area", self.sw, "uID").series
        assert_series_equal(ht_sw_new, ht_sw_old, check_dtype=False, check_names=False)

        quan_sw_new = mm.simpson(
            self.df_tessellation.area, self.graph_diversity, binning="quantiles", k=3
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

        gs_new = mm.simpson(
            self.df_tessellation.area, self.graph_diversity, gini_simpson=True
        )
        gs_old = mm.Simpson(
            self.df_tessellation, "area", self.sw, "uID", gini_simpson=True
        ).series
        assert_series_equal(gs_new, gs_old, check_dtype=False, check_names=False)

        gs_new = mm.simpson(
            self.df_tessellation.area, self.graph_diversity, inverse=True
        )
        gs_old = mm.Simpson(
            self.df_tessellation, "area", self.sw, "uID", inverse=True
        ).series
        assert_series_equal(gs_new, gs_old, check_dtype=False, check_names=False)

        self.df_tessellation["cat"] = list(range(8)) * 18
        cat_new = mm.simpson(
            self.df_tessellation.cat, self.graph_diversity, categorical=True
        )
        cat_old = mm.Simpson(
            self.df_tessellation, "cat", self.sw, "uID", categorical=True
        ).series
        assert_series_equal(cat_new, cat_old, check_dtype=False, check_names=False)

    def test_gini(self):
        full_sw_new = mm.gini(self.df_tessellation["area"], self.graph_diversity)
        full_sw_old = mm.Gini(self.df_tessellation, "area", self.sw, "uID").series
        assert_series_equal(
            full_sw_new, full_sw_old, check_dtype=False, check_names=False
        )

        # ## old and new have different interpolation methods
        ## there need higher rtol
        limit_new = mm.gini(
            self.df_tessellation["area"], self.graph_diversity, q=(10, 90)
        )
        limit_old = mm.Gini(
            self.df_tessellation, "area", self.sw, "uID", rng=(10, 90)
        ).series
        assert_series_equal(
            limit_new, limit_old, rtol=0.3, check_dtype=False, check_names=False
        )

    def test_shannon(self):
        ht_sw_new = mm.shannon(self.df_tessellation["area"], self.graph_diversity)
        ht_sw_old = mm.Shannon(self.df_tessellation, "area", self.sw, "uID").series
        assert_series_equal(ht_sw_new, ht_sw_old, check_dtype=False, check_names=False)

        quan_sw_new = mm.shannon(
            self.df_tessellation["area"], self.graph_diversity, binning="quantiles", k=3
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
        cat_new = mm.shannon(
            self.df_tessellation.cat, self.graph_diversity, categorical=True
        )
        cat_old = mm.Shannon(
            self.df_tessellation, "cat", self.sw, "uID", categorical=True
        ).series
        assert_series_equal(cat_new, cat_old, check_dtype=False, check_names=False)

    def test_unique(self):
        self.df_tessellation["cat"] = list(range(8)) * 18
        un_new = self.graph_diversity.describe(
            self.df_tessellation["cat"], statistics=["nunique"]
        )["nunique"]
        un_old = mm.Unique(self.df_tessellation, "cat", self.sw, "uID").series
        assert_series_equal(un_new, un_old, check_dtype=False, check_names=False)

        self.df_tessellation.loc[0, "cat"] = np.nan
        un_new = self.graph_diversity.describe(
            self.df_tessellation["cat"], statistics=["nunique"]
        )["nunique"]
        un_old = mm.Unique(
            self.df_tessellation, "cat", self.sw, "uID", dropna=True
        ).series
        assert_series_equal(un_new, un_old, check_dtype=False, check_names=False)

    def test_unweighted_percentile(self):
        sw = mm.sw_high(k=3, gdf=self.df_tessellation, ids="uID")
        graph = (
            Graph.build_contiguity(self.df_tessellation)
            .higher_order(k=3, lower_order=True)
            .assign_self_weight()
        )

        perc_new = mm.percentile(self.df_tessellation["area"], graph)
        perc_old = mm.Percentiles(
            self.df_tessellation, "area", sw, "uID", interpolation="hazen"
        ).frame
        assert_frame_equal(perc_new, perc_old, check_dtype=False, check_names=False)

        perc_new = mm.percentile(
            pd.Series(list(range(8)) * 18, index=self.df_tessellation.index), graph
        )
        perc_old = mm.Percentiles(
            self.df_tessellation, list(range(8)) * 18, sw, "uID", interpolation="hazen"
        ).frame
        assert_frame_equal(perc_new, perc_old, check_dtype=False, check_names=False)

        perc_new = mm.percentile(self.df_tessellation["area"], graph, q=[30, 70])
        perc_old = mm.Percentiles(
            self.df_tessellation,
            "area",
            sw,
            "uID",
            interpolation="hazen",
            percentiles=[30, 70],
        ).frame
        assert_frame_equal(perc_new, perc_old, check_dtype=False, check_names=False)

    def test_distance_decay_linearly_weighted_percentiles(self):
        sw = mm.sw_high(k=3, gdf=self.df_tessellation, ids="uID")

        perc_new = mm.percentile(self.df_tessellation["area"], self.decay_graph)
        perc_old = mm.Percentiles(
            self.df_tessellation,
            "area",
            sw,
            "uID",
            weighted="linear",
            verbose=False,
        ).frame

        assert_frame_equal(perc_new, perc_old, check_dtype=False, check_names=False)

        perc_new = mm.percentile(
            self.df_tessellation["area"],
            self.decay_graph,
            q=[30, 70],
        )

        perc_old = mm.Percentiles(
            self.df_tessellation,
            "area",
            sw,
            "uID",
            percentiles=[30, 70],
            weighted="linear",
            verbose=False,
        ).frame
        assert_frame_equal(perc_new, perc_old, check_dtype=False, check_names=False)

    def test_mean_deviation(self):
        street_graph = Graph.build_contiguity(self.df_streets, rook=False)
        y = mm.orientation(self.df_streets)
        deviations_new = mm.mean_deviation(y, street_graph)
        deviations_old = mm.NeighboringStreetOrientationDeviation(
            self.df_streets
        ).series

        assert_series_equal(deviations_new, deviations_old, check_names=False)
