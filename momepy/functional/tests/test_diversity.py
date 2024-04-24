import geopandas as gpd
import pytest
from libpysal.graph import Graph
from packaging.version import Version
from pandas.testing import assert_frame_equal

import momepy as mm

from .conftest import assert_result

GPD_013 = Version(gpd.__version__) >= Version("0.13")


class TestDistribution:
    def setup_method(self):
        test_file_path = mm.datasets.get_path("bubenec")
        self.df_buildings = gpd.read_file(test_file_path, layer="buildings")
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
