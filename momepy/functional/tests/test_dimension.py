import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from libpysal.graph import Graph
from packaging.version import Version
from pandas.testing import assert_series_equal
from shapely import Polygon

import momepy as mm

from .conftest import assert_result

GPD_013 = Version(gpd.__version__) >= Version("0.13")


class TestDimensions:
    def setup_method(self):
        test_file_path = mm.datasets.get_path("bubenec")
        self.df_buildings = gpd.read_file(test_file_path, layer="buildings")
        self.df_streets = gpd.read_file(test_file_path, layer="streets")
        self.df_tessellation = gpd.read_file(test_file_path, layer="tessellation")
        self.df_buildings["height"] = np.linspace(10.0, 30.0, 144)

    def test_volume(self):
        # pandas
        expected = self.df_buildings.area * self.df_buildings["height"]
        pd.testing.assert_series_equal(
            mm.volume(self.df_buildings.area, self.df_buildings["height"]), expected
        )

        # numpy
        expected = self.df_buildings.area.values * self.df_buildings["height"].values
        np.testing.assert_array_equal(
            mm.volume(
                self.df_buildings.area.values, self.df_buildings["height"].values
            ),
            expected,
        )

    def test_floor_area(self):
        expected = self.df_buildings.area * (self.df_buildings["height"] // 3)
        pd.testing.assert_series_equal(
            mm.floor_area(self.df_buildings.area, self.df_buildings["height"]), expected
        )

        expected = self.df_buildings.area * (self.df_buildings["height"] // 5)
        pd.testing.assert_series_equal(
            mm.floor_area(
                self.df_buildings.area, self.df_buildings["height"], floor_height=5
            ),
            expected,
        )

        floor_height = np.repeat(np.array([3, 4]), 72)
        expected = self.df_buildings.area * (
            self.df_buildings["height"] // floor_height
        )
        pd.testing.assert_series_equal(
            mm.floor_area(
                self.df_buildings.area,
                self.df_buildings["height"],
                floor_height=floor_height,
            ),
            expected,
        )

    def test_courtyard_area(self):
        expected = self.df_buildings.geometry.apply(
            lambda geom: Polygon(geom.exterior).area - geom.area
        )
        pd.testing.assert_series_equal(
            mm.courtyard_area(self.df_buildings), expected, check_names=False
        )

    @pytest.mark.skipif(not GPD_013, reason="minimum_bounding_radius() not available")
    def test_longest_axis_length(self):
        expected = self.df_buildings.minimum_bounding_radius() * 2
        pd.testing.assert_series_equal(
            mm.longest_axis_length(self.df_buildings), expected, check_names=False
        )

    def test_perimeter_wall(self):
        result = mm.perimeter_wall(self.df_buildings)
        adj = Graph.build_contiguity(self.df_buildings)
        result_given_graph = mm.perimeter_wall(self.df_buildings, adj)

        pd.testing.assert_series_equal(result, result_given_graph)
        assert result[0] == pytest.approx(137.210, rel=1e-3)

    @pytest.mark.skipif(not GPD_013, reason="no attribute 'segmentize'")
    def test_street_profile(self):
        sp = mm.street_profile(
            self.df_streets,
            self.df_buildings,
            tick_length=50,
            distance=1,
            heights=self.df_buildings["height"],
        )

        expected_w = {
            "count": 35,
            "mean": 43.39405649921903,
            "min": 31.017731525484447,
            "max": 50.0,
        }
        expected_wd = {
            "count": 22,
            "mean": 1.1977356963898373,
            "min": 0.09706119360586668,
            "max": 5.154163996499861,
        }
        expected_o = {
            "count": 35,
            "mean": 0.7186966927475066,
            "min": 0.15270935960591134,
            "max": 1.0,
        }
        expected_h = {
            "count": 22,
            "mean": 16.72499857958264,
            "min": 10.0,
            "max": 28.1969381969382,
        }
        expected_hd = {
            "count": 22,
            "mean": 1.2372251098113227,
            "min": 0.0,
            "max": 7.947097088834963,
        }
        expected_p = {
            "count": 22,
            "mean": 0.43257459410448046,
            "min": 0.20379941361273096,
            "max": 0.7432052069071473,
        }

        assert_result(sp["width"], expected_w, self.df_streets)
        assert_result(sp["width_deviation"], expected_wd, self.df_streets)
        assert_result(sp["openness"], expected_o, self.df_streets)
        assert_result(sp["height"], expected_h, self.df_streets)
        assert_result(sp["height_deviation"], expected_hd, self.df_streets)
        assert_result(sp["hw_ratio"], expected_p, self.df_streets)

    def test_street_profile_infinity(self):
        # avoid infinity
        from shapely import LineString, Point

        blg = gpd.GeoDataFrame(
            {"height": [2, 5]},
            geometry=[
                Point(0, 0).buffer(10, cap_style=3),
                Point(30, 0).buffer(10, cap_style=3),
            ],
        )
        lines = gpd.GeoDataFrame(
            geometry=[LineString([(-8, -8), (8, 8)]), LineString([(15, -10), (15, 10)])]
        )
        assert mm.StreetProfile(lines, blg, "height", 2).p.equals(
            pd.Series([np.nan, 0.35])
        )


class TestDimensionEquivalence:
    def setup_method(self):
        test_file_path = mm.datasets.get_path("bubenec")
        self.df_buildings = gpd.read_file(test_file_path, layer="buildings")
        self.df_streets = gpd.read_file(test_file_path, layer="streets")
        self.df_buildings["height"] = np.linspace(10.0, 30.0, 144)

    @pytest.mark.skipif(not GPD_013, reason="no attribute 'segmentize'")
    def test_street_profile(self):
        sp_new = mm.street_profile(
            self.df_streets,
            self.df_buildings,
            tick_length=50,
            distance=1,
            heights=self.df_buildings["height"],
        )
        sp_old = mm.StreetProfile(
            self.df_streets,
            self.df_buildings,
            tick_length=50,
            distance=1,
            heights=self.df_buildings["height"],
        )

        ## needs tolerance because the generate ticks are different
        assert_series_equal(sp_new["width"], sp_old.w, rtol=1e-2, check_names=False)
        assert_series_equal(
            sp_new["width_deviation"].replace(np.nan, 0),
            sp_old.wd,
            rtol=1,
            check_names=False,
        )
        assert_series_equal(sp_new["openness"], sp_old.o, rtol=1e-1, check_names=False)
        assert_series_equal(
            sp_new["height"].replace(np.nan, 0), sp_old.h, check_names=False, rtol=1e-1
        )
        assert_series_equal(
            sp_new["height_deviation"].replace(np.nan, 0),
            sp_old.hd,
            check_names=False,
            rtol=1e-1,
        )
        assert_series_equal(
            sp_new["hw_ratio"].replace(np.nan, 0), sp_old.p, check_names=False, rtol=1
        )
