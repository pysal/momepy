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
        self.graph = (
            Graph.build_contiguity(self.df_tessellation)
            .higher_order(k=3, lower_order=True)
            .assign_self_weight()
        )

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

    def test_covered_area(self):
        covered_sw = mm.covered_area(self.df_tessellation.area, self.graph)

        covered_sw2 = mm.covered_area(self.df_tessellation.area.values, self.graph)

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

    def test_weighted_char(self):
        weighted = mm.weighted_character(
            self.df_buildings.height, self.df_buildings.area, self.graph
        )
        weighted_expected = {
            "sum": 2703.7438005082204,
            "mean": 18.775998614640418,
            "min": 11.005171818893885,
            "max": 25.424162063245504,
        }
        assert_result(
            weighted,
            weighted_expected,
            self.df_tessellation,
            check_names=False,
            exact=False,
        )


class TestDimensionEquivalence:
    def setup_method(self):
        test_file_path = mm.datasets.get_path("bubenec")
        self.df_buildings = gpd.read_file(test_file_path, layer="buildings")
        self.df_streets = gpd.read_file(test_file_path, layer="streets")
        self.df_tessellation = gpd.read_file(test_file_path, layer="tessellation")
        self.df_buildings["height"] = np.linspace(10.0, 30.0, 144)
        self.sw = mm.sw_high(k=3, gdf=self.df_tessellation, ids="uID")
        self.graph = (
            Graph.build_contiguity(self.df_tessellation)
            .higher_order(k=3, lower_order=True)
            .assign_self_weight()
        )

    def test_covered_area(self):
        covered_sw_new = mm.covered_area(self.df_tessellation.area, self.graph)
        covered_sw_old = mm.CoveredArea(self.df_tessellation, self.sw, "uID").series
        assert_series_equal(
            covered_sw_new, covered_sw_old, check_names=False, check_index_type=False
        )

    def test_weighted_char(self):
        weighted_new = mm.weighted_character(
            self.df_buildings.height, self.df_buildings.area, self.graph
        )
        weighted_old = mm.WeightedCharacter(
            self.df_buildings, "height", self.sw, "uID"
        ).series
        assert_series_equal(
            weighted_new, weighted_old, check_names=False, check_index_type=False
        )
