import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from libpysal.graph import Graph
from packaging.version import Version
from shapely import Polygon

import momepy as mm

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
