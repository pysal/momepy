import geopandas as gpd
import numpy as np
import pandas as pd
from libpysal.graph import Graph
from pandas.testing import assert_series_equal

import momepy as mm

from .conftest import assert_result


class TestIntensity:
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

        self.buildings_graph = Graph.build_contiguity(
            self.df_buildings, rook=False
        ).assign_self_weight()

    def test_courtyards(self):
        courtyards = mm.courtyards(self.df_buildings, self.buildings_graph)
        expected = {"mean": 0.6805555555555556, "sum": 98, "min": 0, "max": 1}
        assert_result(courtyards, expected, self.df_buildings)

    def test_area_ratio(self):
        car_block = mm.area_ratio(
            self.blocks.geometry.area,
            self.df_buildings["area"],
            self.df_buildings["bID"],
        )
        car_block_expected = {
            "mean": 0.27619743196980123,
            "max": 0.35699143584461146,
            "min": 0.12975039475826336,
            "count": 8,
        }
        assert_result(car_block, car_block_expected, self.blocks)

        car = mm.area_ratio(
            self.df_tessellation.geometry.area,
            self.df_buildings["area"],
            self.df_buildings["uID"] - 1,
        )
        car2 = mm.area_ratio(
            self.df_tessellation.set_index("uID").area,
            self.df_buildings.set_index("uID").area,
            self.df_buildings["uID"].values,
        )
        car_expected = {
            "mean": 0.3206556897709747,
            "max": 0.8754071653707558,
            "min": 0.029097983413141276,
            "count": 144,
        }
        assert_result(car, car_expected, self.df_tessellation)
        assert_result(car2, car_expected, self.df_tessellation.set_index("uID"))

        car_sel = mm.area_ratio(
            self.df_tessellation.iloc[10:20]["area"],
            self.df_buildings["area"],
            self.df_tessellation.iloc[10:20]["uID"] - 1,
        )
        car_sel_expected = {
            "mean": 0.3892868062654601,
            "max": 0.5428192449477212,
            "min": 0.22057633949526625,
            "count": 10,
        }
        assert_result(car_sel, car_sel_expected, self.df_tessellation.iloc[10:20])

        far = mm.area_ratio(
            self.df_tessellation.geometry.area,
            self.df_buildings["fl_area"],
            self.df_buildings["uID"] - 1,
        )
        far_expected = {
            "mean": 1.910949846262234,
            "max": 7.003257322966046,
            "min": 0.26188185071827147,
            "count": 144,
        }
        assert_result(far, far_expected, self.df_tessellation)


class TestIntensityEquality:
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

        self.buildings_graph = Graph.build_contiguity(
            self.df_buildings, rook=False
        ).assign_self_weight()

    def test_courtyards(self):
        old_courtyards = mm.Courtyards(self.df_buildings).series
        new_courtyards = mm.courtyards(self.df_buildings, self.buildings_graph)
        assert_series_equal(
            new_courtyards, old_courtyards, check_names=False, check_dtype=False
        )

    def test_area_ratio(self):
        self.blocks["area"] = self.blocks.geometry.area
        car_block_new = mm.area_ratio(
            self.blocks.geometry.area,
            self.df_buildings["area"],
            self.df_buildings["bID"],
        )
        car_block_old = mm.AreaRatio(
            self.blocks, self.df_buildings, "area", "area", "bID"
        ).series
        assert_series_equal(
            car_block_new, car_block_old, check_dtype=False, check_names=False
        )

        car_new = mm.area_ratio(
            self.df_tessellation.geometry.area,
            self.df_buildings["area"],
            self.df_buildings["uID"] - 1,
        )
        car2_new = mm.area_ratio(
            self.df_tessellation.set_index("uID").area,
            self.df_buildings.set_index("uID").area,
            self.df_buildings["uID"].values,
        )
        car_old = mm.AreaRatio(
            self.df_tessellation, self.df_buildings, "area", "area", "uID"
        ).series
        assert_series_equal(car_new, car_old, check_dtype=False, check_names=False)
        assert_series_equal(
            pd.Series(car_old.values, car2_new.index),
            car2_new,
            check_dtype=False,
            check_names=False,
        )

        car_sel = mm.AreaRatio(
            self.df_tessellation.iloc[10:20], self.df_buildings, "area", "area", "uID"
        ).series
        car_sel_new = mm.area_ratio(
            self.df_tessellation.iloc[10:20]["area"],
            self.df_buildings["area"],
            self.df_tessellation.iloc[10:20]["uID"] - 1,
        )

        assert_series_equal(car_sel_new, car_sel, check_dtype=False, check_names=False)

        far_new = mm.area_ratio(
            self.df_tessellation.geometry.area,
            self.df_buildings["fl_area"],
            self.df_buildings["uID"] - 1,
        )

        far_old = mm.AreaRatio(
            self.df_tessellation,
            self.df_buildings,
            self.df_tessellation.area,
            self.df_buildings.fl_area,
            "uID",
        ).series

        assert_series_equal(far_new, far_old, check_dtype=False, check_names=False)
