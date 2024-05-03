import geopandas as gpd
import numpy as np
import pytest
from libpysal.graph import Graph
from pandas.testing import assert_series_equal
from shapely import Point

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

    def test_count(self):
        eib = mm.count(self.blocks, self.df_buildings, self.df_buildings["bID"])
        eib_expected = {"count": 8, "min": 8, "max": 26, "mean": 18.0}
        assert_result(eib, eib_expected, self.blocks)

        weib = mm.count(self.blocks, self.df_buildings, self.df_buildings["bID"], True)
        weib_expected = {
            "count": 8,
            "min": 0.0001518198309906747,
            "max": 0.0006104358352940897,
            "mean": 0.00040170607189454006,
        }
        assert_result(weib, weib_expected, self.blocks)

        weis = mm.count(
            self.df_streets, self.df_buildings, self.df_buildings["nID"], weighted=True
        )
        weis_expected = {
            "count": 35,
            "min": 0.0,
            "max": 0.07663465523823784,
            "mean": 0.020524232642849215,
        }
        assert_result(weis, weis_expected, self.df_streets)

        check_eib = (
            gpd.sjoin(self.df_buildings.drop(columns="bID"), self.blocks)["bID"]
            .value_counts()
            .sort_index()
        )
        assert_series_equal(check_eib, eib, check_names=False)

        point_gdf = gpd.GeoDataFrame(
            {"nID": [0]}, geometry=[Point(1603569.010067892, 6464302.821695424)]
        )
        with pytest.raises(
            TypeError, match="Geometry type does not support weighting."
        ):
            mm.count(point_gdf, self.blocks, point_gdf["nID"], weighted=True)


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

    def test_count(self):
        eib_new = mm.count(self.blocks, self.df_buildings, self.df_buildings["bID"])
        eib_old = mm.Count(self.blocks, self.df_buildings, "bID", "bID").series
        assert_series_equal(eib_new, eib_old, check_names=False, check_dtype=False)

        weib_new = mm.count(
            self.blocks, self.df_buildings, self.df_buildings["bID"], True
        )
        weib_old = mm.Count(
            self.blocks, self.df_buildings, "bID", "bID", weighted=True
        ).series
        assert_series_equal(weib_new, weib_old, check_names=False, check_dtype=False)

        weis_new = mm.count(
            self.df_streets, self.df_buildings, self.df_buildings["nID"], weighted=True
        )
        weis_old = mm.Count(
            self.df_streets, self.df_buildings, "nID", "nID", weighted=True
        ).series
        assert_series_equal(weis_new, weis_old, check_names=False, check_dtype=False)
