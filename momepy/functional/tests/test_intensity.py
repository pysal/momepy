import geopandas as gpd
import numpy as np
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
        check = 0.6805555555555556
        courtyards = mm.courtyards(self.df_buildings, self.buildings_graph)

        expected = {"mean": check}

        assert_result(courtyards, expected, self.df_buildings)


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
