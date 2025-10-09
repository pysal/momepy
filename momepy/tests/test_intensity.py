import geopandas as gpd
import networkx as nx
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
        # self.df_streets["nID"] = mm.unique_id(self.df_streets)
        self.df_buildings["height"] = np.linspace(10.0, 30.0, 144)
        self.df_tessellation["area"] = self.df_tessellation.geometry.area
        self.df_buildings["area"] = self.df_buildings.geometry.area
        self.df_buildings["fl_area"] = mm.floor_area(
            self.df_buildings.area, self.df_buildings["height"]
        )
        self.df_buildings["nID"] = mm.get_nearest_street(
            self.df_buildings, self.df_streets
        )
        self.blocks, ids = mm.generate_blocks(
            self.df_tessellation,
            self.df_streets,
            self.df_buildings,
        )
        self.df_buildings["bID"] = ids
        self.df_tessellation["bID"] = ids

        self.buildings_graph = Graph.build_contiguity(
            self.df_buildings, rook=False
        ).assign_self_weight()

    def test_courtyards(self):
        courtyards = mm.courtyards(self.df_buildings, self.buildings_graph)
        expected = {"mean": 0.6805555555555556, "sum": 98, "min": 0, "max": 1}
        assert_result(courtyards, expected, self.df_buildings)

    def test_courtyards_buffer(self):
        buildings = self.df_buildings.copy()
        buildings["geometry"] = buildings.simplify(0.10)
        new_courtyards = mm.courtyards(buildings, self.buildings_graph)
        old_courtyards = mm.courtyards(self.df_buildings, self.buildings_graph)
        assert (new_courtyards.values != old_courtyards.values).any()

        courtyards = mm.courtyards(buildings, self.buildings_graph, buffer=0.25)
        expected = {"mean": 0.6805555555555556, "sum": 98, "min": 0, "max": 1}
        assert_result(courtyards, expected, self.df_buildings)

    def test_node_density(self):
        g = mm.gdf_to_nx(self.df_streets, integer_labels=True)
        g = mm.node_degree(g)
        nodes, edges, w = mm.nx_to_gdf(g, spatial_weights=True)

        g = mm.node_density(g, radius=3)
        density = pd.Series(nx.get_node_attributes(g, "node_density"))
        expected_density = {
            "count": 29,
            "mean": 0.005534125924228438,
            "max": 0.010177844322387136,
            "min": 0.00427032489140038,
        }
        assert_result(density, expected_density, nodes, check_names=False)

        weighted = pd.Series(nx.get_node_attributes(g, "node_density_weighted"))
        expected_weighted = {
            "count": 29,
            "mean": 0.010090861332429164,
            "max": 0.020355688644774272,
            "min": 0.0077472994887720905,
        }
        assert_result(weighted, expected_weighted, nodes, check_names=False)

        # two API equivalence
        g = mm.gdf_to_nx(self.df_streets, integer_labels=True)
        g = mm.node_degree(g)
        alternative_g = mm.subgraph(g, radius=3)
        alternative_density = pd.Series(
            nx.get_node_attributes(alternative_g, "node_density")
        )
        alternative_weighted = pd.Series(
            nx.get_node_attributes(alternative_g, "node_density_weighted")
        )
        assert_series_equal(alternative_density, density)
        assert_series_equal(alternative_weighted, weighted)

    def test_area_ratio(self):
        def area_ratio(overlay, covering, agg_key):
            res = mm.describe_agg(covering, agg_key)
            return res["sum"] / overlay

        car_block = area_ratio(
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

        assert_result(
            car_block, car_block_expected, self.blocks, exact=False, check_names=False
        )

        car = area_ratio(
            self.df_tessellation.geometry.area,
            self.df_buildings["area"],
            self.df_buildings["uID"] - 1,
        )
        car2 = area_ratio(
            self.df_tessellation.set_index("uID").area,
            self.df_buildings.set_index("uID").area,
            self.df_tessellation.set_index("uID").index,
        )

        car_expected = {
            "mean": 0.3206556897709747,
            "max": 0.8754071653707558,
            "min": 0.029097983413141276,
            "count": 144,
        }
        assert_result(
            car, car_expected, self.df_tessellation, exact=False, check_names=False
        )
        assert_result(
            car2,
            car_expected,
            self.df_tessellation.set_index("uID"),
            exact=False,
            check_names=False,
        )

        car_sel = area_ratio(
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
        assert_result(
            car_sel,
            car_sel_expected,
            self.df_tessellation.iloc[10:20],
            exact=False,
            check_names=False,
        )

        far = area_ratio(
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
        assert_result(
            far, far_expected, self.df_tessellation, exact=False, check_names=False
        )
