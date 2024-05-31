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
        eib = mm.count(self.blocks, self.df_buildings["bID"])
        eib_expected = {"count": 8, "min": 8, "max": 26, "mean": 18.0}
        assert_result(eib, eib_expected, self.blocks)

        weib = mm.count(self.blocks, self.df_buildings["bID"], True)
        weib_expected = {
            "count": 8,
            "min": 0.0001518198309906747,
            "max": 0.0006104358352940897,
            "mean": 0.00040170607189454006,
        }
        assert_result(weib, weib_expected, self.blocks)

        weis = mm.count(self.df_streets, self.df_buildings["nID"], weighted=True)
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
            mm.count(point_gdf, point_gdf["nID"], weighted=True)

    def test_node_density(self):
        nx = mm.gdf_to_nx(self.df_streets, integer_labels=True)
        nx = mm.node_degree(nx)
        nodes, edges, w = mm.nx_to_gdf(nx, spatial_weights=True)
        g = Graph.from_W(w).higher_order(k=3, lower_order=True).assign_self_weight()

        density = mm.node_density(nodes, edges, g)
        expected_density = {
            "count": 29,
            "mean": 0.005534125924228438,
            "max": 0.010177844322387136,
            "min": 0.00427032489140038,
        }
        assert_result(density, expected_density, nodes, check_names=False)

        weighted = mm.node_density(nodes, edges, g, weighted=True)
        expected_weighted = {
            "count": 29,
            "mean": 0.010090861332429164,
            "max": 0.020355688644774272,
            "min": 0.0077472994887720905,
        }
        assert_result(weighted, expected_weighted, nodes, check_names=False)

        island = mm.node_density(nodes, edges, Graph.from_W(w).assign_self_weight())
        expected_island = {
            "count": 29,
            "mean": 0.01026753724860306,
            "max": 0.029319191032027746,
            "min": 0.004808273240207287,
        }
        assert_result(island, expected_island, nodes, check_names=False)

        with pytest.raises(
            ValueError,
            match=("Column node_start is needed in the edges GeoDataframe."),
        ):
            mm.node_density(nodes, nodes, g)

        with pytest.raises(
            ValueError,
            match=("Column node_end is needed in the edges GeoDataframe."),
        ):
            mm.node_density(nodes, edges["node_start"].to_frame(), g)

        with pytest.raises(
            ValueError,
            match=("Column degree is needed in nodes GeoDataframe."),
        ):
            mm.node_density(edges, edges, g, weighted=True)


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
        eib_new = mm.count(self.blocks, self.df_buildings["bID"])
        eib_old = mm.Count(self.blocks, self.df_buildings, "bID", "bID").series
        assert_series_equal(eib_new, eib_old, check_names=False, check_dtype=False)

        weib_new = mm.count(self.blocks, self.df_buildings["bID"], True)
        weib_old = mm.Count(
            self.blocks, self.df_buildings, "bID", "bID", weighted=True
        ).series
        assert_series_equal(weib_new, weib_old, check_names=False, check_dtype=False)

        weis_new = mm.count(self.df_streets, self.df_buildings["nID"], weighted=True)
        weis_old = mm.Count(
            self.df_streets, self.df_buildings, "nID", "nID", weighted=True
        ).series
        assert_series_equal(weis_new, weis_old, check_names=False, check_dtype=False)

    def test_node_density(self):
        nx = mm.gdf_to_nx(self.df_streets, integer_labels=True)
        nx = mm.node_degree(nx)
        nodes, edges, w = mm.nx_to_gdf(nx, spatial_weights=True)
        sw = mm.sw_high(k=3, weights=w)
        g = Graph.from_W(w).higher_order(k=3, lower_order=True).assign_self_weight()

        density_old = mm.NodeDensity(nodes, edges, sw).series
        density_new = mm.node_density(nodes, edges, g)
        assert_series_equal(
            density_old, density_new, check_names=False, check_dtype=False
        )

        weighted_old = mm.NodeDensity(
            nodes, edges, sw, weighted=True, node_degree="degree"
        ).series
        weighted_new = mm.node_density(nodes, edges, g, weighted=True)
        assert_series_equal(
            weighted_old, weighted_new, check_names=False, check_dtype=False
        )

        islands_old = mm.NodeDensity(nodes, edges, w).series
        islands_new = mm.node_density(
            nodes, edges, Graph.from_W(w).assign_self_weight()
        )
        assert_series_equal(
            islands_old, islands_new, check_names=False, check_dtype=False
        )
