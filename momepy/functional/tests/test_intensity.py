import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from libpysal.graph import Graph
from packaging.version import Version
from pandas.testing import assert_series_equal

import momepy as mm

from .conftest import assert_result

PD_210 = Version(pd.__version__) >= Version("2.1.0")


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

    def test_density(self):
        sw = mm.sw_high(k=3, gdf=self.df_tessellation, ids="uID")
        graph = (
            Graph.build_contiguity(self.df_tessellation, rook=False)
            .higher_order(k=3, lower_order=True)
            .assign_self_weight()
        )

        fl_area = graph.describe(self.df_buildings["fl_area"])["sum"]
        tess_area = graph.describe(self.df_tessellation["area"])["sum"]
        dens_new = fl_area / tess_area

        dens_old = mm.Density(
            self.df_tessellation,
            self.df_buildings["fl_area"],
            sw,
            "uID",
            self.df_tessellation.area,
        ).series

        assert_series_equal(
            dens_new,
            dens_old,
            check_names=False,
            check_dtype=False,
            check_index_type=False,
        )

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
