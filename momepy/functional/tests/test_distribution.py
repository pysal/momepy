import geopandas as gpd
from libpysal.graph import Graph
from pandas.testing import assert_series_equal

import momepy as mm

from .test_shape import assert_result


class TestDistribution:
    def setup_method(self):
        test_file_path = mm.datasets.get_path("bubenec")
        self.df_buildings = gpd.read_file(test_file_path, layer="buildings")
        self.df_streets = gpd.read_file(test_file_path, layer="streets")
        self.df_tessellation = gpd.read_file(test_file_path, layer="tessellation")
        self.graph = Graph.build_knn(self.df_buildings.centroid, k=5)
        self.contiguity = Graph.build_contiguity(self.df_buildings)
        self.neighborhood_graph = self.graph.higher_order(3, lower_order=True)
        self.tess_contiguity = Graph.build_contiguity(self.df_tessellation)

    def test_orientation(self):
        expected = {
            "mean": 20.983859394267952,
            "sum": 3021.6757527745854,
            "min": 7.968673890244247,
            "max": 42.329365250279125,
        }
        r = mm.orientation(self.df_buildings)
        assert_result(r, expected, self.df_buildings)

        expected = {
            "mean": 21.176405050561755,
            "sum": 741.1741767696615,
            "min": 0.834911325974133,
            "max": 44.83357900046826,
        }
        r = mm.orientation(self.df_streets)
        assert_result(r, expected, self.df_streets)

    def test_shared_walls(self):
        expected = {
            "mean": 36.87618331446485,
            "sum": 5310.17039728293,
            "min": 0,
            "max": 106.20917523555639,
        }
        r = mm.shared_walls(self.df_buildings)
        assert_result(r, expected, self.df_buildings)

    def test_alignment(self):
        orientation = mm.orientation(self.df_buildings)
        expected = {
            "mean": 2.90842367974375,
            "sum": 418.8130098831,
            "min": 0.03635249292455285,
            "max": 21.32311946014944,
        }
        r = mm.alignment(orientation, self.graph)
        assert_result(r, expected, self.df_buildings)

    def test_neighbor_distance(self):
        expected = {
            "mean": 14.254601392635818,
            "sum": 2052.662600539558,
            "min": 2.0153493186952085,
            "max": 42.164831456311475,
        }
        r = mm.neighbor_distance(self.df_buildings, self.graph)
        assert_result(r, expected, self.df_buildings)

    def test_mean_interbuilding_distance(self):
        expected = {
            "mean": 13.018190603684694,
            "sum": 1874.6194469305958,
            "min": 6.623582625492466,
            "max": 22.513464171665948,
        }
        r = mm.mean_interbuilding_distance(
            self.df_buildings, self.graph, self.neighborhood_graph
        )
        assert_result(r, expected, self.df_buildings)

    def test_building_adjacency(self):
        expected = {
            "mean": 0.3784722222222222,
            "sum": 54.5,
            "min": 0.16666666666666666,
            "max": 0.8333333333333334,
        }
        r = mm.building_adjacency(self.contiguity, self.graph)
        assert_result(r, expected, self.df_buildings, exact=False)

    def test_neighbors(self):
        expected = {
            "mean": 5.180555555555555,
            "sum": 746,
            "min": 2,
            "max": 12,
        }
        r = mm.neighbors(self.df_tessellation, self.tess_contiguity, weighted=False)
        assert_result(r, expected, self.df_buildings, exact=False, check_names=False)

        expected = {
            "mean": 0.029066398893536072,
            "sum": 4.185561440669194,
            "min": 0.008659386154613532,
            "max": 0.08447065801729325,
        }
        r = mm.neighbors(self.df_tessellation, self.tess_contiguity, weighted=True)
        assert_result(r, expected, self.df_buildings, exact=False, check_names=False)


class TestEquality:
    def setup_method(self):
        test_file_path = mm.datasets.get_path("bubenec")
        self.df_buildings = gpd.read_file(test_file_path, layer="buildings").set_index(
            "uID"
        )
        self.df_tessellation = gpd.read_file(
            test_file_path, layer="tessellation"
        ).set_index("uID")
        self.graph = Graph.build_knn(self.df_buildings.centroid, k=5)
        self.df_buildings["orientation"] = mm.orientation(self.df_buildings)
        self.contiguity = Graph.build_contiguity(self.df_buildings)
        self.tessellation_contiguity = Graph.build_contiguity(self.df_tessellation)
        self.neighborhood_graph = self.tessellation_contiguity.higher_order(
            3, lower_order=True
        )

    def test_alignment(self):
        new = mm.alignment(self.df_buildings["orientation"], self.graph)
        old = mm.Alignment(
            self.df_buildings.reset_index(),
            self.graph.to_W(),
            "uID",
            "orientation",
            verbose=False,
        ).series
        assert_series_equal(new, old, check_names=False, check_index=False)

    def test_neighbor_distance(self):
        new = mm.neighbor_distance(self.df_buildings, self.graph)
        old = mm.NeighborDistance(
            self.df_buildings.reset_index(), self.graph.to_W(), "uID", verbose=False
        ).series
        assert_series_equal(new, old, check_names=False, check_index=False)

    def test_mean_interbuilding_distance(self):
        new = mm.mean_interbuilding_distance(
            self.df_buildings, self.tessellation_contiguity, self.neighborhood_graph
        )
        old = mm.MeanInterbuildingDistance(
            self.df_buildings.reset_index(),
            self.tessellation_contiguity.to_W(),
            "uID",
            verbose=False,
        ).series
        assert_series_equal(new, old, check_names=False, check_index=False)

    def test_building_adjacency(self):
        new = mm.building_adjacency(self.contiguity, self.graph)
        old = mm.BuildingAdjacency(
            self.df_buildings.reset_index(), self.graph.to_W(), "uID", verbose=False
        ).series
        assert_series_equal(new, old, check_names=False, check_index=False)

    def test_neighbors(self):
        new = mm.neighbors(
            self.df_tessellation, self.tessellation_contiguity, weighted=False
        )
        old = mm.Neighbors(
            self.df_tessellation.reset_index(),
            self.tessellation_contiguity.to_W(),
            "uID",
            weighted=False,
            verbose=False,
        ).series
        assert_series_equal(new, old, check_names=False, check_index=False)

        new = mm.neighbors(
            self.df_tessellation, self.tessellation_contiguity, weighted=True
        )
        old = mm.Neighbors(
            self.df_tessellation.reset_index(),
            self.tessellation_contiguity.to_W(),
            "uID",
            weighted=True,
            verbose=False,
        ).series
        assert_series_equal(new, old, check_names=False, check_index=False)
