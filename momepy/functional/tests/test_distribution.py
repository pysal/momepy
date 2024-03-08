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
        self.graph = Graph.build_knn(self.df_buildings.centroid, k=5)

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
            "mean": 16.46438739026651,
            "sum": 2370.871784198377,
            "min": 12.279734781239485,
            "max": 25.45874022563638,
        }
        r = mm.mean_interbuilding_distance(self.df_buildings, self.graph)
        assert_result(r, expected, self.df_buildings)


class TestEquality:
    def setup_method(self):
        test_file_path = mm.datasets.get_path("bubenec")
        self.df_buildings = gpd.read_file(test_file_path, layer="buildings").set_index(
            "uID"
        )
        self.graph = Graph.build_knn(self.df_buildings.centroid, k=5)
        self.df_buildings["orientation"] = mm.orientation(self.df_buildings)

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
        new = mm.mean_interbuilding_distance(self.df_buildings, self.graph)
        old = mm.MeanInterbuildingDistance(
            self.df_buildings.reset_index(), self.graph.to_W(), "uID", verbose=False
        ).series
        assert_series_equal(new, old, check_names=False, check_index=False)
