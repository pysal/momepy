import geopandas as gpd
from libpysal.graph import Graph
from pandas.testing import assert_series_equal

import momepy as mm

from .conftest import assert_result


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
        assert_result(r, expected, self.df_buildings, rel=1e-4)

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

    def test_shared_walls_approx(self):
        expected = {
            "mean": 36.87618331446485,
            "sum": 5310.17039728293,
            "min": 0,
            "max": 106.20917523555639,
        }
        tolerance = 0.1
        geometry = self.df_buildings.buffer(-tolerance)
        r = mm.shared_walls(geometry)
        assert (r == 0).all()

        r = mm.shared_walls(geometry, strict=False, tolerance=tolerance + 0.001)

        # check that values are equal to strict version up to 10cm
        assert_result(r, expected, self.df_buildings, rel=1e-1)

    def test_alignment(self):
        orientation = mm.orientation(self.df_buildings)
        expected = {
            "mean": 2.90842367974375,
            "sum": 418.8130098831,
            "min": 0.03635249292455285,
            "max": 21.32311946014944,
        }
        r = mm.alignment(orientation, self.graph)
        assert_result(r, expected, self.df_buildings, check_names=False, rel=1e-4)

    def test_neighbor_distance(self):
        expected = {
            "mean": 14.254601392635818,
            "sum": 2052.662600539558,
            "min": 2.0153493186952085,
            "max": 42.164831456311475,
        }
        r = mm.neighbor_distance(self.df_buildings, self.graph)
        assert_result(r, expected, self.df_buildings, check_names=False)

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

    def test_street_alignment(self):
        building_orientation = mm.orientation(self.df_buildings)
        street_orientation = mm.orientation(self.df_streets)
        street_index = mm.get_nearest_street(self.df_buildings, self.df_streets)
        expected = {
            "mean": 2.024707906317863,
            "sum": 291.5579385097722,
            "min": 0.0061379200252815,
            "max": 20.357934749623894,
        }
        r = mm.street_alignment(building_orientation, street_orientation, street_index)
        assert_result(r, expected, self.df_buildings, rel=1e-3)

    def test_cell_alignment(self):
        df_buildings = self.df_buildings.reset_index()
        df_tessellation = self.df_tessellation.reset_index()
        blgori = mm.orientation(df_buildings)
        tessori = mm.orientation(df_tessellation)

        align = mm.cell_alignment(blgori, tessori)
        align2 = mm.cell_alignment(blgori.values, tessori.values)
        align_expected = {
            "count": 144,
            "mean": 2.604808585700,
            "max": 33.201625570390746,
            "min": 1.722848278973288e-05,
        }
        assert_result(align, align_expected, df_buildings, abs=1e-2)
        assert_series_equal(align, align2, check_names=False)
