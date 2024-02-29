import geopandas as gpd

import momepy as mm

from .test_shape import assert_result


class TestDistribution:
    def setup_method(self):
        test_file_path = mm.datasets.get_path("bubenec")
        self.df_buildings = gpd.read_file(test_file_path, layer="buildings")
        self.df_streets = gpd.read_file(test_file_path, layer="streets")

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
