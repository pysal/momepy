import geopandas as gpd
import pandas as pd
import pytest
from pandas.testing import assert_index_equal, assert_series_equal

import momepy as mm


class TestCOINS:
    def setup_method(self):
        test_file_path = mm.datasets.get_path("bubenec")
        self.gdf = gpd.read_file(test_file_path, layer="streets")

    def test_stroke_gdf(self):
        coins = mm.COINS(self.gdf)
        result = coins.stroke_gdf()

        assert result.shape == (10, 2)

        expected_index = pd.Index(list(range(10)), name="stroke_group")
        assert_index_equal(result.index, expected_index)

        expected_segments = pd.Series(
            [8, 19, 17, 13, 5, 14, 2, 3, 3, 5],
            name="n_segments",
            index=expected_index,
        )
        assert_series_equal(result["n_segments"], expected_segments)

        assert result.length.sum() == pytest.approx(self.gdf.length.sum())

        expected = pd.Series(
            [
                839.5666838320316,
                759.0900425060918,
                744.7579337248078,
                1019.7095084794428,
                562.2466914415573,
                1077.3606756995746,
                193.04063727323836,
                187.49184699173748,
                182.6849740039611,
                382.50195042922803,
            ],
            index=expected_index,
        )
        assert_series_equal(result.length, expected, atol=0.0001, check_exact=False)

    def test_stroke_attribute(self):
        coins = mm.COINS(self.gdf)
        result = coins.stroke_attribute()

        expected = pd.Series(
            [
                0,
                1,
                2,
                3,
                2,
                0,
                4,
                3,
                4,
                4,
                5,
                5,
                5,
                6,
                2,
                2,
                1,
                0,
                5,
                1,
                0,
                0,
                2,
                2,
                3,
                3,
                3,
                7,
                8,
                5,
                5,
                3,
                5,
                1,
                9,
            ]
        )

        assert_series_equal(result, expected)

    def test_premerge(self):
        coins = mm.COINS(self.gdf)
        result = coins._premerge()

        assert result.shape == (89, 8)
        expected_columns = pd.Index(
            [
                "orientation",
                "links_p1",
                "links_p2",
                "best_p1",
                "best_p2",
                "p1_final",
                "p2_final",
                "geometry",
            ]
        )
        assert_index_equal(result.columns, expected_columns)

        assert not result.isna().any().any()
