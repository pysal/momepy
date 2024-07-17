import geopandas as gpd
import pandas as pd
import pytest
from pandas.testing import assert_index_equal, assert_series_equal
from shapely.geometry import LineString

import momepy as mm


class TestCOINS:
    def setup_method(self):
        test_file_path = mm.datasets.get_path("bubenec")
        self.gdf = gpd.read_file(test_file_path, layer="streets")

    def test_stroke_gdf(self):
        coins = mm.COINS(self.gdf)
        result = coins.stroke_gdf()

        assert result.shape == (10, 2)

        expected_index = pd.RangeIndex(stop=10, name="stroke_group")
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

        result, ends = coins.stroke_attribute(return_ends=True)

        expected_ends = pd.Series(
            [
                True,
                False,
                True,
                False,
                True,
                False,
                True,
                True,
                False,
                True,
                True,
                False,
                False,
                True,
                False,
                False,
                False,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
                False,
                False,
                True,
                False,
                True,
                True,
            ]
        )
        assert_series_equal(result, expected)
        assert_series_equal(ends, expected_ends)

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

        assert not result.drop(columns="orientation").isna().any().any()

    def test_sharp_angles(self):
        # test case 1
        a = (0, 0)
        b = (2, 0)
        c = (1, 1)
        d = (2, 1)
        e = (-1, 1)

        line1 = [a, b]
        line2 = [a, c]
        line3 = [a, d]
        line4 = [a, e]

        gdf = gpd.GeoDataFrame(
            {
                "geometry": [
                    LineString(line1),
                    LineString(line2),
                    LineString(line3),
                    LineString(line4),
                ]
            }
        )

        coins = mm.COINS(gdf, angle_threshold=0)
        stroke_attr = coins.stroke_attribute()

        expected_groups = [
            bool(stroke_attr[0] == 0),
            bool(stroke_attr[1] == 1),
            bool(stroke_attr[2] == 2),
            bool(stroke_attr[3] == 0),
            bool(stroke_attr[0] != stroke_attr[2]),
        ]

        assert all(expected_groups)

        # test case 2

        a = (0, 0)
        b = (-1, 1)
        c = (-4, 1)
        d = (-2, -1)
        e = (-1, -2)

        line1 = [a, b]
        line2 = [a, c]
        line3 = [a, d]
        line4 = [a, e]

        gdf = gpd.GeoDataFrame(
            {
                "geometry": [
                    LineString(line1),
                    LineString(line2),
                    LineString(line3),
                    LineString(line4),
                ]
            }
        )

        coins = mm.COINS(gdf, angle_threshold=0)
        stroke_attr = coins.stroke_attribute()

        expected_groups = [
            bool(stroke_attr[0] == 0),
            bool(stroke_attr[1] == 1),
            bool(stroke_attr[2] == 2),
            bool(stroke_attr[3] == 0),
            bool(stroke_attr[0] != stroke_attr[2]),
        ]

        assert all(expected_groups)

        # test case 3
        a = (0, 0)
        b = (1, 1)
        c = (0, 1)

        line1 = [a, b]
        line2 = [a, c]

        gdf = gpd.GeoDataFrame(
            {
                "geometry": [
                    LineString(line1),
                    LineString(line2),
                ]
            }
        )

        # interior angle is 45deg, angle_threshold is 0:
        # expecting both lines to be part of same group
        coins = mm.COINS(gdf, angle_threshold=0)
        stroke_attr = coins.stroke_attribute()
        assert bool(stroke_attr[0] == stroke_attr[1] == 0)

        # interior angle is 45deg, angle_threshold is 46:
        # expecting lines to in different groups
        coins = mm.COINS(gdf, angle_threshold=46)
        stroke_attr = coins.stroke_attribute()
        # expecting both lines to be part of same group
        assert stroke_attr[0] != stroke_attr[1]

    def test_flow_mode(self):
        roads = gpd.GeoSeries.from_wkt(
            [
                "LINESTRING (682705.3550242907 5614078.5083698435, 682708.0849458438 5614073.895229458, 682712.4572257706 5614067.790682519)",  # noqa: E501
                "LINESTRING (682680.7622123861 5614067.175691795, 682695.2357549993 5614044.543851833)",  # noqa: E501
                "LINESTRING (682705.3550242907 5614078.5083698435, 682680.7622123861 5614067.175691795)",  # noqa: E501
                "LINESTRING (682709.3980117479 5614080.294302186, 682705.3550242907 5614078.5083698435)",  # noqa: E501
                "LINESTRING (682677.6759713582 5614091.273667651, 682677.2937893708 5614090.719229713, 682672.358253116 5614084.113239679, 682680.7622123861 5614067.175691795)",  # noqa: E501
            ]
        )
        no_flow = mm.COINS(roads, 120)
        pm = no_flow._premerge()
        assert pm.p1_final.tolist() == [
            "line_break",
            0,
            7,
            4,
            "line_break",
            "line_break",
            5,
            "line_break",
        ]
        assert pm.p2_final.tolist() == [
            1,
            "line_break",
            "line_break",
            "line_break",
            3,
            6,
            "line_break",
            2,
        ]

        flow = mm.COINS(roads, 120, flow_mode=True)
        pm = flow._premerge()
        assert pm.p1_final.tolist() == [
            "line_break",
            0,
            7,
            4,
            "line_break",
            "line_break",
            5,
            6,
        ]
        assert pm.p2_final.tolist() == [
            1,
            "line_break",
            "line_break",
            "line_break",
            3,
            6,
            7,
            2,
        ]
