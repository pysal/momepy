import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
from pandas.testing import assert_frame_equal, assert_series_equal

import momepy as mm

from .conftest import assert_result

GPD_013 = Version(gpd.__version__) >= Version("0.13")


class TestShape:
    def setup_method(self):
        test_file_path = mm.datasets.get_path("bubenec")
        self.df_buildings = gpd.read_file(test_file_path, layer="buildings")
        self.df_streets = gpd.read_file(test_file_path, layer="streets")
        self.df_buildings["height"] = np.linspace(10.0, 30.0, 144)

    def test_form_factor(self):
        expected = {
            "mean": 5.4486362624193,
            "sum": 784.60362178837,
            "min": 4.7712411918919,
            "max": 9.0604207122634,
        }
        r = mm.form_factor(self.df_buildings, self.df_buildings.height)
        assert_result(r, expected, self.df_buildings)

    def test_fractal_dimension(self):
        expected = {
            "mean": 1.0284229071113,
            "sum": 148.09289862402,
            "min": 0.9961153485808,
            "max": 1.1823500218800869,
        }
        r = mm.fractal_dimension(self.df_buildings)
        assert_result(r, expected, self.df_buildings)

    def test_facade_ratio(self):
        expected = {
            "mean": 5.576164283846954,
            "sum": 802.9676568739615,
            "min": 1.693082337025171,
            "max": 11.314007604595293,
        }
        r = mm.facade_ratio(self.df_buildings)
        assert_result(r, expected, self.df_buildings)

    def test_circular_compactness(self):
        expected = {
            "mean": 0.5690762180557374,
            "sum": 81.94697540002619,
            "min": 0.2790908141757502,
            "max": 0.7467862040050506,
        }
        r = mm.circular_compactness(self.df_buildings)
        assert_result(r, expected, self.df_buildings)

    def test_square_compactness(self):
        expected = {
            "mean": 0.8486134470818577,
            "sum": 122.20033637978752,
            "min": 0.18260442238858857,
            "max": 1.0179729958596555,
        }
        r = mm.square_compactness(self.df_buildings)
        assert_result(r, expected, self.df_buildings)

    def test_convexity(self):
        expected = {
            "mean": 0.941622590878818,
            "sum": 135.5936530865498,
            "min": 0.5940675324809443,
            "max": 1,
        }
        r = mm.convexity(self.df_buildings)
        assert_result(r, expected, self.df_buildings)

    def test_courtyard_index(self):
        expected = {
            "mean": 0.0011531885929613557,
            "sum": 0.16605915738643523,
            "min": 0,
            "max": 0.16605915738643523,
        }
        r = mm.courtyard_index(self.df_buildings)
        assert_result(r, expected, self.df_buildings)

        r2 = mm.courtyard_index(self.df_buildings, mm.courtyard_area(self.df_buildings))
        assert_series_equal(r, r2)

    def test_rectangularity(self):
        expected = {
            "mean": 0.8867686143851342,
            "sum": 127.69468047145932,
            "min": 0.4354516872480972,
            "max": 0.9994000284783504,
        }
        r = mm.rectangularity(self.df_buildings)
        assert_result(r, expected, self.df_buildings, rel=1e-3)

    def test_shape_index(self):
        expected = {
            "mean": 0.7517554336770522,
            "sum": 108.25278244949553,
            "min": 0.5282904638319247,
            "max": 0.8641679258136411,
        }
        r = mm.shape_index(self.df_buildings)
        assert_result(r, expected, self.df_buildings)

        r2 = mm.shape_index(
            self.df_buildings, mm.longest_axis_length(self.df_buildings)
        )
        assert_series_equal(r, r2)

    @pytest.mark.skipif(not GPD_013, reason="get_coordinates() not available")
    def test_corners(self):
        expected = {
            "mean": 10.3125,
            "sum": 1485,
            "min": 4,
            "max": 43,
        }
        r = mm.corners(self.df_buildings)
        assert_result(r, expected, self.df_buildings)

        expected = {
            "mean": 10.993055555555555,
            "sum": 1583,
            "min": 4,
            "max": 46,
        }
        r = mm.corners(self.df_buildings, eps=1)
        assert_result(r, expected, self.df_buildings)

        expected = {
            "mean": 10.340277777777779,
            "sum": 1489,
            "min": 4,
            "max": 43,
        }
        r = mm.corners(self.df_buildings, include_interiors=True)
        assert_result(r, expected, self.df_buildings)

    @pytest.mark.skipif(GPD_013, reason="get_coordinates() not available")
    def test_corners_error(self):
        with pytest.raises(ImportError, match="momepy.corners requires geopandas 0.13"):
            mm.corners(self.df_buildings)

    @pytest.mark.skipif(not GPD_013, reason="get_coordinates() not available")
    def test_squareness(self):
        expected = {
            "mean": 5.229888125861968,
            "sum": 753.1038901241234,
            "min": 0.028139949189984748,
            "max": 41.04115291780464,
        }
        r = mm.squareness(self.df_buildings)
        assert_result(r, expected, self.df_buildings)

        expected = {
            "mean": 8.30047963732215,
            "sum": 1195.2690677743897,
            "min": 0.028139949189984748,
            "max": 52.875516523230516,
        }
        r = mm.squareness(self.df_buildings, eps=1)
        assert_result(r, expected, self.df_buildings)

        expected = {
            "mean": 5.32348682306419,
            "sum": 766.5821025212433,
            "min": 0.028139949189984748,
            "max": 41.04115291780464,
        }
        r = mm.squareness(self.df_buildings, include_interiors=True)
        assert_result(r, expected, self.df_buildings)

    @pytest.mark.skipif(GPD_013, reason="get_coordinates() not available")
    def test_squareness_error(self):
        with pytest.raises(
            ImportError, match="momepy.squareness requires geopandas 0.13"
        ):
            mm.squareness(self.df_buildings)

    def test_equivalent_rectangular_index(self):
        expected = {
            "mean": 0.9307591166031689,
            "sum": 134.02931279085632,
            "min": 0.44313707855781465,
            "max": 1.0090920542155701,
        }
        r = mm.equivalent_rectangular_index(self.df_buildings)
        assert_result(r, expected, self.df_buildings)

    def test_elongation(self):
        expected = {
            "mean": 0.8046747233732038,
            "sum": 115.87316016574134,
            "min": 0.2608278715025736,
            "max": 0.9975111632520667,
        }
        r = mm.elongation(self.df_buildings)
        assert_result(r, expected, self.df_buildings, rel=1e-3)

    @pytest.mark.skipif(not GPD_013, reason="get_coordinates() not available")
    def test_centroid_corner_distance(self):
        expected = pd.DataFrame(
            {
                "mean": {
                    "count": 144.0,
                    "mean": 15.5203360663233,
                    "std": 6.093122141583696,
                    "min": 4.854313172218458,
                    "25%": 13.412728528289005,
                    "50%": 15.609796287245382,
                    "75%": 17.69178717016742,
                    "max": 58.76338773692563,
                },
                "std": {
                    "count": 144.0,
                    "mean": 2.829917240070652,
                    "std": 2.823411898664825,
                    "min": 0.0017021571743907728,
                    "25%": 1.2352146286567187,
                    "50%": 2.128549075430912,
                    "75%": 3.757463233035704,
                    "max": 22.922367806062553,
                },
            }
        )
        r = mm.centroid_corner_distance(self.df_buildings)
        assert_frame_equal(r.describe(), expected)

        expected = pd.DataFrame(
            {
                "mean": {
                    "count": 144.0,
                    "mean": 15.454415227361714,
                    "std": 6.123921386925624,
                    "min": 4.854313172218458,
                    "25%": 13.01290636950394,
                    "50%": 15.497887115707538,
                    "75%": 17.63008289695232,
                    "max": 59.2922704452347,
                },
                "std": {
                    "count": 144.0,
                    "mean": 2.8582735179300514,
                    "std": 2.7943793134508357,
                    "min": 0.0017021571743907728,
                    "25%": 1.2296334024258928,
                    "50%": 2.1468383325089695,
                    "75%": 3.757463233035704,
                    "max": 22.924246335383152,
                },
            }
        )
        r = mm.centroid_corner_distance(self.df_buildings, eps=1)
        assert_frame_equal(r.describe(), expected)

        expected = pd.DataFrame(
            {
                "mean": {
                    "count": 144.0,
                    "mean": 15.464353525824205,
                    "std": 5.940640521286903,
                    "min": 4.854313172218458,
                    "25%": 13.412728528289005,
                    "50%": 15.609796287245382,
                    "75%": 17.69178717016742,
                    "max": 58.76338773692563,
                },
                "std": {
                    "count": 144.0,
                    "mean": 2.8992936136803515,
                    "std": 2.8859313144712324,
                    "min": 0.0017021571743907728,
                    "25%": 1.2656882908488725,
                    "50%": 2.2163184044891224,
                    "75%": 3.911029451756276,
                    "max": 22.922367806062553,
                },
            }
        )
        r = mm.centroid_corner_distance(self.df_buildings, include_interiors=True)
        assert_frame_equal(r.describe(), expected)

    @pytest.mark.skipif(GPD_013, reason="get_coordinates() not available")
    def test_centroid_corner_distance_error(self):
        with pytest.raises(
            ImportError, match="momepy.centroid_corner_distance requires geopandas 0.13"
        ):
            mm.centroid_corner_distance(self.df_buildings)

    def test_linearity(self):
        expected = {
            "mean": 0.9976310491404173,
            "sum": 34.91708671991461,
            "min": 0.9801618536039246,
            "max": 1,
        }
        r = mm.linearity(self.df_streets)
        assert_result(r, expected, self.df_streets)

    def test_compactness_weighted_axis(self):
        expected = {
            "mean": 17.592071297082306,
            "sum": 2533.258266779852,
            "min": 3.2803162516228723,
            "max": 208.5887465292061,
        }
        r = mm.compactness_weighted_axis(self.df_buildings)
        assert_result(r, expected, self.df_buildings)

        r2 = mm.compactness_weighted_axis(
            self.df_buildings, mm.longest_axis_length(self.df_buildings)
        )
        assert_series_equal(r, r2)


class TestEquality:
    def setup_method(self):
        test_file_path = mm.datasets.get_path("bubenec")
        self.df_buildings = gpd.read_file(test_file_path, layer="buildings")
        self.df_streets = gpd.read_file(test_file_path, layer="streets")
        self.df_buildings["height"] = np.linspace(10.0, 30.0, 144)

    def test_form_factor(self):
        new = mm.form_factor(self.df_buildings, self.df_buildings.height)
        old = mm.FormFactor(
            self.df_buildings,
            volumes=self.df_buildings.height * self.df_buildings.area,
            heights=self.df_buildings.height,
        ).series
        assert_series_equal(new, old, check_names=False)

    def test_fractal_dimension(self):
        new = mm.fractal_dimension(self.df_buildings)
        old = mm.FractalDimension(self.df_buildings).series
        assert_series_equal(new, old, check_names=False)

    def test_facade_ratio(self):
        new = mm.facade_ratio(self.df_buildings)
        old = mm.VolumeFacadeRatio(self.df_buildings, "height").series
        assert_series_equal(new, old, check_names=False)

    def test_circular_compactness(self):
        new = mm.circular_compactness(self.df_buildings)
        old = mm.CircularCompactness(self.df_buildings).series
        assert_series_equal(new, old, check_names=False)

    def test_square_compactness(self):
        new = mm.square_compactness(self.df_buildings)
        old = mm.SquareCompactness(self.df_buildings).series
        assert_series_equal(new, old, check_names=False)

    def test_convexity(self):
        new = mm.convexity(self.df_buildings)
        old = mm.Convexity(self.df_buildings).series
        assert_series_equal(new, old, check_names=False)

    def test_courtyard_index(self):
        new = mm.courtyard_index(self.df_buildings)
        old = mm.CourtyardIndex(
            self.df_buildings, mm.courtyard_area(self.df_buildings)
        ).series
        assert_series_equal(new, old, check_names=False)

    def test_rectangularity(self):
        new = mm.rectangularity(self.df_buildings)
        old = mm.Rectangularity(self.df_buildings).series
        assert_series_equal(new, old, check_names=False)

    def test_shape_index(self):
        new = mm.shape_index(self.df_buildings)
        old = mm.ShapeIndex(
            self.df_buildings, mm.longest_axis_length(self.df_buildings)
        ).series
        assert_series_equal(new, old, check_names=False)

    @pytest.mark.skipif(not GPD_013, reason="get_coordinates() not available")
    def test_corners(self):
        new = mm.corners(self.df_buildings)
        old = mm.Corners(self.df_buildings).series
        assert_series_equal(new, old, check_names=False)

    @pytest.mark.skipif(not GPD_013, reason="get_coordinates() not available")
    def test_squareness(self):
        new = mm.squareness(self.df_buildings, eps=5)
        old = mm.Squareness(self.df_buildings).series
        assert_series_equal(new, old, check_names=False)

    def test_equivalent_rectangular_index(self):
        new = mm.equivalent_rectangular_index(self.df_buildings)
        old = mm.EquivalentRectangularIndex(self.df_buildings).series
        assert_series_equal(new, old, check_names=False)

    def test_elongation(self):
        new = mm.elongation(self.df_streets)
        old = mm.Elongation(self.df_streets).series
        assert_series_equal(new, old, check_names=False)

    @pytest.mark.skipif(not GPD_013, reason="get_coordinates() not available")
    def test_centroid_corner_distance(self):
        new = mm.centroid_corner_distance(self.df_buildings)
        ccd = mm.CentroidCorners(self.df_buildings)
        mean = ccd.mean
        std = ccd.std
        old = pd.DataFrame({"mean": mean, "std": std})
        assert_frame_equal(new, old, check_names=False)

    def test_linearity(self):
        new = mm.linearity(self.df_streets)
        old = mm.Linearity(self.df_streets).series
        assert_series_equal(new, old, check_names=False)

    def test_compactness_weighted_axis(self):
        new = mm.compactness_weighted_axis(self.df_buildings)
        old = mm.CompactnessWeightedAxis(self.df_buildings).series
        assert_series_equal(new, old, check_names=False)
