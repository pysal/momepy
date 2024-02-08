import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_index_equal, assert_series_equal

import momepy as mm


def assert_result(result, expected, geometry):
    """Check the expected values and types of the result."""
    for key, value in expected.items():
        assert getattr(result, key)() == pytest.approx(value)
    assert isinstance(result, pd.Series)
    assert_index_equal(result.index, geometry.index)


class TestDimensions:
    def setup_method(self):
        test_file_path = mm.datasets.get_path("bubenec")
        self.df_buildings = gpd.read_file(test_file_path, layer="buildings")
        self.df_streets = gpd.read_file(test_file_path, layer="streets")
        self.df_tessellation = gpd.read_file(test_file_path, layer="tessellation")
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
        assert_result(r, expected, self.df_buildings)

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
