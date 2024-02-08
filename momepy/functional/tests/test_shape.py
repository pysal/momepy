import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_index_equal

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
        r = mm.form_factor(self.df_buildings.geometry, self.df_buildings.height)
        assert_result(r, expected, self.df_buildings)

    def test_fractal_dimension(self):
        expected = {
            "mean": 1.0284229071113,
            "sum": 148.09289862402,
            "min": 0.9961153485808,
            "max": 1.1823500218800869,
        }
        r = mm.fractal_dimension(self.df_buildings.geometry)
        assert_result(r, expected, self.df_buildings)

    def test_facade_ratio(self):
        expected = {
            "mean": 5.576164283846954,
            "sum": 802.9676568739615,
            "min": 1.693082337025171,
            "max": 11.314007604595293,
        }
        r = mm.facade_ratio(self.df_buildings.geometry)
        assert_result(r, expected, self.df_buildings)
