import geopandas as gpd
import pandas as pd
import pytest
from pandas.testing import assert_index_equal, assert_series_equal
from shapely.geometry import LineString
import numpy as np

import momepy as mm


class TestStrokeGraph:
    def setup_method(self):
        test_file_path = mm.datasets.get_path("bubenec")
        self.gdf = gpd.read_file(test_file_path, layer="streets")

    def test_get_interior_angle(self):

        a = [172.43389124, 200.04361864]
        b = [-21.00598791, 53.213871]
        result = mm.strokegraph._get_interior_angle(a, b)
        assert result == 62.30218235137648

        a = [1, 0]
        b = [-1, 1]
        result = mm.strokegraph._get_interior_angle(a, b)
        assert result == 45

    def test_get_end_segment(self):

        linestring = LineString([[0, 0], [0, 1], [0, 2], [0, 3], [1, 3]])

        first_segment = np.array([0, -1])
        last_segment = np.array([-1, 0])

        assert all(mm.strokegraph._get_end_segment(linestring, [0, 0]) == first_segment)
        assert all(mm.strokegraph._get_end_segment(linestring, [1, 3]) == last_segment)

        with pytest.raises(ValueError, match="point is not an endpoint of linestring!"):
            mm.strokegraph._get_end_segment(linestring, [0, 2])
