import geopandas as gpd
import networkx as nx
import numpy as np

# import pandas as pd
import pytest

#  from pandas.testing import assert_index_equal, assert_series_equal
from shapely.geometry import LineString

import momepy as mm


class TestContinuity:
    def setup_method(self):
        test_file_path = mm.datasets.get_path("bubenec")
        self.gdf = gpd.read_file(test_file_path, layer="streets")
        self.coins = mm.COINS(self.gdf, angle_threshold=0, flow_mode=False)
        self.continuity_graph = mm.coins_to_nx(self.coins)

    def test_get_interior_angle(self):
        a = [172.43389124, 200.04361864]
        b = [-21.00598791, 53.213871]
        result = mm.continuity._get_interior_angle(a, b)
        assert result == pytest.approx(62.30218235137648)

        a = [1, 0]
        b = [-1, 1]
        result = mm.continuity._get_interior_angle(a, b)
        assert result == pytest.approx(45)

    def test_get_end_segment(self):
        linestring = LineString([[0, 0], [0, 1], [0, 2], [0, 3], [1, 3]])

        first_segment = np.array([0, -1])
        last_segment = np.array([-1, 0])

        assert all(mm.continuity._get_end_segment(linestring, [0, 0]) == first_segment)
        assert all(mm.continuity._get_end_segment(linestring, [1, 3]) == last_segment)

        with pytest.raises(ValueError, match="point is not an endpoint of linestring!"):
            mm.continuity._get_end_segment(linestring, [0, 2])

    def test_coins_to_nx(self):
        assert len(self.continuity_graph.nodes) == 10
        assert len(self.continuity_graph.edges) == 17
        assert nx.get_node_attributes(self.continuity_graph) == {
            0: [0, 3, 15, 27],
            1: [1, 12, 14, 25],
            2: [2, 11, 28, 30],
            3: [4, 5, 6],
            4: [7, 8, 9, 13, 21, 22, 24],
            5: [10],
            6: [16, 17, 18, 23, 29],
            7: [19],
            8: [20],
            9: [26],
        }

    def test_stroke_connectivity(self):
        graph_with_attribute = mm.stroke_connectivity(self.continuity_graph)
        assert nx.get_node_attributes(graph_with_attribute, "stroke_connectivity") == {
            0: 8,
            1: 6,
            2: 8,
            3: 7,
            4: 9,
            5: 1,
            6: 7,
            7: 1,
            8: 2,
            9: 3,
        }

    def test_stroke_access(self):
        graph_with_attribute = mm.stroke_access(self.continuity_graph)
        assert bool(nx.get_node_attributes(graph_with_attribute, "stroke_connectivity"))
        assert bool(nx.get_node_attributes(graph_with_attribute, "stroke_degree"))
        assert nx.get_node_attributes(graph_with_attribute, "stroke_access") == {
            0: 3,
            1: 2,
            2: 4,
            3: 2,
            4: 3,
            5: 0,
            6: 4,
            7: 0,
            8: 0,
            9: 0,
        }

    def test_stroke_orthogonality(self):
        graph_with_attribute = mm.stroke_orthogonality(self.continuity_graph)
        assert bool(nx.get_node_attributes(graph_with_attribute, "stroke_connectivity"))
        assert nx.get_node_attributes(
            graph_with_attribute, "stroke_orthogonality"
        ) == pytest.approx(
            {
                0: np.float64(0.8577408232193254),
                1: np.float64(0.9899635850266516),
                2: np.float64(0.8411759936266772),
                3: np.float64(0.8646755014619606),
                4: np.float64(0.9971868630623598),
                5: np.float64(0.9991299603577619),
                6: np.float64(0.9393327452953741),
                7: np.float64(0.9790865311515882),
                8: np.float64(0.9695975401850382),
                9: np.float64(0.7988075340542938),
            }
        )

    def test_stroke_spacing(self):
        graph_with_attribute = mm.stroke_spacing(self.continuity_graph)
        assert bool(nx.get_node_attributes(graph_with_attribute, "stroke_connectivity"))
        assert nx.get_node_attributes(
            graph_with_attribute, "stroke_spacing"
        ) == pytest.approx(
            {
                0: 104.94583547900395,
                1: 126.51500708434862,
                2: 93.09474171560097,
                3: 80.32095592022247,
                4: 119.70674174439718,
                5: 193.04063727323836,
                6: 145.67278692563468,
                7: 187.49184699173748,
                8: 91.34248700198054,
                9: 127.50065014307602,
            }
        )
