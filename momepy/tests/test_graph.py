import geopandas as gpd
import pytest
from shapely.geometry import LineString

import momepy as mm


class TestGraph:
    def setup_method(self):
        test_file_path = mm.datasets.get_path("bubenec")
        self.df_streets = gpd.read_file(test_file_path, layer="streets")
        self.network = mm.gdf_to_nx(self.df_streets)
        self.network = mm.node_degree(self.network)
        self.dual = mm.gdf_to_nx(self.df_streets, approach="dual")

    def test_node_degree(self):
        deg1 = mm.node_degree(graph=self.network)
        assert deg1.nodes[(1603650.450422848, 6464368.600601688)]["degree"] == 4

    def test_meshedness(self):
        net = mm.meshedness(self.network)
        check = 0.14893617021276595
        assert net.nodes[(1603650.450422848, 6464368.600601688)]["meshedness"] == check
        assert mm.meshedness(self.network, radius=None) == 0.1320754716981132

    def test_mean_node_dist(self):
        net = mm.mean_node_dist(self.network)
        check = 148.8166441307652
        assert net.nodes[(1603650.450422848, 6464368.600601688)]["meanlen"] == check

    def test_cds_length(self):
        net = mm.cds_length(self.network)
        net2 = mm.cds_length(self.network, mode="mean", name="cds_mean")
        sumval = 1753.626758955522
        mean = 219.20334486944023
        assert net.nodes[(1603650.450422848, 6464368.600601688)][
            "cds_len"
        ] == pytest.approx(sumval)
        assert net2.nodes[(1603650.450422848, 6464368.600601688)][
            "cds_mean"
        ] == pytest.approx(mean)
        with pytest.raises(ValueError, match="Mode 'nonexistent' is not supported. "):
            net2 = mm.cds_length(self.network, mode="nonexistent")
        assert mm.cds_length(self.network, radius=None) == pytest.approx(
            2291.4520621447705
        )

    def test_mean_node_degree(self):
        net = mm.mean_node_degree(self.network)
        check = 2.576923076923077
        assert net.nodes[(1603650.450422848, 6464368.600601688)]["mean_nd"] == check
        assert mm.mean_node_degree(self.network, radius=None) == 2.413793103448276

    def test_proportion(self):
        net = mm.proportion(self.network, three="three", four="four", dead="dead")
        three = 0.23076923076923078
        four = 0.2692307692307692
        dead = 0.3076923076923077
        assert net.nodes[(1603650.450422848, 6464368.600601688)]["dead"] == dead
        assert net.nodes[(1603650.450422848, 6464368.600601688)]["four"] == four
        assert net.nodes[(1603650.450422848, 6464368.600601688)]["three"] == three
        glob = mm.proportion(
            self.network, three="three", four="four", dead="dead", radius=None
        )
        assert glob["dead"] == 0.3793103448275862
        assert glob["four"] == 0.2413793103448276
        assert glob["three"] == 0.20689655172413793

    def test_proportion_error(self):
        with pytest.raises(ValueError, match="Nothing to calculate. "):
            mm.proportion(self.network)

    def test_cyclomatic(self):
        net = mm.cyclomatic(self.network)
        check = 7
        assert net.nodes[(1603650.450422848, 6464368.600601688)]["cyclomatic"] == check
        assert mm.cyclomatic(self.network, radius=None) == 7

    def test_edge_node_ratio(self):
        net = mm.edge_node_ratio(self.network)
        check = 1.2307692307692308
        assert (
            net.nodes[(1603650.450422848, 6464368.600601688)]["edge_node_ratio"]
            == check
        )
        assert mm.edge_node_ratio(self.network, radius=None) == 1.206896551724138

    def test_gamma(self):
        net = mm.gamma(self.network)
        check = 0.4444444444444444
        assert net.nodes[(1603650.450422848, 6464368.600601688)]["gamma"] == check
        assert mm.gamma(self.network, radius=None) == 0.43209876543209874

    def test_closeness_centrality(self):
        net = mm.closeness_centrality(self.network, weight="mm_len")
        check = 0.0016066095164175716
        assert net.nodes[(1603650.450422848, 6464368.600601688)][
            "closeness"
        ] == pytest.approx(check)

        net = mm.closeness_centrality(self.network, radius=5, weight=None)
        check = 0.27557319223985893
        assert net.nodes[(1603650.450422848, 6464368.600601688)][
            "closeness"
        ] == pytest.approx(check)
        net2 = mm.closeness_centrality(self.network, weight="mm_len", radius=5)
        check2 = 0.0015544070362478774
        assert net2.nodes[(1603650.450422848, 6464368.600601688)][
            "closeness"
        ] == pytest.approx(check2)

    def test_closeness_centrality_angular(self):
        # GH #463: angular closeness centrality on the dual graph must
        # match the hand calculation reported in the issue. The network is
        # the seven-segment example from the report (segments are the dual
        # nodes, numbered 1-7 in the report). The hand-calculated values
        # are Cc1 = 0.009318217, Cc2 = 0.010 and Cc3 = 0.008379888.
        lines = [
            [
                (297108.3850011958, -785292.0947180807),
                (297180.5067213175, -785260.8175321289),
            ],
            [
                (297244.8884953555, -785230.7877799053),
                (297180.5067213175, -785260.8175321289),
            ],
            [
                (297110.80210419616, -785160.477173293),
                (297180.5067213175, -785260.8175321289),
            ],
            [
                (297042.3441068098, -785190.6891610724),
                (297108.3850011958, -785292.0947180807),
            ],
            [
                (297178.32290327916, -785127.9463243468),
                (297244.8884953555, -785230.7877799053),
            ],
            [
                (297110.80210419616, -785160.477173293),
                (297042.3441068098, -785190.6891610724),
            ],
            [
                (297178.32290327916, -785127.9463243468),
                (297110.80210419616, -785160.477173293),
            ],
        ]
        gdf = gpd.GeoDataFrame(
            geometry=[LineString(line) for line in lines], crs="epsg:32625"
        )
        dual = mm.gdf_to_nx(gdf, approach="dual")

        # nodes are added in the order of the input rows
        nodes = list(dual.nodes)
        for node, line in zip(nodes, lines, strict=True):
            assert dual.nodes[node]["geometry"].equals(LineString(line))

        net = mm.closeness_centrality(dual, weight="angle", verbose=False)
        check = [
            0.009317962,
            0.009997581,
            0.008379147,
            0.008323356,
            0.008300193,
            0.009937100,
            0.009291048,
        ]
        assert [net.nodes[node]["closeness"] for node in nodes] == pytest.approx(
            check, rel=1e-6
        )
        # the first three match the hand calculation from the issue
        assert net.nodes[nodes[0]]["closeness"] == pytest.approx(0.009318217, rel=1e-3)
        assert net.nodes[nodes[1]]["closeness"] == pytest.approx(0.010, rel=1e-3)
        assert net.nodes[nodes[2]]["closeness"] == pytest.approx(0.008379888, rel=1e-3)

        # a path may not turn twice at the same primal node: from segment
        # 1 to segment 3 the direct turn (101.34) is shorter than the
        # shortcut via segment 2 (1.56 + 80.22 = 81.78), which turns twice
        # at the same node without traversing segment 2
        assert mm.graph._angular_path_lengths(dual, nodes[0], "angle")[
            nodes[2]
        ] == pytest.approx(101.3420, rel=1e-4)

        # local (radius) closeness uses the same angular paths
        net = mm.closeness_centrality(dual, weight="angle", radius=1, verbose=False)
        check_radius1 = [
            0.008184492,
            0.008347832,
            0.007414628,
            0.003711288,
            0.003688980,
            0.008324140,
            0.008161716,
        ]
        assert [net.nodes[node]["closeness"] for node in nodes] == pytest.approx(
            check_radius1, rel=1e-6
        )

        # a dual graph without angle attributes falls back to weight 1 per
        # edge, matching networkx Dijkstra behaviour, instead of failing
        dual_no_angle = mm.gdf_to_nx(gdf, approach="dual", angles=False)
        net = mm.closeness_centrality(dual_no_angle, weight="angle", verbose=False)
        topo = mm.closeness_centrality(dual, weight=None, verbose=False)
        assert [net.nodes[node]["closeness"] for node in nodes] == pytest.approx(
            [topo.nodes[node]["closeness"] for node in nodes], rel=1e-9
        )

    def test_betweenness_centrality(self):
        net = mm.betweenness_centrality(self.network)
        net2 = mm.betweenness_centrality(self.network, mode="edges")
        angular = mm.betweenness_centrality(self.dual, weight="angle")
        with pytest.raises(ValueError, match="Mode 'nonexistent' is not supported. "):
            mm.betweenness_centrality(self.network, mode="nonexistent")
        node = 0.2413793103448276
        edge = 0.16995073891625617
        ang_b = 0.16470588235294117
        assert net.nodes[(1603650.450422848, 6464368.600601688)]["betweenness"] == node
        assert (
            net2.edges[
                (1603226.9576840235, 6464160.158361825),
                (1603039.9632033885, 6464087.491175889),
                0,
            ]["betweenness"]
            == edge
        )
        assert (
            angular.nodes[(1603315.3564306537, 6464044.376339891)]["betweenness"]
            == ang_b
        )

        net = mm.betweenness_centrality(self.network, radius=5, weight=None)
        check = 53.74999999999999
        assert net.nodes[(1603650.450422848, 6464368.600601688)]["betweenness"] == check
        net2 = mm.betweenness_centrality(
            self.network, radius=5, weight="mm_len", normalized=True
        )
        check2 = 0.21333333333333335
        assert (
            net2.nodes[(1603650.450422848, 6464368.600601688)]["betweenness"] == check2
        )

    def test_straightness_centrality(self):
        net = mm.straightness_centrality(self.network)
        net2 = mm.straightness_centrality(
            self.network, normalized=False, name="nonnorm"
        )
        g = self.network.copy()
        g.add_node(1)
        net3 = mm.straightness_centrality(g)
        check = 0.8574045143712158
        nonnorm = 0.8574045143712158
        assert (
            net.nodes[(1603650.450422848, 6464368.600601688)]["straightness"] == check
        )
        assert net2.nodes[(1603650.450422848, 6464368.600601688)]["nonnorm"] == nonnorm
        assert (
            net3.nodes[(1603650.450422848, 6464368.600601688)]["straightness"] == check
        )

        net = mm.straightness_centrality(self.network, radius=5, normalized=False)
        check = 0.8614261420539604
        assert (
            net.nodes[(1603650.450422848, 6464368.600601688)]["straightness"] == check
        )

    def test_mean_nodes(self):
        net = mm.straightness_centrality(self.network)
        mm.mean_nodes(net, "straightness")
        edge = 0.8777302256084243
        assert (
            net.edges[
                (1603226.9576840235, 6464160.158361825),
                (1603039.9632033885, 6464087.491175889),
                0,
            ]["straightness"]
            == edge
        )

    def test_clustering(self):
        net = mm.clustering(self.network)
        check = 0.05555555555555555
        assert net.nodes[(1603650.450422848, 6464368.600601688)]["cluster"] == check

    def test_subgraph(self):
        net = mm.subgraph(self.network)
        nodes = mm.nx_to_gdf(net, lines=False)
        cols = [
            "meshedness",
            "cds_length",
            "mean_node_degree",
            "proportion_3",
            "proportion_4",
            "proportion_0",
            "cyclomatic",
            "edge_node_ratio",
            "gamma",
            "local_closeness",
        ]
        for c in cols:
            assert c in nodes.columns
