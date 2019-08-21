import momepy as mm
import geopandas as gpd
import pytest


class TestGraph:

    def setup_method(self):

        test_file_path = mm.datasets.get_path('bubenec')
        self.df_streets = gpd.read_file(test_file_path, layer='streets')
        self.network = mm.gdf_to_nx(self.df_streets)
        self.network = mm.node_degree(self.network)

    def test_node_degree(self):
        deg1 = mm.node_degree(graph=self.network)
        assert deg1.nodes[(1603650.450422848, 6464368.600601688)]['degree'] == 4

    def test_meshedness(self):
        net = mm.meshedness(self.network)
        check = 0.14893617021276595
        assert net.nodes[(1603650.450422848, 6464368.600601688)]['meshedness'] == check

    def test_mean_node_dist(self):
        net = mm.mean_node_dist(self.network)
        check = 148.8166441307652
        assert net.nodes[(1603650.450422848, 6464368.600601688)]['meanlen'] == check

    def test_cds_length(self):
        net = mm.cds_length(self.network)
        net2 = mm.cds_length(self.network, mode='mean', name='cds_mean')
        sumval = 1753.626758955522
        mean = 219.20334486944023
        assert net.nodes[(1603650.450422848, 6464368.600601688)]['cds_len'] == sumval
        assert net2.nodes[(1603650.450422848, 6464368.600601688)]['cds_mean'] == mean

    def test_mean_node_degree(self):
        net = mm.mean_node_degree(self.network)
        check = 2.576923076923077
        assert net.nodes[(1603650.450422848, 6464368.600601688)]['mean_nd'] == check

    def test_proportion(self):
        net = mm.proportion(self.network, three='three', four='four', dead='dead')
        three = 0.23076923076923078
        four = 0.2692307692307692
        dead = 0.3076923076923077
        assert net.nodes[(1603650.450422848, 6464368.600601688)]['dead'] == dead
        assert net.nodes[(1603650.450422848, 6464368.600601688)]['four'] == four
        assert net.nodes[(1603650.450422848, 6464368.600601688)]['three'] == three

    def test_proportion_error(self):
        with pytest.raises(ValueError):
            mm.proportion(self.network)

    def test_cyclomatic(self):
        net = mm.cyclomatic(self.network)
        check = 7
        assert net.nodes[(1603650.450422848, 6464368.600601688)]['cyclomatic'] == check

    def test_edge_node_ratio(self):
        net = mm.edge_node_ratio(self.network)
        check = 1.2307692307692308
        assert net.nodes[(1603650.450422848, 6464368.600601688)]['edge_node_ratio'] == check

    def test_gamma(self):
        net = mm.gamma(self.network)
        check = 0.4444444444444444
        assert net.nodes[(1603650.450422848, 6464368.600601688)]['gamma'] == check

    def test_local_closeness(self):
        net = mm.local_closeness(self.network)
        check = 0.27557319223985893
        assert net.nodes[(1603650.450422848, 6464368.600601688)]['closeness'] == check

    def test_local_closeness_distance(self):
        net = mm.local_closeness(self.network, closeness_distance='mm_len')
        check = 0.0015544070362478774
        assert net.nodes[(1603650.450422848, 6464368.600601688)]['closeness'] == check

    def test_eigenvector(self):
        net = mm.eigenvector(self.network)
        check = 0.26732838680271764
        assert net.nodes[(1603650.450422848, 6464368.600601688)]['eigen'] == check

    def test_clustering(self):
        net = mm.clustering(self.network)
        check = 0.09090909090909091
        assert net.nodes[(1603650.450422848, 6464368.600601688)]['cluster'] == check

    def test_subgraph(self):
        net = mm.subgraph(self.network)
        nodes = mm.nx_to_gdf(net, edges=False)
        cols = ['meshedness', 'cds_length', 'mean_node_degree',
                'proportion_3', 'proportion_4', 'proportion_0', 'cyclomatic',
                'edge_node_ratio', 'gamma', 'local_closeness']
        for c in cols:
            assert c in nodes.columns
