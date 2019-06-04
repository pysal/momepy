import momepy as mm
import geopandas as gpd


class TestConnectivity:

    def setup_method(self):

        test_file_path = mm.datasets.get_path('bubenec')
        self.df_streets = gpd.read_file(test_file_path, layer='streets')
        self.network = mm.gdf_to_nx(self.df_streets)
        self.network = mm.node_degree(self.network)

    def test_meshedness(self):
        net = mm.meshedness(self.network)
        check = 0.022222222222222223
        assert net.nodes[(1603650.450422848, 6464368.600601688)]['meshedness'] == check

    def test_mean_node_dist(self):
        net = mm.mean_node_dist(self.network)
        check = 267.15635764665024
        assert net.nodes[(1603650.450422848, 6464368.600601688)]['meanlen'] == check

    def test_cds_length(self):
        net = mm.cds_length(self.network)
        net = mm.cds_length(self.network, mode='mean', name='cds_mean')
        sumval = 1678.873603417499
        mean = 209.8592004271874
        assert net.nodes[(1603650.450422848, 6464368.600601688)]['cds_len'] == sumval
        assert net.nodes[(1603650.450422848, 6464368.600601688)]['cds_mean'] == mean
