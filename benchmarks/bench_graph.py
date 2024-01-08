import geopandas as gpd

import momepy as mm


class TimeGraph:
    def setup(self):
        test_file_path = mm.datasets.get_path("bubenec")
        self.df_streets = gpd.read_file(test_file_path, layer="streets")
        self.network = mm.gdf_to_nx(self.df_streets)
        self.network = mm.node_degree(self.network)
        self.dual = mm.gdf_to_nx(self.df_streets, approach="dual")

    def time_node_degree(self):
        mm.node_degree(graph=self.network)

    def time_meshedness(self):
        mm.meshedness(self.network)

    def time_mean_node_dist(self):
        mm.mean_node_dist(self.network)

    def time_cds_length(self):
        mm.cds_length(self.network)

    def time_mean_node_degree(self):
        mm.mean_node_degree(self.network)

    def time_proportion(self):
        mm.proportion(self.network, three="three", four="four", dead="dead")

    def time_cyclomatic(self):
        mm.cyclomatic(self.network)

    def time_edge_node_ratio(self):
        mm.edge_node_ratio(self.network)

    def time_gamma(self):
        mm.gamma(self.network)

    def time_closeness_centrality(self):
        mm.closeness_centrality(self.network, weight="mm_len")

    def time_betweenness_centrality_nodes(self):
        mm.betweenness_centrality(self.network)

    def time_betweenness_centrality_edges(self):
        mm.betweenness_centrality(self.network, mode="edges")

    def time_betweenness_centrality_angular(self):
        mm.betweenness_centrality(self.dual, weight="angle")

    def time_betweenness_centrality_local(self):
        mm.betweenness_centrality(self.network, radius=5, weight=None)

    def time_straightness_centrality(self):
        mm.straightness_centrality(self.network)

    def time_clustering(self):
        mm.clustering(self.network)

    def time_subgraph(self):
        mm.subgraph(self.network)
