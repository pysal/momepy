import momepy as mm
import geopandas as gpd
import numpy as np
from libpysal.weights import Queen


class TestDistribution:

    def setup_method(self):

        test_file_path = mm.datasets.get_path('bubenec')
        self.df_buildings = gpd.read_file(test_file_path, layer='buildings')
        self.df_streets = gpd.read_file(test_file_path, layer='streets')
        self.df_tessellation = gpd.read_file(test_file_path, layer='tessellation')
        self.df_buildings['height'] = np.linspace(10., 30., 144)
        self.df_buildings['volume'] = mm.volume(self.df_buildings, 'height')
        self.df_streets['nID'] = mm.unique_id(self.df_streets)
        self.df_buildings['nID'], self.df_tessellation['nID'] = mm.get_network_id(self.df_buildings, self.df_streets,
                                                                                  'uID', 'nID', self.df_tessellation, 'uID')

    def test_orientation(self):
        self.df_buildings['orient'] = mm.orientation(self.df_buildings)
        check = 41.05146788287027
        assert self.df_buildings['orient'][0] == check

    def test_shared_walls_ratio(self):
        self.df_buildings['swr'] = mm.shared_walls_ratio(self.df_buildings, 'uID')
        self.df_buildings['swr_uid'] = mm.shared_walls_ratio(self.df_buildings, range(len(self.df_buildings)))
        self.df_buildings['swr_array'] = mm.shared_walls_ratio(self.df_buildings, 'uID', self.df_buildings.geometry.length)
        check = 0.3424804411228673
        assert self.df_buildings['swr'][10] == check
        assert self.df_buildings['swr_uid'][10] == check
        assert self.df_buildings['swr_array'][10] == check

    def test_street_alignment(self):
        self.df_buildings['orient'] = orient = mm.orientation(self.df_buildings)
        self.df_buildings['street_alignment'] = mm.street_alignment(self.df_buildings, self.df_streets, 'orient', 'nID', 'nID')
        self.df_buildings['street_a_arr'] = mm.street_alignment(self.df_buildings, self.df_streets, orient,
                                                                self.df_buildings['nID'], self.df_streets['nID'])
        check = 0.29073888476702336
        assert self.df_buildings['street_alignment'][0] == check
        assert self.df_buildings['street_a_arr'][0] == check

    def test_cell_alignment(self):
        self.df_buildings['orient'] = mm.orientation(self.df_buildings)
        self.df_tessellation['orient'] = mm.orientation(self.df_tessellation)
        self.df_buildings['c_align'] = mm.cell_alignment(self.df_buildings, self.df_tessellation, 'orient', 'orient', 'uID')
        check = abs(self.df_buildings['orient'][0] -
                    self.df_tessellation[self.df_tessellation['uID'] == self.df_buildings['uID'][0]]['orient'].iloc[0])
        assert self.df_buildings['c_align'][0] == check

    def test_alignment(self):
        self.df_buildings['orient'] = mm.orientation(self.df_buildings)
        self.df_buildings['align'] = mm.alignment(self.df_buildings, 'orient', self.df_tessellation, 'uID')
        sw = Queen.from_dataframe(self.df_tessellation)
        self.df_buildings['align_sw'] = mm.alignment(self.df_buildings, 'orient', self.df_tessellation, 'uID', sw)
        check = 18.299481296455237
        assert self.df_buildings['align'][0] == check
        assert self.df_buildings['align_sw'][0] == check

    def test_neighbour_distance(self):
        self.df_buildings['dist'] = mm.neighbour_distance(self.df_buildings, self.df_tessellation, 'uID')
        sw = Queen.from_dataframe(self.df_tessellation)
        self.df_buildings['dist_sw'] = mm.neighbour_distance(self.df_buildings, self.df_tessellation, 'uID', sw)
        check = 29.18589019096464
        assert self.df_buildings['dist'][0] == check
        assert self.df_buildings['dist_sw'][0] == check

    def test_mean_interbuilding_distance(self):
        self.df_buildings['m_dist'] = mm.mean_interbuilding_distance(self.df_buildings, self.df_tessellation, 'uID')
        sw = Queen.from_dataframe(self.df_tessellation)
        swh = mm.Queen_higher(k=3, geodataframe=self.df_tessellation)
        self.df_buildings['m_dist_sw'] = mm.mean_interbuilding_distance(self.df_buildings, self.df_tessellation, 'uID', sw, swh)
        check = 29.305457092042744
        assert self.df_buildings['m_dist'][0] == check
        assert self.df_buildings['m_dist_sw'][0] == check

    def test_neighbouring_street_orientation_deviation(self):
        self.df_streets['dev'] = mm.neighbouring_street_orientation_deviation(self.df_streets)
        check = 6.060070028371881
        assert self.df_streets['dev'].mean() == check

    def test_building_adjacency(self):
        self.df_buildings['adj'] = mm.building_adjacency(self.df_buildings, self.df_tessellation)
        sw = Queen.from_dataframe(self.df_buildings)
        swh = mm.Queen_higher(k=3, geodataframe=self.df_tessellation)
        self.df_buildings['adj_sw'] = mm.building_adjacency(self.df_buildings, self.df_tessellation, spatial_weights=sw, spatial_weights_higher=swh)
        check = 0.2613824113909074
        assert self.df_buildings['adj'].mean() == check
        assert self.df_buildings['adj_sw'].mean() == check

    def test_neighbours(self):
        self.df_tessellation['nei'] = mm.neighbours(self.df_tessellation)
        sw = Queen.from_dataframe(self.df_tessellation)
        self.df_tessellation['nei_sw'] = mm.neighbours(self.df_tessellation, sw)
        self.df_tessellation['nei_wei'] = mm.neighbours(self.df_tessellation, sw, weighted=True)
        check = 5.180555555555555
        check_w = 0.029066398893536072
        assert self.df_tessellation['nei'].mean() == check
        assert self.df_tessellation['nei_sw'].mean() == check
        assert self.df_tessellation['nei_wei'].mean() == check_w

    def test_node_degree(self):
        graph = mm.gdf_to_nx(self.df_streets)
        deg1 = mm.node_degree(graph=graph, target='graph')
        deg2 = mm.node_degree(geodataframe=self.df_streets, target='graph')
        deg3 = mm.node_degree(graph=graph, target='gdf')
        assert deg1.nodes[(1603650.450422848, 6464368.600601688)]['degree'] == 4
        assert deg2.nodes[(1603650.450422848, 6464368.600601688)]['degree'] == 4
        assert deg3.degree.mean() == 2.0625
