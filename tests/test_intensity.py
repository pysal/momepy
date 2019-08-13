import momepy as mm
import geopandas as gpd
import numpy as np
from libpysal.weights import Queen

import pytest

class TestIntensity:

    def setup_method(self):

        test_file_path = mm.datasets.get_path('bubenec')
        self.df_buildings = gpd.read_file(test_file_path, layer='buildings')
        self.df_streets = gpd.read_file(test_file_path, layer='streets')
        self.df_tessellation = gpd.read_file(test_file_path, layer='tessellation')
        self.df_streets['nID'] = mm.unique_id(self.df_streets)
        self.df_buildings['height'] = np.linspace(10., 30., 144)
        self.df_tessellation['area'] = self.df_tessellation.geometry.area
        self.df_buildings['area'] = self.df_buildings.geometry.area
        self.df_buildings['fl_area'] = mm.floor_area(self.df_buildings, 'height')
        self.df_buildings['nID'] = mm.get_network_id(self.df_buildings, self.df_streets, 'uID', 'nID')
        self.blocks, self.df_buildings['bID'], self.df_tessellation['bID'] = mm.blocks(
            self.df_tessellation, self.df_streets, self.df_buildings, 'bID', 'uID')

    def test_object_area_ratio(self):
        car = mm.object_area_ratio(self.df_tessellation, self.df_buildings, 'area', 'area', 'uID')
        carlr = mm.object_area_ratio(self.df_tessellation, self.df_buildings, 'area', 'area', left_unique_id='uID', right_unique_id='uID')
        check = 0.3206556897709747
        assert car.mean() == check
        assert carlr.mean() == check
        far = mm.object_area_ratio(self.df_tessellation, self.df_buildings, 'area', 'fl_area', 'uID')
        check = 1.910949846262234
        assert far.mean() == check
        with pytest.raises(ValueError):
            car = mm.object_area_ratio(self.df_tessellation, self.df_buildings, 'area', 'area')
        with pytest.raises(ValueError):
            car = mm.object_area_ratio(self.df_tessellation, self.df_buildings, 'area', 'area', left_unique_id='uID')
        with pytest.raises(ValueError):
            car = mm.object_area_ratio(self.df_tessellation, self.df_buildings, 'area', 'area', right_unique_id='uID')

    def test_elements_count(self):
        eib = mm.elements_count(self.blocks, self.df_buildings, 'bID', 'bID')
        weib = mm.elements_count(self.blocks, self.df_buildings, 'bID', 'bID', weighted=True)
        weis = mm.elements_count(self.df_streets, self.df_buildings, 'nID', 'nID', weighted=True)
        check_eib = [13, 14, 8, 26, 24, 17, 23, 19]
        check_weib = 0.00040170607189453996
        assert eib.tolist() == check_eib
        assert weib.mean() == check_weib
        assert weis.mean() == 0.020524232642849215

    def test_courtyards(self):
        courtyards = mm.courtyards(self.df_buildings, 'bID')
        sw = Queen.from_dataframe(self.df_buildings)
        courtyards_wm = mm.courtyards(self.df_buildings, 'bID', sw)
        check = 0.6805555555555556
        assert courtyards.mean() == check
        assert courtyards_wm.mean() == check

    def test_blocks_count(self):
        sw = mm.Queen_higher(k=5, geodataframe=self.df_tessellation, ids='uID')
        count2 = mm.blocks_count(self.df_tessellation, 'bID', sw, 'uID')
        check = 3.142437439120778e-05
        assert count2.mean() == check

    def test_reached(self):
        count = mm.reached(self.df_streets, self.df_buildings, 'nID')
        area = mm.reached(self.df_streets, self.df_buildings, 'nID', mode='sum')
        mean = mm.reached(self.df_streets, self.df_buildings, 'nID', mode='mean')
        std = mm.reached(self.df_streets, self.df_buildings, 'nID', mode='std')
        area_v = mm.reached(self.df_streets, self.df_buildings, 'nID', mode='sum', values='fl_area')
        mean_v = mm.reached(self.df_streets, self.df_buildings, 'nID', mode='mean', values='fl_area')
        std_v = mm.reached(self.df_streets, self.df_buildings, 'nID', mode='std', values='fl_area')
        sw = mm.Queen_higher(k=2, geodataframe=self.df_streets)
        count_sw = mm.reached(self.df_streets, self.df_buildings, 'nID', sw)
        assert max(count) == 18
        assert max(area) == 18085.45897711331
        assert max(count_sw) == 138
        assert max(mean) == 1808.5458977113315
        assert max(std) == 3153.7019229524785
        assert max(area_v) == 79169.31385861784
        assert max(mean_v) == 7916.931385861784
        assert max(std_v) == 8995.18003493457

    def test_node_density(self):
        nx = mm.gdf_to_nx(self.df_streets)
        nx = mm.node_degree(nx)
        nodes, edges, W = mm.nx_to_gdf(nx, spatial_weights=True)
        sw = mm.Queen_higher(k=3, weights=W)
        density = mm.node_density(nodes, edges, sw)
        weighted = mm.node_density(nodes, edges, sw, weighted=True, node_degree='degree')
        assert density.mean() == 0.012690163074599968
        assert weighted.mean() == 0.023207675994368446

    def test_density(self):
        sw = mm.Queen_higher(k=3, geodataframe=self.df_tessellation, ids='uID')
        dens = mm.density(self.df_tessellation, self.df_buildings['fl_area'], sw, 'uID', 'area')
        dens2 = mm.density(self.df_tessellation, self.df_buildings['fl_area'], sw, 'uID')
        check = 1.6615871155383324
        assert dens.mean() == check
        assert dens2.mean() == check
