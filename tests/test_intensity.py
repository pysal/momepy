import momepy as mm
import geopandas as gpd
import numpy as np
from libpysal.weights import Queen


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
        self.df_buildings, self.df_tessellation, self.blocks = mm.blocks(self.df_tessellation, self.df_streets, self.df_buildings, 'bID', 'uID')

    def test_covered_area_ratio(self):
        car = mm.covered_area_ratio(self.df_tessellation, self.df_buildings, 'area', 'area')
        check = 0.3206556897709747
        assert car.mean() == check

    def test_floor_area_ratio(self):
        far = mm.floor_area_ratio(self.df_tessellation, self.df_buildings, 'area', 'fl_area')
        check = 1.910949846262234
        assert far.mean() == check

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

    def test_gross_density(self):
        dens = mm.gross_density(self.df_tessellation, self.df_buildings, 'area', 'fl_area')
        sw = mm.Queen_higher(k=3, geodataframe=self.df_tessellation)
        dens2 = mm.gross_density(self.df_tessellation, self.df_buildings, 'area', 'fl_area', spatial_weights=sw)
        check = 1.6615871155383324
        assert dens.mean() == check
        assert dens2.mean() == check

    def test_blocks_count(self):
        count = mm.blocks_count(self.df_tessellation, 'bID', spatial_weights=None, order=5)
        sw = mm.Queen_higher(k=5, geodataframe=self.df_tessellation)
        count2 = mm.blocks_count(self.df_tessellation, 'bID', spatial_weights=sw)
        check = 3.142437439120778e-05
        assert count.mean() == check
        assert count2.mean() == check

    def test_reached(self):
        count = mm.reached(self.df_streets, self.df_buildings, 'nID')
        area = mm.reached(self.df_streets, self.df_buildings, 'nID', mode='sum')
        sw = mm.Queen_higher(k=2, geodataframe=self.df_streets)
        count_sw = mm.reached(self.df_streets, self.df_buildings, 'nID', sw)
        assert max(count) == 18
        assert max(area) == 18085.45897711331
        assert max(count_sw) == 138
