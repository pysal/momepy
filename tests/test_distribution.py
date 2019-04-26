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
        self.df_buildings['orient'] = orient = mm.orientation(self.df_buildings)
        self.df_tessellation['orient'] = tess_orient = mm.orientation(self.df_tessellation)
        self.df_buildings['c_align'] = mm.cell_alignment(self.df_buildings, self.df_tessellation, 'orient', 'orient', 'uID')
        check = abs(self.df_buildings['orient'][0] -
                    self.df_tessellation[self.df_tessellation['uID'] == self.df_buildings['uID'][0]]['orient'].iloc[0])
        assert self.df_buildings['c_align'][0] == check

    def test_alignment(self):
        self.df_buildings['orient'] = orient = mm.orientation(self.df_buildings)
        self.df_buildings['align'] = mm.alignment(self.df_buildings, 'orient', self.df_tessellation, 'uID')
        sw = Queen.from_dataframe(self.df_tessellation)
        self.df_buildings['align_sw'] = mm.alignment(self.df_buildings, 'orient', self.df_tessellation, 'uID', sw)
        check = 18.299481296455237
        assert self.df_buildings['align'][0] == check
        assert self.df_buildings['align_sw'][0] == check
