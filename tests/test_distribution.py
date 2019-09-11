import momepy as mm
import geopandas as gpd
import numpy as np
from libpysal.weights import Queen

import pytest


class TestDistribution:
    def setup_method(self):

        test_file_path = mm.datasets.get_path("bubenec")
        self.df_buildings = gpd.read_file(test_file_path, layer="buildings")
        self.df_streets = gpd.read_file(test_file_path, layer="streets")
        self.df_tessellation = gpd.read_file(test_file_path, layer="tessellation")
        self.df_buildings["height"] = np.linspace(10.0, 30.0, 144)
        self.df_buildings["volume"] = mm.Volume(self.df_buildings, "height").volume
        self.df_streets["nID"] = mm.unique_id(self.df_streets)
        self.df_buildings["nID"] = mm.get_network_id(
            self.df_buildings, self.df_streets, "uID", "nID"
        )

    def test_orientation(self):
        self.df_buildings["orient"] = mm.orientation(self.df_buildings)
        check = 41.05146788287027
        assert self.df_buildings["orient"][0] == check

    def test_shared_walls_ratio(self):
        self.df_buildings["swr"] = mm.shared_walls_ratio(self.df_buildings, "uID")
        self.df_buildings["swr_uid"] = mm.shared_walls_ratio(
            self.df_buildings, range(len(self.df_buildings))
        )
        self.df_buildings["swr_array"] = mm.shared_walls_ratio(
            self.df_buildings, "uID", self.df_buildings.geometry.length
        )
        check = 0.3424804411228673
        assert self.df_buildings["swr"][10] == check
        assert self.df_buildings["swr_uid"][10] == check
        assert self.df_buildings["swr_array"][10] == check

    def test_street_alignment(self):
        self.df_buildings["orient"] = orient = mm.orientation(self.df_buildings)
        self.df_buildings["street_alignment"] = mm.street_alignment(
            self.df_buildings, self.df_streets, "orient", network_id="nID"
        )
        self.df_buildings["street_alignment2"] = mm.street_alignment(
            self.df_buildings,
            self.df_streets,
            "orient",
            left_network_id="nID",
            right_network_id="nID",
        )
        self.df_buildings["street_a_arr"] = mm.street_alignment(
            self.df_buildings,
            self.df_streets,
            orient,
            left_network_id=self.df_buildings["nID"],
            right_network_id=self.df_streets["nID"],
        )

        with pytest.raises(ValueError):
            self.df_buildings["street_alignment"] = mm.street_alignment(
                self.df_buildings, self.df_streets, "orient"
            )
        with pytest.raises(ValueError):
            self.df_buildings["street_alignment"] = mm.street_alignment(
                self.df_buildings, self.df_streets, "orient", left_network_id="nID"
            )
        with pytest.raises(ValueError):
            self.df_buildings["street_alignment"] = mm.street_alignment(
                self.df_buildings, self.df_streets, "orient", right_network_id="nID"
            )
        check = 0.29073888476702336
        assert self.df_buildings["street_alignment"][0] == check
        assert self.df_buildings["street_alignment2"][0] == check
        assert self.df_buildings["street_a_arr"][0] == check

    def test_cell_alignment(self):
        self.df_buildings["orient"] = blgori = mm.orientation(self.df_buildings)
        self.df_tessellation["orient"] = tessori = mm.orientation(self.df_tessellation)
        self.df_buildings["c_align"] = mm.cell_alignment(
            self.df_buildings, self.df_tessellation, "orient", "orient", "uID", "uID"
        )
        self.df_buildings["c_align_array"] = mm.cell_alignment(
            self.df_buildings, self.df_tessellation, blgori, tessori, "uID", "uID"
        )
        check = abs(
            self.df_buildings["orient"][0]
            - self.df_tessellation[
                self.df_tessellation["uID"] == self.df_buildings["uID"][0]
            ]["orient"].iloc[0]
        )
        assert self.df_buildings["c_align"][0] == check

    def test_alignment(self):
        self.df_buildings["orient"] = mm.orientation(self.df_buildings)
        sw = Queen.from_dataframe(self.df_tessellation, ids="uID")
        self.df_buildings["align_sw"] = mm.alignment(
            self.df_buildings, sw, "uID", self.df_buildings["orient"]
        )
        assert self.df_buildings["align_sw"][0] == 18.299481296455237

    def test_neighbour_distance(self):
        sw = Queen.from_dataframe(self.df_tessellation, ids="uID")
        self.df_buildings["dist_sw"] = mm.neighbour_distance(
            self.df_buildings, sw, "uID"
        )
        check = 29.18589019096464
        assert self.df_buildings["dist_sw"][0] == check

    def test_mean_interbuilding_distance(self):
        sw = Queen.from_dataframe(self.df_tessellation, ids="uID")
        swh = mm.sw_high(k=3, gdf=self.df_tessellation, ids="uID")
        self.df_buildings["m_dist_sw"] = mm.mean_interbuilding_distance(
            self.df_buildings, sw, "uID", swh
        )
        self.df_buildings["m_dist"] = mm.mean_interbuilding_distance(
            self.df_buildings, sw, "uID", order=3
        )
        check = 29.305457092042744
        assert self.df_buildings["m_dist_sw"][0] == check
        assert self.df_buildings["m_dist"][0] == check

    def test_neighbouring_street_orientation_deviation(self):
        self.df_streets["dev"] = mm.neighbouring_street_orientation_deviation(
            self.df_streets
        )
        check = 5.986848512501008
        assert self.df_streets["dev"].mean() == check

    def test_building_adjacency(self):
        sw = Queen.from_dataframe(self.df_buildings, ids="uID")
        swh = mm.sw_high(k=3, gdf=self.df_tessellation, ids="uID")
        self.df_buildings["adj_sw"] = mm.building_adjacency(
            self.df_buildings,
            spatial_weights=sw,
            unique_id="uID",
            spatial_weights_higher=swh,
        )
        self.df_buildings["adj_sw_none"] = mm.building_adjacency(
            self.df_buildings, unique_id="uID", spatial_weights_higher=swh
        )
        check = 0.2613824113909074
        assert self.df_buildings["adj_sw"].mean() == check
        assert self.df_buildings["adj_sw_none"].mean() == check

    def test_neighbours(self):
        sw = Queen.from_dataframe(self.df_tessellation, ids="uID")
        self.df_tessellation["nei_sw"] = mm.neighbours(self.df_tessellation, sw, "uID")
        self.df_tessellation["nei_wei"] = mm.neighbours(
            self.df_tessellation, sw, "uID", weighted=True
        )
        check = 5.180555555555555
        check_w = 0.029066398893536072
        assert self.df_tessellation["nei_sw"].mean() == check
        assert self.df_tessellation["nei_wei"].mean() == check_w
