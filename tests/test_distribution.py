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

    def test_Orientation(self):
        self.df_buildings["orient"] = mm.Orientation(self.df_buildings).o
        check = 41.05146788287027
        assert self.df_buildings["orient"][0] == check

    def test_SharedWallsRatio(self):
        self.df_buildings["swr"] = mm.SharedWallsRatio(self.df_buildings, "uID").swr
        self.df_buildings["swr_uid"] = mm.SharedWallsRatio(
            self.df_buildings, range(len(self.df_buildings))
        ).swr
        self.df_buildings["swr_array"] = mm.SharedWallsRatio(
            self.df_buildings, "uID", self.df_buildings.geometry.length
        ).swr
        check = 0.3424804411228673
        assert self.df_buildings["swr"][10] == check
        assert self.df_buildings["swr_uid"][10] == check
        assert self.df_buildings["swr_array"][10] == check

    def test_StreetAlignment(self):
        self.df_buildings["orient"] = orient = mm.Orientation(self.df_buildings).o
        self.df_buildings["street_alignment"] = mm.StreetAlignment(
            self.df_buildings, self.df_streets, "orient", network_id="nID"
        ).sa
        self.df_buildings["street_alignment2"] = mm.StreetAlignment(
            self.df_buildings,
            self.df_streets,
            "orient",
            left_network_id="nID",
            right_network_id="nID",
        ).sa
        self.df_buildings["street_a_arr"] = mm.StreetAlignment(
            self.df_buildings,
            self.df_streets,
            orient,
            left_network_id=self.df_buildings["nID"],
            right_network_id=self.df_streets["nID"],
        ).sa

        with pytest.raises(ValueError):
            self.df_buildings["street_alignment"] = mm.StreetAlignment(
                self.df_buildings, self.df_streets, "orient"
            )
        with pytest.raises(ValueError):
            self.df_buildings["street_alignment"] = mm.StreetAlignment(
                self.df_buildings, self.df_streets, "orient", left_network_id="nID"
            )
        with pytest.raises(ValueError):
            self.df_buildings["street_alignment"] = mm.StreetAlignment(
                self.df_buildings, self.df_streets, "orient", right_network_id="nID"
            )
        check = 0.29073888476702336
        assert self.df_buildings["street_alignment"][0] == check
        assert self.df_buildings["street_alignment2"][0] == check
        assert self.df_buildings["street_a_arr"][0] == check

    def test_CellAlignment(self):
        self.df_buildings["orient"] = blgori = mm.Orientation(self.df_buildings).o
        self.df_tessellation["orient"] = tessori = mm.Orientation(
            self.df_tessellation
        ).o
        self.df_buildings["c_align"] = mm.CellAlignment(
            self.df_buildings, self.df_tessellation, "orient", "orient", "uID", "uID"
        ).ca
        self.df_buildings["c_align_array"] = mm.CellAlignment(
            self.df_buildings, self.df_tessellation, blgori, tessori, "uID", "uID"
        ).ca
        check = abs(
            self.df_buildings["orient"][0]
            - self.df_tessellation[
                self.df_tessellation["uID"] == self.df_buildings["uID"][0]
            ]["orient"].iloc[0]
        )
        assert self.df_buildings["c_align"][0] == check

    def test_Alignment(self):
        self.df_buildings["orient"] = mm.Orientation(self.df_buildings).o
        sw = Queen.from_dataframe(self.df_tessellation, ids="uID")
        self.df_buildings["align_sw"] = mm.Alignment(
            self.df_buildings, sw, "uID", self.df_buildings["orient"]
        ).a
        assert self.df_buildings["align_sw"][0] == 18.299481296455237

    def test_NeighborDistance(self):
        sw = Queen.from_dataframe(self.df_tessellation, ids="uID")
        self.df_buildings["dist_sw"] = mm.NeighborDistance(
            self.df_buildings, sw, "uID"
        ).nd
        check = 29.18589019096464
        assert self.df_buildings["dist_sw"][0] == check

    def test_MeanInterbuildingDistance(self):
        sw = Queen.from_dataframe(self.df_tessellation, ids="uID")
        swh = mm.sw_high(k=3, gdf=self.df_tessellation, ids="uID")
        self.df_buildings["m_dist_sw"] = mm.MeanInterbuildingDistance(
            self.df_buildings, sw, "uID", swh
        ).mid
        self.df_buildings["m_dist"] = mm.MeanInterbuildingDistance(
            self.df_buildings, sw, "uID", order=3
        ).mid
        check = 29.305457092042744
        assert self.df_buildings["m_dist_sw"][0] == check
        assert self.df_buildings["m_dist"][0] == check

    def test_NeighboringStreetOrientationDeviation(self):
        self.df_streets["dev"] = mm.NeighboringStreetOrientationDeviation(
            self.df_streets
        ).nsod
        check = 5.986848512501008
        assert self.df_streets["dev"].mean() == check

    def test_BuildingAdjacencyy(self):
        sw = Queen.from_dataframe(self.df_buildings, ids="uID")
        swh = mm.sw_high(k=3, gdf=self.df_tessellation, ids="uID")
        self.df_buildings["adj_sw"] = mm.BuildingAdjacency(
            self.df_buildings,
            spatial_weights=sw,
            unique_id="uID",
            spatial_weights_higher=swh,
        ).ba
        self.df_buildings["adj_sw_none"] = mm.BuildingAdjacency(
            self.df_buildings, unique_id="uID", spatial_weights_higher=swh
        ).ba
        check = 0.2613824113909074
        assert self.df_buildings["adj_sw"].mean() == check
        assert self.df_buildings["adj_sw_none"].mean() == check

    def test_Neighbors(self):
        sw = Queen.from_dataframe(self.df_tessellation, ids="uID")
        self.df_tessellation["nei_sw"] = mm.Neighbors(self.df_tessellation, sw, "uID").n
        self.df_tessellation["nei_wei"] = mm.Neighbors(
            self.df_tessellation, sw, "uID", weighted=True
        ).n
        check = 5.180555555555555
        check_w = 0.029066398893536072
        assert self.df_tessellation["nei_sw"].mean() == check
        assert self.df_tessellation["nei_wei"].mean() == check_w
