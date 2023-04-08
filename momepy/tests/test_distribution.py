import geopandas as gpd
import numpy as np
import pytest
from libpysal.weights import Queen

import momepy as mm


class TestDistribution:
    def setup_method(self):
        test_file_path = mm.datasets.get_path("bubenec")
        self.df_buildings = gpd.read_file(test_file_path, layer="buildings")
        self.df_streets = gpd.read_file(test_file_path, layer="streets")
        self.df_tessellation = gpd.read_file(test_file_path, layer="tessellation")
        self.df_buildings["height"] = np.linspace(10.0, 30.0, 144)
        self.df_buildings["volume"] = mm.Volume(self.df_buildings, "height").series
        self.df_streets["nID"] = mm.unique_id(self.df_streets)
        self.df_buildings["nID"] = mm.get_network_id(
            self.df_buildings, self.df_streets, "nID"
        )

    def test_Orientation(self):
        self.df_buildings["orient"] = mm.Orientation(self.df_buildings).series
        check = 41.05146788287027
        assert self.df_buildings["orient"][0] == pytest.approx(check, abs=1e-3)

        self.df_streets["orient"] = mm.Orientation(self.df_streets).series
        check = 40.7607
        assert self.df_streets["orient"][0] == pytest.approx(check)

    def test_SharedWalls(self):
        self.df_buildings["swr"] = mm.SharedWalls(self.df_buildings).series
        nonconsecutive = self.df_buildings.drop(2)
        result = mm.SharedWalls(nonconsecutive).series
        check = 39.395484381507075
        assert self.df_buildings["swr"][10] == check
        assert result[10] == check

    def test_SharedWallsRatio(self):
        self.df_buildings["swr"] = mm.SharedWallsRatio(self.df_buildings).series
        self.df_buildings["swr_array"] = mm.SharedWallsRatio(
            self.df_buildings, perimeters=self.df_buildings.geometry.length
        ).series
        nonconsecutive = self.df_buildings.drop(2)
        result = mm.SharedWallsRatio(nonconsecutive).series
        check = 0.3424804411228673
        assert self.df_buildings["swr"][10] == check
        assert self.df_buildings["swr_array"][10] == check
        assert result[10] == check

    def test_StreetAlignment(self):
        self.df_buildings["orient"] = orient = mm.Orientation(self.df_buildings).series
        self.df_buildings["street_alignment"] = mm.StreetAlignment(
            self.df_buildings, self.df_streets, "orient", network_id="nID"
        ).series
        self.df_buildings["street_alignment2"] = mm.StreetAlignment(
            self.df_buildings,
            self.df_streets,
            "orient",
            left_network_id="nID",
            right_network_id="nID",
        ).series
        self.df_buildings["street_a_arr"] = mm.StreetAlignment(
            self.df_buildings,
            self.df_streets,
            orient,
            left_network_id=self.df_buildings["nID"],
            right_network_id=self.df_streets["nID"],
        ).series

        with pytest.raises(
            ValueError,
            match=(
                "Network ID not set. Use either network_id or "
                "left_network_id and right_network_id."
            ),
        ):
            self.df_buildings["street_alignment"] = mm.StreetAlignment(
                self.df_buildings, self.df_streets, "orient"
            )
        with pytest.raises(ValueError, match="right_network_id not set."):
            self.df_buildings["street_alignment"] = mm.StreetAlignment(
                self.df_buildings, self.df_streets, "orient", left_network_id="nID"
            )
        with pytest.raises(ValueError, match="left_network_id not set."):
            self.df_buildings["street_alignment"] = mm.StreetAlignment(
                self.df_buildings, self.df_streets, "orient", right_network_id="nID"
            )
        check = 0.29073888476702336
        assert self.df_buildings["street_alignment"][0] == pytest.approx(
            check, abs=1e-3
        )
        assert self.df_buildings["street_alignment2"][0] == pytest.approx(
            check, abs=1e-3
        )
        assert self.df_buildings["street_a_arr"][0] == pytest.approx(check, abs=1e-3)

    def test_CellAlignment(self):
        self.df_buildings["orient"] = blgori = mm.Orientation(self.df_buildings).series
        self.df_tessellation["orient"] = tessori = mm.Orientation(
            self.df_tessellation
        ).series
        self.df_buildings["c_align"] = mm.CellAlignment(
            self.df_buildings, self.df_tessellation, "orient", "orient", "uID", "uID"
        ).series
        self.df_buildings["c_align_array"] = mm.CellAlignment(
            self.df_buildings, self.df_tessellation, blgori, tessori, "uID", "uID"
        ).series
        check = abs(
            self.df_buildings["orient"][0]
            - self.df_tessellation[
                self.df_tessellation["uID"] == self.df_buildings["uID"][0]
            ]["orient"].iloc[0]
        )
        assert self.df_buildings["c_align"][0] == pytest.approx(check)

    def test_Alignment(self):
        self.df_buildings["orient"] = mm.Orientation(self.df_buildings).series
        sw = Queen.from_dataframe(self.df_tessellation, ids="uID")
        self.df_buildings["align_sw"] = mm.Alignment(
            self.df_buildings, sw, "uID", self.df_buildings["orient"]
        ).series
        # GH#457 (`minimum_rotated_rectangle` calculation update)
        test_value = 22.744936872392813
        assert self.df_buildings["align_sw"][0] == pytest.approx(test_value)
        sw_drop = Queen.from_dataframe(self.df_tessellation[2:], ids="uID")
        assert (
            mm.Alignment(self.df_buildings, sw_drop, "uID", self.df_buildings["orient"])
            .series.isna()
            .any()
        )

    def test_NeighborDistance(self):
        sw = Queen.from_dataframe(self.df_tessellation, ids="uID")
        self.df_buildings["dist_sw"] = mm.NeighborDistance(
            self.df_buildings, sw, "uID"
        ).series
        check = 29.18589019096464
        assert self.df_buildings["dist_sw"][0] == pytest.approx(check)

        sw_drop = Queen.from_dataframe(self.df_tessellation[:-2], ids="uID")
        self.df_buildings["dist_sw"] = mm.NeighborDistance(
            self.df_buildings, sw_drop, "uID"
        ).series
        check = 29.18589019096464
        assert self.df_buildings["dist_sw"][0] == pytest.approx(check)
        assert self.df_buildings["dist_sw"].isna().any()

    def test_MeanInterbuildingDistance(self):
        sw = Queen.from_dataframe(self.df_tessellation, ids="uID")
        self.df_buildings["m_dist"] = mm.MeanInterbuildingDistance(
            self.df_buildings, sw, "uID", order=3
        ).series
        check = 29.305457092042744
        assert self.df_buildings["m_dist"][0] == pytest.approx(check)
        sw_drop = Queen.from_dataframe(self.df_tessellation[2:], ids="uID")
        assert (
            mm.MeanInterbuildingDistance(self.df_buildings, sw_drop, "uID")
            .series.isna()
            .any()
        )

    def test_NeighboringStreetOrientationDeviation(self):
        self.df_streets["dev"] = mm.NeighboringStreetOrientationDeviation(
            self.df_streets
        ).series
        check = 7.527840590385933
        assert self.df_streets["dev"].mean() == pytest.approx(check)

    def test_BuildingAdjacency(self):
        sw = Queen.from_dataframe(self.df_buildings, ids="uID", silence_warnings=True)
        swh = mm.sw_high(k=3, gdf=self.df_tessellation, ids="uID")
        self.df_buildings["adj_sw"] = mm.BuildingAdjacency(
            self.df_buildings,
            spatial_weights=sw,
            unique_id="uID",
            spatial_weights_higher=swh,
        ).series
        self.df_buildings["adj_sw_none"] = mm.BuildingAdjacency(
            self.df_buildings, unique_id="uID", spatial_weights_higher=swh
        ).series
        check = 0.2613824113909074
        assert self.df_buildings["adj_sw"].mean() == pytest.approx(check)
        assert self.df_buildings["adj_sw_none"].mean() == pytest.approx(check)
        swh_drop = mm.sw_high(k=3, gdf=self.df_tessellation[2:], ids="uID")
        assert (
            mm.BuildingAdjacency(
                self.df_buildings, unique_id="uID", spatial_weights_higher=swh_drop
            )
            .series.isna()
            .any()
        )

    def test_Neighbors(self):
        sw = Queen.from_dataframe(self.df_tessellation, ids="uID")
        sw_drop = Queen.from_dataframe(self.df_tessellation[2:], ids="uID")
        self.df_tessellation["nei_sw"] = mm.Neighbors(
            self.df_tessellation, sw, "uID"
        ).series
        self.df_tessellation["nei_wei"] = mm.Neighbors(
            self.df_tessellation, sw, "uID", weighted=True
        ).series
        check = 5.180555555555555
        check_w = 0.029066398893536072
        assert self.df_tessellation["nei_sw"].mean() == check
        assert self.df_tessellation["nei_wei"].mean() == check_w
        assert mm.Neighbors(self.df_tessellation, sw_drop, "uID").series.isna().any()
