import geopandas as gpd
import numpy as np
import pytest
from libpysal.weights import Queen
from shapely.geometry import Point

import momepy as mm


class TestIntensity:
    def setup_method(self):
        test_file_path = mm.datasets.get_path("bubenec")
        self.df_buildings = gpd.read_file(test_file_path, layer="buildings")
        self.df_streets = gpd.read_file(test_file_path, layer="streets")
        self.df_tessellation = gpd.read_file(test_file_path, layer="tessellation")
        self.df_streets["nID"] = mm.unique_id(self.df_streets)
        self.df_buildings["height"] = np.linspace(10.0, 30.0, 144)
        self.df_tessellation["area"] = self.df_tessellation.geometry.area
        self.df_buildings["area"] = self.df_buildings.geometry.area
        self.df_buildings["fl_area"] = mm.FloorArea(self.df_buildings, "height").series
        self.df_buildings["nID"] = mm.get_network_id(
            self.df_buildings, self.df_streets, "nID"
        )
        blocks = mm.Blocks(
            self.df_tessellation, self.df_streets, self.df_buildings, "bID", "uID"
        )
        self.blocks = blocks.blocks
        self.df_buildings["bID"] = blocks.buildings_id
        self.df_tessellation["bID"] = blocks.tessellation_id

    def test_AreaRatio(self):
        car = mm.AreaRatio(
            self.df_tessellation, self.df_buildings, "area", "area", "uID"
        ).series
        carlr = mm.AreaRatio(
            self.df_tessellation,
            self.df_buildings,
            "area",
            "area",
            left_unique_id="uID",
            right_unique_id="uID",
        ).series
        check = 0.3206556897709747
        assert car.mean() == pytest.approx(check)
        assert carlr.mean() == pytest.approx(check)
        far = mm.AreaRatio(
            self.df_tessellation,
            self.df_buildings,
            self.df_tessellation.area,
            self.df_buildings.fl_area,
            "uID",
        ).series
        check = 1.910949846262234
        assert far.mean() == check
        with pytest.raises(ValueError, match="Unique ID not correctly set."):
            car = mm.AreaRatio(self.df_tessellation, self.df_buildings, "area", "area")
        with pytest.raises(ValueError, match="Unique ID not correctly set."):
            car = mm.AreaRatio(
                self.df_tessellation,
                self.df_buildings,
                "area",
                "area",
                left_unique_id="uID",
            )
        with pytest.raises(ValueError, match="Unique ID not correctly set."):
            car = mm.AreaRatio(
                self.df_tessellation,
                self.df_buildings,
                "area",
                "area",
                right_unique_id="uID",
            )
        car_sel = mm.AreaRatio(
            self.df_tessellation.iloc[10:20], self.df_buildings, "area", "area", "uID"
        ).series
        assert (car_sel.index == self.df_tessellation.iloc[10:20].index).all()
        self.blocks["area"] = self.blocks.geometry.area
        car_block = mm.AreaRatio(self.blocks, self.df_buildings, "area", "area", "bID")
        assert car_block.series.mean() == pytest.approx(0.27619743, rel=1e-8)

    def test_Count(self):
        eib = mm.Count(self.blocks, self.df_buildings, "bID", "bID").series
        weib = mm.Count(
            self.blocks, self.df_buildings, "bID", "bID", weighted=True
        ).series
        weis = mm.Count(
            self.df_streets, self.df_buildings, "nID", "nID", weighted=True
        ).series
        check_eib = [13, 14, 8, 26, 24, 17, 23, 19]
        check_weib = pytest.approx(0.00040170607189453996)
        assert eib.tolist() == check_eib
        assert weib.mean() == check_weib
        assert weis.mean() == pytest.approx(0.020524232642849215)

        point_gdf = gpd.GeoDataFrame(
            {"nID": [0]}, geometry=[Point(1603569.010067892, 6464302.821695424)]
        )
        with pytest.raises(
            TypeError, match="Geometry type does not support weighting."
        ):
            mm.Count(point_gdf, self.blocks, "nID", "bID", weighted=True).series

    def test_Courtyards(self):
        courtyards = mm.Courtyards(self.df_buildings).series
        sw = Queen.from_dataframe(self.df_buildings, silence_warnings=True)
        courtyards_wm = mm.Courtyards(self.df_buildings, sw).series
        check = 0.6805555555555556
        assert courtyards.mean() == check
        assert courtyards_wm.mean() == check

    def test_BlocksCount(self):
        sw = mm.sw_high(k=5, gdf=self.df_tessellation, ids="uID")
        count = mm.BlocksCount(self.df_tessellation, "bID", sw, "uID").series
        count2 = mm.BlocksCount(
            self.df_tessellation, self.df_tessellation.bID, sw, "uID"
        ).series
        unweigthed = mm.BlocksCount(
            self.df_tessellation, "bID", sw, "uID", weighted=False
        ).series
        check = 3.142437439120778e-05
        check2 = 5.222222222222222
        assert count.mean() == check
        assert count2.mean() == check
        assert unweigthed.mean() == check2
        with pytest.raises(
            ValueError, match="Attribute 'weighted' needs to be True or False."
        ):
            count = mm.BlocksCount(
                self.df_tessellation, "bID", sw, "uID", weighted="yes"
            )
        sw_drop = mm.sw_high(k=5, gdf=self.df_tessellation[2:], ids="uID")
        assert (
            mm.BlocksCount(self.df_tessellation, "bID", sw_drop, "uID")
            .series.isna()
            .any()
        )

    def test_Reached(self):
        count = mm.Reached(self.df_streets, self.df_buildings, "nID", "nID").series
        area = mm.Reached(
            self.df_streets,
            self.df_buildings,
            self.df_streets.nID,
            self.df_buildings.nID,
            mode="sum",
        ).series
        mean = mm.Reached(
            self.df_streets, self.df_buildings, "nID", "nID", mode="mean"
        ).series
        std = mm.Reached(
            self.df_streets, self.df_buildings, "nID", "nID", mode="std"
        ).series
        area_v = mm.Reached(
            self.df_streets,
            self.df_buildings,
            "nID",
            "nID",
            mode="sum",
            values="fl_area",
        ).series
        mean_v = mm.Reached(
            self.df_streets,
            self.df_buildings,
            "nID",
            "nID",
            mode="mean",
            values="fl_area",
        ).series
        std_v = mm.Reached(
            self.df_streets,
            self.df_buildings,
            "nID",
            "nID",
            mode="std",
            values="fl_area",
        ).series
        sw = mm.sw_high(k=2, gdf=self.df_streets)
        count_sw = mm.Reached(
            self.df_streets, self.df_buildings, "nID", "nID", sw
        ).series
        assert max(count) == 18
        assert max(area) == 18085.45897711331
        assert max(count_sw) == 138
        assert max(mean) == 1808.5458977113315
        assert max(std) == 3153.7019229524785
        assert max(area_v) == 79169.31385861784
        assert max(mean_v) == 7916.931385861784
        assert max(std_v) == 8995.18003493457

    def test_NodeDensity(self):
        nx = mm.gdf_to_nx(self.df_streets)
        nx = mm.node_degree(nx)
        nodes, edges, W = mm.nx_to_gdf(nx, spatial_weights=True)
        sw = mm.sw_high(k=3, weights=W)
        density = mm.NodeDensity(nodes, edges, sw).series
        weighted = mm.NodeDensity(
            nodes, edges, sw, weighted=True, node_degree="degree"
        ).series
        array = mm.NodeDensity(nodes, edges, W).series
        assert density.mean() == 0.005534125924228438
        assert weighted.mean() == pytest.approx(0.010090861332429164)
        assert array.mean() == 0.01026753724860306

    def test_Density(self):
        sw = mm.sw_high(k=3, gdf=self.df_tessellation, ids="uID")
        dens = mm.Density(
            self.df_tessellation,
            self.df_buildings["fl_area"],
            sw,
            "uID",
            self.df_tessellation.area,
        ).series
        dens2 = mm.Density(
            self.df_tessellation, self.df_buildings["fl_area"], sw, "uID"
        ).series
        check = 1.661587
        assert dens.mean() == pytest.approx(check)
        assert dens2.mean() == pytest.approx(check)
        sw_drop = mm.sw_high(k=3, gdf=self.df_tessellation[2:], ids="uID")
        assert (
            mm.Density(
                self.df_tessellation, self.df_buildings["fl_area"], sw_drop, "uID"
            )
            .series.isna()
            .any()
        )

        # island
        sw.neighbors[1] = []
        dens3 = mm.Density(
            self.df_tessellation,
            self.df_buildings["fl_area"],
            sw,
            "uID",
            self.df_tessellation.area,
        ).series
        assert dens3.mean() == pytest.approx(1.656420)
