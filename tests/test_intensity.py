import geopandas as gpd
import momepy as mm
import numpy as np
import pytest
from libpysal.weights import Queen
from pytest import approx


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
        self.df_buildings["fl_area"] = mm.FloorArea(self.df_buildings, "height").fa
        self.df_buildings["nID"] = mm.get_network_id(
            self.df_buildings, self.df_streets, "uID", "nID"
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
        ).ar
        carlr = mm.AreaRatio(
            self.df_tessellation,
            self.df_buildings,
            "area",
            "area",
            left_unique_id="uID",
            right_unique_id="uID",
        ).ar
        check = 0.3206556897709747
        assert car.mean() == check
        assert carlr.mean() == check
        far = mm.AreaRatio(
            self.df_tessellation,
            self.df_buildings,
            self.df_tessellation.area,
            self.df_buildings.fl_area,
            "uID",
        ).ar
        check = 1.910949846262234
        assert far.mean() == check
        with pytest.raises(ValueError):
            car = mm.AreaRatio(self.df_tessellation, self.df_buildings, "area", "area")
        with pytest.raises(ValueError):
            car = mm.AreaRatio(
                self.df_tessellation,
                self.df_buildings,
                "area",
                "area",
                left_unique_id="uID",
            )
        with pytest.raises(ValueError):
            car = mm.AreaRatio(
                self.df_tessellation,
                self.df_buildings,
                "area",
                "area",
                right_unique_id="uID",
            )

    def test_Count(self):
        eib = mm.Count(self.blocks, self.df_buildings, "bID", "bID").c
        weib = mm.Count(self.blocks, self.df_buildings, "bID", "bID", weighted=True).c
        weis = mm.Count(
            self.df_streets, self.df_buildings, "nID", "nID", weighted=True
        ).c
        check_eib = [13, 14, 8, 26, 24, 17, 23, 19]
        check_weib = 0.00040170607189453996
        assert eib.tolist() == check_eib
        assert weib.mean() == check_weib
        assert weis.mean() == 0.020524232642849215

    def test_Courtyards(self):
        courtyards = mm.Courtyards(self.df_buildings, "bID").c
        sw = Queen.from_dataframe(self.df_buildings)
        courtyards_wm = mm.Courtyards(self.df_buildings, self.df_buildings.bID, sw).c
        check = 0.6805555555555556
        assert courtyards.mean() == check
        assert courtyards_wm.mean() == check

    def test_BlocksCount(self):
        sw = mm.sw_high(k=5, gdf=self.df_tessellation, ids="uID")
        count = mm.BlocksCount(self.df_tessellation, "bID", sw, "uID").bc
        count2 = mm.BlocksCount(
            self.df_tessellation, self.df_tessellation.bID, sw, "uID"
        ).bc
        unweigthed = mm.BlocksCount(
            self.df_tessellation, "bID", sw, "uID", weighted=False
        ).bc
        check = 3.142437439120778e-05
        check2 = 5.222222222222222
        assert count.mean() == check
        assert count2.mean() == check
        assert unweigthed.mean() == check2
        with pytest.raises(ValueError):
            count = mm.BlocksCount(
                self.df_tessellation, "bID", sw, "uID", weighted="yes"
            )

    def test_Reached(self):
        count = mm.Reached(self.df_streets, self.df_buildings, "nID", "nID").r
        area = mm.Reached(
            self.df_streets,
            self.df_buildings,
            self.df_streets.nID,
            self.df_buildings.nID,
            mode="sum",
        ).r
        mean = mm.Reached(
            self.df_streets, self.df_buildings, "nID", "nID", mode="mean"
        ).r
        std = mm.Reached(self.df_streets, self.df_buildings, "nID", "nID", mode="std").r
        area_v = mm.Reached(
            self.df_streets,
            self.df_buildings,
            "nID",
            "nID",
            mode="sum",
            values="fl_area",
        ).r
        mean_v = mm.Reached(
            self.df_streets,
            self.df_buildings,
            "nID",
            "nID",
            mode="mean",
            values="fl_area",
        ).r
        std_v = mm.Reached(
            self.df_streets,
            self.df_buildings,
            "nID",
            "nID",
            mode="std",
            values="fl_area",
        ).r
        sw = mm.sw_high(k=2, gdf=self.df_streets)
        count_sw = mm.Reached(self.df_streets, self.df_buildings, "nID", "nID", sw).r
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
        density = mm.NodeDensity(nodes, edges, sw).nd
        weighted = mm.NodeDensity(
            nodes, edges, sw, weighted=True, node_degree="degree"
        ).nd
        array = mm.NodeDensity(nodes, edges, W).nd
        assert density.mean() == 0.012690163074599968
        assert weighted.mean() == 0.023207675994368446
        assert array.mean() == 0.008554067995928158

    def test_Density(self):
        sw = mm.sw_high(k=3, gdf=self.df_tessellation, ids="uID")
        dens = mm.Density(
            self.df_tessellation,
            self.df_buildings["fl_area"],
            sw,
            "uID",
            self.df_tessellation.area,
        ).d
        dens2 = mm.Density(
            self.df_tessellation, self.df_buildings["fl_area"], sw, "uID"
        ).d
        check = 1.661587
        assert dens.mean() == approx(check)
        assert dens2.mean() == approx(check)
