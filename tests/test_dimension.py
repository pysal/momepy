import geopandas as gpd
import momepy as mm
import numpy as np
import pytest
from momepy import sw_high
from momepy.shape import _make_circle
from pytest import approx
from shapely.geometry import Polygon


class TestDimensions:
    def setup_method(self):

        test_file_path = mm.datasets.get_path("bubenec")
        self.df_buildings = gpd.read_file(test_file_path, layer="buildings")
        self.df_streets = gpd.read_file(test_file_path, layer="streets")
        self.df_tessellation = gpd.read_file(test_file_path, layer="tessellation")
        self.df_buildings["height"] = np.linspace(10.0, 30.0, 144)

    def test_Area(self):
        self.df_buildings["area"] = mm.Area(self.df_buildings).series
        check = self.df_buildings.geometry[0].area
        assert self.df_buildings["area"][0] == check

    def test_Perimeter(self):
        self.df_buildings["perimeter"] = mm.Perimeter(self.df_buildings).series
        check = self.df_buildings.geometry[0].length
        assert self.df_buildings["perimeter"][0] == check

    def test_Volume(self):
        self.df_buildings["area"] = self.df_buildings.geometry.area
        self.df_buildings["volume"] = mm.Volume(
            self.df_buildings, "height", "area"
        ).series
        check = self.df_buildings.geometry[0].area * self.df_buildings.height[0]
        assert self.df_buildings["volume"][0] == check

        area = self.df_buildings.geometry.area
        height = np.linspace(10.0, 30.0, 144)
        self.df_buildings["volume"] = mm.Volume(self.df_buildings, height, area).series
        check = self.df_buildings.geometry[0].area * self.df_buildings.height[0]
        assert self.df_buildings["volume"][0] == check

        self.df_buildings["volume"] = mm.Volume(self.df_buildings, "height").series
        check = self.df_buildings.geometry[0].area * self.df_buildings.height[0]
        assert self.df_buildings["volume"][0] == check

        with pytest.raises(KeyError):
            self.df_buildings["volume"] = mm.Volume(
                self.df_buildings, "height", "nonexistent"
            )

    def test_FloorArea(self):
        self.df_buildings["area"] = self.df_buildings.geometry.area
        self.df_buildings["floor_area"] = mm.FloorArea(
            self.df_buildings, "height", "area"
        ).series
        check = self.df_buildings.geometry[0].area * (self.df_buildings.height[0] // 3)
        assert self.df_buildings["floor_area"][0] == check

        area = self.df_buildings.geometry.area
        height = np.linspace(10.0, 30.0, 144)
        self.df_buildings["floor_area"] = mm.FloorArea(
            self.df_buildings, height, area
        ).series
        assert self.df_buildings["floor_area"][0] == check

        self.df_buildings["floor_area"] = mm.FloorArea(
            self.df_buildings, "height"
        ).series
        assert self.df_buildings["floor_area"][0] == check

        with pytest.raises(KeyError):
            self.df_buildings["floor_area"] = mm.FloorArea(
                self.df_buildings, "height", "nonexistent"
            )

    def test_CourtyardArea(self):
        self.df_buildings["area"] = self.df_buildings.geometry.area
        self.df_buildings["courtyard_area"] = mm.CourtyardArea(
            self.df_buildings, "area"
        ).series
        check = (
            Polygon(self.df_buildings.geometry[80].exterior).area
            - self.df_buildings.geometry[80].area
        )
        assert self.df_buildings["courtyard_area"][80] == check

        area = self.df_buildings.geometry.area
        self.df_buildings["courtyard_area"] = mm.CourtyardArea(
            self.df_buildings, area
        ).series
        assert self.df_buildings["courtyard_area"][80] == check

        self.df_buildings["courtyard_area"] = mm.CourtyardArea(self.df_buildings).series
        assert self.df_buildings["courtyard_area"][80] == check

        with pytest.raises(KeyError):
            self.df_buildings["courtyard_area"] = mm.CourtyardArea(
                self.df_buildings, "nonexistent"
            )

    def test_LongestAxisLength(self):
        self.df_buildings["long_axis"] = mm.LongestAxisLength(self.df_buildings).series
        check = (
            _make_circle(self.df_buildings.geometry[0].convex_hull.exterior.coords)[2]
            * 2
        )
        assert self.df_buildings["long_axis"][0] == check

    def test_AverageCharacter(self):
        spatial_weights = sw_high(k=3, gdf=self.df_tessellation, ids="uID")
        self.df_tessellation["area"] = area = self.df_tessellation.geometry.area
        self.df_tessellation["mesh_ar"] = mm.AverageCharacter(
            self.df_tessellation,
            values="area",
            spatial_weights=spatial_weights,
            unique_id="uID",
            mode="mode",
        ).mode
        self.df_tessellation["mesh_array"] = mm.AverageCharacter(
            self.df_tessellation,
            values=area,
            spatial_weights=spatial_weights,
            unique_id="uID",
            mode="median",
        ).median
        self.df_tessellation["mesh_id"] = mm.AverageCharacter(
            self.df_tessellation,
            spatial_weights=spatial_weights,
            values="area",
            rng=(10, 90),
            unique_id="uID",
        ).mean
        self.df_tessellation["mesh_iq"] = mm.AverageCharacter(
            self.df_tessellation,
            spatial_weights=spatial_weights,
            values="area",
            rng=(25, 75),
            unique_id="uID",
        ).series
        all_m = mm.AverageCharacter(
            self.df_tessellation,
            spatial_weights=spatial_weights,
            values="area",
            unique_id="uID",
        )
        two = mm.AverageCharacter(
            self.df_tessellation,
            spatial_weights=spatial_weights,
            values="area",
            unique_id="uID",
            mode=["mean", "median"],
        )
        with pytest.raises(ValueError):
            self.df_tessellation["mesh_ar"] = mm.AverageCharacter(
                self.df_tessellation,
                values="area",
                spatial_weights=spatial_weights,
                unique_id="uID",
                mode="nonexistent",
            )
        with pytest.raises(ValueError):
            self.df_tessellation["mesh_ar"] = mm.AverageCharacter(
                self.df_tessellation,
                values="area",
                spatial_weights=spatial_weights,
                unique_id="uID",
                mode=["nonexistent", "mean"],
            )
        assert self.df_tessellation["mesh_ar"][0] == approx(249.503, rel=1e-3)
        assert self.df_tessellation["mesh_array"][0] == approx(2623.996, rel=1e-3)
        assert self.df_tessellation["mesh_id"][38] == approx(2250.224, rel=1e-3)
        assert self.df_tessellation["mesh_iq"][38] == approx(2118.609, rel=1e-3)
        assert all_m.mean[0] == approx(2922.957, rel=1e-3)
        assert all_m.median[0] == approx(2623.996, rel=1e-3)
        assert all_m.mode[0] == approx(249.503, rel=1e-3)
        assert all_m.series[0] == approx(2922.957, rel=1e-3)
        assert two.mean[0] == approx(2922.957, rel=1e-3)
        assert two.median[0] == approx(2623.996, rel=1e-3)
        sw_drop = sw_high(k=3, gdf=self.df_tessellation[2:], ids="uID")
        assert (
            mm.AverageCharacter(
                self.df_tessellation,
                values="area",
                spatial_weights=sw_drop,
                unique_id="uID",
            )
            .series.isna()
            .any()
        )

    def test_StreetProfile(self):
        results = mm.StreetProfile(self.df_streets, self.df_buildings, heights="height")
        assert results.w[0] == 46.7290758769204
        assert results.wd[0] == 0.5690458650581023
        assert results.h[0] == 19.454545454545457
        assert results.p[0] == 0.4163263469148574
        assert results.o[0] == 0.9107142857142857
        assert results.hd[0] == 7.720786443287433

        height = np.linspace(10.0, 30.0, 144)
        results2 = mm.StreetProfile(
            self.df_streets, self.df_buildings, heights=height, tick_length=100
        )
        assert results2.w[0] == 67.563796073073
        assert results2.wd[0] == 8.791875291865827
        assert results2.h[0] == 23.756643356643362
        assert results2.p[0] == 0.3516179483306353
        assert results2.o[0] == 0.5535714285714286
        assert results2.hd[0] == 5.526848034418866

        results3 = mm.StreetProfile(self.df_streets, self.df_buildings)
        assert results3.w[0] == 46.7290758769204
        assert results3.wd[0] == 0.5690458650581023
        assert results3.o[0] == 0.9107142857142857

    def test_WeightedCharacter(self):
        sw = sw_high(k=3, gdf=self.df_tessellation, ids="uID")
        weighted = mm.WeightedCharacter(self.df_buildings, "height", sw, "uID").series
        assert weighted[38] == approx(18.301, rel=1e-3)

        self.df_buildings["area"] = self.df_buildings.geometry.area
        sw = sw_high(k=3, gdf=self.df_tessellation, ids="uID")
        weighted = mm.WeightedCharacter(
            self.df_buildings, "height", sw, "uID", "area"
        ).series
        assert weighted[38] == approx(18.301, rel=1e-3)

        area = self.df_buildings.geometry.area
        sw = sw_high(k=3, gdf=self.df_tessellation, ids="uID")
        weighted = mm.WeightedCharacter(
            self.df_buildings, self.df_buildings.height, sw, "uID", area
        ).series
        assert weighted[38] == approx(18.301, rel=1e-3)

        sw_drop = sw_high(k=3, gdf=self.df_tessellation[2:], ids="uID")
        assert (
            mm.WeightedCharacter(self.df_buildings, "height", sw_drop, "uID")
            .series.isna()
            .any()
        )

    def test_CoveredArea(self):
        sw = sw_high(gdf=self.df_tessellation, k=1, ids="uID")
        covered_sw = mm.CoveredArea(self.df_tessellation, sw, "uID").series
        assert covered_sw[0] == approx(24115.667, rel=1e-3)
        sw_drop = sw_high(k=3, gdf=self.df_tessellation[2:], ids="uID")
        assert mm.CoveredArea(self.df_tessellation, sw_drop, "uID").series.isna().any()

    def test_PerimeterWall(self):
        sw = sw_high(gdf=self.df_buildings, k=1)
        wall = mm.PerimeterWall(self.df_buildings).series
        wall_sw = mm.PerimeterWall(self.df_buildings, sw).series
        assert wall[0] == wall_sw[0]
        assert wall[0] == 137.2106961418436

    def test_SegmentsLength(self):
        absol = mm.SegmentsLength(self.df_streets).series
        mean = mm.SegmentsLength(self.df_streets, mean=True).series
        assert max(absol) == 1907.502238338006
        assert max(mean) == 249.5698434867373
