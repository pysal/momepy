import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, Polygon

import momepy as mm
from momepy import sw_high
from momepy.shape import _make_circle


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

        with pytest.raises(KeyError, match="nonexistent"):
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

        with pytest.raises(KeyError, match="nonexistent"):
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

        with pytest.raises(KeyError, match="nonexistent"):
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
        with pytest.raises(ValueError, match="nonexistent is not supported as mode."):
            self.df_tessellation["mesh_ar"] = mm.AverageCharacter(
                self.df_tessellation,
                values="area",
                spatial_weights=spatial_weights,
                unique_id="uID",
                mode="nonexistent",
            )
        with pytest.raises(ValueError, match="nonexistent is not supported as mode."):
            self.df_tessellation["mesh_ar"] = mm.AverageCharacter(
                self.df_tessellation,
                values="area",
                spatial_weights=spatial_weights,
                unique_id="uID",
                mode=["nonexistent", "mean"],
            )
        assert self.df_tessellation["mesh_ar"][0] == pytest.approx(249.503, rel=1e-3)
        assert self.df_tessellation["mesh_array"][0] == pytest.approx(
            2623.996, rel=1e-3
        )
        assert self.df_tessellation["mesh_id"][38] == pytest.approx(2250.224, rel=1e-3)
        assert self.df_tessellation["mesh_iq"][38] == pytest.approx(2118.609, rel=1e-3)
        assert all_m.mean[0] == pytest.approx(2922.957, rel=1e-3)
        assert all_m.median[0] == pytest.approx(2623.996, rel=1e-3)
        assert all_m.mode[0] == pytest.approx(249.503, rel=1e-3)
        assert all_m.series[0] == pytest.approx(2922.957, rel=1e-3)
        assert two.mean[0] == pytest.approx(2922.957, rel=1e-3)
        assert two.median[0] == pytest.approx(2623.996, rel=1e-3)
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
        assert results.w[0] == 47.9039130128257
        assert results.wd[0] == 0.026104885468705645
        assert results.h[0] == 15.26806526806527
        assert results.p[0] == 0.31872271611668607
        assert results.o[0] == 0.9423076923076923
        assert results.hd[0] == 9.124556701878003

        height = np.linspace(10.0, 30.0, 144)
        results2 = mm.StreetProfile(
            self.df_streets, self.df_buildings, heights=height, tick_length=100
        )
        assert results2.w[0] == 70.7214870365335
        assert results2.wd[0] == 8.50508193935929
        assert results2.h[0] == pytest.approx(23.87158296249206)
        assert results2.p[0] == pytest.approx(0.3375435664999579)
        assert results2.o[0] == 0.5769230769230769
        assert results2.hd[0] == pytest.approx(5.9307227575674)

        results3 = mm.StreetProfile(self.df_streets, self.df_buildings)
        assert results3.w[0] == 47.9039130128257
        assert results3.wd[0] == 0.026104885468705645
        assert results3.o[0] == 0.9423076923076923

        # avoid infinity
        blg = gpd.GeoDataFrame(
            dict(height=[2, 5]),
            geometry=[
                Point(0, 0).buffer(10, cap_style=3),
                Point(30, 0).buffer(10, cap_style=3),
            ],
        )
        lines = gpd.GeoDataFrame(
            geometry=[LineString([(-8, -8), (8, 8)]), LineString([(15, -10), (15, 10)])]
        )
        assert mm.StreetProfile(lines, blg, "height", 2).p.equals(
            pd.Series([np.nan, 0.35])
        )

    def test_WeightedCharacter(self):
        sw = sw_high(k=3, gdf=self.df_tessellation, ids="uID")
        weighted = mm.WeightedCharacter(self.df_buildings, "height", sw, "uID").series
        assert weighted[38] == pytest.approx(18.301, rel=1e-3)

        self.df_buildings["area"] = self.df_buildings.geometry.area
        sw = sw_high(k=3, gdf=self.df_tessellation, ids="uID")
        weighted = mm.WeightedCharacter(
            self.df_buildings, "height", sw, "uID", "area"
        ).series
        assert weighted[38] == pytest.approx(18.301, rel=1e-3)

        area = self.df_buildings.geometry.area
        sw = sw_high(k=3, gdf=self.df_tessellation, ids="uID")
        weighted = mm.WeightedCharacter(
            self.df_buildings, self.df_buildings.height, sw, "uID", area
        ).series
        assert weighted[38] == pytest.approx(18.301, rel=1e-3)

        sw_drop = sw_high(k=3, gdf=self.df_tessellation[2:], ids="uID")
        assert (
            mm.WeightedCharacter(self.df_buildings, "height", sw_drop, "uID")
            .series.isna()
            .any()
        )

    def test_CoveredArea(self):
        sw = sw_high(gdf=self.df_tessellation, k=1, ids="uID")
        covered_sw = mm.CoveredArea(self.df_tessellation, sw, "uID").series
        assert covered_sw[0] == pytest.approx(24115.667, rel=1e-3)
        sw_drop = sw_high(k=3, gdf=self.df_tessellation[2:], ids="uID")
        assert mm.CoveredArea(self.df_tessellation, sw_drop, "uID").series.isna().any()

    def test_PerimeterWall(self):
        sw = sw_high(gdf=self.df_buildings, k=1)
        wall = mm.PerimeterWall(self.df_buildings).series
        wall_sw = mm.PerimeterWall(self.df_buildings, sw).series
        assert wall[0] == wall_sw[0]
        assert wall[0] == pytest.approx(137.210, rel=1e-3)

    def test_SegmentsLength(self):
        absol = mm.SegmentsLength(self.df_streets).sum
        mean = mm.SegmentsLength(self.df_streets, mean=True).mean
        assert max(absol) == pytest.approx(1907.502238338006)
        assert max(mean) == pytest.approx(249.5698434867373)
