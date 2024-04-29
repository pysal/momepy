import math

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import MultiLineString, MultiPolygon, Point, Polygon

import momepy as mm
from momepy.shape import _circle_area


class TestShape:
    def setup_method(self):
        test_file_path = mm.datasets.get_path("bubenec")
        self.df_buildings = gpd.read_file(test_file_path, layer="buildings")
        self.df_streets = gpd.read_file(test_file_path, layer="streets")
        self.df_tessellation = gpd.read_file(test_file_path, layer="tessellation")
        self.df_buildings["height"] = np.linspace(10.0, 30.0, 144)
        self.df_buildings["volume"] = mm.Volume(self.df_buildings, "height").series

    def test_FormFactor(self):
        self.df_buildings["ff"] = mm.FormFactor(
            self.df_buildings, "volume", heights="height"
        ).series
        check = 5.4486362624193
        assert self.df_buildings["ff"].mean() == pytest.approx(check)

        self.df_buildings["ff"] = mm.FormFactor(
            self.df_buildings,
            mm.Volume(self.df_buildings, "height").series,
            areas=self.df_buildings.geometry.area,
            heights=self.df_buildings["height"],
        ).series
        assert self.df_buildings["ff"].mean() == pytest.approx(check)

    def test_FractalDimension(self):
        self.df_buildings["fd"] = mm.FractalDimension(self.df_buildings).series
        check = (
            2
            * math.log(self.df_buildings.geometry[0].length / 4)
            / math.log(self.df_buildings.geometry[0].area)
        )
        assert self.df_buildings["fd"][0] == check

        self.df_buildings["fd2"] = mm.FractalDimension(
            self.df_buildings,
            areas=self.df_buildings.geometry.area,
            perimeters=self.df_buildings.geometry.length,
        ).series
        assert self.df_buildings["fd2"][0] == check

    def test_VolumeFacadeRatio(self):
        self.df_buildings["peri"] = self.df_buildings.geometry.length
        self.df_buildings["vfr"] = mm.VolumeFacadeRatio(
            self.df_buildings, "height", "volume", "peri"
        ).series
        check = self.df_buildings.volume[0] / (
            self.df_buildings.peri[0] * self.df_buildings.height[0]
        )
        assert self.df_buildings["vfr"][0] == check

        peri = self.df_buildings.geometry.length
        volume = mm.Volume(self.df_buildings, "height").series
        self.df_buildings["vfr2"] = mm.VolumeFacadeRatio(
            self.df_buildings, "height", volume, peri
        ).series
        assert self.df_buildings["vfr2"][0] == check

        self.df_buildings["peri"] = self.df_buildings.geometry.length
        self.df_buildings["vfr3"] = mm.VolumeFacadeRatio(
            self.df_buildings, "height"
        ).series

        assert self.df_buildings["vfr3"][0] == check

    def test_CircularCompactness(self):
        self.df_buildings["area"] = self.df_buildings.geometry.area
        self.df_buildings["circom"] = mm.CircularCompactness(
            self.df_buildings, "area"
        ).series
        check = self.df_buildings.area[0] / (
            _circle_area(
                list(self.df_buildings.geometry[0].convex_hull.exterior.coords)
            )
        )
        assert self.df_buildings["circom"][0] == check

        area = self.df_buildings.geometry.area
        self.df_buildings["circom2"] = mm.CircularCompactness(
            self.df_buildings, area
        ).series
        assert self.df_buildings["circom2"][0] == check

        self.df_buildings["circom3"] = mm.CircularCompactness(self.df_buildings).series
        assert self.df_buildings["circom3"][0] == check

    def test_SquareCompactness(self):
        self.df_buildings["sqcom"] = mm.SquareCompactness(self.df_buildings).series
        check = (
            (4 * math.sqrt(self.df_buildings.geometry.area[0]))
            / (self.df_buildings.geometry.length[0])
        ) ** 2
        assert self.df_buildings["sqcom"][0] == check

        self.df_buildings["sqcom2"] = mm.SquareCompactness(
            self.df_buildings,
            areas=self.df_buildings.geometry.area,
            perimeters=self.df_buildings.geometry.length,
        ).series

        assert self.df_buildings["sqcom2"][0] == check

    def test_Convexity(self):
        self.df_buildings["conv"] = mm.Convexity(self.df_buildings).series
        check = (
            self.df_buildings.geometry.area[0]
            / self.df_buildings.geometry.convex_hull.area[0]
        )
        assert self.df_buildings["conv"][0] == check

        self.df_buildings["conv2"] = mm.Convexity(
            self.df_buildings, areas=self.df_buildings.geometry.area
        ).series

        assert self.df_buildings["conv2"][0] == check

    def test_CourtyardIndex(self):
        cas = self.df_buildings["cas"] = mm.CourtyardArea(self.df_buildings).series
        self.df_buildings["cix"] = mm.CourtyardIndex(self.df_buildings, "cas").series
        self.df_buildings["cix_array"] = mm.CourtyardIndex(
            self.df_buildings, cas, self.df_buildings.geometry.area
        ).series
        check = self.df_buildings.cas[80] / self.df_buildings.geometry.area[80]
        assert self.df_buildings["cix"][80] == check
        assert self.df_buildings["cix_array"][80] == check

    def test_Rectangularity(self):
        self.df_buildings["rect"] = mm.Rectangularity(self.df_buildings).series
        self.df_buildings["rect_array"] = mm.Rectangularity(
            self.df_buildings, self.df_buildings.geometry.area
        ).series
        check = (
            self.df_buildings.geometry[0].area
            / self.df_buildings.geometry[0].minimum_rotated_rectangle.area
        )
        assert self.df_buildings["rect"][0] == check
        assert self.df_buildings["rect_array"][0] == check

    def test_ShapeIndex(self):
        la = self.df_buildings["la"] = mm.LongestAxisLength(self.df_buildings).series
        self.df_buildings["shape_index"] = mm.ShapeIndex(self.df_buildings, "la").series
        self.df_buildings["shape_index_array"] = mm.ShapeIndex(
            self.df_buildings, la, self.df_buildings.geometry.area
        ).series
        check = math.sqrt(self.df_buildings.area[0] / math.pi) / (
            0.5 * self.df_buildings.la[0]
        )
        assert self.df_buildings["shape_index"][0] == check
        assert self.df_buildings["shape_index_array"][0] == check

    def test_Corners(self):
        self.df_buildings["corners"] = mm.Corners(self.df_buildings).series
        check = 24
        assert self.df_buildings["corners"][0] == check

    def test_Squareness(self):
        self.df_buildings["squ"] = mm.Squareness(self.df_buildings).series
        check = pytest.approx(3.707, rel=1e-3)
        assert self.df_buildings["squ"][0] == check
        self.df_buildings["squ"] = mm.Squareness(self.df_buildings.exterior).series
        assert self.df_buildings["squ"].isna().all()
        df_buildings_multi = self.df_buildings.copy()
        df_buildings_multi["geometry"] = df_buildings_multi["geometry"].apply(
            lambda geom: MultiPolygon([geom])
        )
        self.df_buildings["squm"] = mm.Squareness(df_buildings_multi).series
        assert self.df_buildings["squm"][0] == check

    def test_EquivalentRectangularIndex(self):
        self.df_buildings["eri"] = mm.EquivalentRectangularIndex(
            self.df_buildings
        ).series
        self.df_buildings["eri_array"] = mm.EquivalentRectangularIndex(
            self.df_buildings,
            areas=self.df_buildings.geometry.area,
            perimeters=self.df_buildings.geometry.length,
        ).series
        check = pytest.approx(0.7879, rel=1e-3)
        assert self.df_buildings["eri"][0] == check
        assert self.df_buildings["eri_array"][0] == check

    def test_Elongation(self):
        self.df_buildings["elo"] = mm.Elongation(self.df_buildings).series
        check = pytest.approx(0.908, rel=1e-3)
        assert self.df_buildings["elo"][0] == check

    def test_CentroidCorners(self):
        self.df_buildings.loc[144] = [145, Point(0, 0).buffer(10), 0, 0]
        self.df_buildings.loc[145] = [
            145,
            Polygon([s + (0,) for s in Point(0, 0).buffer(10).exterior.coords]),
            0,
            0,
        ]
        check = pytest.approx(15.961, rel=1e-3)
        check_devs = pytest.approx(3.081, rel=1e-3)
        cc = mm.CentroidCorners(self.df_buildings)
        self.df_buildings["ccd"] = cc.mean
        self.df_buildings["ccddev"] = cc.std
        assert self.df_buildings["ccd"][0] == check
        assert self.df_buildings["ccddev"][0] == check_devs
        df_buildings_multi = self.df_buildings.copy()
        df_buildings_multi["geometry"] = df_buildings_multi["geometry"].apply(
            lambda geom: MultiPolygon([geom])
        )
        cc = mm.CentroidCorners(df_buildings_multi)
        df_buildings_multi["ccd"] = cc.mean
        df_buildings_multi["ccddev"] = cc.std
        assert df_buildings_multi["ccd"][0] == check
        assert df_buildings_multi["ccddev"][0] == check_devs

    def test_Linearity(self):
        self.df_streets["lin"] = mm.Linearity(self.df_streets).series
        euclidean = Point(self.df_streets.geometry[0].coords[0]).distance(
            Point(self.df_streets.geometry[0].coords[-1])
        )
        check = euclidean / self.df_streets.geometry[0].length
        assert self.df_streets["lin"][0] == pytest.approx(check, rel=1e-6)

        self.df_streets.loc[len(self.df_streets)] = MultiLineString(
            [[(0, 0), (-1, 1)], [(10, 10), (11, 11)]]
        )

    def test_CompactnessWeightedAxis(self):
        self.df_buildings["cwa"] = mm.CompactnessWeightedAxis(self.df_buildings).series
        self.df_buildings["cwa_array"] = mm.CompactnessWeightedAxis(
            self.df_buildings,
            areas=self.df_buildings.geometry.area,
            perimeters=self.df_buildings.geometry.length,
            longest_axis=mm.LongestAxisLength(self.df_buildings).series,
        ).series
        check = pytest.approx(26.327, rel=1e-3)
        assert self.df_buildings["cwa"][0] == check
        assert self.df_buildings["cwa_array"][0] == check

    def test__circle_area(self):
        poly = Polygon([(0, 1, 0), (1, 1, 0), (2, 4, 0)])
        check = _circle_area(poly.exterior.coords)
        assert check == pytest.approx(10.210, rel=1e-3)
