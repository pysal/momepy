import momepy as mm
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import math

from momepy.shape import _circle_area


class TestShape:
    def setup_method(self):

        test_file_path = mm.datasets.get_path("bubenec")
        self.df_buildings = gpd.read_file(test_file_path, layer="buildings")
        self.df_streets = gpd.read_file(test_file_path, layer="streets")
        self.df_tessellation = gpd.read_file(test_file_path, layer="tessellation")
        self.df_buildings["height"] = np.linspace(10.0, 30.0, 144)
        self.df_buildings["volume"] = mm.volume(self.df_buildings, "height")

    def test_form_factor(self):
        self.df_buildings["ff"] = mm.form_factor(self.df_buildings, "volume")
        check = (self.df_buildings.geometry[0].area) / (
            self.df_buildings.volume[0] ** (2 / 3)
        )
        assert self.df_buildings["ff"][0] == check

    def test_form_factor_array(self):
        self.df_buildings["ff"] = mm.form_factor(
            self.df_buildings,
            mm.volume(self.df_buildings, "height"),
            areas=self.df_buildings.geometry.area,
        )
        check = (self.df_buildings.geometry[0].area) / (
            self.df_buildings.volume[0] ** (2 / 3)
        )
        assert self.df_buildings["ff"][0] == check

    def test_fractal_dimension(self):
        self.df_buildings["fd"] = mm.fractal_dimension(self.df_buildings)
        check = math.log(self.df_buildings.geometry[0].length / 4) / math.log(
            self.df_buildings.geometry[0].area
        )
        assert self.df_buildings["fd"][0] == check

    def test_fractal_dimension_array(self):
        self.df_buildings["fd"] = mm.fractal_dimension(
            self.df_buildings,
            areas=self.df_buildings.geometry.area,
            perimeters=self.df_buildings.geometry.length,
        )
        check = math.log(self.df_buildings.geometry[0].length / 4) / math.log(
            self.df_buildings.geometry[0].area
        )
        assert self.df_buildings["fd"][0] == check

    def test_volume_facade_ratio(self):
        self.df_buildings["peri"] = self.df_buildings.geometry.length
        self.df_buildings["vfr"] = mm.volume_facade_ratio(
            self.df_buildings, "height", "volume", "peri"
        )
        check = self.df_buildings.volume[0] / (
            self.df_buildings.peri[0] * self.df_buildings.height[0]
        )
        assert self.df_buildings["vfr"][0] == check

    def test_volume_facade_ratio_array(self):
        peri = self.df_buildings.geometry.length
        volume = mm.volume(self.df_buildings, "height")
        self.df_buildings["vfr"] = mm.volume_facade_ratio(
            self.df_buildings, "height", volume, peri
        )
        check = self.df_buildings.volume[0] / (
            self.df_buildings.geometry[0].length * self.df_buildings.height[0]
        )
        assert self.df_buildings["vfr"][0] == check

    def test_volume_facade_ratio_nones(self):
        self.df_buildings["peri"] = self.df_buildings.geometry.length
        self.df_buildings["vfr"] = mm.volume_facade_ratio(self.df_buildings, "height")
        check = self.df_buildings.volume[0] / (
            self.df_buildings.peri[0] * self.df_buildings.height[0]
        )
        assert self.df_buildings["vfr"][0] == check

    def test_circular_compactness(self):
        self.df_buildings["area"] = self.df_buildings.geometry.area
        self.df_buildings["circom"] = mm.circular_compactness(self.df_buildings, "area")
        check = self.df_buildings.area[0] / (
            _circle_area(
                list(self.df_buildings.geometry[0].convex_hull.exterior.coords)
            )
        )
        assert self.df_buildings["circom"][0] == check

    def test_circular_compactness_array(self):
        area = self.df_buildings.geometry.area
        self.df_buildings["circom"] = mm.circular_compactness(self.df_buildings, area)
        check = self.df_buildings.area[0] / (
            _circle_area(
                list(self.df_buildings.geometry[0].convex_hull.exterior.coords)
            )
        )
        assert self.df_buildings["circom"][0] == check

    def test_circular_compactness_nones(self):
        self.df_buildings["circom"] = mm.circular_compactness(self.df_buildings)
        check = self.df_buildings.area[0] / (
            _circle_area(
                list(self.df_buildings.geometry[0].convex_hull.exterior.coords)
            )
        )
        assert self.df_buildings["circom"][0] == check

    def test_square_compactness(self):
        self.df_buildings["sqcom"] = mm.square_compactness(self.df_buildings)
        check = (
            (4 * math.sqrt(self.df_buildings.geometry.area[0]))
            / (self.df_buildings.geometry.length[0])
        ) ** 2
        assert self.df_buildings["sqcom"][0] == check

    def test_square_compactness_array(self):
        self.df_buildings["sqcom"] = mm.square_compactness(
            self.df_buildings,
            areas=self.df_buildings.geometry.area,
            perimeters=self.df_buildings.geometry.length,
        )
        check = (
            (4 * math.sqrt(self.df_buildings.geometry.area[0]))
            / (self.df_buildings.geometry.length[0])
        ) ** 2
        assert self.df_buildings["sqcom"][0] == check

    def test_convexeity(self):
        self.df_buildings["conv"] = mm.convexeity(self.df_buildings)
        check = (
            self.df_buildings.geometry.area[0]
            / self.df_buildings.geometry.convex_hull.area[0]
        )
        assert self.df_buildings["conv"][0] == check

    def test_convexeity_array(self):
        self.df_buildings["conv"] = mm.convexeity(
            self.df_buildings, areas=self.df_buildings.geometry.area
        )
        check = (
            self.df_buildings.geometry.area[0]
            / self.df_buildings.geometry.convex_hull.area[0]
        )
        assert self.df_buildings["conv"][0] == check

    def test_courtyard_index(self):
        cas = self.df_buildings["cas"] = mm.courtyard_area(self.df_buildings)
        self.df_buildings["cix"] = mm.courtyard_index(self.df_buildings, "cas")
        self.df_buildings["cix_array"] = mm.courtyard_index(
            self.df_buildings, cas, self.df_buildings.geometry.area
        )
        check = self.df_buildings.cas[80] / self.df_buildings.geometry.area[80]
        assert self.df_buildings["cix"][80] == check
        assert self.df_buildings["cix_array"][80] == check

    def test_rectangularity(self):
        self.df_buildings["rect"] = mm.rectangularity(self.df_buildings)
        self.df_buildings["rect_array"] = mm.rectangularity(
            self.df_buildings, self.df_buildings.geometry.area
        )
        check = (
            self.df_buildings.geometry[0].area
            / self.df_buildings.geometry[0].minimum_rotated_rectangle.area
        )
        assert self.df_buildings["rect"][0] == check
        assert self.df_buildings["rect_array"][0] == check

    def test_shape_index(self):
        la = self.df_buildings["la"] = mm.longest_axis_length(self.df_buildings)
        self.df_buildings["shape_index"] = mm.shape_index(self.df_buildings, "la")
        self.df_buildings["shape_index_array"] = mm.shape_index(
            self.df_buildings, la, self.df_buildings.geometry.area
        )
        check = math.sqrt(self.df_buildings.area[0] / math.pi) / (
            0.5 * self.df_buildings.la[0]
        )
        assert self.df_buildings["shape_index"][0] == check
        assert self.df_buildings["shape_index_array"][0] == check

    def test_corners(self):
        self.df_buildings["corners"] = mm.corners(self.df_buildings)
        check = 24
        assert self.df_buildings["corners"][0] == check

    def test_squareness(self):
        self.df_buildings["squ"] = mm.squareness(self.df_buildings)
        check = round(3.7075816043359855, 4)
        assert round(self.df_buildings["squ"][0], 4) == check

    def test_equivalent_rectangular_index(self):
        self.df_buildings["eri"] = mm.equivalent_rectangular_index(self.df_buildings)
        self.df_buildings["eri_array"] = mm.equivalent_rectangular_index(
            self.df_buildings,
            areas=self.df_buildings.geometry.area,
            perimeters=self.df_buildings.geometry.length,
        )
        check = 0.7879229963118455
        assert self.df_buildings["eri"][0] == check
        assert self.df_buildings["eri_array"][0] == check

    def test_elongation(self):
        self.df_buildings["elo"] = mm.elongation(self.df_buildings)
        check = 0.9082437463675544
        assert self.df_buildings["elo"][0] == check

    def test_centroid_corners(self):
        means, devs = mm.centroid_corners(self.df_buildings)
        self.df_buildings["ccd"] = means
        self.df_buildings["ccddev"] = devs
        check = 15.961531913184833
        check_devs = 3.0810634305400177
        assert self.df_buildings["ccd"][0] == check
        assert self.df_buildings["ccddev"][0] == check_devs

    def test_linearity(self):
        self.df_streets["lin"] = mm.linearity(self.df_streets)
        euclidean = Point(self.df_streets.geometry[0].coords[0]).distance(
            Point(self.df_streets.geometry[0].coords[-1])
        )
        check = euclidean / self.df_streets.geometry[0].length
        assert self.df_streets["lin"][0] == check

    def test_compactness_weighted_axis(self):
        self.df_buildings["cwa"] = mm.compactness_weighted_axis(self.df_buildings)
        self.df_buildings["cwa_array"] = mm.compactness_weighted_axis(
            self.df_buildings,
            areas=self.df_buildings.geometry.area,
            perimeters=self.df_buildings.geometry.length,
            longest_axis=mm.longest_axis_length(self.df_buildings),
        )
        check = 26.32772969906327
        assert self.df_buildings["cwa"][0] == check
        assert self.df_buildings["cwa_array"][0] == check
