import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from geopandas.testing import assert_geodataframe_equal
from packaging.version import Version
from pandas.testing import assert_index_equal
from shapely import affinity
from shapely.geometry import MultiPoint, Polygon, box

import momepy as mm

GPD_GE_013 = Version(gpd.__version__) >= Version("0.13.0")


class TestElements:
    def setup_method(self):
        test_file_path = mm.datasets.get_path("bubenec")
        self.df_buildings = gpd.read_file(test_file_path, layer="buildings")
        self.df_tessellation = gpd.read_file(test_file_path, layer="tessellation")
        self.df_streets = gpd.read_file(test_file_path, layer="streets")
        self.limit = mm.buffered_limit(self.df_buildings, 50)
        self.enclosures = mm.enclosures(
            self.df_streets, gpd.GeoSeries([self.limit.exterior])
        )

    def test_morphological_tessellation(self):
        tessellation = mm.morphological_tessellation(
            self.df_buildings,
        )
        assert (tessellation.geom_type == "Polygon").all()
        assert tessellation.crs == self.df_buildings.crs
        assert_index_equal(tessellation.index, self.df_buildings.index)

        clipped = mm.morphological_tessellation(
            self.df_buildings,
            clip=self.limit,
        )

        assert (tessellation.geom_type == "Polygon").all()
        assert tessellation.crs == self.df_buildings.crs
        assert_index_equal(tessellation.index, self.df_buildings.index)
        assert clipped.area.sum() < tessellation.area.sum()

        sparser = mm.morphological_tessellation(
            self.df_buildings,
            segment=2,
        )
        if GPD_GE_013:
            assert (
                sparser.get_coordinates().shape[0]
                < tessellation.get_coordinates().shape[0]
            )

    def test_morphological_tessellation_buffer_clip(self):
        tessellation = mm.morphological_tessellation(
            self.df_buildings, clip=self.df_buildings.buffer(50)
        )
        assert (tessellation.geom_type == "Polygon").all()
        assert tessellation.crs == self.df_buildings.crs
        assert_index_equal(tessellation.index, self.df_buildings.index)

    def test_morphological_tessellation_errors(self):
        df = self.df_buildings
        b = df.total_bounds
        x = np.mean([b[0], b[2]])
        y = np.mean([b[1], b[3]])

        df = pd.concat(
            [
                df,
                gpd.GeoDataFrame(
                    {"uID": [145, 146, 147]},
                    geometry=[
                        Polygon([(x, y), (x, y + 1), (x + 1, y)]),
                        MultiPoint([(x, y), (x + 1, y)]).buffer(0.55),
                        affinity.rotate(df.geometry.iloc[0], 12),
                    ],
                    index=[144, 145, 146],
                ),
            ]
        )
        tessellation = mm.morphological_tessellation(
            df,
        )
        assert (tessellation.geom_type == "Polygon").all()
        assert 144 not in tessellation.index
        assert len(tessellation) == len(df) - 1

    def test_enclosed_tessellation(self):
        tessellation = mm.enclosed_tessellation(
            self.df_buildings,
            self.enclosures.geometry,
        )
        assert (tessellation.geom_type == "Polygon").all()
        assert tessellation.crs == self.df_buildings.crs
        assert (self.df_buildings.index.isin(tessellation.index)).all()
        assert np.isin(np.array(range(-11, 0, 1)), tessellation.index).all()

        sparser = mm.enclosed_tessellation(
            self.df_buildings,
            self.enclosures.geometry,
            segment=2,
        )
        if GPD_GE_013:
            assert (
                sparser.get_coordinates().shape[0]
                < tessellation.get_coordinates().shape[0]
            )

        no_threshold_check = mm.enclosed_tessellation(
            self.df_buildings, self.enclosures.geometry, threshold=None, n_jobs=1
        )

        assert_geodataframe_equal(tessellation, no_threshold_check)

        buildings = pd.concat(
            [
                self.df_buildings,
                gpd.GeoDataFrame(
                    {"uID": [145, 146]},
                    geometry=[
                        box(1603283, 6464150, 1603316, 6464234),
                        box(1603293, 6464150, 1603316, 6464244),
                    ],
                    crs=self.df_buildings.crs,
                    index=[144, 145],
                ),
            ]
        )

        threshold_elimination = mm.enclosed_tessellation(
            buildings, self.enclosures.geometry, threshold=0.99, n_jobs=1
        )
        assert not threshold_elimination.index.duplicated().any()
        assert_index_equal(threshold_elimination.index, tessellation.index)
        assert_geodataframe_equal(
            tessellation.sort_values("geometry").reset_index(drop=True),
            threshold_elimination.sort_values("geometry").reset_index(drop=True),
        )

    def test_verify_tessellation(self):
        df = self.df_buildings
        b = df.total_bounds
        x = np.mean([b[0], b[2]])
        y = np.mean([b[1], b[3]])

        df = pd.concat(
            [
                df,
                gpd.GeoDataFrame(
                    {"uID": [145]},
                    geometry=[
                        Polygon([(x, y), (x, y + 1), (x + 1, y)]),
                    ],
                    index=[144],
                ),
            ]
        )
        tessellation = mm.morphological_tessellation(
            df, clip=self.df_streets.buffer(50)
        )
        with (
            pytest.warns(
                UserWarning, match="Tessellation does not fully match buildings"
            ),
            pytest.warns(
                UserWarning, match="Tessellation contains MultiPolygon elements"
            ),
        ):
            collapsed, multi = mm.verify_tessellation(tessellation, df)
        assert_index_equal(collapsed, pd.Index([144]))
        assert_index_equal(
            multi, pd.Index([1, 46, 57, 62, 103, 105, 129, 130, 134, 136, 137])
        )

    def test_get_nearest_street(self):
        streets = self.df_streets.copy()
        nearest = mm.get_nearest_street(self.df_buildings, streets)
        assert len(nearest) == len(self.df_buildings)
        expected = np.array(
            [0, 1, 2, 5, 6, 8, 10, 11, 12, 14, 16, 19, 21, 24, 25, 26, 28, 32, 33, 34]
        )
        expected_counts = np.array(
            [9, 1, 12, 5, 7, 15, 1, 3, 4, 1, 3, 9, 9, 6, 5, 5, 15, 6, 10, 18]
        )
        unique, counts = np.unique(nearest, return_counts=True)
        np.testing.assert_array_equal(unique, expected)
        np.testing.assert_array_equal(counts, expected_counts)

        # induce missing
        nearest = mm.get_nearest_street(self.df_buildings, streets, 10)
        expected = np.array([2.0, 34.0, np.nan])
        expected_counts = np.array([3, 4, 137])
        unique, counts = np.unique(nearest, return_counts=True)
        np.testing.assert_array_equal(unique, expected)
        np.testing.assert_array_equal(counts, expected_counts)

        streets.index = streets.index.astype(str)
        nearest = mm.get_nearest_street(self.df_buildings, streets, 10)
        assert (nearest == None).sum() == 137  # noqa: E711
