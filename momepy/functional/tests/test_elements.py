import geopandas as gpd
import libpysal
import numpy as np
import pandas as pd
import pytest
from geopandas.testing import assert_geodataframe_equal
from packaging.version import Version
from pandas.testing import assert_index_equal, assert_series_equal
from shapely import affinity
from shapely.geometry import MultiPoint, Polygon, box

import momepy as mm

GPD_GE_013 = Version(gpd.__version__) >= Version("0.13.0")
LPS_GE_411 = Version(libpysal.__version__) >= Version("4.11.dev")


class TestElements:
    def setup_method(self):
        test_file_path = mm.datasets.get_path("bubenec")
        self.df_buildings = gpd.read_file(test_file_path, layer="buildings")
        self.df_tessellation = gpd.read_file(test_file_path, layer="tessellation")
        self.df_streets = gpd.read_file(test_file_path, layer="streets")
        self.limit = mm.buffered_limit(self.df_buildings, 50)
        self.enclosures = mm.enclosures(
            self.df_streets,
            gpd.GeoSeries([self.limit.exterior], crs=self.df_streets.crs),
        )

    def test_morphological_tessellation(self):
        tessellation = mm.morphological_tessellation(
            self.df_buildings,
        )
        assert (tessellation.geom_type == "Polygon").all()
        assert tessellation.crs == self.df_buildings.crs
        assert_index_equal(tessellation.index, self.df_buildings.index)
        assert isinstance(tessellation, gpd.GeoDataFrame)

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
                    crs=df.crs,
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
        if GPD_GE_013:
            assert_geodataframe_equal(
                tessellation.sort_values("geometry").reset_index(drop=True),
                threshold_elimination.sort_values("geometry").reset_index(drop=True),
            )

        tessellation_df = mm.enclosed_tessellation(
            self.df_buildings,
            self.enclosures,
        )
        assert_geodataframe_equal(tessellation, tessellation_df)

        custom_index = self.enclosures
        custom_index.index = (custom_index.index + 100).astype(str)
        tessellation_custom_index = mm.enclosed_tessellation(
            self.df_buildings,
            custom_index,
        )
        assert (tessellation_custom_index.geom_type == "Polygon").all()
        assert tessellation_custom_index.crs == self.df_buildings.crs
        assert (self.df_buildings.index.isin(tessellation_custom_index.index)).all()
        assert tessellation_custom_index.enclosure_index.isin(custom_index.index).all()

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
                    crs=df.crs,
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
        assert pd.isna(nearest).sum() == 137  # noqa: E711

    def test_get_nearest_node(self):
        nodes, edges = mm.nx_to_gdf(mm.gdf_to_nx(self.df_streets))
        edge_index = mm.get_nearest_street(self.df_buildings, edges)

        node_index = mm.get_nearest_node(self.df_buildings, nodes, edges, edge_index)

        assert len(node_index) == len(self.df_buildings)
        assert_index_equal(node_index.index, self.df_buildings.index)
        expected = np.array(
            [
                0.0,
                1.0,
                2.0,
                3.0,
                4.0,
                6.0,
                9.0,
                11.0,
                14.0,
                15.0,
                16.0,
                20.0,
                22.0,
                25.0,
            ]
        )
        expected_counts = np.array([9, 31, 12, 10, 11, 2, 23, 8, 2, 8, 3, 6, 12, 7])
        unique, counts = np.unique(node_index, return_counts=True)
        np.testing.assert_array_equal(unique, expected)
        np.testing.assert_array_equal(counts, expected_counts)

    def test_get_nearest_node_missing(self):
        nodes, edges = mm.nx_to_gdf(mm.gdf_to_nx(self.df_streets))
        edge_index = mm.get_nearest_street(self.df_buildings, edges, max_distance=20)

        node_index = mm.get_nearest_node(self.df_buildings, nodes, edges, edge_index)

        assert len(node_index) == len(self.df_buildings)
        assert_index_equal(node_index.index, self.df_buildings.index)
        expected = np.array(
            [1.0, 2.0, 3.0, 4.0, 9.0, 11.0, 14.0, 15.0, 16.0, 20.0, 22.0, 25.0, np.nan]
        )
        expected_counts = np.array([14, 8, 10, 4, 14, 8, 2, 7, 2, 5, 9, 4, 57])
        unique, counts = np.unique(node_index, return_counts=True)
        np.testing.assert_array_equal(unique, expected)
        np.testing.assert_array_equal(counts, expected_counts)

    def test_buffered_limit(self):
        limit = mm.buffered_limit(self.df_buildings, 50)
        assert limit.geom_type == "Polygon"
        assert pytest.approx(limit.area) == 366525.967849688

    @pytest.mark.skipif(not LPS_GE_411, reason="libpysal>=4.11 required")
    def test_buffered_limit_adaptive(self):
        limit = mm.buffered_limit(self.df_buildings, "adaptive")
        assert limit.geom_type == "Polygon"
        assert pytest.approx(limit.area) == 355819.18954170

        limit = mm.buffered_limit(self.df_buildings, "adaptive", max_buffer=30)
        assert limit.geom_type == "Polygon"
        assert pytest.approx(limit.area) == 304200.301833294

        limit = mm.buffered_limit(
            self.df_buildings, "adaptive", min_buffer=30, max_buffer=300
        )
        assert limit.geom_type == "Polygon"
        assert pytest.approx(limit.area) == 357671.831894244

    @pytest.mark.skipif(LPS_GE_411, reason="libpysal>=4.11 required")
    def test_buffered_limit_adaptive_error(self):
        with pytest.raises(
            ImportError, match="Adaptive buffer requires libpysal 4.11 or higher."
        ):
            mm.buffered_limit(self.df_buildings, "adaptive")

    def test_buffered_limit_error(self):
        with pytest.raises(
            ValueError, match="`buffer` must be either 'adaptive' or a number."
        ):
            mm.buffered_limit(self.df_buildings, "invalid")

    def test_blocks(self):
        blocks, tessellation_id = mm.generate_blocks(
            self.df_tessellation, self.df_streets, self.df_buildings
        )
        assert not tessellation_id.isna().any()
        assert len(blocks) == 8

    def test_blocks_inner(self):
        streets = self.df_streets.copy()
        streets.loc[35, "geometry"] = (
            self.df_buildings.geometry.iloc[141]
            .representative_point()
            .buffer(20)
            .exterior
        )
        blocks, tessellation_id = mm.generate_blocks(
            self.df_tessellation, streets, self.df_buildings
        )
        assert not tessellation_id.isna().any()
        assert len(blocks) == 9
        if GPD_GE_013:
            assert len(blocks.sindex.query(blocks.geometry, "overlaps")[0]) == 0
        else:
            assert len(blocks.sindex.query_bulk(blocks.geometry, "overlaps")[0]) == 0

    def test_multi_index(self):
        buildings = self.df_buildings.set_index(["uID", "uID"])
        with pytest.raises(
            ValueError,
            match="MultiIndex is not supported in `momepy.morphological_tessellation`.",
        ):
            mm.morphological_tessellation(buildings)
        with pytest.raises(
            ValueError,
            match="MultiIndex is not supported in `momepy.enclosed_tessellation`.",
        ):
            mm.enclosed_tessellation(buildings, self.enclosures)
        with pytest.raises(
            ValueError,
            match="MultiIndex is not supported in `momepy.verify_tessellation`.",
        ):
            mm.verify_tessellation(buildings, self.enclosures)

        with pytest.raises(
            ValueError,
            match="MultiIndex is not supported in `momepy.get_nearest_node`.",
        ):
            mm.get_nearest_node(
                buildings, self.enclosures, self.enclosures, self.enclosures
            )

        with pytest.raises(
            ValueError, match="MultiIndex is not supported in `momepy.generate_blocks`"
        ):
            mm.generate_blocks(buildings, self.enclosures, self.enclosures)

    def test_tess_single_building_edge_case(self):
        tessellations = mm.enclosed_tessellation(
            self.df_buildings, self.enclosures.geometry, n_jobs=-1
        )
        orig_grouper = tessellations.groupby("enclosure_index")
        idxs = ~self.df_buildings.index.isin(orig_grouper.get_group(8).index)
        idxs[1] = True
        idxs[21] = False
        idxs[23] = False

        new_blg = self.df_buildings[idxs]
        new_blg.loc[22, "geometry"] = new_blg.loc[22, "geometry"].buffer(20)
        new_tess = mm.enclosed_tessellation(new_blg, self.enclosures.geometry, n_jobs=1)

        # assert that buildings 1 and 22 intersect the same enclosure
        inp, res = self.enclosures.sindex.query(
            new_blg.geometry, predicate="intersects"
        )
        assert np.isclose(new_blg.iloc[inp[res == 8]].index.values, [1, 22]).all()

        # assert that there is a tessellation for building 1
        assert 1 in new_tess.index


class TestElementsEquivalence:
    def setup_method(self):
        test_file_path = mm.datasets.get_path("bubenec")
        self.df_buildings = gpd.read_file(test_file_path, layer="buildings")
        self.df_tessellation = gpd.read_file(test_file_path, layer="tessellation")
        self.df_streets = gpd.read_file(test_file_path, layer="streets")
        self.limit = mm.buffered_limit(self.df_buildings, 50)
        self.enclosures = mm.enclosures(
            self.df_streets,
            gpd.GeoSeries([self.limit.exterior], crs=self.df_streets.crs),
        )

    def test_blocks(self):
        blocks, tessellation_id = mm.generate_blocks(
            self.df_tessellation, self.df_streets, self.df_buildings
        )
        res = mm.Blocks(
            self.df_tessellation, self.df_streets, self.df_buildings, "bID", "uID"
        )

        assert_geodataframe_equal(
            blocks.geometry.to_frame(), res.blocks.geometry.to_frame()
        )
        assert_series_equal(
            tessellation_id[tessellation_id.index >= 0], res.buildings_id
        )
        assert_series_equal(tessellation_id, res.tessellation_id)
