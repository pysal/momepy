import uuid

import geopandas as gpd
import libpysal
import numpy as np
import pandas as pd
import pytest
import shapely
from geopandas.testing import assert_geodataframe_equal
from packaging.version import Version
from pandas.testing import assert_index_equal
from shapely import LineString, affinity
from shapely.geometry import MultiPoint, Polygon, box

import momepy as mm

SHPLY_GE_210 = Version(shapely.__version__) >= Version("2.1.0")
LPS_G_4_13_0 = Version(libpysal.__version__) > Version("4.13.0")


class TestElements:
    def setup_method(self):
        test_file_path = mm.datasets.get_path("bubenec")
        self.df_buildings = gpd.read_file(test_file_path, layer="buildings")
        self.df_tessellation = gpd.read_file(test_file_path, layer="tessellation")
        self.df_streets = gpd.read_file(test_file_path, layer="streets")
        self.df_streets["nID"] = range(len(self.df_streets))
        self.limit = mm.buffered_limit(self.df_buildings, 50)
        self.enclosures = mm.enclosures(
            self.df_streets,
            gpd.GeoSeries([self.limit.exterior], crs=self.df_streets.crs),
        )

    def test_morphological_tessellation(self):
        tessellation = mm.morphological_tessellation(self.df_buildings, simplify=False)
        assert (tessellation.geom_type == "Polygon").all()
        assert tessellation.crs == self.df_buildings.crs
        assert_index_equal(tessellation.index, self.df_buildings.index)
        assert isinstance(tessellation, gpd.GeoDataFrame)

        clipped = mm.morphological_tessellation(
            self.df_buildings, clip=self.limit, simplify=False
        )

        assert (tessellation.geom_type == "Polygon").all()
        assert tessellation.crs == self.df_buildings.crs
        assert_index_equal(tessellation.index, self.df_buildings.index)
        assert clipped.area.sum() < tessellation.area.sum()

        sparser = mm.morphological_tessellation(
            self.df_buildings, segment=2, simplify=False
        )
        assert (
            sparser.get_coordinates().shape[0] < tessellation.get_coordinates().shape[0]
        )

    def test_morphological_tessellation_buffer_clip(self):
        tessellation = mm.morphological_tessellation(
            self.df_buildings, clip=self.df_buildings.buffer(50), simplify=False
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
        tessellation = mm.morphological_tessellation(df, simplify=False)
        assert (tessellation.geom_type == "Polygon").all()
        assert 144 not in tessellation.index
        assert len(tessellation) == len(df) - 1

    @pytest.mark.skipif(not SHPLY_GE_210, reason="coverage_simplify required")
    def test_morphological_tessellation_simplify(self):
        simplified = mm.morphological_tessellation(
            self.df_buildings,
        )
        assert simplified.get_coordinates().shape[0] == 4557

        dense = mm.morphological_tessellation(self.df_buildings, simplify=False)
        assert dense.get_coordinates().shape[0] == 47505

    def test_enclosed_tessellation(self):
        tessellation = mm.enclosed_tessellation(
            self.df_buildings, self.enclosures.geometry, simplify=False
        )
        assert (tessellation.geom_type == "Polygon").all()
        assert tessellation.crs == self.df_buildings.crs
        assert (self.df_buildings.index.isin(tessellation.index)).all()
        assert np.isin(np.array(range(-11, 0, 1)), tessellation.index).all()

        sparser = mm.enclosed_tessellation(
            self.df_buildings,
            self.enclosures.geometry,
            simplify=False,
            segment=2,
        )
        assert (
            sparser.get_coordinates().shape[0] < tessellation.get_coordinates().shape[0]
        )

        no_threshold_check = mm.enclosed_tessellation(
            self.df_buildings,
            self.enclosures.geometry,
            simplify=False,
            threshold=None,
            n_jobs=1,
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
            buildings,
            self.enclosures.geometry,
            simplify=False,
            threshold=0.99,
            n_jobs=1,
        )
        assert not threshold_elimination.index.duplicated().any()
        assert_index_equal(threshold_elimination.index, tessellation.index)
        assert_geodataframe_equal(
            tessellation.sort_values("geometry").reset_index(drop=True),
            threshold_elimination.sort_values("geometry").reset_index(drop=True),
        )

        tessellation_df = mm.enclosed_tessellation(
            self.df_buildings,
            self.enclosures,
            simplify=False,
        )
        assert_geodataframe_equal(tessellation, tessellation_df)

        custom_index = self.enclosures
        custom_index.index = (custom_index.index + 100).astype(str)
        tessellation_custom_index = mm.enclosed_tessellation(
            self.df_buildings,
            custom_index,
            simplify=False,
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
            df, clip=self.df_streets.buffer(50), simplify=False
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

    def test_buffered_limit_adaptive(self):
        limit = mm.buffered_limit(self.df_buildings, "adaptive")
        assert limit.geom_type == "Polygon"
        if LPS_G_4_13_0:
            exp = 347096.5835217
        else:
            exp = 355819.1895417
        assert pytest.approx(limit.area) == exp

        limit = mm.buffered_limit(self.df_buildings, "adaptive", max_buffer=30)
        assert limit.geom_type == "Polygon"
        if LPS_G_4_13_0:
            exp = 304712.451361391
        else:
            exp = 304200.301833294
        assert pytest.approx(limit.area) == exp

        limit = mm.buffered_limit(
            self.df_buildings, "adaptive", min_buffer=30, max_buffer=300
        )
        assert limit.geom_type == "Polygon"
        if LPS_G_4_13_0:
            exp = 348777.778371144
        else:
            exp = 357671.831894244
        assert pytest.approx(limit.area) == exp

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
        assert len(blocks.sindex.query(blocks.geometry, "overlaps")[0]) == 0

    @pytest.mark.skipif(not SHPLY_GE_210, reason="coverage_simplify required")
    def test_simplified_tesselations(self):
        n_workers = -1
        tessellations = mm.enclosed_tessellation(
            self.df_buildings,
            self.enclosures.geometry,
            simplify=False,
            n_jobs=n_workers,
        )
        simplified_tessellations = mm.enclosed_tessellation(
            self.df_buildings, self.enclosures.geometry, simplify=True, n_jobs=n_workers
        )
        ## empty enclosures should be unmodified
        assert_geodataframe_equal(
            tessellations[tessellations.index < 0],
            simplified_tessellations[simplified_tessellations.index < 0],
        )
        ## simplification should result in less total points
        orig_points = shapely.get_coordinates(
            tessellations[tessellations.index >= 0].geometry
        ).shape
        simpl_points = shapely.get_coordinates(
            simplified_tessellations[simplified_tessellations.index >= 0].geometry
        ).shape
        assert orig_points > simpl_points

        ## simplification should not modify the external borders of tesselation cells
        orig_grouper = tessellations.groupby("enclosure_index")
        simpl_grouper = simplified_tessellations.groupby("enclosure_index")
        for idx in np.union1d(
            tessellations["enclosure_index"].unique(),
            simplified_tessellations["enclosure_index"].unique(),
        ):
            orig_group = orig_grouper.get_group(idx).dissolve().boundary
            enclosure = self.enclosures.loc[[idx]].dissolve().boundary

            simpl_group = simpl_grouper.get_group(idx).dissolve().boundary

            ## simplified is not different to enclosure
            assert np.isclose(simpl_group.difference(enclosure).area, 0)

            # simplified is not different to original tess
            assert np.isclose(simpl_group.difference(orig_group).area, 0)

    def test_multi_index(self):
        buildings = self.df_buildings.set_index(["uID", "uID"])
        with pytest.raises(
            ValueError,
            match="MultiIndex is not supported in `momepy.morphological_tessellation`.",
        ):
            mm.morphological_tessellation(buildings, simplify=False)
        with pytest.raises(
            ValueError,
            match="MultiIndex is not supported in `momepy.enclosed_tessellation`.",
        ):
            mm.enclosed_tessellation(buildings, self.enclosures, simplify=False)
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
            self.df_buildings, self.enclosures.geometry, simplify=False, n_jobs=-1
        )
        orig_grouper = tessellations.groupby("enclosure_index")
        idxs = ~self.df_buildings.index.isin(orig_grouper.get_group(8).index)
        idxs[1] = True
        idxs[21] = False
        idxs[23] = False

        new_blg = self.df_buildings[idxs]
        new_blg.loc[22, "geometry"] = new_blg.loc[22, "geometry"].buffer(20)
        new_tess = mm.enclosed_tessellation(
            new_blg, self.enclosures.geometry, simplify=False, n_jobs=1
        )

        # assert that buildings 1 and 22 intersect the same enclosure
        inp, res = self.enclosures.sindex.query(
            new_blg.geometry, predicate="intersects"
        )
        assert np.isclose(new_blg.iloc[inp[res == 8]].index.values, [1, 22]).all()

        # assert that there is a tessellation for building 1
        assert 1 in new_tess.index

    def test_enclosures(self):
        basic = mm.enclosures(self.df_streets)
        assert len(basic) == 7
        assert isinstance(basic, gpd.GeoDataFrame)

        limited = mm.enclosures(self.df_streets, self.limit)
        assert len(limited) == 20
        assert isinstance(limited, gpd.GeoDataFrame)

        limited2 = mm.enclosures(
            self.df_streets, gpd.GeoSeries([self.limit], crs=self.df_streets.crs)
        )
        assert len(limited2) == 20
        assert isinstance(limited2, gpd.GeoDataFrame)

        b = self.limit.bounds
        additional_barrier = gpd.GeoSeries(
            [LineString([(b[0], b[1]), (b[2], b[3])])], crs=self.df_streets.crs
        )

        additional = mm.enclosures(
            self.df_streets,
            gpd.GeoSeries([self.limit], crs=self.df_streets.crs),
            [additional_barrier],
        )
        assert len(additional) == 28
        assert isinstance(additional, gpd.GeoDataFrame)

        with pytest.raises(TypeError, match="`additional_barriers` expects a list"):
            additional = mm.enclosures(
                self.df_streets,
                gpd.GeoSeries([self.limit], crs=self.df_streets.crs),
                additional_barrier,
            )

        # test clip
        limit = self.df_streets.dissolve().convex_hull.buffer(-100).item()
        encl = mm.enclosures(
            self.df_streets,
            limit=gpd.GeoSeries([limit], crs=self.df_streets.crs),
            clip=True,
        )
        assert len(encl) == 18

    def test_get_network_ratio(self):
        convex_hull = self.df_streets.dissolve().convex_hull.item()
        enclosures = mm.enclosures(
            self.df_streets, limit=gpd.GeoSeries([convex_hull], crs=self.df_streets.crs)
        )
        enclosed_tess = mm.enclosed_tessellation(
            self.df_buildings, enclosures=enclosures, simplify=False
        )
        links = mm.get_network_ratio(enclosed_tess, self.df_streets, initial_buffer=10)

        assert links.edgeID_values.apply(lambda x: sum(x)).sum() == len(enclosed_tess)
        assert sorted(links.loc[109]["edgeID_keys"]) == [0, 34]

        # ensure index is preserved
        enclosed_tess.index = [str(uuid.uuid4()) for _ in range(len(enclosed_tess))]
        links2 = mm.get_network_ratio(enclosed_tess, self.df_streets, initial_buffer=10)

        assert_index_equal(enclosed_tess.index, links2.index, check_order=False)
        expected_head = [[34], [0, 34], [0], [0, 15, 3, 14, 4, 7], [34]]
        expected_tail = [[8], [16], [8], [32], [21]]

        for i, idx in enumerate(expected_head):
            assert sorted(links2.edgeID_keys.iloc[i]) == sorted(idx)

        for i, idx in enumerate(expected_tail):
            assert sorted(links2.edgeID_keys.tail(5).iloc[i]) == sorted(idx)
