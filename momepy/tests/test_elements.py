import uuid
from random import shuffle

import geopandas as gpd
import numpy as np
import pytest
from geopandas.testing import assert_geodataframe_equal
from packaging.version import Version
from pandas.testing import assert_index_equal
from shapely import affinity
from shapely.geometry import LineString, MultiPoint, Polygon

import momepy as mm

# https://github.com/geopandas/geopandas/issues/2282
GPD_REGR = Version("0.10.2") < Version(gpd.__version__) < Version("0.11")
GPD_10 = Version(gpd.__version__) >= Version("0.10")


class TestElements:
    def setup_method(self):
        test_file_path = mm.datasets.get_path("bubenec")
        self.df_buildings = gpd.read_file(test_file_path, layer="buildings")
        self.df_tessellation = gpd.read_file(test_file_path, layer="tessellation")
        self.df_streets = gpd.read_file(test_file_path, layer="streets")
        self.df_streets["nID"] = range(len(self.df_streets))
        self.limit = mm.buffered_limit(self.df_buildings, 50)
        self.enclosures = mm.enclosures(
            self.df_streets, gpd.GeoSeries([self.limit.exterior])
        )

    def test_Tessellation(self):
        tes = mm.Tessellation(self.df_buildings, "uID", self.limit, segment=2)
        tessellation = tes.tessellation
        assert len(tessellation) == len(self.df_tessellation)
        bands = mm.Tessellation(
            self.df_streets, "nID", mm.buffered_limit(self.df_streets, 50), segment=5
        ).tessellation
        assert len(bands) == len(self.df_streets)

    def test_enclosed_tess(self):
        #  test_enclosed_tessellation
        enc1 = mm.Tessellation(
            self.df_buildings, "uID", enclosures=self.enclosures
        ).tessellation
        assert len(enc1) == 155
        assert isinstance(enc1, gpd.GeoDataFrame)

        enc1_loop = mm.Tessellation(
            self.df_buildings, "uID", enclosures=self.enclosures, use_dask=False
        ).tessellation
        assert len(enc1) == 155
        assert isinstance(enc1, gpd.GeoDataFrame)

        assert len(enc1_loop) == 155
        assert isinstance(enc1_loop, gpd.GeoDataFrame)

        assert_geodataframe_equal(enc1, enc1_loop)

    def test_limit_enclosures_combo_error(self):
        with pytest.raises(ValueError, match="Both `limit` and `enclosures` cannot"):
            mm.Tessellation(
                self.df_buildings, "uID", limit=self.limit, enclosures=self.enclosures
            )

    def test_custom_enclosure_id(self):
        # non-standard enclosure ids
        encl = self.enclosures.copy()
        ids = list(range(len(encl) * 2))
        shuffle(ids)
        encl["eID"] = ids[: len(encl)]
        encl.index = ids[: len(encl)]
        enc = mm.Tessellation(self.df_buildings, "uID", enclosures=encl).tessellation
        assert len(enc) == 155
        assert isinstance(enc, gpd.GeoDataFrame)

    def test_erroroneous_geom(self):
        df = self.df_buildings
        b = df.total_bounds
        x = np.mean([b[0], b[2]])
        y = np.mean([b[1], b[3]])

        df.loc[144] = [145, Polygon([(x, y), (x, y + 1), (x + 1, y)])]
        df.loc[145] = [146, MultiPoint([(x, y), (x + 1, y)]).buffer(0.55)]
        df.loc[146] = [147, affinity.rotate(df.geometry.iloc[0], 12)]

        with pytest.warns(
            UserWarning, match="Tessellation does not fully match buildings."
        ):
            mm.Tessellation(df, "uID", self.limit)

        with pytest.warns(
            UserWarning, match="Tessellation contains MultiPolygon elements."
        ):
            tess = mm.Tessellation(df, "uID", self.limit)
            assert tess.collapsed == {145}
            assert len(tess.multipolygons) == 3

    def test_crs_error(self):
        with pytest.raises(ValueError, match="Geometry is in a geographic CRS"):
            mm.Tessellation(self.df_buildings.to_crs(4326), "uID", self.limit)

    def test_Blocks(self):
        blocks = mm.Blocks(
            self.df_tessellation, self.df_streets, self.df_buildings, "bID", "uID"
        )
        assert not blocks.tessellation_id.isna().any()
        assert not blocks.buildings_id.isna().any()
        assert len(blocks.blocks) == 8

        with pytest.raises(ValueError, match="'uID' column cannot be"):
            mm.Blocks(
                self.df_tessellation, self.df_streets, self.df_buildings, "uID", "uID"
            )

    def test_Blocks_non_default_index(self):
        tessellation = self.df_tessellation.copy()
        tessellation.index = tessellation.index * 3
        buildings = self.df_buildings.copy()
        buildings.index = buildings.index * 5

        blocks = mm.Blocks(tessellation, self.df_streets, buildings, "bID", "uID")

        assert_index_equal(tessellation.index, blocks.tessellation_id.index)
        assert_index_equal(buildings.index, blocks.buildings_id.index)

    def test_Blocks_inner(self):
        streets = self.df_streets.copy()
        streets.loc[35, "geometry"] = (
            self.df_buildings.geometry.iloc[141]
            .representative_point()
            .buffer(20)
            .exterior
        )
        blocks = mm.Blocks(
            self.df_tessellation, streets, self.df_buildings, "bID", "uID"
        )
        assert not blocks.tessellation_id.isna().any()
        assert not blocks.buildings_id.isna().any()
        assert len(blocks.blocks) == 9
        assert (
            len(blocks.blocks.sindex.query_bulk(blocks.blocks.geometry, "overlaps")[0])
            == 0
        )

    def test_get_network_id(self):
        buildings_id = mm.get_network_id(self.df_buildings, self.df_streets, "nID")
        assert not buildings_id.isna().any()

    def test_get_network_id_duplicate(self):
        self.df_buildings["nID"] = range(len(self.df_buildings))
        buildings_id = mm.get_network_id(self.df_buildings, self.df_streets, "nID")
        assert not buildings_id.isna().any()

    @pytest.mark.skipif(GPD_REGR, reason="regression in geopandas")
    def test_get_node_id(self):
        nx = mm.gdf_to_nx(self.df_streets)
        nodes, edges = mm.nx_to_gdf(nx)
        self.df_buildings["nID"] = mm.get_network_id(
            self.df_buildings, self.df_streets, "nID"
        )
        ids = mm.get_node_id(self.df_buildings, nodes, edges, "nodeID", "nID")
        assert not ids.isna().any()

    @pytest.mark.skipif(GPD_REGR, reason="regression in geopandas")
    @pytest.mark.skipif(not GPD_10, reason="requires sindex.nearest")
    def test_get_node_id_ratio(self):
        nx = mm.gdf_to_nx(self.df_streets)
        nodes, edges = mm.nx_to_gdf(nx)

        convex_hull = edges.unary_union.convex_hull
        enclosures = mm.enclosures(edges, limit=gpd.GeoSeries([convex_hull]))
        enclosed_tess = mm.Tessellation(
            self.df_buildings, unique_id="uID", enclosures=enclosures
        ).tessellation
        links = mm.get_network_ratio(enclosed_tess, edges)
        enclosed_tess[links.columns] = links

        ids = mm.get_node_id(
            enclosed_tess,
            nodes,
            edges,
            node_id="nodeID",
            edge_keys="edgeID_keys",
            edge_values="edgeID_values",
        )

        assert not ids.isna().any()

    def test_enclosures(self):
        basic = mm.enclosures(self.df_streets)
        assert len(basic) == 7
        assert isinstance(basic, gpd.GeoDataFrame)

        limited = mm.enclosures(self.df_streets, self.limit)
        assert len(limited) == 20
        assert isinstance(limited, gpd.GeoDataFrame)

        limited2 = mm.enclosures(self.df_streets, gpd.GeoSeries([self.limit]))
        assert len(limited2) == 20
        assert isinstance(limited2, gpd.GeoDataFrame)

        b = self.limit.bounds
        additional_barrier = gpd.GeoSeries([LineString([(b[0], b[1]), (b[2], b[3])])])

        additional = mm.enclosures(
            self.df_streets, gpd.GeoSeries([self.limit]), [additional_barrier]
        )
        assert len(additional) == 28
        assert isinstance(additional, gpd.GeoDataFrame)

        with pytest.raises(TypeError, match="`additional_barriers` expects a list"):
            additional = mm.enclosures(
                self.df_streets, gpd.GeoSeries([self.limit]), additional_barrier
            )

        # test clip
        limit = self.df_streets.unary_union.convex_hull.buffer(-100)
        encl = mm.enclosures(self.df_streets, limit=gpd.GeoSeries([limit]), clip=True)
        assert len(encl) == 18

    @pytest.mark.skipif(not GPD_10, reason="requires sindex.nearest")
    def test_get_network_ratio(self):
        convex_hull = self.df_streets.unary_union.convex_hull
        enclosures = mm.enclosures(self.df_streets, limit=gpd.GeoSeries([convex_hull]))
        enclosed_tess = mm.Tessellation(
            self.df_buildings, unique_id="uID", enclosures=enclosures
        ).tessellation
        links = mm.get_network_ratio(enclosed_tess, self.df_streets, initial_buffer=10)

        assert links.edgeID_values.apply(lambda x: sum(x)).sum() == len(enclosed_tess)
        m = enclosed_tess["uID"] == 110
        assert sorted(links.loc[m].iloc[0]["edgeID_keys"]) == [0, 34]

        # ensure index is preserved
        enclosed_tess.index = [str(uuid.uuid4()) for _ in range(len(enclosed_tess))]
        links2 = mm.get_network_ratio(enclosed_tess, self.df_streets, initial_buffer=10)

        assert_index_equal(enclosed_tess.index, links2.index, check_order=False)
        expected_head = [[0, 34], [34], [34], [0], [0, 15, 3, 14, 4, 7]]
        expected_tail = [[28], [29], [28], [32], [21]]

        for i, idx in enumerate(expected_head):
            assert sorted(links2.edgeID_keys.iloc[i]) == sorted(idx)

        for i, idx in enumerate(expected_tail):
            assert sorted(links2.edgeID_keys.tail(5).iloc[i]) == sorted(idx)

    @pytest.mark.skipif(GPD_10, reason="requires sindex.nearest")
    def test_get_network_ratio_error(self):
        convex_hull = self.df_streets.unary_union.convex_hull
        enclosures = mm.enclosures(self.df_streets, limit=gpd.GeoSeries([convex_hull]))
        enclosed_tess = mm.Tessellation(
            self.df_buildings, unique_id="uID", enclosures=enclosures
        ).tessellation
        with pytest.raises(ImportError, match="`get_network_ratio` requires geopandas"):
            mm.get_network_ratio(enclosed_tess, self.df_streets, initial_buffer=10)
