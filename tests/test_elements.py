import geopandas as gpd
import momepy as mm
import numpy as np
import pytest

from shapely.geometry import Polygon, MultiPoint, LineString
from shapely import affinity

from geopandas.testing import assert_geodataframe_equal


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

        with pytest.raises(ValueError):
            mm.Tessellation(
                self.df_buildings, "uID", limit=self.limit, enclosures=self.enclosures
            )

        enc1_loop = mm.Tessellation(
            self.df_buildings, "uID", enclosures=self.enclosures, use_dask=False
        ).tessellation
        assert len(enc1) == 155
        assert isinstance(enc1, gpd.GeoDataFrame)

        # erroneous geometry
        df = self.df_buildings
        b = df.total_bounds
        x = np.mean([b[0], b[2]])
        y = np.mean([b[1], b[3]])

        df.loc[144] = [145, Polygon([(x, y), (x, y + 1), (x + 1, y)])]
        df.loc[145] = [146, MultiPoint([(x, y), (x + 1, y)]).buffer(0.55)]
        df.loc[146] = [147, affinity.rotate(df.geometry.iloc[0], 12)]

        tess = mm.Tessellation(df, "uID", self.limit)
        assert tess.collapsed == {145}
        assert len(tess.multipolygons) == 3

    def test_Blocks(self):
        blocks = mm.Blocks(
            self.df_tessellation, self.df_streets, self.df_buildings, "bID", "uID"
        )
        assert not blocks.tessellation_id.isna().any()
        assert not blocks.buildings_id.isna().any()
        assert len(blocks.blocks) == 8

        with pytest.raises(ValueError):
            mm.Blocks(
                self.df_tessellation, self.df_streets, self.df_buildings, "uID", "uID"
            )

    def test_get_network_id(self):
        buildings_id = mm.get_network_id(self.df_buildings, self.df_streets, "nID")
        assert not buildings_id.isna().any()

    def test_get_network_id_duplicate(self):
        self.df_buildings["nID"] = range(len(self.df_buildings))
        buildings_id = mm.get_network_id(self.df_buildings, self.df_streets, "nID")
        assert not buildings_id.isna().any()

    def test_get_node_id(self):
        nx = mm.gdf_to_nx(self.df_streets)
        nodes, edges = mm.nx_to_gdf(nx)
        self.df_buildings["nID"] = mm.get_network_id(
            self.df_buildings, self.df_streets, "nID"
        )
        ids = mm.get_node_id(self.df_buildings, nodes, edges, "nodeID", "nID")
        assert not ids.isna().any()

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

        limited = mm.enclosures(self.df_streets, gpd.GeoSeries([self.limit]))
        assert len(limited) == 20
        assert isinstance(limited, gpd.GeoDataFrame)

        b = self.limit.bounds
        additional_barrier = gpd.GeoSeries([LineString([(b[0], b[1]), (b[2], b[3])])])

        additional = mm.enclosures(
            self.df_streets, gpd.GeoSeries([self.limit]), [additional_barrier]
        )
        assert len(additional) == 28
        assert isinstance(additional, gpd.GeoDataFrame)

        with pytest.raises(TypeError):
            additional = mm.enclosures(
                self.df_streets, gpd.GeoSeries([self.limit]), additional_barrier
            )

    def test_get_network_ratio(self):
        convex_hull = self.df_streets.unary_union.convex_hull
        enclosures = mm.enclosures(self.df_streets, limit=gpd.GeoSeries([convex_hull]))
        enclosed_tess = mm.Tessellation(
            self.df_buildings, unique_id="uID", enclosures=enclosures
        ).tessellation
        links = mm.get_network_ratio(enclosed_tess, self.df_streets, initial_buffer=10)

        assert links.edgeID_values.apply(lambda x: sum(x)).sum() == len(enclosed_tess)
        m = enclosed_tess["uID"] == 110
        assert links[m].iloc[0]["edgeID_keys"] == [0, 34]
