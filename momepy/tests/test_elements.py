import uuid

import geopandas as gpd
import pytest
from pandas.testing import assert_index_equal
from shapely import LineString

import momepy as mm


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
            self.df_buildings, enclosures=enclosures
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
