import geopandas as gpd
import momepy as mm
import numpy as np
import pytest

from shapely.geometry import Polygon, MultiPoint, LineString
from shapely import affinity


class TestPreprocessing:
    def setup_method(self):

        test_file_path = mm.datasets.get_path("bubenec")
        self.df_buildings = gpd.read_file(test_file_path, layer="buildings")
        self.df_tessellation = gpd.read_file(test_file_path, layer="tessellation")
        self.df_streets = gpd.read_file(test_file_path, layer="streets")
        self.df_buildings["height"] = np.linspace(10.0, 30.0, 144)

    def test_preprocess(self):
        test_file_path2 = mm.datasets.get_path("tests")
        self.os_buildings = gpd.read_file(test_file_path2, layer="os")
        processed = mm.preprocess(self.os_buildings)
        assert len(processed) == 5

    def test_network_false_nodes(self):
        test_file_path2 = mm.datasets.get_path("tests")
        self.false_network = gpd.read_file(test_file_path2, layer="network")
        self.false_network["vals"] = range(len(self.false_network))
        with pytest.warns(FutureWarning):
            fixed = mm.network_false_nodes(self.false_network)
        assert len(fixed) == 56
        assert isinstance(fixed, gpd.GeoDataFrame)
        assert self.false_network.crs.equals(fixed.crs)
        assert sorted(self.false_network.columns) == sorted(fixed.columns)
        with pytest.warns(FutureWarning):
            fixed_series = mm.network_false_nodes(self.false_network.geometry)
        assert len(fixed_series) == 56
        assert isinstance(fixed_series, gpd.GeoSeries)
        assert self.false_network.crs.equals(fixed_series.crs)
        with pytest.raises(TypeError):
            mm.network_false_nodes(list())
        multiindex = self.false_network.explode()
        with pytest.warns(FutureWarning):
            fixed_multiindex = mm.network_false_nodes(multiindex)
        assert len(fixed_multiindex) == 56
        assert isinstance(fixed, gpd.GeoDataFrame)
        assert sorted(self.false_network.columns) == sorted(fixed.columns)

    def test_remove_false_nodes(self):
        test_file_path2 = mm.datasets.get_path("tests")
        self.false_network = gpd.read_file(test_file_path2, layer="network")
        self.false_network["vals"] = range(len(self.false_network))
        fixed = mm.remove_false_nodes(self.false_network)
        assert len(fixed) == 56
        assert isinstance(fixed, gpd.GeoDataFrame)
        assert self.false_network.crs.equals(fixed.crs)
        assert sorted(self.false_network.columns) == sorted(fixed.columns)
        fixed_series = mm.remove_false_nodes(self.false_network.geometry)
        assert len(fixed_series) == 56
        assert isinstance(fixed_series, gpd.GeoSeries)
        # assert self.false_network.crs.equals(fixed_series.crs) GeoPandas 0.8 BUG
        multiindex = self.false_network.explode()
        fixed_multiindex = mm.remove_false_nodes(multiindex)
        assert len(fixed_multiindex) == 56
        assert isinstance(fixed, gpd.GeoDataFrame)
        assert sorted(self.false_network.columns) == sorted(fixed.columns)

    def test_snap_street_network_edge(self):
        snapped = mm.snap_street_network_edge(
            self.df_streets, self.df_buildings, 20, self.df_tessellation, 70
        )
        snapped_nonedge = mm.snap_street_network_edge(
            self.df_streets, self.df_buildings, 20
        )
        snapped_edge = mm.snap_street_network_edge(
            self.df_streets,
            self.df_buildings,
            20,
            tolerance_edge=70,
            edge=mm.buffered_limit(self.df_buildings, buffer=50),
        )
        assert sum(snapped.geometry.length) == 5980.041004739525
        assert sum(snapped_edge.geometry.length) == 5980.718889937014
        assert sum(snapped_nonedge.geometry.length) < 5980.041004739525

    def test_CheckTessellationInput(self):
        df = self.df_buildings
        df.loc[144, "geometry"] = Polygon([(0, 0), (0, 1), (1, 0)])
        df.loc[145, "geometry"] = MultiPoint([(0, 0), (1, 0)]).buffer(0.55)
        df.loc[146, "geometry"] = affinity.rotate(df.geometry.iloc[0], 12)
        check = mm.CheckTessellationInput(self.df_buildings)
        assert len(check.collapse) == 1
        assert len(check.split) == 1
        assert len(check.overlap) == 2

        check = mm.CheckTessellationInput(self.df_buildings, collapse=False)
        assert len(check.split) == 1
        assert len(check.overlap) == 2

        check = mm.CheckTessellationInput(self.df_buildings, split=False)
        assert len(check.collapse) == 1
        assert len(check.overlap) == 2

        check = mm.CheckTessellationInput(self.df_buildings, overlap=False)
        assert len(check.collapse) == 1
        assert len(check.split) == 1

        check = mm.CheckTessellationInput(self.df_buildings, shrink=0)
        assert len(check.collapse) == 0
        assert len(check.split) == 0
        assert len(check.overlap) == 4

    def test_close_gaps(self):
        l1 = LineString([(1, 0), (2, 1)])
        l2 = LineString([(2.1, 1), (3, 2)])
        l3 = LineString([(3.1, 2), (4, 0)])
        l4 = LineString([(4.1, 0), (5, 0)])
        l5 = LineString([(5.1, 0), (6, 0)])
        df = gpd.GeoDataFrame(geometry=[l1, l2, l3, l4, l5])

        closed = mm.close_gaps(df, 0.25)
        assert len(closed) == len(df)

        merged = mm.remove_false_nodes(closed)
        assert len(merged) == 1
        assert merged.length[0] == pytest.approx(7.0502, rel=1e-3)
