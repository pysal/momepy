import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

import momepy as mm
from momepy import sw_high


class TestDiversity:
    def setup_method(self):
        test_file_path = mm.datasets.get_path("bubenec")
        self.df_buildings = gpd.read_file(test_file_path, layer="buildings")
        self.df_streets = gpd.read_file(test_file_path, layer="streets")
        self.df_tessellation = gpd.read_file(test_file_path, layer="tessellation")
        self.df_buildings["height"] = np.linspace(10.0, 30.0, 144)
        self.df_tessellation["area"] = mm.Area(self.df_tessellation).series
        self.sw = sw_high(k=3, gdf=self.df_tessellation, ids="uID")
        self.sw.neighbors[100] = []
        self.sw_drop = sw_high(k=3, gdf=self.df_tessellation[2:], ids="uID")

    def test_Range(self):
        full_sw = mm.Range(self.df_tessellation, "area", self.sw, "uID").series
        assert full_sw[0] == pytest.approx(8255.372, rel=1e-3)
        area = self.df_tessellation["area"]
        full2 = mm.Range(self.df_tessellation, area, self.sw, "uID").series
        assert full2[0] == pytest.approx(8255.372, rel=1e-3)
        limit = mm.Range(
            self.df_tessellation, "area", self.sw, "uID", rng=(10, 90)
        ).series
        assert limit[0] == pytest.approx(4122.139, rel=1e-3)
        assert (
            mm.Range(self.df_tessellation, "area", self.sw_drop, "uID")
            .series.isna()
            .any()
        )

    def test_Theil(self):
        full_sw = mm.Theil(self.df_tessellation, "area", self.sw, "uID").series
        assert full_sw[0] == pytest.approx(0.25744684)
        limit = mm.Theil(
            self.df_tessellation,
            self.df_tessellation.area,
            self.sw,
            "uID",
            rng=(10, 90),
        ).series
        assert limit[0] == pytest.approx(0.1330295)
        zeros = mm.Theil(
            self.df_tessellation, np.zeros(len(self.df_tessellation)), self.sw, "uID"
        ).series
        assert zeros[0] == 0
        assert (
            mm.Theil(self.df_tessellation, "area", self.sw_drop, "uID")
            .series.isna()
            .any()
        )

    def test_Simpson(self):
        ht_sw = mm.Simpson(self.df_tessellation, "area", self.sw, "uID").series
        assert ht_sw[0] == 0.385
        quan_sw = mm.Simpson(
            self.df_tessellation,
            self.df_tessellation.area,
            self.sw,
            "uID",
            binning="quantiles",
            k=3,
        ).series
        assert quan_sw[0] == 0.395
        with pytest.raises(ValueError):
            ht_sw = mm.Simpson(
                self.df_tessellation, "area", self.sw, "uID", binning="nonexistent"
            )
        assert (
            mm.Simpson(self.df_tessellation, "area", self.sw_drop, "uID")
            .series.isna()
            .any()
        )
        gs = mm.Simpson(
            self.df_tessellation, "area", self.sw, "uID", gini_simpson=True
        ).series
        assert gs[0] == 1 - 0.385

        inv = mm.Simpson(
            self.df_tessellation, "area", self.sw, "uID", inverse=True
        ).series
        assert inv[0] == 1 / 0.385

        self.df_tessellation["cat"] = list(range(8)) * 18
        cat = mm.Simpson(
            self.df_tessellation, "cat", self.sw, "uID", categorical=True
        ).series
        assert cat[0] == pytest.approx(0.15)

        cat2 = mm.Simpson(
            self.df_tessellation,
            "cat",
            self.sw,
            "uID",
            categorical=True,
        ).series
        assert cat2[0] == pytest.approx(0.15)

    def test_Gini(self):
        full_sw = mm.Gini(self.df_tessellation, "area", self.sw, "uID").series
        assert full_sw[0] == pytest.approx(0.3945388)
        limit = mm.Gini(
            self.df_tessellation, "area", self.sw, "uID", rng=(10, 90)
        ).series
        assert limit[0] == pytest.approx(0.28532814)
        self.df_tessellation["negative"] = (
            self.df_tessellation.area - self.df_tessellation.area.mean()
        )
        with pytest.raises(ValueError):
            mm.Gini(self.df_tessellation, "negative", self.sw, "uID").series
        assert (
            mm.Gini(self.df_tessellation, "area", self.sw_drop, "uID")
            .series.isna()
            .any()
        )

    def test_Shannon(self):
        ht_sw = mm.Shannon(self.df_tessellation, "area", self.sw, "uID").series
        assert ht_sw[0] == 1.094056456831614
        quan_sw = mm.Shannon(
            self.df_tessellation,
            self.df_tessellation.area,
            self.sw,
            "uID",
            binning="quantiles",
            k=3,
        ).series
        assert quan_sw[0] == 0.9985793315873921
        with pytest.raises(ValueError):
            ht_sw = mm.Shannon(
                self.df_tessellation, "area", self.sw, "uID", binning="nonexistent"
            )
        assert (
            mm.Shannon(self.df_tessellation, "area", self.sw_drop, "uID")
            .series.isna()
            .any()
        )

        self.df_tessellation["cat"] = list(range(8)) * 18
        cat = mm.Shannon(
            self.df_tessellation, "cat", self.sw, "uID", categorical=True
        ).series
        assert cat[0] == pytest.approx(1.973)

        cat2 = mm.Shannon(
            self.df_tessellation,
            "cat",
            self.sw,
            "uID",
            categorical=True,
            categories=range(15),
        ).series
        assert cat2[0] == pytest.approx(1.973)

    def test_Unique(self):
        self.df_tessellation["cat"] = list(range(8)) * 18
        un = mm.Unique(self.df_tessellation, "cat", self.sw, "uID").series
        assert un[0] == 8
        un = mm.Unique(self.df_tessellation, list(range(8)) * 18, self.sw, "uID").series
        assert un[0] == 8
        un = mm.Unique(self.df_tessellation, "cat", self.sw_drop, "uID").series
        assert un.isna().any()
        assert un[5] == 8

        self.df_tessellation.loc[0, "cat"] = np.nan
        un = mm.Unique(self.df_tessellation, "cat", self.sw, "uID", dropna=False).series
        assert un[0] == 9

        un = mm.Unique(self.df_tessellation, "cat", self.sw, "uID", dropna=True).series
        assert un[0] == 8

    def test_Percentile(self):
        perc = mm.Percentiles(self.df_tessellation, "area", self.sw, "uID").frame
        assert np.all(
            perc.loc[0].values - np.array([1085.11492833, 2623.9962661, 4115.47168328])
            < 0.00001
        )
        perc = mm.Percentiles(
            self.df_tessellation, list(range(8)) * 18, self.sw, "uID"
        ).frame
        assert np.all(perc.loc[0].values == np.array([1.0, 3.5, 6.0]))

        perc = mm.Percentiles(
            self.df_tessellation, "area", self.sw, "uID", percentiles=[30, 70]
        ).frame
        assert np.all(
            perc.loc[0].values - np.array([1218.98841575, 3951.35531166]) < 0.00001
        )

        perc = mm.Percentiles(
            self.df_tessellation,
            "area",
            self.sw,
            "uID",
            weighted="linear",
        ).frame
        assert np.all(
            perc.loc[0].values - np.array([997.8086922, 2598.84036762, 4107.14201011])
            < 0.00001
        )

        perc = mm.Percentiles(
            self.df_tessellation,
            "area",
            self.sw,
            "uID",
            percentiles=[30, 70],
            weighted="linear",
        ).frame
        assert np.all(
            perc.loc[0].values - np.array([1211.83227008, 3839.99083097]) < 0.00001
        )

        _data = {"uID": [9999], "area": 1.0}
        _pgon = [Polygon(((0, 0), (0, 1), (1, 1), (1, 0)))]
        _gdf = gpd.GeoDataFrame(_data, index=[9999], geometry=_pgon)

        perc = mm.Percentiles(
            pd.concat([self.df_tessellation, _gdf]),
            "area",
            self.sw,
            "uID",
        ).frame
        np.testing.assert_array_equal(np.isnan(perc.loc[9999]), np.ones(3, dtype=bool))

        perc = mm.Percentiles(
            pd.concat([_gdf, self.df_tessellation]),
            "area",
            self.sw,
            "uID",
            weighted="linear",
        ).frame
        np.testing.assert_array_equal(np.isnan(perc.loc[9999]), np.ones(3, dtype=bool))

        with pytest.raises(ValueError, match="'nonsense' is not a valid"):
            mm.Percentiles(
                self.df_tessellation,
                "area",
                self.sw,
                "uID",
                weighted="nonsense",
            )
