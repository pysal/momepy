import momepy as mm
import geopandas as gpd
import numpy as np

from momepy import sw_high

import pytest
from pytest import approx


class TestDiversity:
    def setup_method(self):

        test_file_path = mm.datasets.get_path("bubenec")
        self.df_buildings = gpd.read_file(test_file_path, layer="buildings")
        self.df_streets = gpd.read_file(test_file_path, layer="streets")
        self.df_tessellation = gpd.read_file(test_file_path, layer="tessellation")
        self.df_buildings["height"] = np.linspace(10.0, 30.0, 144)
        self.df_tessellation["area"] = mm.area(self.df_tessellation)
        self.sw = sw_high(k=3, gdf=self.df_tessellation, ids="uID")

    def test_rng(self):
        full_sw = mm.rng(self.df_tessellation, "area", self.sw, "uID")
        assert full_sw[0] == approx(8255.372, rel=1e-3)
        area = self.df_tessellation["area"]
        full2 = mm.rng(self.df_tessellation, area, self.sw, "uID")
        assert full2[0] == approx(8255.372, rel=1e-3)
        limit = mm.rng(self.df_tessellation, "area", self.sw, "uID", rng=(10, 90))
        assert limit[0] == approx(4122.139, rel=1e-3)

    def test_theil(self):
        full_sw = mm.theil(self.df_tessellation, "area", self.sw, "uID")
        assert full_sw[0] == approx(0.25744684)
        limit = mm.theil(
            self.df_tessellation,
            self.df_tessellation.area,
            self.sw,
            "uID",
            rng=(10, 90),
        )
        assert limit[0] == approx(0.1330295)
        zeros = mm.theil(self.df_tessellation, np.zeros(len(self.df_tessellation)), self.sw, "uID")
        assert zeros[0] == 0

    def test_simpson(self):
        ht_sw = mm.simpson(self.df_tessellation, "area", self.sw, "uID")
        assert ht_sw[0] == 0.385
        quan_sw = mm.simpson(
            self.df_tessellation,
            self.df_tessellation.area,
            self.sw,
            "uID",
            binning="quantiles",
            k=3,
        )
        assert quan_sw[0] == 0.395
        with pytest.raises(ValueError):
            ht_sw = mm.simpson(
                self.df_tessellation, "area", self.sw, "uID", binning="nonexistent"
            )

    def test_gini(self):
        full_sw = mm.gini(self.df_tessellation, "area", self.sw, "uID")
        assert full_sw[0] == approx(0.3945388)
        limit = mm.gini(self.df_tessellation, "area", self.sw, "uID", rng=(10, 90))
        assert limit[0] == approx(0.28532814)
        self.df_tessellation["negative"] = (
            self.df_tessellation.area - self.df_tessellation.area.mean()
        )
        with pytest.raises(ValueError):
            mm.gini(self.df_tessellation, "negative", self.sw, "uID")
