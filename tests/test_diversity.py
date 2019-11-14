import geopandas as gpd
import momepy as mm
import numpy as np
import pytest
from momepy import sw_high
from pytest import approx

from distutils.version import LooseVersion

try:
    import mapclassify
    MC_21 = str(mapclassify.__version__) <= LooseVersion("2.1.0")
except ImportError:
    import pysal
    MC_21 = str(pysal.__version__) <= LooseVersion("2.1.0")


class TestDiversity:
    def setup_method(self):

        test_file_path = mm.datasets.get_path("bubenec")
        self.df_buildings = gpd.read_file(test_file_path, layer="buildings")
        self.df_streets = gpd.read_file(test_file_path, layer="streets")
        self.df_tessellation = gpd.read_file(test_file_path, layer="tessellation")
        self.df_buildings["height"] = np.linspace(10.0, 30.0, 144)
        self.df_tessellation["area"] = mm.Area(self.df_tessellation).series
        self.sw = sw_high(k=3, gdf=self.df_tessellation, ids="uID")

    def test_Range(self):
        full_sw = mm.Range(self.df_tessellation, "area", self.sw, "uID").series
        assert full_sw[0] == approx(8255.372, rel=1e-3)
        area = self.df_tessellation["area"]
        full2 = mm.Range(self.df_tessellation, area, self.sw, "uID").series
        assert full2[0] == approx(8255.372, rel=1e-3)
        limit = mm.Range(
            self.df_tessellation, "area", self.sw, "uID", rng=(10, 90)
        ).series
        assert limit[0] == approx(4122.139, rel=1e-3)

    def test_Theil(self):
        full_sw = mm.Theil(self.df_tessellation, "area", self.sw, "uID").series
        assert full_sw[0] == approx(0.25744684)
        limit = mm.Theil(
            self.df_tessellation,
            self.df_tessellation.area,
            self.sw,
            "uID",
            rng=(10, 90),
        ).series
        assert limit[0] == approx(0.1330295)
        zeros = mm.Theil(
            self.df_tessellation, np.zeros(len(self.df_tessellation)), self.sw, "uID"
        ).series
        assert zeros[0] == 0

    @pytest.mark.skipif(MC_21, reason="requires mapclassify < 2.1")
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

    def test_Gini(self):
        full_sw = mm.Gini(self.df_tessellation, "area", self.sw, "uID").series
        assert full_sw[0] == approx(0.3945388)
        limit = mm.Gini(
            self.df_tessellation, "area", self.sw, "uID", rng=(10, 90)
        ).series
        assert limit[0] == approx(0.28532814)
        self.df_tessellation["negative"] = (
            self.df_tessellation.area - self.df_tessellation.area.mean()
        )
        with pytest.raises(ValueError):
            mm.Gini(self.df_tessellation, "negative", self.sw, "uID").series
