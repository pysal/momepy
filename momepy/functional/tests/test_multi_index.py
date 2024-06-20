import geopandas as gpd
import numpy as np
from packaging.version import Version

import momepy as mm

GPD_013 = Version(gpd.__version__) >= Version("0.13")


class TestShape:
    def setup_method(self):
        test_file_path = mm.datasets.get_path("bubenec")
        self.df_buildings = gpd.read_file(test_file_path, layer="buildings")
        self.df_streets = gpd.read_file(test_file_path, layer="streets")
        self.df_tessellation = gpd.read_file(test_file_path, layer="tessellation")
        self.df_buildings["height"] = np.linspace(10.0, 30.0, 144)
        self.df_buildings["area"] = self.df_buildings.geometry.area
        blocks = mm.Blocks(
            self.df_tessellation, self.df_streets, self.df_buildings, "bID", "uID"
        )
        self.df_buildings["bID"] = blocks.buildings_id
        self.df_tessellation["bID"] = blocks.tessellation_id
        self.df_buildings_multi = self.df_buildings.set_index(["bID", "uID"])

    def test_dimension(self):
        dimension_functions = [
            "volume",
            "floor_area",
            "courtyard_area",
            "longest_axis_length",
            "street_profile",
        ]

        dimension_functions_args = [
            (self.df_buildings["area"], self.df_buildings["height"]),
            (self.df_buildings["area"], self.df_buildings["height"]),
            (self.df_buildings,),
            (self.df_buildings,),
            (self.df_streets, self.df_buildings),
        ]

        dimension_functions_args_multi = [
            (self.df_buildings_multi["area"], self.df_buildings_multi["height"]),
            (self.df_buildings_multi["area"], self.df_buildings_multi["height"]),
            (self.df_buildings_multi,),
            (self.df_buildings_multi,),
            (self.df_streets, self.df_buildings),
        ]

        for fname, fargs, fargs_multi in zip(
            dimension_functions,
            dimension_functions_args,
            dimension_functions_args_multi,
            strict=True,
        ):
            func = getattr(mm, fname)
            res = func(*fargs)
            res_multi = func(*fargs_multi)

            assert np.allclose(res.values, res_multi.values, equal_nan=True)
