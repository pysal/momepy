import geopandas as gpd
import libpysal
import pytest

import momepy as mm


class TestWeights:
    def setup_method(self):
        test_file_path = mm.datasets.get_path("bubenec")
        self.df_buildings = gpd.read_file(test_file_path, layer="buildings")
        self.df_tessellation = gpd.read_file(test_file_path, layer="tessellation")
        self.df_tessellation["area"] = mm.Area(self.df_tessellation).series

    def test_sw_high(self):
        first_order = libpysal.weights.Queen.from_dataframe(
            self.df_tessellation, use_index=False
        )
        from_sw = mm.sw_high(2, gdf=None, weights=first_order)
        from_df = mm.sw_high(2, gdf=self.df_tessellation)
        rook = mm.sw_high(2, gdf=self.df_tessellation, contiguity="rook")
        check = sorted([133, 134, 111, 112, 113, 114, 115, 121, 125])
        assert sorted(from_sw.neighbors[0]) == check
        assert sorted(from_df.neighbors[0]) == check
        assert sorted(rook.neighbors[0]) == check

        with pytest.raises(
            AttributeError, match="GeoDataFrame or spatial weights must be given."
        ):
            mm.sw_high(2, gdf=None, weights=None)

        with pytest.raises(
            ValueError, match="is not supported. Use 'queen' or 'rook'."
        ):
            mm.sw_high(2, gdf=self.df_tessellation, contiguity="nonexistent")

    def test_DistanceBand(self):
        lp = libpysal.weights.DistanceBand.from_dataframe(self.df_buildings, 100)
        lp_ids = libpysal.weights.DistanceBand.from_dataframe(
            self.df_buildings, 100, ids="uID"
        )
        db = mm.DistanceBand(self.df_buildings, 100)
        db_ids = mm.DistanceBand(self.df_buildings, 100, ids="uID")

        for k in range(len(self.df_buildings)):
            assert k in db.neighbors.keys()  # noqa: SIM118
            assert sorted(lp.neighbors[k]) == sorted(db.neighbors[k])
        for k in self.df_buildings.uID:
            assert k in db_ids.neighbors.keys()  # noqa: SIM118
            assert sorted(lp_ids.neighbors[k]) == sorted(db_ids.neighbors[k])

        db_cent_false = mm.DistanceBand(self.df_buildings, 100, centroid=False)
        assert sorted(db_cent_false.neighbors[0]) == sorted(
            [111, 112, 115, 130, 125, 133, 114, 120, 134, 113, 121]
        )
