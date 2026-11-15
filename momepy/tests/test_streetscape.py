import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely

import momepy
from momepy.streetscape import _compute_slope_from_arrays


class TestStreetscape:
    def setup_method(self):
        self.streets = gpd.read_file(
            momepy.datasets.get_path("bubenec"), layer="streets"
        ).to_crs(5514)
        self.buildings = gpd.read_file(
            momepy.datasets.get_path("bubenec"), layer="buildings"
        ).to_crs(5514)
        self.plots = gpd.read_file(
            momepy.datasets.get_path("bubenec"), layer="plots"
        ).to_crs(5514)

        self.buildings["category"] = np.repeat(
            np.arange(0, 6), self.buildings.shape[0] // 6
        )
        self.buildings["height"] = np.linspace(12, 30, self.buildings.shape[0])

    def test_minimal(self):
        sc = momepy.Streetscape(self.streets, self.buildings)

        street_df = sc.street_level()
        point_df = sc.point_level()

        assert street_df.shape == (35, 77)
        assert point_df.shape == (1277, 26)

    def test_no_dtm(self):
        sc = momepy.Streetscape(self.streets, self.buildings)
        sc.compute_plots(self.plots)

        street_df = sc.street_level()
        point_df = sc.point_level()

        assert street_df.shape == (35, 96)
        assert point_df.shape == (1277, 34)

    def test_no_plots(self):
        rioxarray = pytest.importorskip("rioxarray")

        dtm = rioxarray.open_rasterio(momepy.datasets.get_path("bubenec"), layer="dtm")
        sc = momepy.Streetscape(self.streets, self.buildings)
        sc.compute_slope(dtm)

        street_df = sc.street_level()
        point_df = sc.point_level()

        assert street_df.shape == (35, 81)
        assert point_df.shape == (1277, 26)

    def test_slope_without_sightline_points(self):
        slope = _compute_slope_from_arrays(
            np.array([0, 0]),
            np.array([3, 4]),
            10,
            15,
            np.empty((0, 2)),
            np.empty(0),
            -9999,
        )

        np.testing.assert_allclose(slope[:2], [45, 1])
        assert slope[2:] == (1, True)

    def test_all_values(self):
        rioxarray = pytest.importorskip("rioxarray")

        dtm = rioxarray.open_rasterio(momepy.datasets.get_path("bubenec"), layer="dtm")
        sc = momepy.Streetscape(
            self.streets, self.buildings, category_col="category", height_col="height"
        )
        sc.compute_plots(self.plots)
        sc.compute_slope(dtm)

        street_df = sc.street_level()
        point_df = sc.point_level()

        assert street_df.shape == (35, 106)
        assert point_df.shape == (1277, 34)

        np.testing.assert_allclose(
            street_df.drop(
                columns=[
                    "geometry",
                    "left_seq_sb_index",
                    "right_seq_sb_index",
                    "left_plot_seq_sb_index",
                    "right_plot_seq_sb_index",
                ]
            )
            .median()
            .to_numpy(),
            [
                40.0,
                1.0,
                5.0,
                49.52091382052582,
                40.36256501570919,
                69.83386039170725,
                3.030007045293202,
                12.059253149679153,
                10.21810727846502,
                0.9342180499746456,
                8.235235059863397,
                5.466598786768599,
                50.0,
                50.0,
                71.7417882891403,
                0.4790861794741746,
                5.219298794685249,
                3.600796457720469,
                40.48734963733774,
                21.572308177329184,
                13.986169015643167,
                0.0,
                0.20704299360657683,
                0.5525628119571071,
                0.0,
                0.17926085794362198,
                0.4308082829508534,
                40.5132645982187,
                20.579992651038328,
                11.529709821041505,
                0.0,
                0.17926085794362198,
                0.4306405791896738,
                12.272727272727275,
                13.309090909090907,
                16.65734265734266,
                0.0,
                0.0,
                0.13115783637106365,
                0.6678662504484538,
                0.9123068730095277,
                1.2807710270557002,
                0.0,
                0.050560668046915766,
                0.07389640863894119,
                139.05332097308423,
                600.0,
                0.0,
                40.0,
                7.628569795246345,
                1.3404001620791903,
                0.36363636363636365,
                0.8888888888888888,
                0.0,
                0.0,
                0.9420168067226891,
                0.8888888888888888,
                0.36363636363636365,
                0.8888888888888888,
                0.0,
                0.0,
                0.932967032967033,
                0.8888888888888888,
                0.007707360211947403,
                0.0222791933330667,
                0.042730470886721494,
                0.0062570850358775664,
                0.12697554970552732,
                0.19715820572676396,
                0.0,
                0.5,
                0.5,
                120.32016819730187,
                3.97464175672102e-05,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                35.0,
                35.0,
                10.200056087083096,
                10.531177039571478,
                9.657233436559752,
                0.08546898018857566,
                0.04661944373922308,
                0.03884953644935257,
                25.560049303992262,
                24.019915363283637,
                24.00729812249979,
                1.3470143628193392,
                1.0986002800935302,
                1.2473399517163581,
                0.18589437276353585,
                0.1866655749518955,
                0.20448652117425886,
                0.5716914086061647,
                0.009978565533223297,
                40.0,
                1.0,
            ],
        )

        np.testing.assert_allclose(
            point_df.drop(
                columns=[
                    "geometry",
                    "left_seq_sb_index",
                    "right_seq_sb_index",
                    "left_plot_seq_sb_index",
                    "right_plot_seq_sb_index",
                ]
            )
            .median()
            .to_numpy(),
            [
                1.0,
                50.0,
                0.0,
                10.676224480322063,
                19.04895104895105,
                1.4613218911757093,
                0.0,
                1.0,
                50.0,
                0.0,
                10.569809303648983,
                18.923076923076923,
                1.7138139154950125,
                0.0,
                300.0,
                300.0,
                9.463190880155867,
                24.97048079487916,
                9.296934793752289,
                23.15900597369601,
                2.0,
                37.110487558011485,
                1.0,
                10.706691344842483,
                19.23776223776224,
                1.723948224477644,
                6.432950514651803,
                9.885960548059535,
                24.735688410200858,
            ],
        )

        assert (
            street_df.columns
            == [
                "N",
                "n_l",
                "n_r",
                "left_os",
                "right_os",
                "os",
                "left_os_std",
                "right_os_std",
                "os_std",
                "left_os_mad",
                "right_os_mad",
                "os_mad",
                "left_os_med",
                "right_os_med",
                "os_med",
                "left_os_mad_med",
                "right_os_mad_med",
                "os_mad_med",
                "left_sb",
                "right_sb",
                "sb",
                "left_sb_std",
                "right_sb_std",
                "sb_std",
                "left_sb_mad",
                "right_sb_mad",
                "sb_mad",
                "left_sb_med",
                "right_sb_med",
                "sb_med",
                "left_sb_mad_med",
                "right_sb_mad_med",
                "sb_mad_med",
                "left_h",
                "right_h",
                "H",
                "left_h_std",
                "right_h_std",
                "H_std",
                "left_hw",
                "right_hw",
                "HW",
                "left_hw_std",
                "right_hw_std",
                "HW_std",
                "csosva",
                "tan",
                "tan_std",
                "n_tan_ratio",
                "tan_ratio",
                "tan_ratio_std",
                "par_tot",
                "par_rel",
                "left_par_tot",
                "right_par_tot",
                "left_par_rel",
                "right_par_rel",
                "par_tot_15",
                "par_rel_15",
                "left_par_tot_15",
                "right_par_tot_15",
                "left_par_rel_15",
                "right_par_rel_15",
                "left_built_freq",
                "right_built_freq",
                "built_freq",
                "left_built_coverage",
                "right_built_coverage",
                "built_coverage",
                "left_seq_sb_index",
                "right_seq_sb_index",
                "nodes_degree_1",
                "nodes_degree_4",
                "nodes_degree_3_5_plus",
                "street_length",
                "windingness",
                "building_prevalence[0]",
                "building_prevalence[1]",
                "building_prevalence[2]",
                "building_prevalence[3]",
                "building_prevalence[4]",
                "building_prevalence[5]",
                "left_plot_count",
                "right_plot_count",
                "plot_sb",
                "left_plot_sb",
                "right_plot_sb",
                "plot_freq",
                "left_plot_freq",
                "right_plot_freq",
                "plot_depth",
                "left_plot_depth",
                "right_plot_depth",
                "plot_WD_ratio",
                "left_plot_WD_ratio",
                "right_plot_WD_ratio",
                "plot_WP_ratio",
                "left_plot_WP_ratio",
                "right_plot_WP_ratio",
                "left_plot_seq_sb_index",
                "right_plot_seq_sb_index",
                "slope_degree",
                "slope_percent",
                "n_slopes",
                "slope_valid",
                "geometry",
            ]
        ).all()

        assert (
            point_df.columns
            == [
                "geometry",
                "left_os_count",
                "left_os",
                "left_sb_count",
                "left_sb",
                "left_h",
                "left_hw",
                "left_bc",
                "right_os_count",
                "right_os",
                "right_sb_count",
                "right_sb",
                "right_h",
                "right_hw",
                "right_bc",
                "front_sb",
                "back_sb",
                "left_seq_sb_index",
                "right_seq_sb_index",
                "left_plot_seq_sb",
                "left_plot_seq_sb_depth",
                "right_plot_seq_sb",
                "right_plot_seq_sb_depth",
                "left_plot_seq_sb_index",
                "right_plot_seq_sb_index",
                "os_count",
                "os",
                "sb_count",
                "sb",
                "h",
                "hw",
                "bc",
                "plot_seq_sb",
                "plot_seq_sb_depth",
            ]
        ).all()

    def test_first_building(self):
        # GH776
        streets = gpd.GeoDataFrame(geometry=[shapely.LineString([(0, 0), (60, 0)])])
        blds = [
            shapely.box(0, 5, 30, 20),
            shapely.box(30, 8, 60, 20),
            shapely.box(0, -30, 60, -6),
        ]
        b = gpd.GeoDataFrame({"height": [10.0] * len(blds)}, geometry=blds)
        pts = momepy.Streetscape(streets, b, height_col="height").point_level()
        expected = pd.Series(
            np.repeat([5.0, 8.0], 10),
            index=pd.Index(
                np.zeros(20, dtype=int),
                name="street_index",
            ),
            name="left_sb",
        )
        pd.testing.assert_series_equal(pts["left_sb"], expected)
