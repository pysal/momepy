import geopandas as gpd
import numpy as np
import pytest

import momepy


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
                49.52091382053271,
                40.36256501570919,
                69.83386039161877,
                3.0300070452496355,
                12.059253149635389,
                10.21810727846115,
                0.9342180499612095,
                8.235235059925058,
                5.466598786795851,
                50.0,
                50.0,
                71.7417882891403,
                0.47908617946728604,
                5.219298794549362,
                3.6007964573641646,
                40.487349637520765,
                21.087695047127575,
                13.986169016184006,
                0.0,
                0.10383427236125595,
                0.5186793716655227,
                0.0,
                0.08919263116228532,
                0.39624778346488937,
                40.513264598873654,
                20.579992651038328,
                11.52970982277211,
                0.0,
                0.08891354078509374,
                0.39268806547246066,
                12.272727272727275,
                13.309090909090907,
                16.65734265734266,
                0.0,
                0.0,
                0.111635608832882,
                0.6678662504155345,
                0.9123068730113607,
                1.28077102705215,
                0.0,
                0.017397828672440434,
                0.05759654889470471,
                139.05332097291637,
                600.0,
                0.0,
                40.0,
                7.628569795201426,
                1.3404001620798687,
                0.36363636363636365,
                0.898989898989899,
                0.0,
                0.0,
                0.9420168067226891,
                0.8903903903903904,
                0.36363636363636365,
                0.898989898989899,
                0.0,
                0.0,
                0.932967032967033,
                0.8903903903903904,
                0.007707360211947403,
                0.022279193333061414,
                0.04070216572227663,
                0.006257085035661018,
                0.1269755497050381,
                0.19715820572615364,
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
                10.200056087159462,
                10.531177039571478,
                9.65723343618015,
                0.0854689801885757,
                0.0466194437392231,
                0.03884953644935259,
                25.560049303845382,
                24.0199153637509,
                24.007298122345833,
                1.3470143628325295,
                1.0986002812885625,
                1.2473399517104464,
                0.18589437276314322,
                0.18666557494673167,
                0.2044865211771481,
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
                10.654982043105191,
                19.111888111888113,
                1.4711633374979423,
                0.0,
                1.0,
                50.0,
                0.0,
                10.535728185318764,
                19.04895104895105,
                1.7192769653490183,
                0.0,
                300.0,
                300.0,
                9.463190880206646,
                24.97048079338272,
                9.296934793860625,
                23.159005973830855,
                2.0,
                37.11048755809363,
                1.0,
                10.65171713340952,
                19.3006993006993,
                1.7635576674363311,
                6.4329505145817825,
                9.885960548533543,
                24.735688410357433,
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
