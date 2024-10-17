"""
Original implementation by Alessandro Araldi and the team of University of Cote d'Azur.

Adapted for use in momepy by Marek Novotny and Martin Fleischmann.

Note that the implementation in most cases follows the original code, resulting in
certain performance issues compared to the rest of momepy.
"""

import math
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely import LineString, MultiLineString, MultiPoint, Point

import momepy


class Streetscape:
    """Streetscape analysis based on sightlines

    TODO: explain what it does.

    This is a direct implementation of the algorithm proposed in
    :cite:`araldi2024multi`.

    Parameters
    ----------
    streets : gpd.GeoDataFrame
        GeoDataFrame containing LineString geometry representing streets
    buildings : gpd.GeoDataFrame
        GeoDataFrame containing Polygon geometry representing buildings
    sightline_length : float, optional
        length of the sightline generated at each sightline point perpendiculary to
        the street geometry, by default 50
    tangent_length : float, optional
        length of the sightline generated at each sightline point tangentially to
        the street geometry, by default 300
    sightline_spacing : float, optional
        approximate distance between sightline points generated along streets,
        by default 3
    intersection_offset : float, optional
        Offset to use at the beginning and the end of each LineString. The first
        sightline point is generated at this distance from the start and the last
        one is generated at this distance from the end of each geometry,
        by default 0.5
    angle_tolerance : float, optional
        Maximum angle between sightlines that does not require infill lines to be
        generated, by default 5
    height_col : str, optional
        name of a column of the buildings DataFrame containing the information
        about the building height in meters.
    category_col : str, optional
        name of a column of the buildings DataFrame containing the information
        about the building category encoded as integer labels.

    Examples
    --------
    Given only streets and buildings, you can already measure the majority of
    characters:

    >>> sc = momepy.Streetscape(streets, buildings)

    The resulting data can be extracted either on a street level:

    >>> street_df = sc.street_level()

    Or for all individual sightline points:

    >>> point_df = sc.point_level()

    If you have access to plots, you can additionally measure plot-based data:

    >>> sc.compute_plots(plots)

    If you have a digital terrain model, you can measure slope-based data:

    >>> sc.compute_slope(dtm)

    Notes
    -----
    momepy offers also a simplified way of anlysing streetscape using the
    :func:`momepy.street_profile` function. That is able to compute significantly less
    characters but is several orders of magnitude faster.
    """

    def __init__(
        self,
        streets: gpd.GeoDataFrame,
        buildings: gpd.GeoDataFrame,
        sightline_length: float = 50,
        tangent_length: float = 300,
        sightline_spacing: float = 3,
        intersection_offset: float = 0.5,
        angle_tolerance: float = 5,
        height_col: str | None = None,
        category_col: str | None = None,
    ) -> None:
        """Streetscape analysis based on sightlines"""
        self.sightline_length = sightline_length
        self.tangent_length = tangent_length
        self.sightline_spacing = sightline_spacing
        self.intersection_offset = intersection_offset
        self.angle_tolerance = angle_tolerance
        self.height_col = height_col
        self.category_col = category_col
        self.building_categories_count = (
            buildings[category_col].nunique() if category_col else 0
        )

        self.SIGHTLINE_LEFT = 0
        self.SIGHTLINE_RIGHT = 1
        self.SIGHTLINE_FRONT = 2
        self.SIGHTLINE_BACK = 3

        self.sightline_length_PER_SIGHT_TYPE = [
            sightline_length,
            sightline_length,
            tangent_length,
            tangent_length,
        ]

        streets = streets.copy()
        streets.geometry = shapely.force_2d(streets.geometry)

        nodes, edges = momepy.nx_to_gdf(
            momepy.node_degree(momepy.gdf_to_nx(streets, preserve_index=True))
        )
        edges["n1_degree"] = nodes.degree.loc[edges.node_start].values
        edges["n2_degree"] = nodes.degree.loc[edges.node_end].values
        edges["dead_end_left"] = edges["n1_degree"] == 1
        edges["dead_end_right"] = edges["n2_degree"] == 1
        edges["street_index"] = edges.index

        self.streets = edges

        buildings = buildings.copy()
        buildings["street_index"] = np.arange(len(buildings))
        self.buildings = buildings

        self.rtree_buildings = self.buildings.sindex

        self._compute_sightline_indicators_full()

    # return empty list if no sight line could be build du to total road length
    def _compute_sightlines(
        self,
        line: LineString,
        dead_end_start,
        dead_end_end,
    ):
        # FIRST PART : PERPENDICULAR SIGHTLINES #

        # Calculate the number of profiles to generate
        line_length = line.length

        remaining_length = line_length - 2 * self.intersection_offset
        if remaining_length < self.sightline_spacing:
            # no sight line
            return (
                gpd.GeoDataFrame(columns=["geometry", "point_id", "sight_type"]),
                [],
                [],
            )

        distances = [self.intersection_offset]
        nb_inter_nodes = int(math.floor(remaining_length / self.sightline_spacing))
        offset = remaining_length / nb_inter_nodes
        distance = self.intersection_offset

        for _ in range(0, nb_inter_nodes):
            distance = distance + offset
            distances.append(distance)

        results_sight_points = []
        results_sight_points_distances = []
        results_sightlines = []

        previous_sigh_line_left = None
        previous_sigh_line_right = None

        # semi_ortho_segment_size = self.sightline_spacing/2
        semi_ortho_segment_size = self.intersection_offset / 2

        sightline_index = 0

        last_pure_sightline_left_position_in_array = -1

        field_geometry = 0
        field_uid = 1

        # SECOND PART : TANGENT SIGHTLINES #

        # Start iterating along the line
        for distance in distances:
            # Get the start, mid and end points for this segment

            seg_st = line.interpolate(distance - semi_ortho_segment_size)
            seg_mid = line.interpolate(distance)
            seg_end = line.interpolate(distance + semi_ortho_segment_size)

            # Get a displacement vector for this segment
            vec = np.array(
                [
                    [
                        seg_end.x - seg_st.x,
                    ],
                    [
                        seg_end.y - seg_st.y,
                    ],
                ]
            )

            # Rotate the vector 90 deg clockwise and 90 deg counter clockwise
            rot_anti = np.array([[0, -1], [1, 0]])
            rot_clock = np.array([[0, 1], [-1, 0]])
            vec_anti = np.dot(rot_anti, vec)
            vec_clock = np.dot(rot_clock, vec)

            # Normalise the perpendicular vectors
            len_anti = ((vec_anti**2).sum()) ** 0.5
            vec_anti = vec_anti / len_anti
            len_clock = ((vec_clock**2).sum()) ** 0.5
            vec_clock = vec_clock / len_clock

            # Scale them up to the profile length
            vec_anti = vec_anti * self.sightline_length
            vec_clock = vec_clock * self.sightline_length

            # Calculate displacements from midpoint
            prof_st = (seg_mid.x + float(vec_anti[0]), seg_mid.y + float(vec_anti[1]))
            prof_end = (
                seg_mid.x + float(vec_clock[0]),
                seg_mid.y + float(vec_clock[1]),
            )

            results_sight_points.append(seg_mid)
            results_sight_points_distances.append(distance)

            sightline_left = LineString([seg_mid, prof_st])
            sightline_right = LineString([seg_mid, prof_end])

            # append LEFT sight line
            rec = [
                sightline_left,  # field_geometry
                sightline_index,  # field_uid
                self.SIGHTLINE_LEFT,  # FIELD_type
            ]
            results_sightlines.append(rec)

            # back up for dead end population
            last_pure_sightline_left_position_in_array = len(results_sightlines) - 1

            # append RIGHT sight line
            rec = [
                sightline_right,  # field_geometry
                sightline_index,  # field_uid
                self.SIGHTLINE_RIGHT,  # FIELD_type
            ]
            results_sightlines.append(rec)

            line_tan_back = LineString(
                [
                    seg_mid,
                    rotate(prof_end[0], prof_end[1], seg_mid.x, seg_mid.y, rad_90),
                ]
            )
            line_tan_front = LineString(
                [seg_mid, rotate(prof_st[0], prof_st[1], seg_mid.x, seg_mid.y, rad_90)]
            )

            # extends tanline to reach parametrized width
            line_tan_back = extend_line_end(line_tan_back, self.tangent_length)
            line_tan_front = extend_line_end(line_tan_front, self.tangent_length)

            # append tangent sigline front view
            rec = [
                line_tan_back,  # field_geometry
                sightline_index,  # FIELD_type
                self.SIGHTLINE_BACK,
            ]
            results_sightlines.append(rec)

            # append tangent sigline front view
            rec = [
                line_tan_front,  # field_geometry
                sightline_index,  # field_uid
                self.SIGHTLINE_FRONT,
            ]
            results_sightlines.append(rec)

            # THIRD PART: SIGHTLINE ENRICHMENT #

            # Populate lost space between consecutive sight lines with high deviation
            # (>angle_tolerance)
            if previous_sigh_line_left is not None:
                for this_line, prev_line, side in [
                    (sightline_left, previous_sigh_line_left, self.SIGHTLINE_LEFT),
                    (sightline_right, previous_sigh_line_right, self.SIGHTLINE_RIGHT),
                ]:
                    # angle between consecutive sight line
                    deviation = round(lines_angle(prev_line, this_line), 1)
                    # DEBUG_VALUES.append([this_line.coords[1],deviation])
                    # condition 1: large deviation
                    if abs(deviation) <= self.angle_tolerance:
                        continue
                    # condition 1: consecutive sight lines do not intersect

                    if this_line.intersects(prev_line):
                        continue

                    nb_new_sightlines = int(
                        math.floor(abs(deviation) / self.angle_tolerance)
                    )
                    nb_new_sightlines_this = nb_new_sightlines // 2
                    nb_new_sightlines_prev = nb_new_sightlines - nb_new_sightlines_this
                    delta_angle = deviation / (nb_new_sightlines)
                    theta_rad = np.deg2rad(delta_angle)

                    # add S2 new sight line on previous one
                    angle = 0
                    for _ in range(0, nb_new_sightlines_this):
                        angle -= theta_rad
                        x0 = this_line.coords[0][0]
                        y0 = this_line.coords[0][1]
                        x = this_line.coords[1][0]
                        y = this_line.coords[1][1]
                        new_line = LineString(
                            [this_line.coords[0], rotate(x, y, x0, y0, angle)]
                        )
                        rec = [
                            new_line,  # field_geometry
                            sightline_index,  # field_uid
                            side,  # FIELD_type
                        ]
                        results_sightlines.append(rec)

                        # add S2 new sight line on this current sight line
                    angle = 0
                    for _ in range(0, nb_new_sightlines_prev):
                        angle += theta_rad
                        x0 = prev_line.coords[0][0]
                        y0 = prev_line.coords[0][1]
                        x = prev_line.coords[1][0]
                        y = prev_line.coords[1][1]
                        new_line = LineString(
                            [prev_line.coords[0], rotate(x, y, x0, y0, angle)]
                        )
                        rec = [
                            new_line,  # field_geometry
                            sightline_index - 1,  # field_uid
                            side,  # FIELD_type
                        ]
                        results_sightlines.append(rec)

            # ==

            # iterate
            previous_sigh_line_left = sightline_left
            previous_sigh_line_right = sightline_right

            sightline_index += 1

        # ==
        # SPECIFIC ENRICHMENT FOR SIGHTPOINTS corresponding to DEAD ENDs
        # ==
        if dead_end_start or dead_end_end:
            for prev_sg, this_sg, dead_end in [
                (
                    results_sightlines[0],
                    results_sightlines[1],
                    dead_end_start,
                ),
                (
                    results_sightlines[last_pure_sightline_left_position_in_array + 1],
                    results_sightlines[last_pure_sightline_left_position_in_array],
                    dead_end_end,
                ),
            ]:
                if not dead_end:
                    continue
                # angle between consecutive dead end sight line LEFT and RIGHT (~180)
                prev_line = prev_sg[field_geometry]  # FIRST sight line LEFT side
                this_line = this_sg[field_geometry]  # FIRST sight line LEFT side

                # special case --> dead end .. so 180 °
                deviation = 180

                nb_new_sightlines = int(
                    math.floor(abs(deviation) / self.angle_tolerance)
                )
                nb_new_sightlines_this = nb_new_sightlines // 2
                nb_new_sightlines_prev = nb_new_sightlines - nb_new_sightlines_this
                delta_angle = deviation / (nb_new_sightlines)
                theta_rad = np.deg2rad(delta_angle)

                # add S2 new sight line on previous one
                angle = 0
                for _ in range(0, nb_new_sightlines_this):
                    angle -= theta_rad
                    x0 = this_line.coords[0][0]
                    y0 = this_line.coords[0][1]
                    x = this_line.coords[1][0]
                    y = this_line.coords[1][1]
                    new_line = LineString(
                        [this_line.coords[0], rotate(x, y, x0, y0, angle)]
                    )

                    rec = [
                        new_line,  # field_geometry
                        this_sg[field_uid],  # field_uid
                        self.SIGHTLINE_LEFT,
                    ]
                    results_sightlines.append(rec)

                    # add S2 new sight line on this current sight line
                angle = 0
                for _ in range(0, nb_new_sightlines_prev):
                    angle += theta_rad
                    x0 = prev_line.coords[0][0]
                    y0 = prev_line.coords[0][1]
                    x = prev_line.coords[1][0]
                    y = prev_line.coords[1][1]
                    new_line = LineString(
                        [prev_line.coords[0], rotate(x, y, x0, y0, angle)]
                    )
                    rec = [
                        new_line,  # field_geometry
                        prev_sg[field_uid],  # field_uid
                        self.SIGHTLINE_RIGHT,
                    ]
                    results_sightlines.append(rec)
            # ==
        return (
            gpd.GeoDataFrame(
                results_sightlines, columns=["geometry", "point_id", "sight_type"]
            ),
            results_sight_points,
            results_sight_points_distances,
        )

    def _compute_sigthlines_indicators(self, street_row, optimize_on=True):
        street_uid = street_row.street_index
        street_geom = street_row.geometry

        gdf_sightlines, sightlines_points, results_sight_points_distances = (
            self._compute_sightlines(
                street_geom, street_row.dead_end_left, street_row.dead_end_right
            )
        )

        # per street sightpoints indicators
        current_street_uid = street_uid
        current_street_sightlines_points = sightlines_points
        current_street_left_os_count = []
        current_street_left_os = []
        current_street_left_sb_count = []
        current_street_left_sb = []
        current_street_left_h = []
        current_street_left_hw = []
        current_street_right_os_count = []
        current_street_right_os = []
        current_street_right_sb_count = []
        current_street_right_sb = []
        current_street_right_h = []
        current_street_right_hw = []

        current_street_left_bc = []
        current_street_right_bc = []

        # SPARSE STORAGE (one value if set back is OK ever in intersightline)
        current_street_left_seq_sb_ids = []
        current_street_left_seq_sb_categories = []
        current_street_right_seq_sb_ids = []
        current_street_right_seq_sb_categories = []

        current_street_front_sb = []
        current_street_back_sb = []

        # [Expanded] each time a sight line or intersight line occured
        left_seq_sightlines_end_points = []
        right_seq_sightlines_end_points = []

        if sightlines_points is None:
            current_street_sightlines_points = []
            return [
                current_street_uid,
                current_street_sightlines_points,
                current_street_left_os_count,
                current_street_left_os,
                current_street_left_sb_count,
                current_street_left_sb,
                current_street_left_h,
                current_street_left_hw,
                current_street_left_bc,
                current_street_left_seq_sb_ids,
                current_street_left_seq_sb_categories,
                current_street_right_os_count,
                current_street_right_os,
                current_street_right_sb_count,
                current_street_right_sb,
                current_street_right_h,
                current_street_right_hw,
                current_street_right_bc,
                current_street_right_seq_sb_ids,
                current_street_right_seq_sb_categories,
                current_street_front_sb,
                current_street_back_sb,
                left_seq_sightlines_end_points,
                right_seq_sightlines_end_points,
            ], None

        # ------- SIGHT LINES
        # Extract building in SIGHTLINES buffer (e.g: 50m)

        # iterate throught sightlines groups.
        # Eeach sigh points could have many sub sighpoint in case of snail effect)
        for _, group in gdf_sightlines.groupby("point_id"):
            front_sl_tan_sb = self.tangent_length
            back_sl_tan_sb = self.tangent_length
            left_sl_count = 0
            left_sl_distance_total = 0
            left_sl_building_count = 0
            left_sl_building_sb_total = 0
            left_sl_building_sb_height_total = 0
            right_sl_count = 0
            right_sl_distance_total = 0
            right_sl_building_count = 0
            right_sl_building_sb_total = 0
            right_sl_building_sb_height_total = 0

            left_sl_cr_total = 0
            right_sl_cr_total = 0

            # iterate throught each sightline links to the sigh point:
            # LEFT(1-*),RIGHT(1-*),FRONT(1), BACK(1)
            for row_s in group.itertuples(index=False):
                sightline_geom = row_s.geometry
                sightline_side = row_s.sight_type
                sightline_length = self.sightline_length_PER_SIGHT_TYPE[sightline_side]
                # extract possible candidates
                if optimize_on and sightline_side >= self.SIGHTLINE_FRONT:
                    # = OPTIM TEST
                    # cut tan line in 3 block (~100m)
                    length_3 = sightline_geom.length / 3.0
                    a = sightline_geom.coords[0]
                    b = sightline_geom.coords[-1]
                    end_points = [
                        sightline_geom.interpolate(length_3),
                        sightline_geom.interpolate(length_3 * 2),
                        b,
                    ]

                    gdf_sightline_buildings = None
                    start_point = a
                    for end_point in end_points:
                        sub_line = LineString([start_point, end_point])
                        gdf_sightline_buildings = self.buildings.iloc[
                            self.rtree_buildings.query(sub_line, predicate="intersects")
                        ]
                        if len(gdf_sightline_buildings) > 0:
                            break
                        start_point = end_point
                else:
                    gdf_sightline_buildings = self.buildings.iloc[
                        self.rtree_buildings.query(
                            sightline_geom, predicate="intersects"
                        )
                    ]

                s_pt1 = shapely.get_point(sightline_geom, 0)
                endpoint = shapely.get_point(sightline_geom, -1)

                # agregate
                match_sl_distance = (
                    sightline_length  # set max distance if no polygon intersect
                )
                match_sl_building_id = None
                match_sl_building_category = None
                match_sl_building_height = 0

                sl_cr_total = 0
                for _, res in gdf_sightline_buildings.iterrows():
                    # building geom
                    geom = res.geometry
                    isect = sightline_geom.intersection(geom.exterior)
                    if not isect.is_empty:
                        dist = s_pt1.distance(isect)
                        if dist < match_sl_distance:
                            match_sl_distance = dist
                            match_sl_building_id = res.street_index
                            match_sl_building_height = (
                                res[self.height_col] if self.height_col else np.nan
                            )
                            match_sl_building_category = (
                                res[self.category_col] if self.category_col else None
                            )

                        # coverage ratio between sight line and candidate building
                        # (geom: building geom)
                        _coverage_isec = sightline_geom.intersection(geom)
                        # display(type(coverage_isec))
                        sl_cr_total += _coverage_isec.length

                if sightline_side == self.SIGHTLINE_LEFT:
                    left_sl_count += 1
                    left_seq_sightlines_end_points.append(endpoint)
                    left_sl_distance_total += match_sl_distance
                    left_sl_cr_total += sl_cr_total
                    if match_sl_building_id:
                        left_sl_building_count += 1
                        left_sl_building_sb_total += match_sl_distance
                        left_sl_building_sb_height_total += match_sl_building_height
                        # PREVALENCE: Emit each time a new setback or INTER-setback is
                        # found (campact storage structure)
                        current_street_left_seq_sb_ids.append(match_sl_building_id)
                        current_street_left_seq_sb_categories.append(
                            match_sl_building_category
                        )

                elif sightline_side == self.SIGHTLINE_RIGHT:
                    right_sl_count += 1
                    right_seq_sightlines_end_points.append(endpoint)
                    right_sl_distance_total += match_sl_distance
                    right_sl_cr_total += sl_cr_total
                    if match_sl_building_id:
                        right_sl_building_count += 1
                        right_sl_building_sb_total += match_sl_distance
                        right_sl_building_sb_height_total += match_sl_building_height
                        # PREVALENCE: Emit each time a new setback or INTER-setback is
                        # found (campact storage structure)
                        current_street_right_seq_sb_ids.append(match_sl_building_id)
                        current_street_right_seq_sb_categories.append(
                            match_sl_building_category
                        )

                elif sightline_side == self.SIGHTLINE_BACK:
                    back_sl_tan_sb = match_sl_distance
                elif sightline_side == self.SIGHTLINE_FRONT:
                    front_sl_tan_sb = match_sl_distance

            # LEFT
            left_os_count = left_sl_count
            left_os = left_sl_distance_total / left_os_count
            left_sb_count = left_sl_building_count
            left_sb = np.nan
            left_h = np.nan
            left_hw = np.nan
            if left_sb_count != 0:
                left_sb = left_sl_building_sb_total / left_sb_count
                left_h = left_sl_building_sb_height_total / left_sb_count
                # HACk if sb = 0 --> 10cm
                left_hw = left_h / max(left_sb, 0.1)
            left_cr = left_sl_cr_total / left_os_count
            # RIGHT
            right_os_count = right_sl_count
            right_os = right_sl_distance_total / right_os_count
            right_sb_count = right_sl_building_count
            right_sb = np.nan
            right_h = np.nan
            right_hw = np.nan
            if right_sb_count != 0:
                right_sb = right_sl_building_sb_total / right_sb_count
                right_h = right_sl_building_sb_height_total / right_sb_count
                # HACk if sb = 0 --> 10cm
                right_hw = right_h / max(right_sb, 0.1)
            right_cr = right_sl_cr_total / right_os_count

            current_street_left_os_count.append(left_os_count)
            current_street_left_os.append(left_os)
            current_street_left_sb_count.append(left_sb_count)
            current_street_left_sb.append(left_sb)
            current_street_left_h.append(left_h)
            current_street_left_hw.append(left_hw)
            current_street_right_os_count.append(right_os_count)
            current_street_right_os.append(right_os)
            current_street_right_sb_count.append(right_sb_count)
            current_street_right_sb.append(right_sb)
            current_street_right_h.append(right_h)
            current_street_right_hw.append(right_hw)
            # FRONT / BACK
            current_street_front_sb.append(front_sl_tan_sb)
            current_street_back_sb.append(back_sl_tan_sb)
            # COverage ratio Built up
            current_street_left_bc.append(left_cr)
            current_street_right_bc.append(right_cr)

        return [
            current_street_uid,
            current_street_sightlines_points,
            current_street_left_os_count,
            current_street_left_os,
            current_street_left_sb_count,
            current_street_left_sb,
            current_street_left_h,
            current_street_left_hw,
            current_street_left_bc,
            current_street_left_seq_sb_ids,
            current_street_left_seq_sb_categories,
            current_street_right_os_count,
            current_street_right_os,
            current_street_right_sb_count,
            current_street_right_sb,
            current_street_right_h,
            current_street_right_hw,
            current_street_right_bc,
            current_street_right_seq_sb_ids,
            current_street_right_seq_sb_categories,
            current_street_front_sb,
            current_street_back_sb,
            left_seq_sightlines_end_points,
            right_seq_sightlines_end_points,
        ], gdf_sightlines

    def _compute_sightline_indicators_full(self):
        values = []

        for street_row in self.streets[
            ["street_index", "geometry", "dead_end_left", "dead_end_right"]
        ].itertuples(index=False):
            indicators, _ = self._compute_sigthlines_indicators(street_row)
            values.append(indicators)

        df = pd.DataFrame(
            values,
            columns=[
                "street_index",
                "sightline_points",
                "left_os_count",
                "left_os",
                "left_sb_count",
                "left_sb",
                "left_h",
                "left_hw",
                "left_bc",
                "left_seq_sb_ids",
                "left_seq_sb_categories",
                "right_os_count",
                "right_os",
                "right_sb_count",
                "right_sb",
                "right_h",
                "right_hw",
                "right_bc",
                "right_seq_sb_ids",
                "right_seq_sb_categories",
                "front_sb",
                "back_sb",
                "left_seq_os_endpoints",
                "right_seq_os_endpoints",
            ],
        )
        df = df.set_index("street_index")

        df["nodes_degree_1"] = self.streets.apply(
            lambda row: (
                (1 if row.n1_degree == 1 else 0) + (1 if row.n2_degree == 1 else 0)
            )
            / 2,
            axis=1,
        )

        df["nodes_degree_4"] = self.streets.apply(
            lambda row: (
                (1 if row.n1_degree == 4 else 0) + (1 if row.n2_degree == 4 else 0)
            )
            / 2,
            axis=1,
        )

        df["nodes_degree_3_5_plus"] = self.streets.apply(
            lambda row: (
                (1 if row.n1_degree == 3 or row.n1_degree >= 5 else 0)
                + (1 if row.n2_degree == 3 or row.n2_degree >= 5 else 0)
            )
            / 2,
            axis=1,
        )
        df["street_length"] = self.streets.length
        df["windingness"] = 1 - momepy.linearity(self.streets)

        self._sightline_indicators = df

    def _compute_sigthlines_plot_indicators_one_side(
        self, sightline_points, os_count, seq_os_endpoint
    ):
        parcel_sb_count = []
        parcel_seq_sb_ids = []
        parcel_seq_sb = []
        parcel_seq_sb_depth = []

        n = len(sightline_points)
        if n == 0:
            parcel_sb_count = [0] * n
            return [
                parcel_sb_count,
                parcel_seq_sb_ids,
                parcel_seq_sb,
                parcel_seq_sb_depth,
            ]

        idx_end_point = 0

        for sight_point, os_count_ in zip(sightline_points, os_count, strict=False):
            n_sightlines_touching = 0
            for _ in range(os_count_):
                sightline_geom = LineString(
                    [sight_point, seq_os_endpoint[idx_end_point]]
                )
                s_pt1 = Point(sightline_geom.coords[0])

                gdf_items = self.plots.iloc[
                    self.rtree_parcels.query(sightline_geom, predicate="intersects")
                ]

                match_distance = (
                    self.sightline_length  # set max distance if no polygon intersect
                )
                match_id = None
                match_geom = None

                if not gdf_items.empty:
                    _distances = gdf_items.exterior.intersection(
                        sightline_geom
                    ).distance(s_pt1)
                    match_id = _distances.idxmin()
                    match_distance = _distances.min()
                    match_geom = gdf_items.geometry[match_id]

                # ---------------
                # result in intersightline
                if match_id is not None:
                    n_sightlines_touching += 1
                    parcel_seq_sb_ids.append(match_id)
                    parcel_seq_sb.append(match_distance)
                    # compute depth of plot intersect sighline etendue
                    if not match_geom.is_valid:
                        match_geom = match_geom.buffer(0)
                    isec = match_geom.intersection(
                        extend_line_end(
                            sightline_geom, self.sightline_plot_depth_extension
                        )
                    )
                    if (not isinstance(isec, LineString)) and (
                        not isinstance(isec, MultiLineString)
                    ):
                        raise Exception("Not allowed: intersection is not of type Line")
                    parcel_seq_sb_depth.append(isec.length)

                # ------- iterate
                idx_end_point += 1

            parcel_sb_count.append(n_sightlines_touching)

        return [parcel_sb_count, parcel_seq_sb_ids, parcel_seq_sb, parcel_seq_sb_depth]

    def compute_plots(
        self, plots: gpd.GeoDataFrame, sightline_plot_depth_extension: float = 300
    ) -> None:
        """Compute plot-based characters

        Parameters
        ----------
        plots : gpd.GeoDataFrame
            plots represented as polygons
        sightline_plot_depth_extension : float, optional
            depth of the sightline extension, by default 300

        Examples
        --------
        >>> sc = momepy.Streetscape(streets, buildings)
        >>> sc.compute_plots(plots)
        """
        self.sightline_plot_depth_extension = sightline_plot_depth_extension

        self.rtree_parcels = plots.sindex
        plots = plots.copy()
        plots["parcel_id"] = np.arange(len(plots))
        self.plots = plots
        self.plots["perimeter"] = self.plots.length

        values = []

        for uid, row in self._sightline_indicators.iterrows():
            sightline_values = [uid]

            side_values = self._compute_sigthlines_plot_indicators_one_side(
                row.sightline_points, row.left_os_count, row.left_seq_os_endpoints
            )
            sightline_values += side_values

            side_values = self._compute_sigthlines_plot_indicators_one_side(
                row.sightline_points, row.right_os_count, row.right_seq_os_endpoints
            )
            sightline_values += side_values

            values.append(sightline_values)

        df = pd.DataFrame(
            values,
            columns=[
                "street_index",
                "left_parcel_sb_count",
                "left_parcel_seq_sb_ids",
                "left_parcel_seq_sb",
                "left_parcel_seq_sb_depth",
                "right_parcel_sb_count",
                "right_parcel_seq_sb_ids",
                "right_parcel_seq_sb",
                "right_parcel_seq_sb_depth",
            ],
        )
        df = df.set_index("street_index").join(self._sightline_indicators.street_length)

        self._plot_indicators = df

    def _aggregate_plots(self):
        values = []

        for street_uid, row in self._plot_indicators.iterrows():
            left_parcel_sb_count = row.left_parcel_sb_count
            left_parcel_seq_sb_ids = row.left_parcel_seq_sb_ids
            left_parcel_seq_sb = row.left_parcel_seq_sb
            left_parcel_seq_sb_depth = row.left_parcel_seq_sb_depth
            right_parcel_sb_count = row.right_parcel_sb_count
            right_parcel_seq_sb_ids = row.right_parcel_seq_sb_ids
            right_parcel_seq_sb = row.right_parcel_seq_sb
            right_parcel_seq_sb_depth = row.right_parcel_seq_sb_depth
            street_length = row.street_length

            n = len(left_parcel_sb_count)
            if n == 0:
                values.append(
                    [
                        street_uid,
                        0,
                        0,  # np_l, np_r
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ]
                )
                continue

            left_parcel_seq_sb_depth = [
                d if d >= 1 else 1 for d in left_parcel_seq_sb_depth
            ]
            right_parcel_seq_sb_depth = [
                d if d >= 1 else 1 for d in right_parcel_seq_sb_depth
            ]

            left_unique_ids = set(left_parcel_seq_sb_ids)
            right_unique_ids = set(right_parcel_seq_sb_ids)
            all_unique_ids = left_unique_ids.union(right_unique_ids)

            left_parcel_freq = len(left_unique_ids) / street_length
            right_parcel_freq = len(right_unique_ids) / street_length
            parcel_freq = len(all_unique_ids) / street_length

            # compute sightline weights
            left_sight_weight = []
            # iterate all sight point
            for sb_count in left_parcel_sb_count:
                if sb_count != 0:
                    w = 1.0 / sb_count
                    for _ in range(sb_count):
                        left_sight_weight.append(w)

            right_sight_weight = []
            # iterate all sight point
            for sb_count in right_parcel_sb_count:
                if sb_count != 0:
                    w = 1.0 / sb_count
                    for _ in range(sb_count):
                        right_sight_weight.append(w)

            # build depth dataframe with interzsighline weight
            df_depth = [
                [parcel_id, w, sb, depth, self.SIGHTLINE_LEFT]
                for parcel_id, w, sb, depth in zip(
                    left_parcel_seq_sb_ids,
                    left_sight_weight,
                    left_parcel_seq_sb,
                    left_parcel_seq_sb_depth,
                    strict=False,
                )
            ]
            df_depth += [
                [parcel_id, w, sb, depth, self.SIGHTLINE_RIGHT]
                for parcel_id, w, sb, depth in zip(
                    right_parcel_seq_sb_ids,
                    right_sight_weight,
                    right_parcel_seq_sb,
                    right_parcel_seq_sb_depth,
                    strict=False,
                )
            ]

            df_depth = pd.DataFrame(
                df_depth, columns=["parcel_id", "w", "sb", "depth", "side"]
            ).set_index("parcel_id")
            df_depth["w_sb"] = df_depth.w * df_depth.sb
            df_depth["w_depth"] = df_depth.w * df_depth.depth

            df_depth_left = df_depth[df_depth.side == self.SIGHTLINE_LEFT]
            df_depth_right = df_depth[df_depth.side == self.SIGHTLINE_RIGHT]

            np_l = int(df_depth_left.w.sum())
            np_r = int(df_depth_right.w.sum())
            np_lr = np_l + np_r

            left_parcel_sb = (
                df_depth_left.w_sb.sum() / np_l if np_l > 0 else self.sightline_length
            )
            right_parcel_sb = (
                df_depth_right.w_sb.sum() / np_r if np_r > 0 else self.sightline_length
            )
            parcel_sb = (
                df_depth.w_sb.sum() / np_lr if np_lr > 0 else self.sightline_length
            )

            left_parcel_depth = df_depth_left.w_depth.sum() / np_l if np_l > 0 else 0
            right_parcel_depth = df_depth_right.w_depth.sum() / np_r if np_r > 0 else 0
            parcel_depth = df_depth.w_depth.sum() / np_lr if np_lr > 0 else 0

            wd_ratio_list = []
            wp_ratio_list = []
            # TODO: this thing is pretty terrible and needs to be completely redone
            # It is a massive bottleneck
            for df in [df_depth, df_depth_left, df_depth_right]:
                if len(df) == 0:
                    wd_ratio_list.append(0)
                    wp_ratio_list.append(0)
                    continue

                df = (
                    df[["w", "w_depth"]]
                    .groupby(level=0)
                    .aggregate(
                        nb=pd.NamedAgg(column="w", aggfunc=len),
                        w_sum=pd.NamedAgg(column="w", aggfunc="sum"),
                        w_depth=pd.NamedAgg(column="w_depth", aggfunc="mean"),
                    )
                )

                df = df.join(self.plots.perimeter)
                sum_nb = df.nb.sum()

                wd_ratio = (
                    (df.w_sum * self.sightline_spacing * df.nb) / df.w_depth
                ).sum() / sum_nb
                wp_ratio = (
                    (df.w_sum * self.sightline_spacing * df.nb) / df.perimeter
                ).sum() / sum_nb
                wd_ratio_list.append(wd_ratio)
                wp_ratio_list.append(wp_ratio)

            values.append(
                [
                    street_uid,
                    np_l,
                    np_r,
                    parcel_sb,
                    left_parcel_sb,
                    right_parcel_sb,
                    parcel_freq,
                    left_parcel_freq,
                    right_parcel_freq,
                    parcel_depth,
                    left_parcel_depth,
                    right_parcel_depth,
                ]
                + wd_ratio_list
                + wp_ratio_list
            )

        columns = [
            "uid",
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
        ]

        self._aggregate_plot_data = pd.DataFrame(values, columns=columns).set_index(
            "uid"
        )

    def _compute_slope(self, road_row):
        start = road_row.sl_start  # Point z
        end = road_row.sl_end  # Point z
        slp = road_row.sl_points  # Multipoint z

        if slp is None:
            # Case when there is no sight line point (e.g. when the road is too short)
            # just computes slope between start and end
            if start.z == self.NODATA_RASTER or end.z == self.NODATA_RASTER:
                # Case when there is at least one invalid z coord
                return 0, 0, 0, False
            slope_percent = abs(start.z - end.z) / shapely.distance(start, end)
            slope_degree = math.degrees(math.atan(slope_percent))

            return slope_percent, slope_degree, 1, True

        # From Multipoint z to Point z list
        slp_list = list(slp.geoms)

        points = []

        points.append(start)
        # From Point z list to all points list
        for p in slp_list:
            points.append(p)
        points.append(end)

        # number of points
        nb_points = len([start]) + len([end]) + len(slp_list)

        # temporary variables to store inter slope values
        sum_slope_percent = 0
        sum_slope_radian = 0
        sum_nb_points = 0

        # if there is one or more sight line points
        for i in range(1, nb_points - 1):
            a = points[i - 1]
            b = points[i + 1]

            if a.z == self.NODATA_RASTER or b.z == self.NODATA_RASTER:
                # Case when there is no valid z coord in slpoint
                continue

            sum_nb_points += 1
            inter_slope_percent = abs(a.z - b.z) / shapely.distance(a, b)

            sum_slope_percent += inter_slope_percent
            sum_slope_radian += math.atan(inter_slope_percent)

        if sum_nb_points == 0:
            # Case when no slpoint has a valid z coord
            # Unable to compute slope
            return 0, 0, 0, False

        # compute mean of inter slopes
        slope_percent = sum_slope_percent / sum_nb_points
        slope_degree = math.degrees(sum_slope_radian / sum_nb_points)

        return slope_degree, slope_percent, sum_nb_points, True

    def compute_slope(self, raster) -> None:
        """Compute slope-based characters

        Requires Xarray and Xvec packages.

        Parameters
        ----------
        raster : xarray.DataArray
            xarray.DataArray object (optimally read via rioxarray), supported by Xvec's
            ``extract_points`` method, representing digital terrain model.

        Examples
        --------
        >>> sc = momepy.Streetscape(streets, buildings)
        >>> sc.compute_slope(dtm)
        """
        try:
            import rioxarray  # noqa: F401
            import xvec  # noqa: F401
        except ImportError as err:
            raise ImportError(
                "compute_slope requires rioxarray and xvec packages."
            ) from err

        self.NODATA_RASTER = raster.rio.nodata

        start_points = shapely.get_point(self.streets.geometry, 0)
        end_points = shapely.get_point(self.streets.geometry, -1)

        # Extract z coords from raster
        z_start = (
            raster.drop_vars("spatial_ref")
            .xvec.extract_points(points=start_points, x_coords="x", y_coords="y")
            .xvec.to_geopandas()
        )
        z_start = z_start.rename(
            columns={k: "z" for k in z_start.columns.drop("geometry")}
        )

        # Append z values to points
        z_start["start_point_3d"] = shapely.points(
            *shapely.get_coordinates(start_points.geometry).T, z=z_start["z"]
        )

        # Extract z coords from raster
        z_end = (
            raster.drop_vars("spatial_ref")
            .xvec.extract_points(points=end_points, x_coords="x", y_coords="y")
            .xvec.to_geopandas()
        )
        z_end = z_end.rename(columns={k: "z" for k in z_end.columns.drop("geometry")})

        # Append z values to points
        z_end["end_point_3d"] = shapely.points(
            *shapely.get_coordinates(end_points.geometry).T, z=z_end["z"]
        )

        z_points_list = []

        for row in self._sightline_indicators["sightline_points"].apply(
            lambda x: MultiPoint(x) if x else None
        ):
            if row is not None:
                points = row.geoms

                z_points = (
                    raster.drop_vars("spatial_ref")
                    .xvec.extract_points(points=points, x_coords="x", y_coords="y")
                    .xvec.to_geopandas()
                )
                z_points = z_points.rename(
                    columns={k: "z" for k in z_points.columns.drop("geometry")}
                )

                z_points["geometry"] = shapely.points(
                    *shapely.get_coordinates(z_points.geometry).T, z=z_points["z"]
                )
                z_points = z_points.drop(columns="z")

                multipoint = MultiPoint(z_points["geometry"].tolist())

            else:
                multipoint = None

            z_points_list.append(multipoint)

        sightlines = pd.concat(
            [z_start[["start_point_3d"]], z_end[["end_point_3d"]]], axis=1
        )

        sightlines = sightlines.rename(
            columns={"start_point_3d": "sl_start", "end_point_3d": "sl_end"}
        )

        sightlines["sl_points"] = z_points_list

        slope_values = []

        for _, road_row in sightlines.iterrows():
            slope_degree, slope_percent, n_slopes, slope_valid = self._compute_slope(
                road_row
            )

            slope_values.append([slope_degree, slope_percent, n_slopes, slope_valid])

        self.slope = pd.DataFrame(
            slope_values,
            columns=["slope_degree", "slope_percent", "n_slopes", "slope_valid"],
        )

    # 0.5 contribution if parralel with previous sightpoint setback
    # 0.5 contribution if parralel with next sightpoint setback
    def _compute_parallelism_factor(self, side_sb, side_sb_count, max_distance=999):
        if side_sb_count is None or len(side_sb_count) == 0:
            return []
        is_parralel_with_next = []
        for sb_a, sb_a_count, sb_b, sb_b_count in zip(
            side_sb[0:-1],
            side_sb_count[0:-1],
            side_sb[1:],
            side_sb_count[1:],
            strict=False,
        ):
            if sb_a_count == 0 or sb_b_count == 0:
                is_parralel_with_next.append(False)
                continue
            if max_distance is None or max(sb_a, sb_b) <= max_distance:
                is_parralel_with_next.append(
                    abs(sb_a - sb_b) < self.sightline_spacing / 3
                )
            else:
                is_parralel_with_next.append(False)
        # choice for last point
        is_parralel_with_next.append(False)

        result = []
        prev_parralel = False
        for next_parralel in is_parralel_with_next:
            # Ajouter condition su
            factor = 0
            if prev_parralel:  # max_distance
                # STOP
                factor += 0.5
            if next_parralel:
                factor += 0.5
            result.append(factor)
            prev_parralel = next_parralel

        return result

    def _compute_parallelism_indicators(
        self,
        left_sb,
        left_sb_count,
        right_sb,
        right_sb_count,
        n,
        n_l,
        n_r,
        max_distance=None,
    ):
        parallel_left_factors = self._compute_parallelism_factor(
            left_sb, left_sb_count, max_distance
        )
        parallel_right_factors = self._compute_parallelism_factor(
            right_sb, right_sb_count, max_distance
        )

        parallel_left_total = sum(parallel_left_factors)
        parallel_right_total = sum(parallel_right_factors)

        ind_left_par_tot = parallel_left_total / (n - 1) if n > 1 else np.nan
        ind_left_par_rel = parallel_left_total / (n_l - 1) if n_l > 1 else np.nan

        ind_right_par_tot = parallel_right_total / (n - 1) if n > 1 else np.nan
        ind_right_par_rel = parallel_right_total / (n_r - 1) if n_r > 1 else np.nan

        ind_par_tot = np.nan
        if n > 1:
            ind_par_tot = (parallel_left_total + parallel_right_total) / (2 * n - 2)

        ind_par_rel = np.nan
        if n_l > 1 or n_r > 1:
            ind_par_rel = (parallel_left_total + parallel_right_total) / (
                max(1, n_l) + max(1, n_r) - 2
            )

        return (
            ind_left_par_tot,
            ind_left_par_rel,
            ind_right_par_tot,
            ind_right_par_rel,
            ind_par_tot,
            ind_par_rel,
        )

    def street_level(self) -> gpd.GeoDataFrame:
        """Extract data on a street level.

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame with streetscape data linked to street geometry.

        Examples
        --------
        >>> sc = momepy.Streetscape(streets, buildings)
        >>> sc.steet_level()
        """
        values = []

        for street_uid, row in self._sightline_indicators.iterrows():
            street_length = row.street_length

            left_os_count = row.left_os_count
            left_os = row.left_os
            left_sb_count = row.left_sb_count
            left_sb = row.left_sb
            left_h = row.left_h
            left_hw = row.left_hw
            right_os = row.right_os
            right_sb_count = row.right_sb_count
            right_sb = row.right_sb
            right_h = row.right_h
            right_hw = row.right_hw

            left_bc = row.left_bc
            left_seq_sb_ids = row.left_seq_sb_ids

            right_bc = row.right_bc
            right_seq_sb_ids = row.right_seq_sb_ids

            front_sb = row.front_sb
            back_sb = row.back_sb

            n = len(left_os_count)
            if n == 0:
                continue

            # ------------------------
            # OPENNESS
            # ------------------------
            sum_left_os = np.sum(left_os)
            sum_right_os = np.sum(right_os)

            ind_left_os = sum_left_os / n
            ind_right_os = sum_right_os / n
            ind_os = ind_left_os + ind_right_os  # ==(left_os+right_os)/n

            full_os = [le + r for le, r in zip(left_os, right_os, strict=False)]
            # mediane >> med
            ind_left_os_med = np.median(left_os)
            ind_right_os_med = np.median(right_os)
            ind_os_med = np.median(full_os)

            # OPENNESS ROUGHNESS
            sum_square_error_left_os = np.sum(
                [(os - ind_left_os) ** 2 for os in left_os]
            )
            sum_square_error_right_os = np.sum(
                [(os - ind_right_os) ** 2 for os in right_os]
            )
            sum_abs_error_left_os = np.sum([abs(os - ind_left_os) for os in left_os])
            sum_abs_error_right_os = np.sum([abs(os - ind_right_os) for os in right_os])
            ind_os_std = math.sqrt(
                (sum_square_error_left_os + sum_square_error_right_os) / (2 * n - 1)
            )
            ind_os_mad = (sum_abs_error_left_os + sum_abs_error_right_os) / (2 * n)

            ind_left_os_std = 0  # default
            ind_right_os_std = 0  # default
            ind_left_os_mad = 0  # default
            ind_right_os_mad = 0  # default

            ind_left_os_mad = sum_abs_error_left_os / n
            ind_right_os_mad = sum_abs_error_right_os / n
            if n > 1:
                ind_left_os_std = math.sqrt((sum_square_error_left_os) / (n - 1))
                ind_right_os_std = math.sqrt((sum_square_error_right_os) / (n - 1))

            sum_abs_error_left_os_med = np.sum(
                [abs(os - ind_left_os_med) for os in left_os]
            )
            sum_abs_error_right_os_med = np.sum(
                [abs(os - ind_right_os_med) for os in right_os]
            )
            ind_left_os_mad_med = sum_abs_error_left_os_med / n
            ind_right_os_mad_med = sum_abs_error_right_os_med / n
            ind_os_mad_med = (
                sum_abs_error_left_os_med + sum_abs_error_right_os_med
            ) / (2 * n)

            # ------------------------
            # SETBACK
            # ------------------------
            rel_left_sb = [x for x in left_sb if not math.isnan(x)]
            rel_right_sb = [x for x in right_sb if not math.isnan(x)]
            n_l = len(rel_left_sb)
            n_r = len(rel_right_sb)
            n_l_plus_r = n_l + n_r
            sum_left_sb = np.sum(rel_left_sb)
            sum_right_sb = np.sum(rel_right_sb)

            # SETBACK default values
            ind_left_sb = sum_left_sb / n_l if n_l > 0 else self.sightline_length
            ind_right_sb = sum_right_sb / n_r if n_r > 0 else self.sightline_length
            ind_sb = (
                (sum_left_sb + sum_right_sb) / (n_l_plus_r)
                if n_l_plus_r > 0
                else self.sightline_length
            )

            sum_square_error_left_sb = np.sum(
                [(x - ind_left_sb) ** 2 for x in rel_left_sb]
            )
            sum_square_error_right_sb = np.sum(
                [(x - ind_right_sb) ** 2 for x in rel_right_sb]
            )

            ind_left_sb_std = (
                math.sqrt(sum_square_error_left_sb / (n_l - 1)) if n_l > 1 else 0
            )
            ind_right_sb_std = (
                math.sqrt(sum_square_error_right_sb / (n_r - 1)) if n_r > 1 else 0
            )
            ind_sb_std = (
                math.sqrt(
                    (sum_square_error_left_sb + sum_square_error_right_sb)
                    / (n_l_plus_r - 1)
                )
                if n_l_plus_r > 1
                else 0
            )

            # medianes
            ind_left_sb_med = (
                np.median(rel_left_sb) if n_l > 0 else self.sightline_length
            )
            ind_right_sb_med = (
                np.median(rel_right_sb) if n_r > 0 else self.sightline_length
            )
            ind_sb_med = (
                np.median(np.concatenate([rel_left_sb, rel_right_sb]))
                if n_l_plus_r > 0
                else self.sightline_length
            )

            # mad
            sum_abs_error_left_sb = np.sum([abs(x - ind_left_sb) for x in rel_left_sb])
            sum_abs_error_right_sb = np.sum(
                [abs(x - ind_right_sb) for x in rel_right_sb]
            )
            ind_left_sb_mad = sum_abs_error_left_sb / n_l if n_l > 0 else 0
            ind_right_sb_mad = sum_abs_error_right_sb / n_r if n_r > 0 else 0
            ind_sb_mad = (
                (sum_abs_error_left_sb + sum_abs_error_right_sb) / (n_l_plus_r)
                if n_l_plus_r > 0
                else 0
            )

            # mad_med
            sum_abs_error_left_sb_med = np.sum(
                [abs(x - ind_left_sb_med) for x in rel_left_sb]
            )
            sum_abs_error_right_sb_med = np.sum(
                [abs(x - ind_right_sb_med) for x in rel_right_sb]
            )
            ind_left_sb_mad_med = sum_abs_error_left_sb_med / n_l if n_l > 0 else 0
            ind_right_sb_mad_med = sum_abs_error_right_sb_med / n_r if n_r > 0 else 0
            ind_sb_mad_med = (
                (sum_abs_error_left_sb_med + sum_abs_error_right_sb_med) / (n_l_plus_r)
                if n_l_plus_r > 0
                else 0
            )

            # ------------------------
            # HEIGHT
            # ------------------------
            rel_left_h = [x for x in left_h if not math.isnan(x)]
            rel_right_h = [x for x in right_h if not math.isnan(x)]
            sum_left_h = np.sum(rel_left_h)
            sum_right_h = np.sum(rel_right_h)

            # HEIGHT AVERAGE default values
            ind_left_h = sum_left_h / n_l if n_l > 0 else 0
            ind_right_h = sum_right_h / n_r if n_r > 0 else 0
            ind_h = (sum_left_h + sum_right_h) / (n_l_plus_r) if n_l_plus_r > 0 else 0

            sum_square_error_left_h = np.sum(
                [(x - ind_left_h) ** 2 for x in rel_left_h]
            )
            sum_square_error_right_h = np.sum(
                [(x - ind_right_h) ** 2 for x in rel_right_h]
            )

            ind_left_h_std = (
                math.sqrt(sum_square_error_left_h / (n_l - 1)) if n_l > 1 else 0
            )
            ind_right_h_std = (
                math.sqrt(sum_square_error_right_h / (n_r - 1)) if n_r > 1 else 0
            )
            ind_h_std = (
                math.sqrt(
                    (sum_square_error_left_h + sum_square_error_right_h)
                    / (n_l_plus_r - 1)
                )
                if n_l_plus_r > 1
                else 0
            )

            # ------------------------
            # CRosS_SECTION_PROPORTIOn (cross sectionnal ratio)
            # ------------------------
            rel_left_hw = [x for x in left_hw if not math.isnan(x)]
            rel_right_hw = [x for x in right_hw if not math.isnan(x)]
            sum_left_hw = np.sum(rel_left_hw)
            sum_right_hw = np.sum(rel_right_hw)

            ind_left_hw = sum_left_hw / n_l if n_l > 0 else 0
            ind_right_hw = sum_right_hw / n_r if n_r > 0 else 0
            ind_hw = (
                (sum_left_hw + sum_right_hw) / (n_l_plus_r) if n_l_plus_r > 0 else 0
            )

            sum_square_error_left_hw = np.sum(
                [(x - ind_left_hw) ** 2 for x in rel_left_hw]
            )
            sum_square_error_right_hw = np.sum(
                [(x - ind_right_hw) ** 2 for x in rel_right_hw]
            )

            ind_left_hw_std = (
                math.sqrt(sum_square_error_left_hw / (n_l - 1)) if n_l > 1 else 0
            )
            ind_right_hw_std = (
                math.sqrt(sum_square_error_right_hw / (n_r - 1)) if n_r > 1 else 0
            )
            ind_hw_std = (
                math.sqrt(
                    (sum_square_error_left_hw + sum_square_error_right_hw)
                    / (n_l_plus_r - 1)
                )
                if n_l_plus_r > 1
                else 0
            )

            # --------------------------------
            # CRosS_SECTIONNAL OPEn VIEW ANGLE
            # --------------------------------
            left_angles = [
                np.rad2deg(np.arctan(hw)) if not math.isnan(hw) else 0 for hw in left_hw
            ]
            right_angles = [
                np.rad2deg(np.arctan(hw)) if not math.isnan(hw) else 0
                for hw in right_hw
            ]

            angles = [
                180 - gamma_l - gamma_r
                for gamma_l, gamma_r in zip(left_angles, right_angles, strict=False)
            ]
            ind_csosva = sum(angles) / n

            # ------------------------
            # TANGENTE Ratio (front+back/os if setback exists)
            # ------------------------
            all_tan = []
            all_tan_ratio = []
            for f, b, lf, r in zip(front_sb, back_sb, left_os, right_os, strict=False):
                tan_value = f + b
                all_tan.append(tan_value)
                if not math.isnan(lf) and not math.isnan(r):
                    all_tan_ratio.append(tan_value / (lf + r))

            # Tan
            ind_tan = np.sum(all_tan) / n
            ind_tan_std = 0
            if n > 1:
                ind_tan_std = math.sqrt(
                    np.sum([(x - ind_tan) ** 2 for x in all_tan]) / (n - 1)
                )

            # Tan ratio
            ind_tan_ratio = 0
            ind_tan_ratio_std = 0
            n_tan_ratio = len(all_tan_ratio)
            if n_tan_ratio > 0:
                ind_tan_ratio = np.sum(all_tan_ratio) / n_tan_ratio
                if n_tan_ratio > 1:
                    ind_tan_ratio_std = math.sqrt(
                        np.sum([(x - ind_tan_ratio) ** 2 for x in all_tan_ratio])
                        / (n_tan_ratio - 1)
                    )

            # version de l'indictaur sans horizon (max = sightline_length)
            (
                ind_left_par_tot,
                ind_left_par_rel,
                ind_right_par_tot,
                ind_right_par_rel,
                ind_par_tot,
                ind_par_rel,
            ) = self._compute_parallelism_indicators(
                left_sb,
                left_sb_count,
                right_sb,
                right_sb_count,
                n,
                n_l,
                n_r,
                max_distance=None,
            )

            # version de l'indictaur a 15 mètres maximum
            (
                ind_left_par_tot_15,
                ind_left_par_rel_15,
                ind_right_par_tot_15,
                ind_right_par_rel_15,
                ind_par_tot_15,
                ind_par_rel_15,
            ) = self._compute_parallelism_indicators(
                left_sb,
                left_sb_count,
                right_sb,
                right_sb_count,
                n,
                n_l,
                n_r,
                max_distance=15,
            )

            # Built frequency
            ind_left_built_freq = len(set(left_seq_sb_ids)) / street_length
            ind_right_built_freq = len(set(right_seq_sb_ids)) / street_length
            ind_built_freq = (
                len(set(left_seq_sb_ids + right_seq_sb_ids)) / street_length
            )

            # Built coverage
            ind_left_built_coverage = np.mean(left_bc) / self.sightline_length
            ind_right_built_coverage = np.mean(right_bc) / self.sightline_length
            ind_built_coverage = (
                ind_left_built_coverage + ind_right_built_coverage
            ) / 2

            # Built category prevvvalence

            values.append(
                [
                    street_uid,
                    n,
                    n_l,
                    n_r,
                    ind_left_os,
                    ind_right_os,
                    ind_os,
                    ind_left_os_std,
                    ind_right_os_std,
                    ind_os_std,
                    ind_left_os_mad,
                    ind_right_os_mad,
                    ind_os_mad,
                    ind_left_os_med,
                    ind_right_os_med,
                    ind_os_med,
                    ind_left_os_mad_med,
                    ind_right_os_mad_med,
                    ind_os_mad_med,
                    ind_left_sb,
                    ind_right_sb,
                    ind_sb,
                    ind_left_sb_std,
                    ind_right_sb_std,
                    ind_sb_std,
                    ind_left_sb_mad,
                    ind_right_sb_mad,
                    ind_sb_mad,
                    ind_left_sb_med,
                    ind_right_sb_med,
                    ind_sb_med,
                    ind_left_sb_mad_med,
                    ind_right_sb_mad_med,
                    ind_sb_mad_med,
                    ind_left_h,
                    ind_right_h,
                    ind_h,
                    ind_left_h_std,
                    ind_right_h_std,
                    ind_h_std,
                    ind_left_hw,
                    ind_right_hw,
                    ind_hw,
                    ind_left_hw_std,
                    ind_right_hw_std,
                    ind_hw_std,
                    ind_csosva,
                    ind_tan,
                    ind_tan_std,
                    n_tan_ratio,
                    ind_tan_ratio,
                    ind_tan_ratio_std,
                    ind_par_tot,
                    ind_par_rel,
                    ind_left_par_tot,
                    ind_right_par_tot,
                    ind_left_par_rel,
                    ind_right_par_rel,
                    ind_par_tot_15,
                    ind_par_rel_15,
                    ind_left_par_tot_15,
                    ind_right_par_tot_15,
                    ind_left_par_rel_15,
                    ind_right_par_rel_15,
                    ind_left_built_freq,
                    ind_right_built_freq,
                    ind_built_freq,
                    ind_left_built_coverage,
                    ind_right_built_coverage,
                    ind_built_coverage,
                ]
            )

        df = (
            pd.DataFrame(
                values,
                columns=[
                    "street_index",
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
                ],
            )
            .set_index("street_index")
            .join(
                self._sightline_indicators[
                    [
                        "nodes_degree_1",
                        "nodes_degree_4",
                        "nodes_degree_3_5_plus",
                        "street_length",
                        "windingness",
                    ]
                ]
            )
        )

        if self.category_col:
            self._compute_prevalences()
            df = df.join(self.prevalences)

        if hasattr(self, "plots"):
            self._aggregate_plots()
            df = df.join(self._aggregate_plot_data)

        if hasattr(self, "slope"):
            df = df.join(self.slope)

        return df.set_geometry(self.streets.geometry)

    def _compute_building_category_prevalence_indicators(
        self, sb_count, seq_sb_categories
    ):
        sb_sequence_id = 0
        category_total_weight = 0
        category_counters = np.zeros(self.building_categories_count)
        for sb_count_ in sb_count:
            if sb_count_ == 0:
                continue
            # add sight line contribution relative to snail effect
            sb_weight = 1 / sb_count_
            category_total_weight += 1
            for _ in range(sb_count_):
                category_counters[seq_sb_categories[sb_sequence_id]] += sb_weight
                sb_sequence_id += 1

        return category_counters, category_total_weight

    def _compute_prevalences(self):
        values = []

        for street_uid, row in self._sightline_indicators.iterrows():
            left_seq_sb_categories = row.left_seq_sb_categories
            left_sb_count = row.left_sb_count
            right_seq_sb_categories = row.right_seq_sb_categories
            right_sb_count = row.right_sb_count

            # left right totalizer
            left_category_indicators, left_category_total_weight = (
                self._compute_building_category_prevalence_indicators(
                    left_sb_count, left_seq_sb_categories
                )
            )
            right_category_indicators, right_category_total_weight = (
                self._compute_building_category_prevalence_indicators(
                    right_sb_count, right_seq_sb_categories
                )
            )

            # global  totalizer
            category_indicators = (
                left_category_indicators + right_category_indicators
            )  # numpy #add X+Y = Z wxhere zi=xi+yi
            category_total_weight = (
                left_category_total_weight + right_category_total_weight
            )

            left_category_indicators = (
                left_category_indicators / left_category_total_weight
                if left_category_total_weight != 0
                else left_category_indicators
            )
            right_category_indicators = (
                right_category_indicators / right_category_total_weight
                if right_category_total_weight != 0
                else right_category_indicators
            )
            category_indicators = (
                category_indicators / category_total_weight
                if category_total_weight != 0
                else category_indicators
            )

            values.append([street_uid] + list(category_indicators))

        columns = ["street_index"] + [
            f"building_prevalence[{clazz}]"
            for clazz in range(self.building_categories_count)
        ]
        self.prevalences = pd.DataFrame(values, columns=columns).set_index(
            "street_index"
        )

    def point_level(self) -> gpd.GeoDataFrame:
        """Extract data on a sightline point level.

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame with streetscape data linked to points representing
            origins of sightlines.

        Examples
        --------
        >>> sc = momepy.Streetscape(streets, buildings)
        >>> sc.point_level()
        """
        point_data = self._sightline_indicators[
            [
                "sightline_points",
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
            ]
        ]
        point_data = point_data.explode(point_data.columns.tolist())
        for col in point_data.columns[1:]:
            point_data[col] = pd.to_numeric(point_data[col])

        inds = [
            "os_count",
            "os",
            "sb_count",
            "sb",
            "h",
            "hw",
            "bc",
        ]

        if hasattr(self, "plots"):
            # process parcel data
            left_parcel_seq_sb = []
            left_parcel_seq_sb_depth = []
            right_parcel_seq_sb = []
            right_parcel_seq_sb_depth = []

            # we occasionally have more sightlines per point, so we need to average
            # values
            for row in self._plot_indicators.itertuples():
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", "Mean of empty slice", RuntimeWarning
                    )
                    left_inds = np.cumsum(row.left_parcel_sb_count)[:-1]
                    left_parcel_seq_sb.append(
                        [
                            np.nanmean(x)
                            for x in np.split(row.left_parcel_seq_sb, left_inds)
                        ]
                    )
                    left_parcel_seq_sb_depth.append(
                        [
                            np.nanmean(x)
                            for x in np.split(row.left_parcel_seq_sb_depth, left_inds)
                        ]
                    )

                    right_inds = np.cumsum(row.right_parcel_sb_count)[:-1]
                    right_parcel_seq_sb.append(
                        [
                            np.nanmean(x)
                            for x in np.split(row.right_parcel_seq_sb, right_inds)
                        ]
                    )
                    right_parcel_seq_sb_depth.append(
                        [
                            np.nanmean(x)
                            for x in np.split(row.right_parcel_seq_sb_depth, right_inds)
                        ]
                    )

            point_parcel_data = pd.DataFrame(
                {
                    "left_plot_seq_sb": left_parcel_seq_sb,
                    "left_plot_seq_sb_depth": left_parcel_seq_sb_depth,
                    "right_plot_seq_sb": right_parcel_seq_sb,
                    "right_plot_seq_sb_depth": right_parcel_seq_sb_depth,
                },
                index=self._plot_indicators.index,
            )
            point_data = pd.concat(
                [
                    point_data,
                    point_parcel_data.explode(
                        point_parcel_data.columns.tolist()
                    ).astype(float),
                ],
                axis=1,
            )
            inds.extend(
                [
                    "plot_seq_sb",
                    "plot_seq_sb_depth",
                ]
            )

        for ind in inds:
            if "count" in ind:
                sums = point_data[[f"left_{ind}", f"right_{ind}"]].sum(axis=1)
                nan_mask = (
                    point_data[[f"left_{ind}", f"right_{ind}"]].isna().all(axis=1)
                )
                sums[nan_mask] = np.nan
                point_data[ind] = sums
            else:
                point_data[ind] = point_data[[f"left_{ind}", f"right_{ind}"]].mean(
                    axis=1
                )

        return point_data.set_geometry(
            "sightline_points", crs=self.streets.crs
        ).rename_geometry("geometry")


def rotate(x, y, xo, yo, theta):  # rotate x,y around xo,yo by theta (rad)
    xr = math.cos(theta) * (x - xo) - math.sin(theta) * (y - yo) + xo
    yr = math.sin(theta) * (x - xo) + math.cos(theta) * (y - yo) + yo
    return [xr, yr]


rad_90 = np.deg2rad(90)


def extend_line_end(line, distance):
    coords = line.coords
    nbp = len(coords)

    len_ext = distance + 1  # eps

    # extend line start point
    ax, ay = coords[0]
    bx, by = coords[1]

    # extend line end point
    ax, ay = coords[nbp - 1]
    bx, by = coords[nbp - 2]
    len_ab = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
    xe = ax + (ax - bx) / len_ab * len_ext
    ye = ay + (ay - by) / len_ab * len_ext
    return LineString(coords[0 : nbp - 1] + [[xe, ye]])


def lines_angle(l1, l2):
    v1_a = l1.coords[0]
    v1_b = l1.coords[-1]
    v2_a = l2.coords[0]
    v2_b = l2.coords[-1]
    start_x = v1_b[0] - v1_a[0]
    start_y = v1_b[1] - v1_a[1]
    dest_x = v2_b[0] - v2_a[0]
    dest_y = v2_b[1] - v2_a[1]
    ahab = math.atan2((dest_y), (dest_x))
    ahao = math.atan2((start_y), (start_x))

    ab = ahab - ahao
    # calc radian
    if ab > math.pi:
        angle = ab + (-2 * math.pi)
    else:
        if ab < 0 - math.pi:
            angle = ab + (2 * math.pi)
        else:
            angle = ab + 0

    return math.degrees(angle)
