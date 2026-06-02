"""
Original implementation by Alessandro Araldi and the team of University of Cote d'Azur.

Adapted for use in momepy by Marek Novotny and Martin Fleischmann.

Note that the implementation in most cases follows the original code, resulting in
certain performance issues compared to the rest of momepy.
"""

import math

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely import LineString

import momepy


class Streetscape:
    """Streetscape analysis based on sightlines

    The class is designed for morphological streetscape analysis, focusing on generating
    and analyzing streetscape measures based on sight points and sightlines. It places
    sight points at regular intervals along streets (``sightline_spacing``) and
    generates four rays from each point: two perpendicular and two tangent to the
    street. These rays intersect with building features, allowing the function to
    capture various point-based and street-based metrics.

    The function returns the following measures:

    Point-level measures (for cross-sectional streetscape analysis):

    * ``os``: Open Space [m]
    * ``sb``: Setback distance from the street edge [m],
    * ``h``: Building Height [m] (if height data is provided ``height_col``),
    * ``hw``: Height-to-Width Ratio [-] (if height data is provided ``height_col``),
    * ``tan``: Visual depth along the street[m],
    * ``tan_ratio``: Tangent-Width Ratio[-],
    * ``csosva``: Cross-sectional Sky View Factor[°],
    * ``cr``: Building Coverage Ratio [%]

    Street-level measures (summarizing the entire street element):

    * ``par_tot``: Total Parallel Façades [-],
    * ``par_rel``: RelativecParallel Façades [-],
    * ``par_tot_15``: Total Parallel Façades within 15 meters[%],
    * ``par_rel_15``: Relative Parallel Façades within 15 meters[%],
    * ``built_freq``: Building Frequency,
    * ``building_prevalence_Ti``: Prevalence of specific building types [%] (if
        building classification is provided ``category_col``).

    The method also provides the distribution of measures along the street, providing:

    * ``_*_std``: Standard Deviation,
    * ``_*_mad``: Mean Absolute Deviation,
    * ``_*_med``: Median,
    * ``_*_mad_med``: Median Absolute Deviation

    for the following indicators: ``os``, ``sb``, ``h``, ``hw``, ``cr``, ``tan``.

    If no building or feature is found within the specified tick length, the function a
    ssigns a theoretical maximum value for ``os`` and ``sb`` or ``0/NaN`` depending on
    the variable.

    The function also provides the distribution of measures separately for the two sides
    of the street (right and left, assigned arbitrarily). Indicators for the left side
    are preceded by ``left_*`` and for the right side by ``right_*`` At the same time,
    you can retrieve the index of buildings that intersected each sightline via
    ``*_seq_sb_index``.

    Additional street-level measures are: ``street_length`` (Street Length),
    ``windingness`` (Windingness or Curvature of Streets, equal to 1 -
    :py:func:`~momepy.linearity`), ``nodes_degree_1`` (Degree 1 Nodes, representing dead
    ends), ``nodes_degree_4`` (Degree 4 Nodes, representing intersections with 4
    connections), and ``nodes_degree_3_5_plus`` (Degree 3-5 or More Nodes, representing
    intersections with 3, 5, or more connections).

    This is a direct implementation of the algorithm proposed in
    :cite:`araldi2024multi`. When using the implementation, please cite
    :cite:`araldi2025Streetscape`.

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

    >>> sc = momepy.Streetscape(streets, buildings)  # doctest: +SKIP

    The resulting data can be extracted either on a street level:

    >>> street_df = sc.street_level()  # doctest: +SKIP

    Or for all individual sightline points:

    >>> point_df = sc.point_level()  # doctest: +SKIP

    If you have access to plots, you can additionally measure plot-based data:

    >>> sc.compute_plots(plots)  # doctest: +SKIP

    If you have a digital terrain model, you can measure slope-based data:

    >>> sc.compute_slope(dtm)  # doctest: +SKIP

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
        self._sightline_lengths_by_type = np.asarray(
            self.sightline_length_PER_SIGHT_TYPE, dtype=float
        )

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
        self._building_geometries = self.buildings.geometry.array
        self._building_ids = self.buildings["street_index"].to_numpy()
        self._building_heights = (
            self.buildings[height_col].to_numpy() if height_col else None
        )
        self._building_categories = (
            self.buildings[category_col].to_numpy() if category_col else None
        )

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
                np.empty(0, dtype=object),
                np.empty(0, dtype=int),
                np.empty(0, dtype=int),
                [],
                [],
            )

        nb_inter_nodes = int(math.floor(remaining_length / self.sightline_spacing))
        distances = np.linspace(
            self.intersection_offset,
            line_length - self.intersection_offset,
            nb_inter_nodes + 1,
        )

        # semi_ortho_segment_size = self.sightline_spacing/2
        semi_ortho_segment_size = self.intersection_offset / 2

        seg_st = shapely.line_interpolate_point(
            line, distances - semi_ortho_segment_size
        )
        seg_mid = shapely.line_interpolate_point(line, distances)
        seg_end = shapely.line_interpolate_point(
            line, distances + semi_ortho_segment_size
        )

        mid_coords = shapely.get_coordinates(seg_mid)
        direction = shapely.get_coordinates(seg_end) - shapely.get_coordinates(seg_st)
        direction = direction / np.linalg.norm(direction, axis=1)[:, np.newaxis]

        vec_anti = np.column_stack((-direction[:, 1], direction[:, 0]))
        vec_clock = np.column_stack((direction[:, 1], -direction[:, 0]))
        prof_st = mid_coords + vec_anti * self.sightline_length
        prof_end = mid_coords + vec_clock * self.sightline_length

        tangent_length = self.sightline_length + self.tangent_length + 1
        tan_back = mid_coords + direction * tangent_length
        tan_front = mid_coords - direction * tangent_length

        n_points = len(distances)
        base_point_ids = np.repeat(np.arange(n_points), 4)
        base_sight_types = np.tile(
            [
                self.SIGHTLINE_LEFT,
                self.SIGHTLINE_RIGHT,
                self.SIGHTLINE_BACK,
                self.SIGHTLINE_FRONT,
            ],
            n_points,
        )

        start_coords = np.repeat(mid_coords, 4, axis=0)
        end_coords = np.empty_like(start_coords)
        end_coords[0::4] = prof_st
        end_coords[1::4] = prof_end
        end_coords[2::4] = tan_back
        end_coords[3::4] = tan_front

        sightline_geometries = list(
            shapely.linestrings(np.stack((start_coords, end_coords), axis=1))
        )
        point_ids = base_point_ids.tolist()
        sight_types = base_sight_types.tolist()

        # SECOND PART : TANGENT SIGHTLINES #

        # Start iterating along the line
        for sightline_index in range(n_points):
            sightline_left = sightline_geometries[sightline_index * 4]
            sightline_right = sightline_geometries[sightline_index * 4 + 1]

            # THIRD PART: SIGHTLINE ENRICHMENT #

            # Populate lost space between consecutive sight lines with high deviation
            # (>angle_tolerance)
            if sightline_index > 0:
                for this_line, prev_line, side, this_coords, prev_coords in [
                    (
                        sightline_left,
                        sightline_geometries[(sightline_index - 1) * 4],
                        self.SIGHTLINE_LEFT,
                        (mid_coords[sightline_index], prof_st[sightline_index]),
                        (mid_coords[sightline_index - 1], prof_st[sightline_index - 1]),
                    ),
                    (
                        sightline_right,
                        sightline_geometries[(sightline_index - 1) * 4 + 1],
                        self.SIGHTLINE_RIGHT,
                        (mid_coords[sightline_index], prof_end[sightline_index]),
                        (
                            mid_coords[sightline_index - 1],
                            prof_end[sightline_index - 1],
                        ),
                    ),
                ]:
                    # angle between consecutive sight line
                    deviation = round(
                        lines_angle_from_coords(prev_coords, this_coords), 1
                    )
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
                        x0, y0 = this_coords[0]
                        x, y = this_coords[1]
                        new_line = LineString(
                            [this_coords[0], rotate(x, y, x0, y0, angle)]
                        )
                        sightline_geometries.append(new_line)
                        point_ids.append(sightline_index)
                        sight_types.append(side)

                        # add S2 new sight line on this current sight line
                    angle = 0
                    for _ in range(0, nb_new_sightlines_prev):
                        angle += theta_rad
                        x0, y0 = prev_coords[0]
                        x, y = prev_coords[1]
                        new_line = LineString(
                            [prev_coords[0], rotate(x, y, x0, y0, angle)]
                        )
                        sightline_geometries.append(new_line)
                        point_ids.append(sightline_index - 1)
                        sight_types.append(side)

        # ==
        # SPECIFIC ENRICHMENT FOR SIGHTPOINTS corresponding to DEAD ENDs
        # ==
        if dead_end_start or dead_end_end:
            for prev_line, prev_point_id, this_line, this_point_id, dead_end in [
                (
                    sightline_geometries[0],
                    point_ids[0],
                    sightline_geometries[1],
                    point_ids[1],
                    dead_end_start,
                ),
                (
                    sightline_geometries[(n_points - 1) * 4 + 1],
                    point_ids[(n_points - 1) * 4 + 1],
                    sightline_geometries[(n_points - 1) * 4],
                    point_ids[(n_points - 1) * 4],
                    dead_end_end,
                ),
            ]:
                if not dead_end:
                    continue

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

                    sightline_geometries.append(new_line)
                    point_ids.append(this_point_id)
                    sight_types.append(self.SIGHTLINE_LEFT)

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
                    sightline_geometries.append(new_line)
                    point_ids.append(prev_point_id)
                    sight_types.append(self.SIGHTLINE_RIGHT)
            # ==
        return (
            np.asarray(sightline_geometries, dtype=object),
            np.asarray(point_ids, dtype=int),
            np.asarray(sight_types, dtype=int),
            list(seg_mid),
            distances.tolist(),
        )

    def _compute_building_sightline_matches(self, sightlines, sight_types):
        """Compute nearest building hits and coverage for a street's sightlines."""
        lines = sightlines
        n_sightlines = len(lines)

        match_distances = self._sightline_lengths_by_type.take(sight_types)
        match_building_ids = np.empty(n_sightlines, dtype=object)
        match_building_ids[:] = None
        match_building_heights = np.full(n_sightlines, np.nan, dtype=float)
        match_building_categories = np.empty(n_sightlines, dtype=object)
        match_building_categories[:] = None
        coverage_lengths = np.zeros(n_sightlines, dtype=float)
        endpoints = shapely.get_point(lines, -1)

        sightline_ix, building_ix = self.rtree_buildings.query(
            lines, predicate="intersects"
        )
        if len(sightline_ix) == 0:
            return (
                match_distances,
                match_building_ids,
                match_building_heights,
                match_building_categories,
                coverage_lengths,
                endpoints,
            )

        line_hits = lines.take(sightline_ix)
        building_hits = self._building_geometries.take(building_ix)
        coverage = shapely.length(shapely.intersection(line_hits, building_hits))
        coverage_lengths = np.bincount(
            sightline_ix, weights=coverage, minlength=n_sightlines
        )

        starts = shapely.get_point(line_hits, 0)
        exteriors = shapely.get_exterior_ring(building_hits)
        exterior_intersections = shapely.intersection(line_hits, exteriors)
        hit_mask = ~shapely.is_empty(exterior_intersections)
        if not hit_mask.any():
            return (
                match_distances,
                match_building_ids,
                match_building_heights,
                match_building_categories,
                coverage_lengths,
                endpoints,
            )

        hit_sightline_ix = sightline_ix[hit_mask]
        hit_building_ix = building_ix[hit_mask]
        hit_distances = shapely.distance(
            starts[hit_mask], exterior_intersections[hit_mask]
        )

        # Stable sorting preserves the first candidate chosen by the original loop
        # whenever multiple buildings have the same distance from a sightline origin.
        order = np.argsort(hit_distances, kind="stable")
        sorted_sightline_ix = hit_sightline_ix[order]
        _, first_positions = np.unique(sorted_sightline_ix, return_index=True)
        nearest_positions = order[first_positions]

        nearest_sightline_ix = hit_sightline_ix[nearest_positions]
        nearest_building_ix = hit_building_ix[nearest_positions]
        nearest_distances = hit_distances[nearest_positions]
        closer_than_default = nearest_distances < match_distances[nearest_sightline_ix]

        nearest_sightline_ix = nearest_sightline_ix[closer_than_default]
        nearest_building_ix = nearest_building_ix[closer_than_default]
        nearest_distances = nearest_distances[closer_than_default]

        match_distances[nearest_sightline_ix] = nearest_distances
        match_building_ids[nearest_sightline_ix] = self._building_ids[
            nearest_building_ix
        ]

        if self.height_col:
            match_building_heights[nearest_sightline_ix] = self._building_heights[
                nearest_building_ix
            ]
        if self.category_col:
            match_building_categories[nearest_sightline_ix] = self._building_categories[
                nearest_building_ix
            ]

        return (
            match_distances,
            match_building_ids,
            match_building_heights,
            match_building_categories,
            coverage_lengths,
            endpoints,
        )

    def _compute_sigthlines_indicators(self, street_row, _optimize_on=True):
        street_uid = street_row.street_index
        street_geom = street_row.geometry

        (
            sightlines,
            point_ids,
            sight_types,
            sightlines_points,
            _results_sight_points_distances,
        ) = self._compute_sightlines(
            street_geom, street_row.dead_end_left, street_row.dead_end_right
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

        left_ids_all = []
        right_ids_all = []

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
                left_ids_all,
                right_ids_all,
            ], None

        if len(sightlines_points) == 0:
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
                left_ids_all,
                right_ids_all,
            ], None

        (
            match_distances,
            match_building_ids,
            match_building_heights,
            match_building_categories,
            coverage_lengths,
            endpoints,
        ) = self._compute_building_sightline_matches(sightlines, sight_types)

        sort_order = np.argsort(point_ids, kind="stable")
        sorted_point_ids = point_ids[sort_order]
        _, group_starts = np.unique(sorted_point_ids, return_index=True)
        group_ends = np.r_[group_starts[1:], len(sort_order)]

        # iterate throught sightlines groups.
        # Eeach sigh points could have many sub sighpoint in case of snail effect)
        for group_start, group_end in zip(group_starts, group_ends, strict=False):
            front_sl_tan_sb = self.tangent_length
            back_sl_tan_sb = self.tangent_length
            left_sl_count = 0
            left_sl_distance_total = 0
            left_sl_building_count = 0
            left_sl_building_sb = []
            left_sl_building_sb_heights = []
            right_sl_count = 0
            right_sl_distance_total = 0
            right_sl_building_count = 0
            right_sl_building_sb = []
            right_sl_building_sb_heights = []

            left_sl_cr_total = 0
            right_sl_cr_total = 0

            left_ids = []
            right_ids = []

            # iterate throught each sightline links to the sigh point:
            # LEFT(1-*),RIGHT(1-*),FRONT(1), BACK(1)
            for sightline_pos in sort_order[group_start:group_end]:
                sightline_side = sight_types[sightline_pos]
                match_sl_distance = match_distances[sightline_pos]
                match_sl_building_id = match_building_ids[sightline_pos]
                match_sl_building_height = match_building_heights[sightline_pos]
                match_sl_building_category = match_building_categories[sightline_pos]
                sl_cr_total = coverage_lengths[sightline_pos]
                endpoint = endpoints[sightline_pos]

                if sightline_side == self.SIGHTLINE_LEFT:
                    left_sl_count += 1
                    left_seq_sightlines_end_points.append(endpoint)
                    left_sl_distance_total += match_sl_distance
                    left_sl_cr_total += sl_cr_total
                    if match_sl_building_id:
                        left_sl_building_count += 1
                        left_sl_building_sb.append(match_sl_distance)
                        left_sl_building_sb_heights.append(match_sl_building_height)
                        # PREVALENCE: Emit each time a new setback or INTER-setback is
                        # found (campact storage structure)
                        current_street_left_seq_sb_ids.append(match_sl_building_id)
                        current_street_left_seq_sb_categories.append(
                            match_sl_building_category
                        )
                        left_ids.append(match_sl_building_id)
                    else:
                        left_ids.append(np.nan)

                elif sightline_side == self.SIGHTLINE_RIGHT:
                    right_sl_count += 1
                    right_seq_sightlines_end_points.append(endpoint)
                    right_sl_distance_total += match_sl_distance
                    right_sl_cr_total += sl_cr_total
                    if match_sl_building_id:
                        right_sl_building_count += 1
                        right_sl_building_sb.append(match_sl_distance)
                        right_sl_building_sb_heights.append(match_sl_building_height)
                        # PREVALENCE: Emit each time a new setback or INTER-setback is
                        # found (campact storage structure)
                        current_street_right_seq_sb_ids.append(match_sl_building_id)
                        current_street_right_seq_sb_categories.append(
                            match_sl_building_category
                        )
                        right_ids.append(match_sl_building_id)
                    else:
                        right_ids.append(np.nan)

                elif sightline_side == self.SIGHTLINE_BACK:
                    back_sl_tan_sb = match_sl_distance
                elif sightline_side == self.SIGHTLINE_FRONT:
                    front_sl_tan_sb = match_sl_distance

            left_ids_all.append(left_ids)
            right_ids_all.append(right_ids)

            # LEFT
            left_os_count = left_sl_count
            left_os = left_sl_distance_total / left_os_count
            left_sb_count = left_sl_building_count
            left_sb = np.nan
            left_h = np.nan
            left_hw = np.nan
            if left_sl_building_sb:
                left_sb = np.nanmean(left_sl_building_sb)
            if not np.isnan(left_sl_building_sb_heights).all():
                left_h = np.nanmean(left_sl_building_sb_heights)
            if not np.isnan([left_sb, left_h]).all():
                # HACK if sb = 0 --> 10cm
                left_hw = left_h / max(left_sb, 0.1)
            left_cr = left_sl_cr_total / left_os_count
            # RIGHT
            right_os_count = right_sl_count
            right_os = right_sl_distance_total / right_os_count
            right_sb_count = right_sl_building_count
            right_sb = np.nan
            right_h = np.nan
            right_hw = np.nan
            if right_sl_building_sb:
                right_sb = np.nanmean(right_sl_building_sb)
            if not np.isnan(right_sl_building_sb_heights).all():
                right_h = np.nanmean(right_sl_building_sb_heights)
            if not np.isnan([right_sb, right_h]).all():
                # HACK if sb = 0 --> 10cm
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
            left_ids_all,
            right_ids_all,
        ], None

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
                "left_ids",
                "right_ids",
            ],
        )
        df = df.set_index("street_index")

        df["nodes_degree_1"] = self.streets.apply(
            lambda row: (
                ((1 if row.n1_degree == 1 else 0) + (1 if row.n2_degree == 1 else 0))
                / 2
            ),
            axis=1,
        )

        df["nodes_degree_4"] = self.streets.apply(
            lambda row: (
                ((1 if row.n1_degree == 4 else 0) + (1 if row.n2_degree == 4 else 0))
                / 2
            ),
            axis=1,
        )

        df["nodes_degree_3_5_plus"] = self.streets.apply(
            lambda row: (
                (
                    (1 if row.n1_degree == 3 or row.n1_degree >= 5 else 0)
                    + (1 if row.n2_degree == 3 or row.n2_degree >= 5 else 0)
                )
                / 2
            ),
            axis=1,
        )
        df["street_length"] = self.streets.length
        df["windingness"] = 1 - momepy.linearity(self.streets)

        self._sightline_indicators = df

    def _compute_sigthlines_plot_indicators_one_side(
        self, sightline_points, os_count, seq_os_endpoint
    ):
        n = len(sightline_points)
        if n == 0:
            return [
                [0] * n,
                [],
                [],
                [],
                [],
            ]

        os_count = np.asarray(os_count, dtype=int)
        total_sightlines = os_count.sum()
        if total_sightlines == 0:
            return [
                [0] * n,
                [],
                [],
                [],
                [[] for _ in range(n)],
            ]

        point_ids = np.repeat(np.arange(n), os_count)
        start_coords = shapely.get_coordinates(
            np.asarray(sightline_points, dtype=object)
        )[point_ids]
        end_coords = shapely.get_coordinates(seq_os_endpoint)
        sightlines = shapely.linestrings(np.stack((start_coords, end_coords), axis=1))

        sightline_ix, plot_ix = self.rtree_parcels.query(
            sightlines, predicate="intersects"
        )
        match_distances = np.full(total_sightlines, self.sightline_length, dtype=float)
        match_ids = np.full(total_sightlines, np.nan, dtype=object)
        match_plot_ix = np.full(total_sightlines, -1, dtype=int)

        if len(sightline_ix) > 0:
            line_hits = sightlines.take(sightline_ix)
            plot_hits = self._plot_geometries.take(plot_ix)
            exterior_intersections = shapely.intersection(
                line_hits, shapely.get_exterior_ring(plot_hits)
            )
            hit_mask = ~shapely.is_empty(exterior_intersections)

            if hit_mask.any():
                hit_sightline_ix = sightline_ix[hit_mask]
                hit_plot_ix = plot_ix[hit_mask]
                hit_distances = shapely.distance(
                    shapely.points(start_coords[hit_sightline_ix]),
                    exterior_intersections[hit_mask],
                )
                order = np.argsort(hit_distances, kind="stable")
                sorted_sightline_ix = hit_sightline_ix[order]
                _, first_positions = np.unique(sorted_sightline_ix, return_index=True)
                nearest_positions = order[first_positions]
                nearest_sightline_ix = hit_sightline_ix[nearest_positions]
                nearest_plot_ix = hit_plot_ix[nearest_positions]

                match_distances[nearest_sightline_ix] = hit_distances[nearest_positions]
                match_ids[nearest_sightline_ix] = self._plot_ids[nearest_plot_ix]
                match_plot_ix[nearest_sightline_ix] = nearest_plot_ix

        matched = match_plot_ix != -1
        parcel_sb_count = np.bincount(point_ids[matched], minlength=n).tolist()
        parcel_seq_sb_ids = match_ids[matched].tolist()
        parcel_seq_sb = match_distances[matched].tolist()

        parcel_seq_sb_depth = []
        if matched.any():
            matched_starts = start_coords[matched]
            matched_ends = end_coords[matched]
            direction = matched_ends - matched_starts
            direction = direction / np.linalg.norm(direction, axis=1)[:, np.newaxis]
            extended_ends = matched_ends + direction * (
                self.sightline_plot_depth_extension + 1
            )
            extended_sightlines = shapely.linestrings(
                np.stack((matched_starts, extended_ends), axis=1)
            )
            matched_plots = self._plot_geometries.take(match_plot_ix[matched])
            invalid = ~shapely.is_valid(matched_plots)
            if invalid.any():
                matched_plots = matched_plots.copy()
                matched_plots[invalid] = shapely.buffer(matched_plots[invalid], 0)
            intersections = shapely.intersection(matched_plots, extended_sightlines)
            bad_type = ~np.isin(shapely.get_type_id(intersections), [1, 5])
            if bad_type.any():
                raise Exception("Not allowed: intersection is not of type Line")
            parcel_seq_sb_depth = shapely.length(intersections).tolist()

        split_indices = np.cumsum(os_count)[:-1]
        parcel_ids_all = [ids.tolist() for ids in np.split(match_ids, split_indices)]

        return [
            parcel_sb_count,
            parcel_seq_sb_ids,
            parcel_seq_sb,
            parcel_seq_sb_depth,
            parcel_ids_all,
        ]

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
        >>> sc = momepy.Streetscape(streets, buildings)  # doctest: +SKIP
        >>> sc.compute_plots(plots)  # doctest: +SKIP
        """
        self.sightline_plot_depth_extension = sightline_plot_depth_extension

        self.rtree_parcels = plots.sindex
        plots = plots.copy()
        plots["parcel_id"] = np.arange(len(plots))
        self.plots = plots
        self.plots["perimeter"] = self.plots.length
        self._plot_geometries = self.plots.geometry.array
        self._plot_ids = self.plots.index.to_numpy()
        self._plot_perimeters = self.plots["perimeter"]

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
                "left_parcel_ids",
                "right_parcel_sb_count",
                "right_parcel_seq_sb_ids",
                "right_parcel_seq_sb",
                "right_parcel_seq_sb_depth",
                "right_parcel_ids",
            ],
        )
        df = df.set_index("street_index").join(self._sightline_indicators.street_length)

        self._plot_indicators = df
        self._aggregate_plot_data = None

    def _aggregate_plots(self):
        values = []

        for street_uid, row in self._plot_indicators.drop(
            columns=["left_parcel_ids", "right_parcel_ids"]
        ).iterrows():
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
                        {},
                        {},
                    ]
                )
                continue

            left_parcel_seq_sb = np.asarray(left_parcel_seq_sb, dtype=float)
            right_parcel_seq_sb = np.asarray(right_parcel_seq_sb, dtype=float)
            left_parcel_seq_sb_depth = np.maximum(
                np.asarray(left_parcel_seq_sb_depth, dtype=float), 1
            )
            right_parcel_seq_sb_depth = np.maximum(
                np.asarray(right_parcel_seq_sb_depth, dtype=float), 1
            )

            left_unique_ids = set(left_parcel_seq_sb_ids)
            right_unique_ids = set(right_parcel_seq_sb_ids)
            all_unique_ids = left_unique_ids.union(right_unique_ids)

            left_parcel_freq = len(left_unique_ids) / street_length
            right_parcel_freq = len(right_unique_ids) / street_length
            parcel_freq = len(all_unique_ids) / street_length

            left_sight_weight = _weights_from_counts(left_parcel_sb_count)
            right_sight_weight = _weights_from_counts(right_parcel_sb_count)

            left_w_sb = left_sight_weight * left_parcel_seq_sb
            right_w_sb = right_sight_weight * right_parcel_seq_sb
            left_w_depth = left_sight_weight * left_parcel_seq_sb_depth
            right_w_depth = right_sight_weight * right_parcel_seq_sb_depth

            np_l = int(left_sight_weight.sum())
            np_r = int(right_sight_weight.sum())
            np_lr = np_l + np_r

            left_parcel_sb = (
                left_w_sb.sum() / np_l if np_l > 0 else self.sightline_length
            )
            right_parcel_sb = (
                right_w_sb.sum() / np_r if np_r > 0 else self.sightline_length
            )
            parcel_sb = (
                (left_w_sb.sum() + right_w_sb.sum()) / np_lr
                if np_lr > 0
                else self.sightline_length
            )

            left_parcel_depth = left_w_depth.sum() / np_l if np_l > 0 else 0
            right_parcel_depth = right_w_depth.sum() / np_r if np_r > 0 else 0
            parcel_depth = (
                (left_w_depth.sum() + right_w_depth.sum()) / np_lr if np_lr > 0 else 0
            )

            all_parcel_seq_sb_ids = left_parcel_seq_sb_ids + right_parcel_seq_sb_ids
            all_sight_weight = np.concatenate((left_sight_weight, right_sight_weight))
            all_w_depth = np.concatenate((left_w_depth, right_w_depth))

            all_wd_wp = _plot_wd_wp_ratio(
                all_parcel_seq_sb_ids,
                all_sight_weight,
                all_w_depth,
                self.sightline_spacing,
                self._plot_perimeters,
            )
            left_wd_wp = _plot_wd_wp_ratio(
                left_parcel_seq_sb_ids,
                left_sight_weight,
                left_w_depth,
                self.sightline_spacing,
                self._plot_perimeters,
            )
            right_wd_wp = _plot_wd_wp_ratio(
                right_parcel_seq_sb_ids,
                right_sight_weight,
                right_w_depth,
                self.sightline_spacing,
                self._plot_perimeters,
            )
            wd_ratio_list = [all_wd_wp[0], left_wd_wp[0], right_wd_wp[0]]
            wp_ratio_list = [all_wd_wp[1], left_wd_wp[1], right_wd_wp[1]]

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
                + [set(left_parcel_seq_sb_ids), set(right_parcel_seq_sb_ids)]
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
            "left_plot_seq_sb_index",
            "right_plot_seq_sb_index",
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
        >>> sc = momepy.Streetscape(streets, buildings)  # doctest: +SKIP
        >>> sc.compute_slope(dtm)  # doctest: +SKIP
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
        sightline_points = self._sightline_indicators["sightline_points"]
        sightline_counts = sightline_points.apply(len).to_numpy()
        flattened_sightline_points = [
            point for points in sightline_points for point in points
        ]

        all_points = np.concatenate(
            [
                np.asarray(start_points, dtype=object),
                np.asarray(end_points, dtype=object),
                np.asarray(flattened_sightline_points, dtype=object),
            ]
        )
        z_values = (
            raster.drop_vars("spatial_ref")
            .xvec.extract_points(points=all_points, x_coords="x", y_coords="y")
            .to_numpy()
            .reshape(-1)
        )

        n_streets = len(self.streets)
        z_start = z_values[:n_streets]
        z_end = z_values[n_streets : 2 * n_streets]
        z_sightline_points = z_values[2 * n_streets :]

        start_coords = shapely.get_coordinates(start_points)
        end_coords = shapely.get_coordinates(end_points)
        sightline_coords = (
            shapely.get_coordinates(flattened_sightline_points)
            if flattened_sightline_points
            else np.empty((0, 2), dtype=float)
        )
        split_indices = np.cumsum(sightline_counts)[:-1]

        slope_values = [
            _compute_slope_from_arrays(
                start_coords[i],
                end_coords[i],
                z_start[i],
                z_end[i],
                sight_coords,
                sight_z,
                self.NODATA_RASTER,
            )
            for i, (sight_coords, sight_z) in enumerate(
                zip(
                    np.split(sightline_coords, split_indices),
                    np.split(z_sightline_points, split_indices),
                    strict=False,
                )
            )
        ]

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
        >>> sc = momepy.Streetscape(streets, buildings)  # doctest: +SKIP
        >>> sc.steet_level()  # doctest: +SKIP
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

            left_os = np.asarray(left_os, dtype=float)
            right_os = np.asarray(right_os, dtype=float)
            left_sb = np.asarray(left_sb, dtype=float)
            right_sb = np.asarray(right_sb, dtype=float)
            left_h = np.asarray(left_h, dtype=float)
            right_h = np.asarray(right_h, dtype=float)
            left_hw = np.asarray(left_hw, dtype=float)
            right_hw = np.asarray(right_hw, dtype=float)
            left_bc = np.asarray(left_bc, dtype=float)
            right_bc = np.asarray(right_bc, dtype=float)
            front_sb = np.asarray(front_sb, dtype=float)
            back_sb = np.asarray(back_sb, dtype=float)

            # ------------------------
            # OPENNESS
            # ------------------------
            sum_left_os = left_os.sum()
            sum_right_os = right_os.sum()

            ind_left_os = sum_left_os / n
            ind_right_os = sum_right_os / n
            ind_os = ind_left_os + ind_right_os  # ==(left_os+right_os)/n

            full_os = left_os + right_os
            # mediane >> med
            ind_left_os_med = np.median(left_os)
            ind_right_os_med = np.median(right_os)
            ind_os_med = np.median(full_os)

            # OPENNESS ROUGHNESS
            sum_square_error_left_os = np.square(left_os - ind_left_os).sum()
            sum_square_error_right_os = np.square(right_os - ind_right_os).sum()
            sum_abs_error_left_os = np.abs(left_os - ind_left_os).sum()
            sum_abs_error_right_os = np.abs(right_os - ind_right_os).sum()
            ind_os_std = math.sqrt(
                (sum_square_error_left_os + sum_square_error_right_os) / (2 * n - 1)
            )
            ind_os_mad = (sum_abs_error_left_os + sum_abs_error_right_os) / (2 * n)

            ind_left_os_std = 0.0  # default
            ind_right_os_std = 0.0  # default
            ind_left_os_mad = 0.0  # default
            ind_right_os_mad = 0.0  # default

            ind_left_os_mad = sum_abs_error_left_os / n
            ind_right_os_mad = sum_abs_error_right_os / n
            if n > 1:
                ind_left_os_std = math.sqrt((sum_square_error_left_os) / (n - 1))
                ind_right_os_std = math.sqrt((sum_square_error_right_os) / (n - 1))

            sum_abs_error_left_os_med = np.abs(left_os - ind_left_os_med).sum()
            sum_abs_error_right_os_med = np.abs(right_os - ind_right_os_med).sum()
            ind_left_os_mad_med = sum_abs_error_left_os_med / n
            ind_right_os_mad_med = sum_abs_error_right_os_med / n
            ind_os_mad_med = (
                sum_abs_error_left_os_med + sum_abs_error_right_os_med
            ) / (2 * n)

            # ------------------------
            # SETBACK
            # ------------------------
            rel_left_sb = left_sb[~np.isnan(left_sb)]
            rel_right_sb = right_sb[~np.isnan(right_sb)]
            n_l = len(rel_left_sb)
            n_r = len(rel_right_sb)
            n_l_plus_r = n_l + n_r
            sum_left_sb = rel_left_sb.sum()
            sum_right_sb = rel_right_sb.sum()

            # SETBACK default values
            ind_left_sb = sum_left_sb / n_l if n_l > 0 else self.sightline_length
            ind_right_sb = sum_right_sb / n_r if n_r > 0 else self.sightline_length
            ind_sb = (
                (sum_left_sb + sum_right_sb) / (n_l_plus_r)
                if n_l_plus_r > 0
                else self.sightline_length
            )

            sum_square_error_left_sb = np.square(rel_left_sb - ind_left_sb).sum()
            sum_square_error_right_sb = np.square(rel_right_sb - ind_right_sb).sum()

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
                np.median(np.concatenate((rel_left_sb, rel_right_sb)))
                if n_l_plus_r > 0
                else self.sightline_length
            )

            # mad
            sum_abs_error_left_sb = np.abs(rel_left_sb - ind_left_sb).sum()
            sum_abs_error_right_sb = np.abs(rel_right_sb - ind_right_sb).sum()
            ind_left_sb_mad = sum_abs_error_left_sb / n_l if n_l > 0 else 0
            ind_right_sb_mad = sum_abs_error_right_sb / n_r if n_r > 0 else 0
            ind_sb_mad = (
                (sum_abs_error_left_sb + sum_abs_error_right_sb) / (n_l_plus_r)
                if n_l_plus_r > 0
                else 0
            )

            # mad_med
            sum_abs_error_left_sb_med = np.abs(rel_left_sb - ind_left_sb_med).sum()
            sum_abs_error_right_sb_med = np.abs(rel_right_sb - ind_right_sb_med).sum()
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
            rel_left_h = left_h[~np.isnan(left_h)]
            rel_right_h = right_h[~np.isnan(right_h)]
            sum_left_h = rel_left_h.sum()
            sum_right_h = rel_right_h.sum()

            # HEIGHT AVERAGE default values
            ind_left_h = sum_left_h / n_l if n_l > 0 else 0
            ind_right_h = sum_right_h / n_r if n_r > 0 else 0
            ind_h = (sum_left_h + sum_right_h) / (n_l_plus_r) if n_l_plus_r > 0 else 0

            sum_square_error_left_h = np.square(rel_left_h - ind_left_h).sum()
            sum_square_error_right_h = np.square(rel_right_h - ind_right_h).sum()

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
            rel_left_hw = left_hw[~np.isnan(left_hw)]
            rel_right_hw = right_hw[~np.isnan(right_hw)]
            sum_left_hw = rel_left_hw.sum()
            sum_right_hw = rel_right_hw.sum()

            ind_left_hw = sum_left_hw / n_l if n_l > 0 else 0
            ind_right_hw = sum_right_hw / n_r if n_r > 0 else 0
            ind_hw = (
                (sum_left_hw + sum_right_hw) / (n_l_plus_r) if n_l_plus_r > 0 else 0
            )

            sum_square_error_left_hw = np.square(rel_left_hw - ind_left_hw).sum()
            sum_square_error_right_hw = np.square(rel_right_hw - ind_right_hw).sum()

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
            left_angles = np.nan_to_num(np.rad2deg(np.arctan(left_hw)), nan=0)
            right_angles = np.nan_to_num(np.rad2deg(np.arctan(right_hw)), nan=0)
            ind_csosva = (180 - left_angles - right_angles).sum() / n

            # ------------------------
            # TANGENTE Ratio (front+back/os if setback exists)
            # ------------------------
            all_tan = front_sb + back_sb
            tan_ratio_mask = ~np.isnan(left_os) & ~np.isnan(right_os)
            all_tan_ratio = all_tan[tan_ratio_mask] / (
                left_os[tan_ratio_mask] + right_os[tan_ratio_mask]
            )

            # Tan
            ind_tan = all_tan.sum() / n
            ind_tan_std = 0.0
            if n > 1:
                ind_tan_std = math.sqrt(np.square(all_tan - ind_tan).sum() / (n - 1))

            # Tan ratio
            ind_tan_ratio = 0.0
            ind_tan_ratio_std = 0.0
            n_tan_ratio = len(all_tan_ratio)
            if n_tan_ratio > 0:
                ind_tan_ratio = all_tan_ratio.sum() / n_tan_ratio
                if n_tan_ratio > 1:
                    ind_tan_ratio_std = math.sqrt(
                        np.square(all_tan_ratio - ind_tan_ratio).sum()
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
                    set(left_seq_sb_ids),
                    set(right_seq_sb_ids),
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
                    "left_seq_sb_index",
                    "right_seq_sb_index",
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
            if self._aggregate_plot_data is None:
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
        >>> sc = momepy.Streetscape(streets, buildings)  # doctest: +SKIP
        >>> sc.point_level()  # doctest: +SKIP
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
                "left_ids",
                "right_ids",
            ]
        ]

        point_data = point_data.explode(point_data.columns.tolist())
        point_data["left_ids"] = point_data["left_ids"].apply(_ids_to_set)
        point_data["right_ids"] = point_data["right_ids"].apply(_ids_to_set)
        point_data = point_data.rename(
            columns={"left_ids": "left_seq_sb_index", "right_ids": "right_seq_sb_index"}
        )
        for col in point_data.columns[1:-2]:
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
            left_ids = []
            right_ids = []

            # we occasionally have more sightlines per point, so we need to average
            # values
            for row in self._plot_indicators.itertuples():
                left_parcel_seq_sb.append(
                    _nanmean_by_counts(row.left_parcel_seq_sb, row.left_parcel_sb_count)
                )
                left_parcel_seq_sb_depth.append(
                    _nanmean_by_counts(
                        row.left_parcel_seq_sb_depth, row.left_parcel_sb_count
                    )
                )
                right_parcel_seq_sb.append(
                    _nanmean_by_counts(
                        row.right_parcel_seq_sb, row.right_parcel_sb_count
                    )
                )
                right_parcel_seq_sb_depth.append(
                    _nanmean_by_counts(
                        row.right_parcel_seq_sb_depth, row.right_parcel_sb_count
                    )
                )
                left_ids.append(row.left_parcel_ids)
                right_ids.append(row.right_parcel_ids)

            point_parcel_data = pd.DataFrame(
                {
                    "left_plot_seq_sb": left_parcel_seq_sb,
                    "left_plot_seq_sb_depth": left_parcel_seq_sb_depth,
                    "right_plot_seq_sb": right_parcel_seq_sb,
                    "right_plot_seq_sb_depth": right_parcel_seq_sb_depth,
                    "left_plot_seq_sb_index": left_ids,
                    "right_plot_seq_sb_index": right_ids,
                },
                index=self._plot_indicators.index,
            )
            point_parcel_data = point_parcel_data.explode(
                point_parcel_data.columns.tolist()
            )
            point_parcel_data[
                point_parcel_data.columns.drop(
                    ["left_plot_seq_sb_index", "right_plot_seq_sb_index"]
                )
            ] = point_parcel_data[
                point_parcel_data.columns.drop(
                    ["left_plot_seq_sb_index", "right_plot_seq_sb_index"]
                )
            ].astype(float)
            point_data = pd.concat(
                [point_data, point_parcel_data],
                axis=1,
            )

            point_data["left_plot_seq_sb_index"] = point_data[
                "left_plot_seq_sb_index"
            ].apply(_ids_to_set)
            point_data["right_plot_seq_sb_index"] = point_data[
                "right_plot_seq_sb_index"
            ].apply(_ids_to_set)
            inds.extend(
                [
                    "plot_seq_sb",
                    "plot_seq_sb_depth",
                ]
            )

        for ind in inds:
            left_values = point_data[f"left_{ind}"].to_numpy(dtype=float)
            right_values = point_data[f"right_{ind}"].to_numpy(dtype=float)
            if "count" in ind:
                sums = np.nansum((left_values, right_values), axis=0)
                sums[np.isnan(left_values) & np.isnan(right_values)] = np.nan
                point_data[ind] = sums
            else:
                valid_count = (~np.isnan(left_values)).astype(int) + (
                    ~np.isnan(right_values)
                ).astype(int)
                sums = np.nansum((left_values, right_values), axis=0)
                values = np.full_like(sums, np.nan)
                np.divide(sums, valid_count, out=values, where=valid_count != 0)
                point_data[ind] = values

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
    return lines_angle_from_coords((v1_a, v1_b), (v2_a, v2_b))


def lines_angle_from_coords(c1, c2):
    v1_a, v1_b = c1
    v2_a, v2_b = c2
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


def _ids_to_set(ids):
    if isinstance(ids, list):
        return {i for i in ids if i is not None and i == i}
    return {ids}


def _nanmean_by_counts(values, counts):
    counts = np.asarray(counts, dtype=int)
    out = np.full(len(counts), np.nan, dtype=float)
    if len(counts) == 0 or len(values) == 0:
        return out.tolist()

    values = np.asarray(values, dtype=float)
    ends = np.cumsum(counts)
    starts = ends - counts
    valid_groups = counts > 0
    if not valid_groups.any():
        return out.tolist()

    starts = starts[valid_groups]
    values_no_nan = np.nan_to_num(values, nan=0)
    valid_values = ~np.isnan(values)

    sums = np.add.reduceat(values_no_nan, starts)
    valid_counts = np.add.reduceat(valid_values.astype(int), starts)
    group_means = np.full(len(starts), np.nan, dtype=float)
    np.divide(sums, valid_counts, out=group_means, where=valid_counts != 0)
    out[valid_groups] = group_means
    return out.tolist()


def _weights_from_counts(counts):
    counts = np.asarray(counts, dtype=int)
    valid_counts = counts[counts != 0]
    if len(valid_counts) == 0:
        return np.empty(0, dtype=float)
    return np.repeat(1 / valid_counts, valid_counts)


def _plot_wd_wp_ratio(parcel_ids, weights, weighted_depth, spacing, perimeters):
    if len(parcel_ids) == 0:
        return 0, 0

    parcel_ids = np.asarray(parcel_ids)
    unique_ids, inverse = np.unique(parcel_ids, return_inverse=True)
    nb = np.bincount(inverse)
    w_sum = np.bincount(inverse, weights=weights)
    w_depth = np.bincount(inverse, weights=weighted_depth) / nb
    perimeter = perimeters.reindex(unique_ids).to_numpy()
    sum_nb = nb.sum()

    wd_ratio = ((w_sum * spacing * nb) / w_depth).sum() / sum_nb
    wp_ratio = ((w_sum * spacing * nb) / perimeter).sum() / sum_nb
    return wd_ratio, wp_ratio


def _compute_slope_from_arrays(
    start_coord,
    end_coord,
    start_z,
    end_z,
    sightline_coords,
    sightline_z,
    nodata,
):
    if len(sightline_z) == 0:
        if start_z == nodata or end_z == nodata:
            return 0, 0, 0, False
        slope_percent = abs(start_z - end_z) / np.linalg.norm(start_coord - end_coord)
        slope_degree = math.degrees(math.atan(slope_percent))
        return slope_percent, slope_degree, 1, True

    coords = np.vstack((start_coord, sightline_coords, end_coord))
    z = np.concatenate(([start_z], sightline_z, [end_z]))

    left_z = z[:-2]
    right_z = z[2:]
    valid = (left_z != nodata) & (right_z != nodata)
    n_slopes = int(valid.sum())
    if n_slopes == 0:
        return 0, 0, 0, False

    distances = np.linalg.norm(coords[2:] - coords[:-2], axis=1)
    slopes = np.abs(left_z - right_z) / distances
    slopes = slopes[valid]
    slope_percent = slopes.sum() / n_slopes
    slope_degree = math.degrees(np.arctan(slopes).sum() / n_slopes)

    return slope_degree, slope_percent, n_slopes, True
