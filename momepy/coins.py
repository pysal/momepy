"""
This Python script generates strokes from edge geodataframes, mainly roads.

Author: Pratyush Tripathy
Date: 29 February 2020
Version: 0.2

Adapted for momepy by: Andres Morfin, Niki Patrinopoulou, and Ioannis Daramouskas
Date: May 29, 2021
"""

import collections
import math

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString, MultiLineString


class COINS:

    """
    Calculates natural continuity and hierarchy of street networks in a given
    GeoDataFrame using the COINS algorithm.

    For details on the algorithms refer to the original paper :cite:`tripathy2020open`.

    This is a reimplementation of the original script from
    https://github.com/PratyushTripathy/COINS

    ``COINS`` can return final stroke geometry (``.stroke_gdf()``) or a pandas
    Series encoding stroke groups onto the original input geometry
    (``.stroke_attribute()``).

    Parameters
    ----------
    edge_gdf : GeoDataFrame
        A GeoDataFrame containing edge geometry of a street network.
        ``edge_gdf`` should ideally not contain MultiLineStrings.
    angle_threshold : int, float (default 0)
        The angle threshold for the COINS algorithm. Segments will only be considered
        a part of the same street if the deflection angle is above the threshold.

    Examples
    --------

    Initialise a ``COINS`` class. This step will compute the topology.

    >>> coins = momepy.COINS(streets)

    To get final stroke geometry:

    >>> stroke_gdf = coins.stroke_gdf()

    To get a Series encoding stroke groups:

    >>> stroke_attr = coins.stroke_attribute()

    Notes
    -----
    The LineStrings of the ``edge_gdf`` are not expected to overlap. If you are creating
    it using OSMnx, don't forget to cast the graph to undirected using
    ``osmnx.get_undirected(G)`` prior converting it to a GeoDataFrame.
    """

    def __init__(self, edge_gdf, angle_threshold=0):
        self.edge_gdf = edge_gdf
        self.gdf_projection = self.edge_gdf.crs
        self.already_merged = False

        # get indices of original gdf
        self.uv_index = range(len(self.edge_gdf.index))

        # get line segments from edge gdf
        self.lines = [list(value[1].coords) for value in edge_gdf.geometry.items()]

        # split edges into line segments
        self._split_lines()

        # create unique_id for each individual line segment
        self._unique_id()

        # compute edge connectivity table
        self._get_links()

        # find best link at every point for both lines
        self._best_link()

        # cross check best links and enter angle threshold for connectivity
        self._cross_check_links(angle_threshold)

    def _premerge(self):
        """
        Return a GeoDataFrame containing the individual segments with all underlying
        information. The result is useful for debugging purposes.
        """
        return self._create_gdf_premerge()

    def stroke_gdf(self):
        """Return a GeoDataFrame containing merged final stroke geometry."""
        if not self.already_merged:
            self._merge_lines()
        return self._create_gdf_strokes()

    def stroke_attribute(self):
        """
        Return a pandas Series encoding stroke groups onto the original input geometry.
        """
        if not self.already_merged:
            self._merge_lines()
        return self._add_gdf_stroke_attributes()

    def _split_lines(self):
        out_line = []
        self.temp_array = []
        n = 0
        # Iterate through the lines and split the edges
        idx = 0
        for line in self.lines:
            for part in _list_to_pairs(line):
                out_line.append(
                    [
                        part,
                        _compute_orientation(part),
                        [],
                        [],
                        [],
                        [],
                        [],
                        [],
                        self.uv_index[idx],
                    ]
                )
                # merge the coordinates as a string, this will help
                # in finding adjacent edges in the function below
                self.temp_array.append(
                    [n, f"{part[0][0]}_{part[0][1]}", f"{part[1][0]}_{part[1][1]}"]
                )
                n += 1
            idx += 1

        self.split = out_line

    def _unique_id(self):
        # Loop through split lines, assign unique ID, and
        # store inside a list along with the connectivity dictionary
        self.unique = dict(enumerate(self.split))

    def _get_links(self):
        self.temp_array = np.array(self.temp_array, dtype=object)

        items = collections.defaultdict(set)
        for i, vertex in enumerate(self.temp_array[:, 1]):
            items[vertex].add(i)
        for i, vertex in enumerate(self.temp_array[:, 2]):
            items[vertex].add(i)

        p1 = []
        for i, vertex in enumerate(self.temp_array[:, 1]):
            item = list(items[vertex])

            item.remove(i)
            p1.append(item)

        p2 = []
        for i, vertex in enumerate(self.temp_array[:, 2]):
            item = list(items[vertex])

            item.remove(i)

            p2.append(item)

        self.result = list(zip(range(len(p1)), p1, p2))

        for a in self.result:
            n = a[0]
            self.unique[n][2] = a[1]
            self.unique[n][3] = a[2]

    def _best_link(self):
        self.angle_pairs = {}
        for edge in range(0, len(self.unique)):
            p1_angle_set = []
            p2_angle_set = []

            # Instead of computing the angle between the two segments twice,
            # this method calculates it once and stores in a dictionary for
            # both the keys. The key is already present in the dictionary so
            # it does not calculate a second time.
            for link1 in self.unique[edge][2]:
                self.angle_pairs["%d_%d" % (edge, link1)] = _angle_between_two_lines(
                    self.unique[edge][0], self.unique[link1][0]
                )
                p1_angle_set.append(self.angle_pairs["%d_%d" % (edge, link1)])

            for link2 in self.unique[edge][3]:
                self.angle_pairs["%d_%d" % (edge, link2)] = _angle_between_two_lines(
                    self.unique[edge][0], self.unique[link2][0]
                )
                p2_angle_set.append(self.angle_pairs["%d_%d" % (edge, link2)])

            # Among the adjacent segments deflection angle values, check
            # for the maximum value at both the ends. The segment with
            # the maximum angle is stored in the attributes to be cross-checked
            # later before finalising the segments at both the ends.
            if len(p1_angle_set) != 0:
                val1, idx1 = max((val, idx) for (idx, val) in enumerate(p1_angle_set))
                self.unique[edge][4] = self.unique[edge][2][idx1], val1
            else:
                self.unique[edge][4] = "dead_end"

            if len(p2_angle_set) != 0:
                val2, idx2 = max((val, idx) for (idx, val) in enumerate(p2_angle_set))
                self.unique[edge][5] = self.unique[edge][3][idx2], val2
            else:
                self.unique[edge][5] = "dead_end"

    def _cross_check_links(self, angle_threshold):
        for edge in range(0, len(self.unique)):
            best_p1 = self.unique[edge][4][0]
            best_p2 = self.unique[edge][5][0]

            if (
                isinstance(best_p1, int)
                and edge in [self.unique[best_p1][4][0], self.unique[best_p1][5][0]]
                and self.angle_pairs["%d_%d" % (edge, best_p1)] > angle_threshold
            ):
                self.unique[edge][6] = best_p1
            else:
                self.unique[edge][6] = "line_break"

            if (
                isinstance(best_p2, int)
                and edge in [self.unique[best_p2][4][0], self.unique[best_p2][5][0]]
                and self.angle_pairs["%d_%d" % (edge, best_p2)] > angle_threshold
            ):
                self.unique[edge][7] = best_p2
            else:
                self.unique[edge][7] = "line_break"

    def _merge_lines(self):
        self.merging_list = []
        self.merged = []
        self.edge_idx = []

        self.result = [
            _merge_lines_loop(n, self.unique) for n in range(len(self.unique))
        ]

        for temp_list in self.result:
            if temp_list not in self.merging_list:
                self.merging_list.append(temp_list)
                self.merged.append(
                    {_list_to_tuple(self.unique[key][0]) for key in temp_list}
                )

                # assign stroke number to edge from argument
                self.edge_idx.append({self.unique[key][8] for key in temp_list})

        self.merged = dict(enumerate(self.merged))
        self.edge_idx = dict(enumerate(self.edge_idx))
        self.already_merged = True

    # Export geodataframes, 3 options
    def _create_gdf_premerge(self):
        my_list = []

        for parts in range(0, len(self.unique)):
            # get all segment points and make line
            line_list = _tuple_to_list(self.unique[parts][0])
            geom_line = LineString([(line_list[0]), (line_list[1])])

            # get other values for premerged
            _unique_id = parts
            orientation = self.unique[parts][1]
            links_p1 = self.unique[parts][2]
            links_p2 = self.unique[parts][3]
            best_p1 = self.unique[parts][4]
            best_p2 = self.unique[parts][5]
            p1_final = self.unique[parts][6]
            p2_final = self.unique[parts][7]

            my_list.append(
                [
                    _unique_id,
                    orientation,
                    links_p1,
                    links_p2,
                    best_p1,
                    best_p2,
                    p1_final,
                    p2_final,
                    geom_line,
                ]
            )

        edge_gdf = gpd.GeoDataFrame(
            my_list,
            columns=[
                "_unique_id",
                "orientation",
                "links_p1",
                "links_p2",
                "best_p1",
                "best_p2",
                "p1_final",
                "p2_final",
                "geometry",
            ],
            crs=self.gdf_projection,
        )
        edge_gdf.set_index("_unique_id", inplace=True)

        return edge_gdf

    def _create_gdf_strokes(self):
        my_list = []

        for a in self.merged:
            # get all segment points and make line strings
            linelist = _tuple_to_list(list(self.merged[a]))
            list_lines_segments = []

            for b in linelist:
                list_lines_segments.append(LineString(b))

            geom_multi_line = shapely.line_merge(MultiLineString(list_lines_segments))

            # get other values for gdf
            id_value = a
            n_segments = len(self.merged[a])

            my_list.append([id_value, n_segments, geom_multi_line])

        edge_gdf = gpd.GeoDataFrame(
            my_list,
            columns=["stroke_group", "n_segments", "geometry"],
            crs=self.gdf_projection,
        )
        edge_gdf.set_index("stroke_group", inplace=True)

        return edge_gdf

    def _add_gdf_stroke_attributes(self):
        # Invert self.edge_idx to get a dictionary where the key is
        # the original edge index and the value is the group
        inv_edges = {
            value: key for key in self.edge_idx for value in self.edge_idx[key]
        }

        stroke_group_attributes = []

        for edge in self.uv_index:
            stroke_group_attributes.append(inv_edges[edge])

        return pd.Series(stroke_group_attributes, index=self.edge_gdf.index)


def _tuple_to_list(line):
    """
    The imported shapefile lines comes as tuple, whereas the export requires list,
    this function converts tuples inside lines to lists.
    """
    for a in range(0, len(line)):
        line[a] = list(line[a])
    return line


def _list_to_tuple(line):
    for a in range(0, len(line)):
        line[a] = tuple(line[a])
    return tuple(line)


def _list_to_pairs(in_list):
    """Split a line at every point."""
    out_list = []
    index = 0
    for index in range(0, len(in_list) - 1):
        temp_list = [list(in_list[index]), list(in_list[index + 1])]
        out_list.append(temp_list)
    return out_list


def _compute_angle(point1, point2):
    """Calculates the angle between two points in space."""
    height = abs(point2[1] - point1[1])
    base = abs(point2[0] - point1[0])
    angle = round(math.degrees(math.atan(height / base)), 3)
    return angle


def _compute_orientation(line):
    """Calculates the orientation of a line segment. Point1 is
    the lower one on the y-axes and vice versa for Point2.
    """
    point1 = line[1]
    point2 = line[0]

    # If the latutide of a point is less and the longitude is more, or
    # If the latitude of a point is more and the longitude is less, then
    # the point is oriented leftward and wil have negative orientation.
    if ((point2[0] > point1[0]) and (point2[1] < point1[1])) or (
        (point2[0] < point1[0]) and (point2[1] > point1[1])
    ):
        return -_compute_angle(point1, point2)

    # if the latitudes are same, the line is horizontal
    elif point2[1] == point1[1]:
        return 0

    # if the longitudes are same, the line is vertical
    elif point2[0] == point1[0]:
        return 90

    return _compute_angle(point1, point2)


def _points_set_angle(line1, line2):
    """Calculate the acute joining angle between two given set of points."""
    l1orien = _compute_orientation(line1)
    l2orien = _compute_orientation(line2)

    if ((l1orien > 0) and (l2orien < 0)) or ((l1orien < 0) and (l2orien > 0)):
        return abs(l1orien) + abs(l2orien)

    elif ((l1orien > 0) and (l2orien > 0)) or ((l1orien < 0) and (l2orien < 0)):
        theta1 = abs(l1orien) + 180 - abs(l2orien)
        theta2 = abs(l2orien) + 180 - abs(l1orien)
        if theta1 < theta2:
            return theta1
        else:
            return theta2

    elif (l1orien == 0) or (l2orien == 0):
        if l1orien < 0:
            return 180 - abs(l1orien)
        elif l2orien < 0:
            return 180 - abs(l2orien)
        else:
            return 180 - (
                abs(_compute_orientation(line1)) + abs(_compute_orientation(line2))
            )

    elif l1orien == l2orien:
        return 180


def _angle_between_two_lines(line1, line2):
    """Calculate the joining angle between two line segments."""
    l1p1, l1p2 = line1
    l2p1, l2p2 = line2
    l1orien = _compute_orientation(line1)
    l2orien = _compute_orientation(line2)

    # If both lines have same orientation, return 180 If one of the
    # lines is zero, exception for that If both the lines are on same side
    # of the horizontal plane, calculate 180-(sumOfOrientation) If both the
    # lines are on same side of the vertical plane, calculate pointSetAngle.
    if l1orien == l2orien:
        angle = 180

    elif (l1orien == 0) or (l2orien == 0):
        angle = _points_set_angle(line1, line2)

    elif l1p1 == l2p1:
        if ((l1p1[1] > l1p2[1]) and (l1p1[1] > l2p2[1])) or (
            (l1p1[1] < l1p2[1]) and (l1p1[1] < l2p2[1])
        ):
            angle = 180 - (abs(l1orien) + abs(l2orien))
        else:
            angle = _points_set_angle([l1p1, l1p2], [l2p1, l2p2])

    elif l1p1 == l2p2:
        if ((l1p1[1] > l2p1[1]) and (l1p1[1] > l1p2[1])) or (
            (l1p1[1] < l2p1[1]) and (l1p1[1] < l1p2[1])
        ):
            angle = 180 - (abs(l1orien) + abs(l2orien))
        else:
            angle = _points_set_angle([l1p1, l1p2], [l2p2, l2p1])

    elif l1p2 == l2p1:
        if ((l1p2[1] > l1p1[1]) and (l1p2[1] > l2p2[1])) or (
            (l1p2[1] < l1p1[1]) and (l1p2[1] < l2p2[1])
        ):
            angle = 180 - (abs(l1orien) + abs(l2orien))
        else:
            angle = _points_set_angle([l1p2, l1p1], [l2p1, l2p2])

    elif l1p2 == l2p2:
        if ((l1p2[1] > l1p1[1]) and (l1p2[1] > l2p1[1])) or (
            (l1p2[1] < l1p1[1]) and (l1p2[1] < l2p1[1])
        ):
            angle = 180 - (abs(l1orien) + abs(l2orien))
        else:
            angle = _points_set_angle([l1p2, l1p1], [l2p2, l2p1])

    return angle


def _merge_lines_loop(n, unique_dict):
    outlist = set()
    current_edge1 = n

    outlist.add(current_edge1)

    while True:
        if (
            isinstance(unique_dict[current_edge1][6], int)
            and unique_dict[current_edge1][6] not in outlist
        ):
            current_edge1 = unique_dict[current_edge1][6]
            outlist.add(current_edge1)
        elif (
            isinstance(unique_dict[current_edge1][7], int)
            and unique_dict[current_edge1][7] not in outlist
        ):
            current_edge1 = unique_dict[current_edge1][7]
            outlist.add(current_edge1)
        else:
            break

    current_edge1 = n
    while True:
        if (
            isinstance(unique_dict[current_edge1][7], int)
            and unique_dict[current_edge1][7] not in outlist
        ):
            current_edge1 = unique_dict[current_edge1][7]
            outlist.add(current_edge1)
        elif (
            isinstance(unique_dict[current_edge1][6], int)
            and unique_dict[current_edge1][6] not in outlist
        ):
            current_edge1 = unique_dict[current_edge1][6]
            outlist.add(current_edge1)
        else:
            break

    outlist = list(outlist)
    outlist.sort()
    return outlist
