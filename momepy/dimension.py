#!/usr/bin/env python

# dimension.py
# definitions of dimension characters

import math

import geopandas as gpd
import numpy as np
import pandas as pd
import scipy as sp
import shapely
from packaging.version import Version
from tqdm.auto import tqdm

from .shape import _circle_radius
from .utils import deprecated, removed

GPD_GE_10 = Version(gpd.__version__) >= Version("1.0dev")


__all__ = [
    "Area",
    "Perimeter",
    "Volume",
    "FloorArea",
    "CourtyardArea",
    "LongestAxisLength",
    "AverageCharacter",
    "StreetProfile",
    "WeightedCharacter",
    "CoveredArea",
    "PerimeterWall",
    "SegmentsLength",
]


@removed("`.area` attribute of a GeoDataFrame")
class Area:
    """
    Calculates the area of each object in a given GeoDataFrame. It can be used for any
    suitable element (building footprint, plot, tessellation, block). It is a simple
    wrapper for GeoPandas ``.area`` for the consistency of momepy.

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing objects to analyse.

    Attributes
    ----------
    series : Series
        A Series containing resulting values.
    gdf : GeoDataFrame
        The original GeoDataFrame.

    Examples
    --------
    >>> buildings = gpd.read_file(momepy.datasets.get_path('bubenec'),
    ...                           layer='buildings')
    >>> buildings['area'] = momepy.Area(buildings).series
    >>> buildings.area[0]
    728.5574947044363

    """

    def __init__(self, gdf):
        self.gdf = gdf
        self.series = self.gdf.geometry.area


@removed("`.length` attribute of a GeoDataFrame")
class Perimeter:
    """
    Calculates perimeter of each object in a given GeoDataFrame. It can be used for any
    suitable element (building footprint, plot, tessellation, block). It is a simple
    wrapper for GeoPandas ``.length`` for the consistency of momepy.

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing objects to analyse.

    Attributes
    ----------
    series : Series
        A Series containing resulting values.
    gdf : GeoDataFrame
        The original GeoDataFrame.

    Examples
    --------
    >>> buildings = gpd.read_file(momepy.datasets.get_path('bubenec'),
    ...                           layer='buildings')
    >>> buildings['perimeter'] = momepy.Perimeter(buildings).series
    >>> buildings.perimeter[0]
    137.18630991119903
    """

    def __init__(self, gdf):
        self.gdf = gdf
        self.series = self.gdf.geometry.length


@deprecated("volume")
class Volume:
    """
    Calculates the volume of each object in a
    given GeoDataFrame based on its height and area.

    .. math::
        area * height

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing objects to analyse.
    heights : str, list, np.array, pd.Series
        The name of the dataframe column, ``np.array``, or ``pd.Series``
        where height values are stored.
    areas : str, list, np.array, pd.Series (default None)
        The name of the dataframe column, ``np.array``, or ``pd.Series``
        where area values are stored. If set to ``None``, this will calculate
        areas during the process without saving them separately.

    Attributes
    ----------
    series : Series
        A Series containing resulting values.
    gdf : GeoDataFrame
        The original GeoDataFrame.
    heights : Series
        A Series containing used heights values.
    areas : GeoDataFrame
        A Series containing used areas values.

    Examples
    --------
    >>> buildings['volume'] = momepy.Volume(buildings, heights='height_col').series
    >>> buildings.volume[0]
    7285.5749470443625

    >>> buildings['volume'] = momepy.Volume(buildings, heights='height_col',
    ...                                     areas='area_col').series
    >>> buildings.volume[0]
    7285.5749470443625
    """

    def __init__(self, gdf, heights, areas=None):
        self.gdf = gdf

        gdf = gdf.copy()
        if not isinstance(heights, str):
            gdf["mm_h"] = heights
            heights = "mm_h"
        self.heights = gdf[heights]

        if areas is not None:
            if not isinstance(areas, str):
                gdf["mm_a"] = areas
                areas = "mm_a"
            self.areas = gdf[areas]
        else:
            self.areas = gdf.geometry.area
        try:
            self.series = self.areas * self.heights

        except KeyError as err:
            raise KeyError(
                "Column not found. Define heights and areas or set areas to None."
            ) from err


@deprecated("floor_area")
class FloorArea:
    """
    Calculates floor area of each object based on height and area. The number of
    floors is simplified into the formula: height / 3. It is assumed that on
    average one floor is approximately 3 metres.

    .. math::
        area * \\frac{height}{3}

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing objects to analyse.
    heights : str, list, np.array, pd.Series
        The name of the dataframe column, ``np.array``, or ``pd.Series``
        where height values are stored.
    areas : str, list, np.array, pd.Series (default None)
        The name of the dataframe column, ``np.array``, or ``pd.Series``
        where area values are stored. If set to ``None``, this will calculate
        areas during the process without saving them separately.

    Attributes
    ----------
    series : Series
        A Series containing resulting values.
    gdf : GeoDataFrame
        The original GeoDataFrame.
    heights : Series
        A Series containing used heights values.
    areas : GeoDataFrame
        A Series containing used areas values.

    Examples
    --------
    >>> buildings['floor_area'] = momepy.FloorArea(buildings,
    ...                                            heights='height_col').series
    Calculating floor areas...
    Floor areas calculated.
    >>> buildings.floor_area[0]
    2185.672484113309

    >>> buildings['floor_area'] = momepy.FloorArea(buildings, heights='height_col',
    ...                                            areas='area_col').series
    >>> buildings.floor_area[0]
    2185.672484113309
    """

    def __init__(self, gdf, heights, areas=None):
        self.gdf = gdf

        gdf = gdf.copy()
        if not isinstance(heights, str):
            gdf["mm_h"] = heights
            heights = "mm_h"
        self.heights = gdf[heights]

        if areas is not None:
            if not isinstance(areas, str):
                gdf["mm_a"] = areas
                areas = "mm_a"
            self.areas = gdf[areas]
        else:
            self.areas = gdf.geometry.area
        try:
            self.series = self.areas * (self.heights // 3)

        except KeyError as err:
            raise KeyError(
                "Column not found. Define heights and areas or set areas to None."
            ) from err


@deprecated("courtyard_area")
class CourtyardArea:
    """
    Calculates area of holes within geometry - area of courtyards.

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing objects to analyse.
    areas : str, list, np.array, pd.Series (default None)
        The name of the dataframe column, ``np.array``, or ``pd.Series``
        where area values are stored. If set to ``None``, this will calculate
        areas during the process without saving them separately.

    Attributes
    ----------
    series : Series
        Series containing resulting values.
    gdf : GeoDataFrame
        The original GeoDataFrame.
    areas : GeoDataFrame
        A Series containing used areas values.

    Examples
    --------
    >>> buildings['courtyard_area'] = momepy.CourtyardArea(buildings).series
    >>> buildings.courtyard_area[80]
    353.33274206543274
    """

    def __init__(self, gdf, areas=None):
        self.gdf = gdf

        gdf = gdf.copy()
        if areas is None:
            areas = gdf.geometry.area

        if not isinstance(areas, str):
            gdf["mm_a"] = areas
            areas = "mm_a"
        self.areas = gdf[areas]

        exts = shapely.area(
            shapely.polygons(shapely.get_exterior_ring(gdf.geometry.array))
        )

        self.series = pd.Series(exts - gdf[areas], index=gdf.index)


@deprecated("longest_axis_length")
class LongestAxisLength:
    """
    Calculates the length of the longest axis of object. Axis is defined as a
    diameter of minimal circumscribed circle around the convex hull. It does
    not have to be fully inside an object.

    .. math::
        \\max \\left\\{d_{1}, d_{2}, \\ldots, d_{n}\\right\\}

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing objects to analyse.

    Attributes
    ----------
    series : Series
        A Series containing resulting values.
    gdf : GeoDataFrame
        The original GeoDataFrame.

    Examples
    --------
    >>> buildings['lal'] = momepy.LongestAxisLength(buildings).series
    >>> buildings.lal[0]
    40.2655616057102
    """

    def __init__(self, gdf):
        self.gdf = gdf
        hulls = gdf.geometry.convex_hull.exterior
        self.series = hulls.apply(lambda g: _circle_radius(list(g.coords))) * 2


@removed("`.describe()` method of libpysal.graph.Graph")
class AverageCharacter:
    """
    Calculates the average of a character within a set
    neighbourhood defined in ``spatial_weights``. Can be
    set to ``mean``, ``median`` or ``mode``. ``mean`` is defined as:

    .. math::
        \\frac{1}{n}\\left(\\sum_{i=1}^{n} value_{i}\\right)

    Adapted from :cite:`hausleitner2017`.

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing a morphological tessellation.
    values : str, list, np.array, pd.Series
        The name of the dataframe column, ``np.array``, or ``pd.Series``
        where character values are stored.
    unique_id : str
        The name of the column with unique ID used as the ``spatial_weights`` index.
    spatial_weights : libpysal.weights
        A spatial weights matrix.
    rng : tuple, list, optional (default None)
        A two-element sequence containing floats between 0 and 100 (inclusive)
        that are the percentiles over which to compute the range.
        The order of the elements is not important.
    mode : str (default 'all')
        The mode of average calculation. It can be set to ``'all'``, ``'mean'``,
        ``'median'``, or ``'mode'`` or a list of any of the options.
    verbose : bool (default True)
        If ``True``, shows progress bars in loops and indication of steps.

    Attributes
    ----------
    series : Series
        A Series containing resulting mean values.
    mean : Series
        A Series containing resulting mean values.
    median : Series
        A Series containing resulting median values.
    mode : Series
        A Series containing resulting mode values.
    gdf : GeoDataFrame
        The original GeoDataFrame.
    values : GeoDataFrame
        A Series containing used values.
    sw : libpysal.weights
        The spatial weights matrix.
    id : Series
        A Series containing used unique ID.
    rng : tuple
        The range.
    modes : str
        The mode.

    Examples
    --------
    >>> sw = libpysal.weights.DistanceBand.from_dataframe(tessellation,
    ...                                                   threshold=100,
    ...                                                   silence_warnings=True,
    ...                                                   ids='uID')
    >>> tessellation['mean_area'] = momepy.AverageCharacter(tessellation,
    ...                                                     values='area',
    ...                                                     spatial_weights=sw,
    ...                                                     unique_id='uID').mean
    100%|██████████| 144/144 [00:00<00:00, 1433.32it/s]
    >>> tessellation.mean_area[0]
    4823.1334436678835
    """

    def __init__(
        self,
        gdf,
        values,
        spatial_weights,
        unique_id,
        rng=None,
        mode="all",
        verbose=True,
    ):
        self.gdf = gdf
        self.sw = spatial_weights
        self.id = gdf[unique_id]
        self.rng = rng
        self.modes = mode

        if rng:
            from momepy import limit_range

        data = gdf.copy()
        if values is not None and not isinstance(values, str):
            data["mm_v"] = values
            values = "mm_v"
        self.values = data[values]

        data = data.set_index(unique_id)[values]

        means = []
        medians = []
        modes = []

        allowed = ["mean", "median", "mode"]

        if mode == "all":
            mode = allowed
        elif isinstance(mode, list):
            for m in mode:
                if m not in allowed:
                    raise ValueError(f"{m} is not supported as mode.")
        elif isinstance(mode, str):
            if mode not in allowed:
                raise ValueError(f"{mode} is not supported as mode.")
            mode = [mode]

        for index in tqdm(data.index, total=data.shape[0], disable=not verbose):
            if index in spatial_weights.neighbors:
                neighbours = [index]
                neighbours += spatial_weights.neighbors[index]

                values_list = data.loc[neighbours]

                if rng:
                    values_list = limit_range(values_list.values, rng=rng)
                if "mean" in mode:
                    means.append(np.mean(values_list))
                if "median" in mode:
                    medians.append(np.median(values_list))
                if "mode" in mode:
                    modes.append(sp.stats.mode(values_list, keepdims=False)[0])

            else:
                if "mean" in mode:
                    means.append(np.nan)
                if "median" in mode:
                    medians.append(np.nan)
                if "mode" in mode:
                    modes.append(np.nan)

        if "mean" in mode:
            self.series = self.mean = pd.Series(means, index=gdf.index)
        if "median" in mode:
            self.median = pd.Series(medians, index=gdf.index)
        if "mode" in mode:
            self.mode = pd.Series(modes, index=gdf.index)


@deprecated("street_profile")
class StreetProfile:
    """
    Calculates the street profile characters. This functions
    returns a dictionary with widths, standard deviation of width, openness, heights,
    standard deviation of height and ratio height/width. The algorithm generates
    perpendicular lines to the ``right`` dataframe features every ``distance`` and
    measures values on intersections with features of ``left``. If no feature is
    reached within ``tick_length`` its value is set as width (being a theoretical
    maximum).

    Derived from :cite:`araldi2019`.

    Parameters
    ----------
    left : GeoDataFrame
        A GeoDataFrame containing streets to analyse.
    right : GeoDataFrame
        A GeoDataFrame containing buildings along the streets.
        Only Polygon geometries are currently supported.
    heights: str, list, np.array, pd.Series (default None)
        The name of the buildings dataframe column, ``np.array``, or ``pd.Series``
        where building height are stored. If set to ``None``,
        height and ratio height/width will not be calculated.
    distance : int (default 10)
        The distance between perpendicular ticks.
    tick_length : int (default 50)
        The length of ticks.

    Attributes
    ----------
    w : Series
        A Series containing street profile width values.
    wd : Series
        A Series containing street profile standard deviation values.
    o : Series
        A Series containing street profile openness values.
    h : Series
        A Series containing street profile heights
        values that is returned only when ``heights`` is set.
    hd : Series
        A Series containing street profile heights standard deviation
        values that is returned only when ``heights`` is set.
    p : Series
        A Series containing street profile height/width ratio
        values that is returned only when ``heights`` is set.
    left : GeoDataFrame
        The original left GeoDataFrame.
    right : GeoDataFrame
        The original right GeoDataFrame.
    distance : int
        The distance between perpendicular ticks.
    tick_length : int
        The length of ticks.
    heights : GeoDataFrame
        A Series containing used height values.

    Examples
    --------
    >>> street_prof = momepy.StreetProfile(streets_df, buildings_df, heights='height')
    100%|██████████| 33/33 [00:02<00:00, 15.66it/s]
    >>> streets_df['width'] = street_prof.w
    >>> streets_df['deviations'] = street_prof.wd
    """

    def __init__(
        self,
        left,
        right,
        heights=None,
        distance=10,
        tick_length=50,
    ):
        self.left = left
        self.right = right
        self.distance = distance
        self.tick_length = tick_length

        lines = left.geometry.array

        list_points = np.empty((0, 2))
        ids = []
        end_markers = []

        lengths = shapely.length(lines)
        for ix, (line, length) in enumerate(zip(lines, lengths, strict=True)):
            pts = shapely.line_interpolate_point(
                line, np.linspace(0, length, num=int((length) // distance))
            )
            list_points = np.append(list_points, shapely.get_coordinates(pts), axis=0)
            if len(pts) > 1:
                ids += [ix] * len(pts) * 2
                markers = [True] + ([False] * (len(pts) - 2)) + [True]
                end_markers += markers
            elif len(pts) == 1:
                end_markers += [True]
                ids += [ix] * 2

        ticks = []
        for num, (pt, end) in enumerate(zip(list_points, end_markers, strict=True), 1):
            if end:
                ticks.append([pt, pt])
                ticks.append([pt, pt])

            else:
                angle = self._get_angle(pt, list_points[num])
                line_end_1 = self._get_point1(pt, angle, tick_length / 2)
                angle = self._get_angle(line_end_1, pt)
                line_end_2 = self._get_point2(line_end_1, angle, tick_length)
                ticks.append([line_end_1, pt])
                ticks.append([line_end_2, pt])

        ticks = shapely.linestrings(ticks)

        inp, res = shapely.STRtree(right.geometry).query(ticks, predicate="intersects")
        intersections = shapely.intersection(ticks[inp], right.geometry.array[res])
        distances = shapely.distance(
            intersections, shapely.points(list_points[inp // 2])
        )
        inp_uni, inp_cts = np.unique(inp, return_counts=True)
        splitter = np.cumsum(inp_cts)[:-1]
        dist_per_res = np.split(distances, splitter)
        inp_per_res = np.split(res, splitter)

        min_distances = []
        min_inds = []
        for dis, ind in zip(dist_per_res, inp_per_res, strict=True):
            min_distances.append(np.min(dis))
            min_inds.append(ind[np.argmin(dis)])

        dists = np.zeros((len(ticks),))
        dists[:] = np.nan
        dists[inp_uni] = min_distances

        if heights is not None:
            if isinstance(heights, str):
                heights = self.heights = right[heights]
            elif not isinstance(heights, pd.Series):
                heights = self.heights = pd.Series(heights)

            blgs = np.zeros((len(ticks),))
            blgs[:] = None
            blgs[inp_uni] = min_inds
            do_heights = True
        else:
            do_heights = False

        ids = np.array(ids)
        widths = []
        openness = []
        deviations = []
        heights_list = []
        heights_deviations_list = []

        for i in range(len(left)):
            f = ids == i
            s = dists[f]
            lefts = s[::2]
            rights = s[1::2]
            left_mean = np.nanmean(lefts) if ~np.isnan(lefts).all() else tick_length / 2
            right_mean = (
                np.nanmean(rights) if ~np.isnan(rights).all() else tick_length / 2
            )
            widths.append(np.mean([left_mean, right_mean]) * 2)

            f_sum = (f).sum()
            s_nan = np.isnan(s)

            openness_score = np.nan if not f_sum else s_nan.sum() / f_sum
            openness.append(openness_score)

            deviation_score = np.nan if s_nan.all() else np.nanstd(s)
            deviations.append(deviation_score)

            if do_heights:
                b = blgs[f]
                h = heights.iloc[b[~np.isnan(b)]]
                heights_list.append(h.mean())
                heights_deviations_list.append(h.std())

        self.w = pd.Series(widths, index=left.index)
        self.wd = pd.Series(deviations, index=left.index).fillna(
            0
        )  # fill for empty intersections
        self.o = pd.Series(openness, index=left.index).fillna(1)

        if do_heights:
            self.h = pd.Series(heights_list, index=left.index).fillna(
                0
            )  # fill for empty intersections
            self.hd = pd.Series(heights_deviations_list, index=left.index).fillna(
                0
            )  # fill for empty intersections
            self.p = self.h / self.w.replace(0, np.nan)  # replace to avoid np.inf

    # http://wikicode.wikidot.com/get-angle-of-line-between-two-points
    # https://glenbambrick.com/tag/perpendicular/
    # angle between two points
    def _get_angle(self, pt1, pt2):
        """
        pt1, pt2 : tuple
        """
        x_diff = pt2[0] - pt1[0]
        y_diff = pt2[1] - pt1[1]
        return math.degrees(math.atan2(y_diff, x_diff))

    # start and end points of chainage tick
    # get the first end point of a tick
    def _get_point1(self, pt, bearing, dist):
        """
        pt : tuple
        """
        angle = bearing + 90
        bearing = math.radians(angle)
        x = pt[0] + dist * math.cos(bearing)
        y = pt[1] + dist * math.sin(bearing)
        return (x, y)

    # get the second end point of a tick
    def _get_point2(self, pt, bearing, dist):
        """
        pt : tuple
        """
        bearing = math.radians(bearing)
        x = pt[0] + dist * math.cos(bearing)
        y = pt[1] + dist * math.sin(bearing)
        return (x, y)


@deprecated("weighted_character")
class WeightedCharacter:
    """
    Calculates the weighted character. Character weighted by the area
    of the objects within neighbors defined in ``spatial_weights``.

    .. math::
        \\frac{\\sum_{i=1}^{n} {character_{i} * area_{i}}}{\\sum_{i=1}^{n} area_{i}}

    Adapted from :cite:`dibble2017`.

    Parameters
    ----------
    gdf : GeoDataFrame
        The GeoDataFrame containing objects to analyse.
    values : str, list, np.array, pd.Series
        The name of the ``gdf`` dataframe column, ``np.array``, or
        ``pd.Series`` where the characters to be weighted are stored.
    spatial_weights : libpysal.weights
        A spatial weights matrix. If ``None``, Queen contiguity matrix
        of set order will be calculated based on left.
    unique_id : str
        The name of the column with unique ID used as ``spatial_weights`` index.
    areas : str, list, np.array, pd.Series (default None)
        The name of the left dataframe column, ``np.array``, or ``pd.Series``
        where the area values are stored.
    verbose : bool (default True)
        If ``True``, shows progress bars in loops and indication of steps.


    Attributes
    ----------
    series : Series
        A Series containing resulting values.
    gdf : GeoDataFrame
        A original GeoDataFrame.
    values : GeoDataFrame
        A Series containing used values.
    areas : GeoDataFrame
        Series containing used areas.
    sw : libpysal.weights
        The spatial weights matrix.
    id : Series
        A Series containing used unique ID.

    Examples
    --------
    >>> sw = libpysal.weights.DistanceBand.from_dataframe(tessellation_df,
    ...                                                   threshold=100,
    ...                                                   silence_warnings=True)
    >>> buildings_df['w_height_100'] = momepy.WeightedCharacter(buildings_df,
    ...                                                         values='height',
    ...                                                         spatial_weights=sw,
    ...                                                         unique_id='uID').series
    100%|██████████| 144/144 [00:00<00:00, 361.60it/s]
    """

    def __init__(
        self, gdf, values, spatial_weights, unique_id, areas=None, verbose=True
    ):
        self.gdf = gdf
        self.sw = spatial_weights
        self.id = gdf[unique_id]

        data = gdf.copy()
        if areas is None:
            areas = gdf.geometry.area

        if not isinstance(areas, str):
            data["mm_a"] = areas
            areas = "mm_a"
        if not isinstance(values, str):
            data["mm_vals"] = values
            values = "mm_vals"

        self.areas = data[areas]
        self.values = data[values]

        data = data.set_index(unique_id)[[values, areas]]

        results_list = []
        for index in tqdm(data.index, total=data.shape[0], disable=not verbose):
            if index in spatial_weights.neighbors:
                neighbours = [index]
                neighbours += spatial_weights.neighbors[index]

                subset = data.loc[neighbours]
                results_list.append(
                    (sum(subset[values] * subset[areas])) / (sum(subset[areas]))
                )
            else:
                results_list.append(np.nan)

        self.series = pd.Series(results_list, index=gdf.index)


@removed("`.describe()` method of libpysal.graph.Graph")
class CoveredArea:
    """
    Calculates the area covered by neighbours, which is total area covered
    by neighbours defined in ``spatial_weights`` and the element itself.

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing Polygon geometries.
    spatial_weights : libpysal.weights
        A spatial weights matrix.
    unique_id : str
        The name of the column with unique ID used as ``spatial_weights`` index.
    verbose : bool (default True)
        If ``True``, shows progress bars in loops and indication of steps.

    Attributes
    ----------
    series : Series
        A Series containing resulting values.
    gdf : GeoDataFrame
        The original GeoDataFrame.
    sw : libpysal.weights
        The spatial weights matrix.
    id : Series
        A Series containing used unique ID.

    Examples
    --------
    >>> sw = momepy.sw_high(k=3, gdf=tessellation_df, ids='uID')
    >>> tessellation_df['covered'] = mm.CoveredArea(tessellation_df, sw, 'uID').series
    100%|██████████| 144/144 [00:00<00:00, 549.15it/s]

    """

    def __init__(self, gdf, spatial_weights, unique_id, verbose=True):
        self.gdf = gdf
        self.sw = spatial_weights
        self.id = gdf[unique_id]

        data = gdf
        area = data.set_index(unique_id).geometry.area

        results_list = []
        for index in tqdm(area.index, total=area.shape[0], disable=not verbose):
            if index in spatial_weights.neighbors:
                neighbours = [index]
                neighbours += spatial_weights.neighbors[index]

                areas = area.loc[neighbours]
                results_list.append(sum(areas))
            else:
                results_list.append(np.nan)

        self.series = pd.Series(results_list, index=gdf.index)


@deprecated("perimeter_wall")
class PerimeterWall:
    """
    Calculate the perimeter wall length of the joined structure.

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing objects to analyse.
    spatial_weights : libpysal.weights, optional
        A spatial weights matrix. If ``None``, Queen contiguity matrix will
        be calculated based on ``gdf``. It is to denote adjacent buildings.
    verbose : bool (default True)
        If ``True``, shows progress bars in loops and indication of steps.

    Attributes
    ----------
    series : Series
        A Series containing resulting values.
    gdf : GeoDataFrame
        The original GeoDataFrame.
    sw : libpysal.weights
        The spatial weights matrix.

    Examples
    --------
    >>> buildings_df['wall_length'] = mm.PerimeterWall(buildings_df).series
    Calculating spatial weights...
    Spatial weights ready...
    100%|██████████| 144/144 [00:00<00:00, 4171.39it/s]

    Notes
    -----
    The ``spatial_weights`` keyword argument should be
    based on *position*, not unique ID.

    It might take a while to compute this character.
    """

    def __init__(self, gdf, spatial_weights=None, verbose=True):
        self.gdf = gdf

        if spatial_weights is None:
            print("Calculating spatial weights...") if verbose else None
            from libpysal.weights import Queen

            spatial_weights = Queen.from_dataframe(
                gdf, silence_warnings=True, use_index=False
            )
            print("Spatial weights ready...") if verbose else None
        self.sw = spatial_weights

        # dict to store walls for each uID
        walls = {}
        components = pd.Series(spatial_weights.component_labels, index=range(len(gdf)))
        geom = gdf.geometry

        for i in tqdm(range(gdf.shape[0]), total=gdf.shape[0], disable=not verbose):
            # if the id is already present in walls, continue (avoid repetition)
            if i in walls:
                continue
            else:
                comp = spatial_weights.component_labels[i]
                to_join = components[components == comp].index
                joined = geom.iloc[to_join]
                # buffer to avoid multipolygons where buildings touch by corners only
                dissolved = (
                    joined.buffer(0.01).union_all()
                    if GPD_GE_10
                    else joined.buffer(0.01).unary_union
                )
                for b in to_join:
                    walls[b] = dissolved.exterior.length

        results_list = []
        for i in tqdm(range(gdf.shape[0]), total=gdf.shape[0], disable=not verbose):
            results_list.append(walls[i])
        self.series = pd.Series(results_list, index=gdf.index)


@removed("`.describe()` or `.lag()` methods of libpysal.graph.Graph")
class SegmentsLength:
    """
    Calculate the cummulative and/or mean length of segments. Length of segments
    within set topological distance from each of them. Reached topological distance
    should be captured by ``spatial_weights``.  If ``mean=False`` it will compute
    sum of length, if ``mean=True`` it will compute sum and mean.

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing streets (edges) to analyse.
    spatial_weights : libpysal.weights, optional
        A spatial weights matrix. If ``None``, Queen contiguity
        matrix will be calculated based on streets.
    mean : bool, optional
        If ``mean=False`` it will compute sum of length, if ``mean=True``
         it will compute sum and mean.
    verbose : bool (default True)
        If ``True``, shows progress bars in loops and indication of steps.

    Attributes
    ----------
    series : Series
        A Series containing resulting total lengths.
    mean : Series
        A Series containing resulting total lengths.
    sum : Series
        A Series containing resulting total lengths.
    gdf : GeoDataFrame
        The original GeoDataFrame
    sw : libpysal.weights
        The spatial weights matrix.

    Examples
    --------
    >>> streets_df['length_neighbours'] = mm.SegmentsLength(streets_df, mean=True).mean
    Calculating spatial weights...
    Spatial weights ready...

    Notes
    -----
    The ``spatial_weights`` keyword argument should be based on *index*, not unique ID.
    """

    def __init__(self, gdf, spatial_weights=None, mean=False, verbose=True):
        self.gdf = gdf

        if spatial_weights is None:
            print("Calculating spatial weights...") if verbose else None
            from libpysal.weights import Queen

            spatial_weights = Queen.from_dataframe(
                gdf, silence_warnings=True, use_index=False
            )
            print("Spatial weights ready...") if verbose else None
        self.sw = spatial_weights

        lenghts = gdf.geometry.length

        sums = []
        means = []
        for index in tqdm(gdf.index, total=gdf.shape[0], disable=not verbose):
            neighbours = [index]
            neighbours += spatial_weights.neighbors[index]

            dims = lenghts.iloc[neighbours]
            if mean:
                means.append(np.mean(dims))
            sums.append(sum(dims))

        self.series = self.sum = pd.Series(sums, index=gdf.index)
        if mean:
            self.mean = pd.Series(means, index=gdf.index)
