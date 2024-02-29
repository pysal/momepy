import geopandas as gpd
import numpy as np
import shapely
from geopandas import GeoDataFrame, GeoSeries
from packaging.version import Version
from pandas import Series

__all__ = ["orientation", "shared_walls"]

GPD_GE_013 = Version(gpd.__version__) >= Version("0.13.0")


def orientation(geometry: GeoDataFrame | GeoSeries) -> Series:
    """Calculate the orientation of object.

    The 'orientation' is defined as the deviation of orientation of the bounding
    rectangle from cardinal directions. As such it is within a range 0 - 45. The
    orientation of LineStrings is represented by the orientation of the line connecting
    the first and the last point of the segment.

    Adapted from :cite:`schirmer2015`.

    Parameters
    ----------
    geometry : GeoDataFrame | GeoSeries
        A GeoDataFrame or GeoSeries containing polygons to analyse.

    Returns
    -------
    Series
    """
    geom_types = geometry.geom_type
    poly_mask = geom_types.str.contains("Polygon")
    line_mask = geom_types.str.contains("Line")

    result = np.full(len(geometry), np.nan, dtype=float)

    if poly_mask.any():
        bboxes = shapely.minimum_rotated_rectangle(geometry.geometry.loc[poly_mask])
        coords = shapely.get_coordinates(bboxes)
        pt0 = coords[::5]
        pt1 = coords[1::5]
        angle = np.arctan2(pt1[:, 0] - pt0[:, 0], pt1[:, 1] - pt0[:, 1])
        result[poly_mask] = np.degrees(angle)

    if line_mask.any():
        first = shapely.get_point(geometry.geometry, 0)
        last = shapely.get_point(geometry.geometry, -1)
        pt0 = shapely.get_coordinates(first)
        pt1 = shapely.get_coordinates(last)
        angle = np.arctan2(pt1[:, 0] - pt0[:, 0], pt1[:, 1] - pt0[:, 1])
        result[line_mask] = np.degrees(angle)

    return Series(
        np.abs((result + 45) % 90 - 45),
        index=geometry.index,
        dtype=float,
        name="orientation",
    )


def shared_walls(geometry: GeoDataFrame | GeoSeries) -> Series:
    """Calculate the length of shared walls of adjacent elements (typically buildings).

    Note that data needs to be topologically correct. Overlapping polygons will lead to
    incorrect results.

    Adapted from :cite:`hamaina2012a`.

    Parameters
    ----------
    geometry : GeoDataFrame | GeoSeries
        A GeoDataFrame or GeoSeries containing polygons to analyse.

    Returns
    -------
    Series
    """
    if GPD_GE_013:
        inp, res = geometry.sindex.query(geometry.geometry, predicate="touches")
    else:
        inp, res = geometry.sindex.query_bulk(geometry.geometry, predicate="touches")
    left = geometry.geometry.take(inp).reset_index(drop=True)
    right = geometry.geometry.take(res).reset_index(drop=True)
    intersections = left.intersection(right).length
    walls = intersections.groupby(inp).sum()
    walls.index = geometry.index.take(walls.index)

    results = Series(0.0, index=geometry.index, name="shared_walls")
    results.loc[walls.index] = walls
    return results
