import numpy as np
import pandas as pd
import shapely

__all__ = [
    "volume",
    "floor_area",
    "courtyard_area",
    "longest_axis_length",
    "perimeter_wall",
]


def volume(area, height):
    """
    Calculates volume of each object in given GeoDataFrame based on its height and area.

    .. math::
        area * height

    Parameters
    ----------
    area : array_like
        array of areas
    height : array_like
        array of heights

    Returns
    -------
    array_like
    """
    return area * height


def floor_area(area, height, floor_height=3):
    """Calculates floor area of each object based on height and area.

    The number of
    floors is simplified into the formula: ``height // floor_height``. B
    y default one floor is approximated to 3 metres.

    .. math::
        area * \\frac{height}{floor_height}


    Parameters
    ----------
    area : array_like
        array of areas
    height : array_like
        array of heights
    floor_height : float | array_like, optional
        float denoting the uniform floor height or an array_like reflecting the building
        height by geometry, by default 3

    Returns
    -------
    array_like
    """
    return area * (height // floor_height)


def courtyard_area(gdf):
    """Calculates area of holes within geometry - area of courtyards.

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing objects to analyse.

    Returns
    -------
    pandas.Series
    """
    return pd.Series(
        shapely.area(shapely.polygons(shapely.get_exterior_ring(gdf.geometry.array)))
        - gdf.area,
        index=gdf.index,
        name="courtyard_area",
    )


def longest_axis_length(gdf):
    """Calculates the length of the longest axis of object.

    Axis is defined as a
    diameter of minimal bounding circle around the geometry. It does
    not have to be fully inside an object.

    .. math::
        \\max \\left\\{d_{1}, d_{2}, \\ldots, d_{n}\\right\\}

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing objects to analyse.

    Returns
    -------
    pandas.Series
    """
    return shapely.minimum_bounding_radius(gdf.geometry) * 2


def perimeter_wall(gdf, graph=None):
    """
    Calculate the perimeter wall length the joined structure.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects to analyse
    graph : libpysal.graph.Graph, optional
        Graph encoding Queen contiguity of ``gdf``

    Returns
    -------
    pandas.Series
    """

    if graph is None:
        from libpysal.graph import Graph

        graph = Graph.build_contiguity(gdf)

    isolates = graph.isolates

    # measure perimeter walls of connected components while ignoring isolates
    blocks = gdf.drop(isolates)
    component_perimeter = (
        blocks[[blocks.geometry.name]]
        .set_geometry(blocks.buffer(0.01))
        .dissolve(by=graph.component_labels.drop(isolates))
        .exterior.length
    )

    # combine components with isolates
    results = pd.Series(np.nan, index=gdf.index, name="perimeter_wall")
    results.loc[isolates] = gdf.geometry[isolates].exterior.length
    results.loc[results.index.drop(isolates)] = component_perimeter.loc[
        graph.component_labels.loc[results.index.drop(isolates)]
    ].values

    return results
