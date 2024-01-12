import numpy as np
import shapely
from geopandas import GeoDataFrame, GeoSeries
from libpysal.graph import Graph
from numpy.typing import NDArray
from pandas import Series

__all__ = [
    "volume",
    "floor_area",
    "courtyard_area",
    "longest_axis_length",
    "perimeter_wall",
]


def volume(
    area: NDArray[np.float_] | Series,
    height: NDArray[np.float_] | Series,
) -> NDArray[np.float_] | Series:
    """
    Calculates volume of each object in given GeoDataFrame based on its height and area.

    .. math::
        area * height

    Parameters
    ----------
    area : NDArray[np.float_] | Series
        array of areas
    height : NDArray[np.float_] | Series
        array of heights

    Returns
    -------
    NDArray[np.float_] | Series
        array of a type depending on the input
    """
    return area * height


def floor_area(
    area: NDArray[np.float_] | Series,
    height: NDArray[np.float_] | Series,
    floor_height: float | NDArray[np.float_] | Series = 3,
) -> NDArray[np.float_] | Series:
    """Calculates floor area of each object based on height and area.

    The number of
    floors is simplified into the formula: ``height // floor_height``. B
    y default one floor is approximated to 3 metres.

    .. math::
        area * \\frac{height}{floor_height}


    Parameters
    ----------
    area : NDArray[np.float_] | Series
        array of areas
    height : NDArray[np.float_] | Series
        array of heights
    floor_height : float | NDArray[np.float_] | Series, optional
        float denoting the uniform floor height or an aarray reflecting the building
        height by geometry, by default 3

    Returns
    -------
    NDArray[np.float_] | Series
        array of a type depending on the input
    """
    return area * (height // floor_height)


def courtyard_area(gdf: GeoDataFrame | GeoSeries) -> Series:
    """Calculates area of holes within geometry - area of courtyards.

    Parameters
    ----------
    gdf : GeoDataFrame | GeoSeries
        A GeoDataFrame or GeoSeries containing polygons to analyse.

    Returns
    -------
    Series
    """
    return Series(
        shapely.area(shapely.polygons(shapely.get_exterior_ring(gdf.geometry.array)))
        - gdf.area,
        index=gdf.index,
        name="courtyard_area",
    )


def longest_axis_length(gdf: GeoDataFrame | GeoSeries) -> Series:
    """Calculates the length of the longest axis of object.

    Axis is defined as a
    diameter of minimal bounding circle around the geometry. It does
    not have to be fully inside an object.

    .. math::
        \\max \\left\\{d_{1}, d_{2}, \\ldots, d_{n}\\right\\}

    Parameters
    ----------
    gdf : GeoDataFrame | GeoSeries
        A GeoDataFrame or GeoSeries containing polygons to analyse.

    Returns
    -------
    Series
    """
    return shapely.minimum_bounding_radius(gdf.geometry) * 2


def perimeter_wall(gdf: GeoDataFrame | GeoSeries, graph: Graph | None = None) -> Series:
    """
    Calculate the perimeter wall length the joined structure.

    Parameters
    ----------
    gdf : GeoDataFrame | GeoSeries
        A GeoDataFrame or GeoSeries containing polygons to analyse.
    graph : Graph | None, optional
        Graph encoding Queen contiguity of ``gdf``. If ``None`` Queen contiguity is
        built on the fly.

    Returns
    -------
    Series
    """

    if graph is None:
        graph = Graph.build_contiguity(gdf)

    isolates = graph.isolates

    # measure perimeter walls of connected components while ignoring isolates
    blocks = gdf.drop(isolates)
    component_perimeter = (
        blocks[[blocks.geometry.name]]
        .set_geometry(blocks.buffer(0.01))  # type: ignore
        .dissolve(by=graph.component_labels.drop(isolates))
        .exterior.length
    )

    # combine components with isolates
    results = Series(np.nan, index=gdf.index, name="perimeter_wall")
    results.loc[isolates] = gdf.geometry[isolates].exterior.length
    results.loc[results.index.drop(isolates)] = component_perimeter.loc[
        graph.component_labels.loc[results.index.drop(isolates)]
    ].values

    return results
