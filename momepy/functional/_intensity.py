import numpy as np
import pandas as pd
import shapely
from geopandas import GeoDataFrame, GeoSeries
from libpysal.graph import Graph
from pandas import Series

__all__ = ["courtyards", "density"]


def courtyards(geometry: GeoDataFrame | GeoSeries, graph: Graph) -> Series:
    """Calculate the number of courtyards within the joined structure.

    Adapted from :cite:`schirmer2015`.

    Parameters
    ----------
    geometry : GeoDataFrame
        A GeoDataFrame containing objects to analyse.
    graph : libpysal.graph.Graph
        A spatial weights matrix for the geodataframe,
        it is used to denote adjacent buildings.

    Returns
    -------
    Series
        A Series containing the resulting values.

    Examples
    --------
    >>> courtyards = mm.calculate_courtyards(buildings_df, graph)
    """

    def _calculate_courtyards(group):
        """helper function to carry out the per group calculations"""
        return shapely.get_num_interior_rings(
            shapely.union_all(shapely.buffer(group.values, 0.01))
        )

    # calculate per group courtyards
    temp_result = (
        geometry["geometry"]
        .groupby(graph.component_labels.values)
        .apply(_calculate_courtyards)
    )
    # assign group courtyard values to each building in the group
    ## note: this is faster than using transform directly
    result = Series(
        temp_result.loc[graph.component_labels.values].values, index=geometry.index
    )

    return result


def density(values, areas, graph) -> pd.Series:
    """
    Calculate the gross density.

    .. math::
        \\frac{\\sum \\text {values}}{\\sum \\text {areas}}

    Adapted from :cite:`dibble2017`.

    Parameters
    ----------
    values : pd.Series, pd.DataFrame
        The character values for density calculations.
        The index is used to arrange the final results.
    areas : np.array, pd.Series
        The area values for the density calculations,
        an ``np.array``, or ``pd.Series``.
    graph : libpysal.graph.Graph
        A spatial weights matrix for the geodataframe,
        it is used to denote adjacent elements.

    Returns
    -------
    DataFrame


    Examples
    --------
    >>> tessellation_df['floor_area_dens'] = mm.density(tessellation_df['floor_area'],
    ...                                                 tessellation_df['area'],
    ...                                                 graph)
    """

    if isinstance(values, np.ndarray):
        values = pd.DataFrame(values)
    elif isinstance(values, pd.Series):
        values = values.to_frame()

    if isinstance(areas, np.ndarray):
        areas = pd.Series(values)

    stats = graph.apply(
        pd.concat((values, areas.rename("area")), axis=1),
        lambda x: (x.loc[:, x.columns != "area"].sum() / x["area"].sum()),
    )
    result = pd.DataFrame(
        np.full(values.shape, np.nan), index=values.index, columns=values.columns
    )
    result[values.columns] = stats[values.columns]
    return result
