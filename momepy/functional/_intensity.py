import numpy as np
import pandas as pd
import shapely
from geopandas import GeoDataFrame, GeoSeries
from libpysal.graph import Graph
from pandas import Series

__all__ = ["courtyards", "area_ratio"]


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


def area_ratio(
    left: Series, right: Series, right_group_key: Series | np.ndarray
) -> pd.Series:
    """
    Calculate covered area ratio or floor area ratio of objects.
    .. math::
        \\textit{covering object area} \\over \\textit{covered object area}

    Adapted from :cite:`schirmer2015`.

    Parameters
    ----------
    left : Series
        A GeoDataFrame with the areas of the objects being covered (e.g. land unit).
    right : Series
        A GeoDataFrame with the areas of the covering objects (e.g. building).
    right_group_key: np.array | pd.Series
        The group key that assigns objects from ``right`` to ``left``.

    Returns
    -------
    Series


    Examples
    --------
    >>> tessellation_df['CAR'] = mm.area_ratio(tessellation_df['area'],
    ...                                       buildings_df['area'],
    ...                                       buildings_df['uID'])
    """

    if isinstance(right_group_key, np.ndarray):
        right_group_key = pd.Series(right_group_key, index=right.index)

    results = pd.Series(np.nan, left.index)
    stats = (
        right.loc[right_group_key.index.values].groupby(right_group_key.values).sum()
    )
    results.loc[stats.index.values] = stats.values

    return results / left.values
