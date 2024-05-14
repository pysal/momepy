import numpy as np
import pandas as pd
import shapely
from geopandas import GeoDataFrame, GeoSeries
from libpysal.graph import Graph
from pandas import Series

__all__ = ["courtyards", "node_density"]


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


def node_density(
    left: GeoDataFrame, right: GeoDataFrame, graph: Graph, weighted: bool = False
) -> Series:
    """Calculate the density of nodes neighbours on street network defined in
    ``graph``.

    Calculated as the number of neighbouring
    nodes / cumulative length of street network within neighbours.
    ``node_start``,  ``node_end``, is standard output of
    :py:func:`momepy.nx_to_gdf` and is compulsory for ``right`` to have
    these columns.
    If ``weighted``, a ``degree`` column is also required in ``left``.

    Adapted from :cite:`dibble2017`.

    Parameters
    ----------
    left : GeoDataFrame
        A GeoDataFrame containing nodes of street network.
    right : GeoDataFrame
        A GeoDataFrame containing edges of street network.
    graph :  libpysal.graph.Graph
        A spatial weights matrix capturing relationship between nodes.
    weighted : bool (default False)
        If ``True``, density will take into account node degree as ``k-1``.

    Returns
    -------
    Series
        A Series containing resulting values.

    Examples
    --------
    >>> nodes['density'] = mm.node_density(nodes, edges, graph)
    """

    required_cols = ["node_start", "node_end"]
    if not np.isin(required_cols, right.columns).all():
        raise ValueError(
            f"Columns { *required_cols, } are needed in "
            "right GeoDataframe for the calculations."
        )
    if weighted and ("degree" not in left.columns):
        raise ValueError(
            "Degree column is needed in " "left GeoDataframe for density calculations."
        )

    def _calc_nodedensity(group, edges):
        """ "Helper function to calculate group values."""
        neighbours = group.index.values
        locs = np.in1d(edges["node_start"], neighbours) & np.in1d(
            edges["node_end"], neighbours
        )
        lengths = edges.loc[locs].geometry.length.sum()
        return group.sum() / lengths if lengths else 0

    if weighted:
        summation_values = left["degree"] - 1
    else:
        summation_values = pd.Series(np.ones(left.shape[0]), index=left.index)

    return graph.apply(summation_values, _calc_nodedensity, edges=right)
