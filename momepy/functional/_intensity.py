import numpy as np
import pandas as pd
import shapely
from geopandas import GeoDataFrame, GeoSeries
from libpysal.graph import Graph
from pandas import Series

__all__ = ["courtyards", "node_density", "block_counts"]


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
    nodes: GeoDataFrame, edges: GeoDataFrame, graph: Graph, weighted: bool = False
) -> Series:
    """Calculate the density of a node's neighbours (for all nodes)
    on the street network defined in ``graph``.

    Calculated as the number of neighbouring
    nodes / cumulative length of street network within neighbours.
    ``node_start``,  ``node_end``, is standard output of
    :py:func:`momepy.nx_to_gdf` and is compulsory for ``edges`` to have
    these columns.
    If ``weighted``, a ``degree`` column is also required in ``nodes``.

    Adapted from :cite:`dibble2017`.

    Parameters
    ----------
    nodes : GeoDataFrame
        A GeoDataFrame containing nodes of a street network.
    edges : GeoDataFrame
        A GeoDataFrame containing edges of a street network.
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
    for col in required_cols:
        if col not in edges.columns:
            raise ValueError(f"Column {col} is needed in the edges GeoDataframe.")

    if weighted and ("degree" not in nodes.columns):
        raise ValueError("Column degree is needed in nodes GeoDataframe.")

    def _calc_nodedensity(group, edges):
        """Helper function to calculate group values."""
        neighbours = group.index.values
        locs = np.in1d(edges["node_start"], neighbours) & np.in1d(
            edges["node_end"], neighbours
        )
        lengths = edges.loc[locs].geometry.length.sum()
        return group.sum() / lengths if lengths else 0

    if weighted:
        summation_values = nodes["degree"] - 1
    else:
        summation_values = pd.Series(np.ones(nodes.shape[0]), index=nodes.index)

    return graph.apply(summation_values, _calc_nodedensity, edges=edges)


def block_counts(
    group_key: pd.Series | np.ndarray, graph: Graph, areas=None
) -> pd.Series:
    """
    Calculates the weighted number of blocks. The number of blocks within neighbours
    defined in ``graph`` divided by the area covered by the neighbours.

    Adapted from :cite:`dibble2017`.

    Parameters
    ----------
    group_key: | pd.Series
        The group key that denotes block membership.
        has to be Series, to index the results.
    graph : libpysal.graph.Graph
        A spatial weights matrix for ``left``.
    areas : Series, default None
        Areas of the tesslations, if areas is none return pure count (``False``).

    Returns
    -------
    Series


    Examples
    --------
    >>> tessellation_df['blocks_within_4'] = mm.block_counts(
    ...                                       df_tessellation['bID'],
    ...                                       graph,
    ...                                       buildings_df['uID'])
    """

    if areas is not None:
        stats = graph.apply(
            pd.concat([group_key, areas], axis=1),
            lambda x: x.iloc[:, 0].unique().shape[0] / x.iloc[:, 1].sum(),
        )

    else:
        stats = graph.apply(group_key, lambda x: x.unique().shape[0])

    results = pd.Series(np.nan, group_key.index)
    results.loc[stats.index.values] = stats.values

    return results
