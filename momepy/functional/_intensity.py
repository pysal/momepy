import numpy as np
import pandas as pd
import shapely
from geopandas import GeoDataFrame, GeoSeries
from libpysal.graph import Graph
from pandas import Series

__all__ = ["courtyards", "node_density", "count"]


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


def count(
    left: GeoDataFrame,
    right: GeoDataFrame,
    right_group_key: Series | np.ndarray,
    weighted=False,
) -> Series:
    """Calculate the number of elements within an aggregated structure.

    Aggregated structures can typically be blocks, street segments, or street nodes
    (their snappedd objects). The right gdf has to have a unique ID of aggregated
    structures assigned before hand (e.g. using :py:func:`momepy.get_network_id`).
    If ``weighted=True``, the number of elements will be divided by the area of
    length (based on geometry type) of aggregated elements, to return relative value.

    .. math::
        \\sum_{i \\in aggr} (n_i);\\space \\frac{\\sum_{i \\in aggr} (n_i)}{area_{aggr}}

    Adapted from :cite:`hermosilla2012` and :cite:`feliciotti2018`.

    Parameters
    ----------
    left : GeoDaStaFrame
        A GeoDataFrame containing aggregation to analyse.
    right : GeoDataFrame
        A GeoDataFrame containing objects to analyse.
    right_group_key: np.array | pd.Series
        The group key that assigns objects from ``right`` to ``left``.
    weighted : bool (default False)
        If ``True``, count will be divided by the area or length.

    Returns
    -------
    Series

    Examples
    --------
    >>> blocks_df['buildings_count'] = mm.count(blocks_df,
    ...                                         buildings_df,
    ...                                         buildings_df['bID'])
    """

    if isinstance(right_group_key, np.ndarray):
        right_group_key = Series(right_group_key, index=right.index)

    stats = (
        right.loc[right_group_key.index.values].groupby(right_group_key.values).size()
    )

    # fill missing values with 0s
    results = Series(0, left.index)
    results.loc[stats.index.values] = stats.values
    if weighted:
        if left.geometry[0].geom_type in ["Polygon", "MultiPolygon"]:
            results = results / left.geometry.area
        elif left.geometry[0].geom_type in ["LineString", "MultiLineString"]:
            results = results / left.geometry.length
        else:
            raise TypeError("Geometry type does not support weighting.")

    return results
