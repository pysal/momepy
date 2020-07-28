#!/usr/bin/env python
# -*- coding: utf-8 -*-

# connectivity.py
# definitions of connectivity characters
import math
import warnings

import networkx as nx
import numpy as np
from tqdm import tqdm

__all__ = [
    "node_degree",
    "meshedness",
    "mean_node_dist",
    "cds_length",
    "mean_node_degree",
    "proportion",
    "cyclomatic",
    "edge_node_ratio",
    "gamma",
    "clustering",
    "local_closeness_centrality",
    "closeness_centrality",
    "betweenness_centrality",
    "local_betweenness_centrality",
    "local_straightness_centrality",
    "straightness_centrality",
    "subgraph",
    "mean_nodes",
]


def node_degree(graph, name="degree"):
    """
    Calculates node degree for each node.

    Wrapper around ``networkx.degree()``.

    Parameters
    ----------
    graph : networkx.Graph
        Graph representing street network.
        Ideally generated from GeoDataFrame using :func:`momepy.gdf_to_nx`
    name : str (default 'degree')
        calculated attribute name

    Returns
    -------
    Graph
        networkx.Graph

    Examples
    --------
    >>> network_graph = mm.node_degree(network_graph)
    """
    netx = graph.copy()

    degree = dict(nx.degree(netx))
    nx.set_node_attributes(netx, degree, name)

    return netx


def _meshedness(graph):
    """
    Calculates meshedness of a graph.
    """
    e = graph.number_of_edges()
    v = graph.number_of_nodes()
    return (e - v + 1) / (2 * v - 5)


def meshedness(graph, radius=5, name="meshedness", distance=None, verbose=True):
    """
    Calculates meshedness for subgraph around each node if radius is set, or for
    whole graph, if ``radius=None``.

    Subgraph is generated around each node within set radius. If ``distance=None``,
    radius will define topological distance, otherwise it uses values in distance
    attribute.

    .. math::
        \\alpha=\\frac{e-v+1}{2 v-5}

    where :math:`e` is the number of edges in subgraph and :math:`v` is the number of nodes in subgraph.

    Adapted from :cite:`feliciotti2018`.


    Parameters
    ----------
    graph : networkx.Graph
        Graph representing street network.
        Ideally generated from GeoDataFrame using :func:`momepy.gdf_to_nx`
    radius: int, optional
        Include all neighbors of distance <= radius from n
    name : str, optional
        calculated attribute name
    distance : str, optional
        Use specified edge data key as distance.
        For example, setting ``distance=’weight’`` will use the edge ``weight`` to
        measure the distance from the node n.
    verbose : bool (default True)
        if True, shows progress bars in loops and indication of steps

    Returns
    -------
    Graph
        networkx.Graph if radius is set
    float
        meshedness for graph if ``radius=None``

    Examples
    --------
    >>> network_graph = mm.meshedness(network_graph, radius=800, distance='edge_length')
    """
    netx = graph.copy()

    if radius:
        for n in tqdm(netx, total=len(netx), disable=not verbose):
            sub = nx.ego_graph(
                netx, n, radius=radius, distance=distance
            )  # define subgraph of steps=radius
            netx.nodes[n][name] = _meshedness(
                sub
            )  # save value calulated for subgraph to node

        return netx

    return _meshedness(netx)


def mean_node_dist(graph, name="meanlen", length="mm_len", verbose=True):
    """
    Calculates mean distance to neighbouring nodes.

    Mean of values in ``length`` attribute.

    Parameters
    ----------
    graph : networkx.Graph
        Graph representing street network.
        Ideally generated from GeoDataFrame using :func:`momepy.gdf_to_nx`
    name : str, optional
        calculated attribute name
    length : str, optional
        name of attribute of segment length (geographical)
    verbose : bool (default True)
        if True, shows progress bars in loops and indication of steps

    Returns
    -------
    Graph
        networkx.Graph

    Examples
    --------
    >>> network_graph = mm.mean_node_dist(network_graph)

    """
    netx = graph.copy()

    for n, nbrs in tqdm(netx.adj.items(), total=len(netx), disable=not verbose):
        lengths = []
        for nbr, keydict in nbrs.items():
            for key, eattr in keydict.items():
                lengths.append(eattr[length])
        netx.nodes[n][name] = np.mean(lengths)

    return netx


def _cds_length(graph, mode, length):
    """
    Calculates cul-de-sac length in a graph.
    """
    lens = []
    for u, v, k, cds in graph.edges.data("cdsbool", keys=True):
        if cds:
            lens.append(graph[u][v][k][length])
    if mode == "sum":
        return sum(lens)
    if mode == "mean":
        return np.mean(lens)
    raise ValueError("Mode {} is not supported. Use 'sum' or 'mean'.".format(mode))


def cds_length(
    graph,
    radius=5,
    mode="sum",
    name="cds_len",
    degree="degree",
    length="mm_len",
    distance=None,
    verbose=True,
):
    """
    Calculates length of cul-de-sacs for subgraph around each node if radius is set, or for
    whole graph, if ``radius=None``.

    Subgraph is generated around each node within set radius. If ``distance=None``,
    radius will define topological distance, otherwise it uses values in distance
    attribute.


    Parameters
    ----------
    graph : networkx.Graph
        Graph representing street network.
        Ideally generated from GeoDataFrame using :func:`momepy.gdf_to_nx`
    radius : int
        Include all neighbors of distance <= radius from n
    mode : str (default 'sum')
        if ``'sum'``, calculate total length, if ``'mean'`` calculate mean length
    name : str, optional
        calculated attribute name
    degree : str
        name of attribute of node degree (:py:func:`momepy.node_degree`)
    length : str, optional
        name of attribute of segment length (geographical)
    distance : str, optional
        Use specified edge data key as distance.
        For example, setting ``distance=’weight’`` will use the edge ``weight`` to
        measure the distance from the node n.
    verbose : bool (default True)
        if True, shows progress bars in loops and indication of steps


    Returns
    -------
    Graph
        networkx.Graph if radius is set
    float
        length of cul-de-sacs for graph if ``radius=None``

    Examples
    --------
    >>> network_graph = mm.cds_length(network_graph, radius=9, mode='mean')
    """
    # node degree needed beforehand
    netx = graph.copy()

    for u, v, k in netx.edges(keys=True):
        if netx.nodes[u][degree] == 1 or netx.nodes[v][degree] == 1:
            netx[u][v][k]["cdsbool"] = True
        else:
            netx[u][v][k]["cdsbool"] = False

    if radius:
        for n in tqdm(netx, total=len(netx), disable=not verbose):
            sub = nx.ego_graph(
                netx, n, radius=radius, distance=distance
            )  # define subgraph of steps=radius
            netx.nodes[n][name] = _cds_length(
                sub, mode=mode, length=length
            )  # save value calculated for subgraph to node

        return netx

    return _cds_length(netx, mode=mode, length=length)


def _mean_node_degree(graph, degree):
    """
    Calculates mean node degree in a graph.
    """
    return np.mean(list(dict(graph.nodes(degree)).values()))


def mean_node_degree(
    graph, radius=5, name="mean_nd", degree="degree", distance=None, verbose=True
):
    """
    Calculates mean node degree for subgraph around each node if radius is set, or for
    whole graph, if ``radius=None``.

    Subgraph is generated around each node within set radius. If ``distance=None``,
    radius will define topological distance, otherwise it uses values in ``distance``
    attribute.


    Parameters
    ----------
    graph : networkx.Graph
        Graph representing street network.
        Ideally generated from GeoDataFrame using :func:`momepy.gdf_to_nx`
    radius: int
        radius defining the extent of subgraph
    name : str, optional
        calculated attribute name
    degree : str
        name of attribute of node degree (:py:func:`momepy.node_degree`)
    distance : str, optional
        Use specified edge data key as distance.
        For example, setting ``distance=’weight’`` will use the edge ``weight`` to
        measure the distance from the node n.
    verbose : bool (default True)
        if True, shows progress bars in loops and indication of steps

    Returns
    -------
    Graph
        networkx.Graph if radius is set
    float
        mean node degree for graph if ``radius=None``

    Examples
    --------
    >>> network_graph = mm.mean_node_degree(network_graph, radius=3)
    """
    netx = graph.copy()

    if radius:
        for n in tqdm(netx, total=len(netx), disable=not verbose):
            sub = nx.ego_graph(
                netx, n, radius=radius, distance=distance
            )  # define subgraph of steps=radius
            netx.nodes[n][name] = _mean_node_degree(sub, degree=degree)

        return netx

    return _mean_node_degree(netx, degree=degree)


def _proportion(graph, degree):
    """
    Calculates the proportion of intersection types in a graph.
    """
    import collections

    values = list(dict(graph.nodes(degree)).values())
    counts = collections.Counter(values)
    return counts


def proportion(
    graph,
    radius=5,
    three=None,
    four=None,
    dead=None,
    degree="degree",
    distance=None,
    verbose=True,
):
    """
    Calculates the proportion of intersection types for subgraph around each node if radius is set, or for
    whole graph, if ``radius=None``.

    Subgraph is generated around each node within set radius. If ``distance=None``,
    radius will define topological distance, otherwise it uses values in ``distance``
    attribute.

    Parameters
    ----------
    graph : networkx.Graph
        Graph representing street network.
        Ideally generated from GeoDataFrame using :func:`momepy.gdf_to_nx`
    radius: int
        Include all neighbors of distance <= radius from n
    three : str, optional
        attribute name for 3-way intersections proportion
    four : str, optional
        attribute name for 4-way intersections proportion
    dead : str, optional
        attribute name for deadends proportion
    degree : str
        name of attribute of node degree (:py:func:`momepy.node_degree`)
    distance : str, optional
        Use specified edge data key as distance.
        For example, setting ``distance=’weight’`` will use the edge ``weight`` to
        measure the distance from the node n.
    verbose : bool (default True)
        if True, shows progress bars in loops and indication of steps

    Returns
    -------
    Graph
        networkx.Graph if radius is set
    dict
        dict with proportions for graph if ``radius=None``

    Examples
    --------
    >>> network_graph = mm.proportion(network_graph, three='threeway', four='fourway', dead='deadends')
    """
    if not three and not four and not dead:
        raise ValueError(
            "Nothing to calculate. Define names for at least one proportion to be calculated: three, four, dead."
        )
    netx = graph.copy()

    if radius:
        for n in tqdm(netx, total=len(netx), disable=not verbose):
            sub = nx.ego_graph(
                netx, n, radius=radius, distance=distance
            )  # define subgraph of steps=radius
            counts = _proportion(sub, degree=degree)
            if three:
                netx.nodes[n][three] = counts[3] / len(sub)
            if four:
                netx.nodes[n][four] = counts[4] / len(sub)
            if dead:
                netx.nodes[n][dead] = counts[1] / len(sub)
        return netx

    # add example to docs explaining keys
    counts = _proportion(netx, degree=degree)
    result = {}
    if three:
        result[three] = counts[3] / len(netx)
    if four:
        result[four] = counts[4] / len(netx)
    if dead:
        result[dead] = counts[1] / len(netx)

    return result


def _cyclomatic(graph):
    """
    Calculates the cyclomatic complexity of a graph.
    """
    e = graph.number_of_edges()
    v = graph.number_of_nodes()
    return e - v + 1


def cyclomatic(graph, radius=5, name="cyclomatic", distance=None, verbose=True):
    """
    Calculates cyclomatic complexity for subgraph around each node if radius is set, or for
    whole graph, if ``radius=None``.

    Subgraph is generated around each node within set radius. If ``distance=None``,
    radius will define topological distance, otherwise it uses values in ``distance``
    attribute.

    .. math::
        \\alpha=e-v+1

    where :math:`e` is the number of edges in subgraph and :math:`v` is the number of nodes in subgraph.

    Adapted from :cite:`bourdic2012`.

    Parameters
    ----------
    graph : networkx.Graph
        Graph representing street network.
        Ideally generated from GeoDataFrame using :func:`momepy.gdf_to_nx`
    radius: int
        Include all neighbors of distance <= radius from n
    name : str, optional
        calculated attribute name
    distance : str, optional
        Use specified edge data key as distance.
        For example, setting ``distance=’weight’`` will use the edge ``weight`` to
        measure the distance from the node n.
    verbose : bool (default True)
        if True, shows progress bars in loops and indication of steps

    Returns
    -------
    Graph
        networkx.Graph if radius is set
    float
        cyclomatic complexity for graph if ``radius=None``

    Examples
    --------
    >>> network_graph = mm.cyclomatic(network_graph, radius=3)
    """
    netx = graph.copy()

    if radius:
        for n in tqdm(netx, total=len(netx), disable=not verbose):
            sub = nx.ego_graph(
                netx, n, radius=radius, distance=distance
            )  # define subgraph of steps=radius
            netx.nodes[n][name] = _cyclomatic(
                sub
            )  # save value calulated for subgraph to node

        return netx

    return _cyclomatic(netx)


def _edge_node_ratio(graph):
    """
    Calculates edge / node ratio of a graph.
    """
    e = graph.number_of_edges()
    v = graph.number_of_nodes()
    return e / v


def edge_node_ratio(
    graph, radius=5, name="edge_node_ratio", distance=None, verbose=True
):
    """
    Calculates edge / node ratio for subgraph around each node if radius is set, or for
    whole graph, if ``radius=None``.

    Subgraph is generated around each node within set radius. If ``distance=None``,
    radius will define topological distance, otherwise it uses values in ``distance``
    attribute.

    .. math::
        \\alpha=e/v

    where :math:`e` is the number of edges in subgraph and :math:`v` is the number of nodes in subgraph.

    Adapted from :cite:`dibble2017`.

    Parameters
    ----------
    graph : networkx.Graph
        Graph representing street network.
        Ideally generated from GeoDataFrame using :func:`momepy.gdf_to_nx`
    radius: int
        Include all neighbors of distance <= radius from n
    name : str, optional
        calculated attribute name
    distance : str, optional
        Use specified edge data key as distance.
        For example, setting ``distance=’weight’`` will use the edge ``weight`` to
        measure the distance from the node n.
    verbose : bool (default True)
        if True, shows progress bars in loops and indication of steps

    Returns
    -------
    Graph
        networkx.Graph if radius is set
    float
        edge / node ratio for graph if ``radius=None``

    Examples
    --------
    >>> network_graph = mm.edge_node_ratio(network_graph, radius=3)
    """
    netx = graph.copy()

    if radius:
        for n in tqdm(netx, total=len(netx), disable=not verbose):
            sub = nx.ego_graph(
                netx, n, radius=radius, distance=distance
            )  # define subgraph of steps=radius
            netx.nodes[n][name] = _edge_node_ratio(
                sub
            )  # save value calulated for subgraph to node

        return netx

    return _edge_node_ratio(netx)


def _gamma(graph):
    """
    Calculates gamma index of a graph.
    """
    e = graph.number_of_edges()
    v = graph.number_of_nodes()
    if v == 2:
        return np.nan
    return e / (3 * (v - 2))  # save value calulated for subgraph to node


def gamma(graph, radius=5, name="gamma", distance=None, verbose=True):
    """
    Calculates connectivity gamma index for subgraph around each node if radius is set, or for
    whole graph, if ``radius=None``.

    Subgraph is generated around each node within set radius. If ``distance=None``,
    radius will define topological distance, otherwise it uses values in ``distance``
    attribute.

    .. math::
        \\alpha=\\frac{e}{3(v-2)}

    where :math:`e` is the number of edges in subgraph and :math:`v` is the number of nodes in subgraph.

    Adapted from :cite:`dibble2017`.

    Parameters
    ----------
    graph : networkx.Graph
        Graph representing street network.
        Ideally generated from GeoDataFrame using :func:`momepy.gdf_to_nx`
    radius: int
        Include all neighbors of distance <= radius from n
    name : str, optional
        calculated attribute name
    distance : str, optional
        Use specified edge data key as distance.
        For example, setting ``distance=’weight’`` will use the edge ``weight`` to
        measure the distance from the node n.
    verbose : bool (default True)
        if True, shows progress bars in loops and indication of steps

    Returns
    -------
    Graph
        networkx.Graph if radius is set
    float
        gamma index for graph if ``radius=None``

    Examples
    --------
    >>> network_graph = mm.gamma(network_graph, radius=3)

    """
    netx = graph.copy()

    if radius:
        for n in tqdm(netx, total=len(netx), disable=not verbose):
            sub = nx.ego_graph(
                netx, n, radius=radius, distance=distance
            )  # define subgraph of steps=radius
            netx.nodes[n][name] = _gamma(sub)

        return netx

    return _gamma(netx)


def clustering(graph, name="cluster"):
    """
    Calculates the squares clustering coefficient for nodes.

    Wrapper around ``networkx.square_clustering``.


    Parameters
    ----------
    graph : networkx.Graph
        Graph representing street network.
        Ideally generated from GeoDataFrame using :func:`momepy.gdf_to_nx`
    name : str, optional
        calculated attribute name

    Returns
    -------
    Graph
        networkx.Graph

    Examples
    --------
    >>> network_graph = mm.clustering(network_graph)
    """
    netx = graph.copy()

    vals = nx.square_clustering(netx)
    nx.set_node_attributes(netx, vals, name)

    return netx


def _closeness_centrality(G, u=None, length=None, wf_improved=True, len_graph=None):
    r"""Compute closeness centrality for nodes. Slight adaptation of networkx
    `closeness_centrality` to allow normalisation for local closeness.
    Adapted script used in networkx.

    Closeness centrality [1]_ of a node `u` is the reciprocal of the
    average shortest path distance to `u` over all `n-1` reachable nodes.

    .. math::

        C(u) = \frac{n - 1}{\sum_{v=1}^{n-1} d(v, u)},

    where `d(v, u)` is the shortest-path distance between `v` and `u`,
    and `n` is the number of nodes that can reach `u`. Notice that the
    closeness distance function computes the incoming distance to `u`
    for directed graphs. To use outward distance, act on `G.reverse()`.

    Notice that higher values of closeness indicate higher centrality.

    Wasserman and Faust propose an improved formula for graphs with
    more than one connected component. The result is "a ratio of the
    fraction of actors in the group who are reachable, to the average
    distance" from the reachable actors [2]_. You might think this
    scale factor is inverted but it is not. As is, nodes from small
    components receive a smaller closeness value. Letting `N` denote
    the number of nodes in the graph,

    .. math::

        C_{WF}(u) = \frac{n-1}{N-1} \frac{n - 1}{\sum_{v=1}^{n-1} d(v, u)},

    Parameters
    ----------
    G : graph
      A NetworkX graph

    u : node, optional
      Return only the value for node u

    distance : edge attribute key, optional (default=None)
      Use the specified edge attribute as the edge distance in shortest
      path calculations

    len_graph : int
        length of complete graph

    Returns
    -------
    nodes : dictionary
      Dictionary of nodes with closeness centrality as the value.

    References
    ----------
    .. [1] Linton C. Freeman: Centrality in networks: I.
       Conceptual clarification. Social Networks 1:215-239, 1979.
       http://leonidzhukov.ru/hse/2013/socialnetworks/papers/freeman79-centrality.pdf
    .. [2] pg. 201 of Wasserman, S. and Faust, K.,
       Social Network Analysis: Methods and Applications, 1994,
       Cambridge University Press.
    """

    if length is not None:
        import functools

        # use Dijkstra's algorithm with specified attribute as edge weight
        path_length = functools.partial(
            nx.single_source_dijkstra_path_length, weight=length
        )
    else:
        path_length = nx.single_source_shortest_path_length

    nodes = [u]
    closeness_centrality = {}
    for n in nodes:
        sp = dict(path_length(G, n))
        totsp = sum(sp.values())
        if totsp > 0.0 and len(G) > 1:
            closeness_centrality[n] = (len(sp) - 1.0) / totsp
            # normalize to number of nodes-1 in connected part
            s = (len(sp) - 1.0) / (len_graph - 1)
            closeness_centrality[n] *= s
        else:
            closeness_centrality[n] = 0.0

    return closeness_centrality[u]


def local_closeness_centrality(
    graph, radius=5, name="closeness", distance=None, weight=None
):
    """
    Calculates local closeness for each node based on the defined distance.

    Subgraph is generated around each node within set radius. If ``distance=None``,
    radius will define topological distance, otherwise it uses values in ``distance``
    attribute. Based on ``networkx.closeness_centrality``.

    Local closeness centrality of a node `u` is the reciprocal of the
    average shortest path distance to `u` over all `n-1` nodes within subgraph.

    .. math::

        C(u) = \\frac{n - 1}{\\sum_{v=1}^{n-1} d(v, u)},

    where :math:`d(v, u)` is the shortest-path distance between :math:`v` and :math:`u`,
    and :math:`n` is the number of nodes that can reach :math:`u`.

    Adapted from :cite:`porta2006`.


    Parameters
    ----------
    graph : networkx.Graph
        Graph representing street network.
        Ideally generated from GeoDataFrame using :func:`momepy.gdf_to_nx`
    radius: int
        Include all neighbors of distance <= radius from n
    name : str, optional
        calculated attribute name
    distance : str, optional
        Use specified edge data key as distance.
        For example, setting ``distance=’weight’`` will use the edge ``weight`` to
        measure the distance from the node n during ego_graph generation.
    weight : str, optional
      Use the specified edge attribute as the edge distance in shortest
      path calculations in closeness centrality algorithm

    Returns
    -------
    Graph
        networkx.Graph

    Examples
    --------
    >>> network_graph = mm.local_closeness_centrality(network_graph, radius=400, distance='edge_length')

    """
    warnings.warn(
        "local_closeness_centrality() is deprecated and will be removed in momepy 0.4.0. "
        "Use closeness_centrality() instead.",
        FutureWarning,
    )

    return closeness_centrality(
        graph=graph, radius=radius, name=name, distance=distance, weight=weight
    )


def closeness_centrality(
    graph,
    name="closeness",
    weight="mm_len",
    radius=None,
    distance=None,
    verbose=True,
    **kwargs
):
    """
    Calculates the closeness centrality for nodes.

    Wrapper around ``networkx.closeness_centrality``.

    Closeness centrality of a node `u` is the reciprocal of the
    average shortest path distance to `u` over all `n-1` nodes within reachable nodes.

    .. math::

        C(u) = \\frac{n - 1}{\\sum_{v=1}^{n-1} d(v, u)},

    where :math:`d(v, u)` is the shortest-path distance between :math:`v` and :math:`u`,
    and :math:`n` is the number of nodes that can reach :math:`u`.


    Parameters
    ----------
    graph : networkx.Graph
        Graph representing street network.
        Ideally generated from GeoDataFrame using :func:`momepy.gdf_to_nx`
    name : str, optional
        calculated attribute name
    weight : str (default 'mm_len')
        attribute holding the weight of edge (e.g. length, angle)
    radius: int
        Include all neighbors of distance <= radius from n
    distance : str, optional
        Use specified edge data key as distance.
        For example, setting ``distance=’weight’`` will use the edge ``weight`` to
        measure the distance from the node n during ego_graph generation.
    verbose : bool (default True)
        if True, shows progress bars in loops and indication of steps
    **kwargs
        kwargs for ``networkx.closeness_centrality``

    Returns
    -------
    Graph
        networkx.Graph

    Examples
    --------
    >>> network_graph = mm.closeness_centrality(network_graph)
    """
    netx = graph.copy()

    if radius:
        lengraph = len(netx)
        for n in tqdm(netx, total=len(netx), disable=not verbose):
            sub = nx.ego_graph(
                netx, n, radius=radius, distance=distance
            )  # define subgraph of steps=radius
            netx.nodes[n][name] = _closeness_centrality(
                sub, n, length=weight, len_graph=lengraph
            )
    else:
        vals = nx.closeness_centrality(netx, distance=weight, **kwargs)
        nx.set_node_attributes(netx, vals, name)

    return netx


def betweenness_centrality(
    graph,
    name="betweenness",
    mode="nodes",
    weight="mm_len",
    endpoints=True,
    radius=None,
    distance=None,
    normalized=False,
    verbose=True,
    **kwargs
):
    """
    Calculates the shortest-path betweenness centrality for nodes.

    Wrapper around ``networkx.betweenness_centrality`` or ``networkx.edge_betweenness_centrality``.

    Betweenness centrality of a node `v` is the sum of the
    fraction of all-pairs shortest paths that pass through `v`

    .. math::

       c_B(v) =\\sum_{s,t \\in V} \\frac{\\sigma(s, t|v)}{\\sigma(s, t)}

    where `V` is the set of nodes, :math:`\\sigma(s, t)` is the number of
    shortest :math:`(s, t)`-paths,  and :math:`\\sigma(s, t|v)` is the number of
    those paths  passing through some  node `v` other than `s, t`.
    If `s = t`, :math:`\\sigma(s, t) = 1`, and if `v` in `{s, t}``,
    :math:`\\sigma(s, t|v) = 0`.

    Betweenness centrality of an edge `e` is the sum of the
    fraction of all-pairs shortest paths that pass through `e`

    .. math::

       c_B(e) =\\sum_{s,t \\in V} \\frac{\\sigma(s, t|e)}{\\sigma(s, t)}

    where `V` is the set of nodes, :math:`\\sigma(s, t)` is the number of
    shortest :math:`(s, t)`-paths, and :math:`\\sigma(s, t|e)` is the number of
    those paths passing through edge `e`.

    Adapted from :cite:`porta2006`.

    Parameters
    ----------
    graph : networkx.Graph
        Graph representing street network.
        Ideally generated from GeoDataFrame using :func:`momepy.gdf_to_nx`
    name : str, optional
        calculated attribute name
    mode : str, default 'nodes'
        mode of betweenness calculation. 'node' for node-based, 'edges' for edge-based
    weight : str (default 'mm_len')
        attribute holding the weight of edge (e.g. length, angle)
    radius: int
        Include all neighbors of distance <= radius from n
    distance : str, optional
        Use specified edge data key as distance.
        For example, setting ``distance=’weight’`` will use the edge ``weight`` to
        measure the distance from the node n during ego_graph generation.
    normalized : bool, optional
        If True the betweenness values are normalized by `2/((n-1)(n-2))`,
        where n is the number of nodes in subgraph.
    verbose : bool (default True)
        if True, shows progress bars in loops and indication of steps
    **kwargs
        kwargs for ``networkx.betweenness_centrality`` or ``networkx.edge_betweenness_centrality``

    Returns
    -------
    Graph
        networkx.Graph

    Examples
    --------
    >>> network_graph = mm.betweenness_centrality(network_graph)

    Notes
    -----
    In case of angular betweenness, implementation is based on "Tasos Implementation".
    """
    netx = graph.copy()

    # has to be Graph not MultiGraph as MG is not supported by networkx2.4
    G = nx.Graph()
    for u, v, k, data in netx.edges(data=True, keys=True):
        if G.has_edge(u, v):
            if G[u][v][weight] > netx[u][v][k][weight]:
                nx.set_edge_attributes(G, {(u, v): data})
        else:
            G.add_edge(u, v, **data)

    if radius:
        for n in tqdm(G, total=len(G), disable=not verbose):
            sub = nx.ego_graph(
                G, n, radius=radius, distance=distance
            )  # define subgraph of steps=radius
            netx.nodes[n][name] = nx.betweenness_centrality(
                sub, weight=weight, normalized=normalized, **kwargs
            )[n]

    elif mode == "nodes":
        vals = nx.betweenness_centrality(
            G, weight=weight, endpoints=endpoints, **kwargs
        )
        nx.set_node_attributes(netx, vals, name)
    elif mode == "edges":
        vals = nx.edge_betweenness_centrality(G, weight=weight, **kwargs)
        for u, v, k in netx.edges(keys=True):
            try:
                val = vals[u, v]
            except KeyError:
                val = vals[v, u]
            netx[u][v][k][name] = val
    else:
        raise ValueError(
            "Mode {} is not supported. Use 'nodes' or 'edges'.".format(mode)
        )

    return netx


def local_betweenness_centrality(
    graph,
    radius=5,
    name="betweenness",
    distance=None,
    weight=None,
    normalized=False,
    **kwargs
):
    """
    Calculates the shortest-path betweenness centrality for nodes within subgraph.

    Subgraph is generated around each node within set radius. If ``distance=None``,
    radius will define topological distance, otherwise it uses values in ``distance``
    attribute. Based on ``networkx.betweenness_centrality``.

    Betweenness centrality of a node `v` is the sum of the
    fraction of all-pairs shortest paths that pass through `v`

    .. math::

       c_B(v) =\\sum_{s,t \\in V} \\frac{\\sigma(s, t|v)}{\\sigma(s, t)}

    where `V` is the set of nodes, :math:`\\sigma(s, t)` is the number of
    shortest :math:`(s, t)`-paths,  and :math:`\\sigma(s, t|v)` is the number of
    those paths  passing through some  node `v` other than `s, t`.
    If `s = t`, :math:`\\sigma(s, t) = 1`, and if `v` in `{s, t}``,
    :math:`\\sigma(s, t|v) = 0`.


    Parameters
    ----------
    graph : networkx.Graph
        Graph representing street network.
        Ideally generated from GeoDataFrame using :func:`momepy.gdf_to_nx`
    radius: int
        Include all neighbors of distance <= radius from n
    name : str, optional
        calculated attribute name
    distance : str, optional
        Use specified edge data key as distance.
        For example, setting ``distance=’weight’`` will use the edge ``weight`` to
        measure the distance from the node n during ego_graph generation.
    weight : str, optional
        Use the specified edge attribute as the edge distance in shortest
        path calculations in closeness centrality algorithm
    normalized : bool, optional
        If True the betweenness values are normalized by `2/((n-1)(n-2))`,
        where n is the number of nodes in subgraph.
    **kwargs
        kwargs for ``networkx.betweenness_centrality_subset``

    Returns
    -------
    Graph
        networkx.Graph

    Examples
    --------
    >>> network_graph = mm.local_betweenness_centrality(network_graph, radius=800, distance='edge_length')

    """
    warnings.warn(
        "local_betweenness_centrality() is deprecated and will be removed in momepy 0.4.0. "
        "Use betweenness_centrality() instead.",
        FutureWarning,
    )

    return betweenness_centrality(
        graph,
        radius=radius,
        name=name,
        distance=distance,
        weight=weight,
        normalized=normalized,
        **kwargs
    )


def _euclidean(n, m):
    """helper for straightness"""
    return math.sqrt((n[0] - m[0]) ** 2 + (n[1] - m[1]) ** 2)


def _straightness_centrality(G, weight, normalized=True):
    """
    Calculates straightness centrality.
    """
    straightness_centrality = {}

    for n in G.nodes():
        straightness = 0
        sp = nx.single_source_dijkstra_path_length(G, n, weight=weight)

        if len(sp) > 0 and len(G) > 1:
            for target in sp:
                if n != target:
                    network_dist = sp[target]
                    euclidean_dist = _euclidean(n, target)
                    straightness = straightness + (euclidean_dist / network_dist)
            straightness_centrality[n] = straightness * (1.0 / (len(G) - 1.0))
            # normalize to number of nodes-1 in connected part
            if normalized:
                if len(sp) > 1:
                    s = (len(G) - 1.0) / (len(sp) - 1.0)
                    straightness_centrality[n] *= s
                else:
                    straightness_centrality[n] = 0
        else:
            straightness_centrality[n] = 0.0
    return straightness_centrality


def straightness_centrality(
    graph,
    weight="mm_len",
    normalized=True,
    name="straightness",
    radius=None,
    distance=None,
    verbose=True,
):
    """
    Calculates the straightness centrality for nodes.

    .. math::
        C_{S}(i)=\\frac{1}{n-1} \\sum_{j \\in V, j \\neq i} \\frac{d_{i j}^{E u}}{d_{i j}}

    where :math:`\\mathrm{d}^{\\mathrm{E} \\mathrm{u}}_{\\mathrm{ij}}` is the Euclidean distance
    between nodes `i` and `j` along a straight line.

    Adapted from :cite:`porta2006`.

    Parameters
    ----------
    graph : networkx.Graph
        Graph representing street network.
        Ideally generated from GeoDataFrame using :func:`momepy.gdf_to_nx`
    weight : str (default 'mm_len')
        attribute holding length of edge
    normalized : bool
        normalize to number of nodes-1 in connected part (for local straightness
        is recommended to set to normalized False)
    name : str, optional
        calculated attribute name
    radius: int
        Include all neighbors of distance <= radius from n
    distance : str, optional
        Use specified edge data key as distance.
        For example, setting ``distance=’weight’`` will use the edge ``weight`` to
        measure the distance from the node n during ego_graph generation.
    verbose : bool (default True)
        if True, shows progress bars in loops and indication of steps

    Returns
    -------
    Graph
        networkx.Graph

    Examples
    --------
    >>> network_graph = mm.straightness_centrality(network_graph)
    """
    netx = graph.copy()

    if radius:
        for n in tqdm(netx, total=len(netx), disable=not verbose):
            sub = nx.ego_graph(
                netx, n, radius=radius, distance=distance
            )  # define subgraph of steps=radius
            netx.nodes[n][name] = _straightness_centrality(
                sub, weight=weight, normalized=normalized
            )[n]
    else:
        vals = _straightness_centrality(netx, weight=weight, normalized=normalized)
        nx.set_node_attributes(netx, vals, name)

    return netx


def local_straightness_centrality(
    graph, radius=5, name="straightness", distance=None, weight="mm_len"
):
    """
    Calculates local straightness for each node based on the defined distance.

    Subgraph is generated around each node within set radius. If ``distance=None``,
    radius will define topological distance, otherwise it uses values in ``distance``
    attribute.

    .. math::
        C_{S}(i)=\\frac{1}{n-1} \\sum_{j \\in V, j \\neq i} \\frac{d_{i j}^{E u}}{d_{i j}}

    where :math:`\\mathrm{d}^{\\mathrm{E} \\mathrm{u}}_{\\mathrm{ij}}` is the Euclidean distance
    between nodes `i` and `j` along a straight line.


    Parameters
    ----------
    graph : networkx.Graph
        Graph representing street network.
        Ideally generated from GeoDataFrame using :func:`momepy.gdf_to_nx`
    radius: int
        Include all neighbors of distance <= radius from n
    name : str, optional
        calculated attribute name
    distance : str, optional
        Use specified edge data key as distance.
        For example, setting ``distance=’weight’`` will use the edge ``weight`` to
        measure the distance from the node n during ego_graph generation.
    weight : str, optional
      Use the specified edge attribute as the edge distance in shortest
      path calculations in closeness centrality algorithm

    Returns
    -------
    Graph
        networkx.Graph


    Examples
    --------
    >>> network_graph = mm.local_straightness_centrality(network_graph, radius=400, distance='edge_length')

    """
    warnings.warn(
        "local_straightness_centrality() is deprecated and will be removed in momepy 0.4.0. "
        "Use straightness_centrality() instead.",
        FutureWarning,
    )

    return straightness_centrality(
        graph=graph, radius=radius, name=name, distance=distance, weight=weight
    )


def subgraph(
    graph,
    radius=5,
    distance=None,
    meshedness=True,
    cds_length=True,
    mode="sum",
    degree="degree",
    length="mm_len",
    mean_node_degree=True,
    proportion={3: True, 4: True, 0: True},
    cyclomatic=True,
    edge_node_ratio=True,
    gamma=True,
    local_closeness=True,
    closeness_weight=None,
    verbose=True,
):
    """
    Calculates all subgraph-based characters.

    Generating subgraph might be a time consuming activity. If we want to use the same
    subgraph for more characters, ``subgraph`` allows this by generating subgraph and
    then analysing it using selected options.


    Parameters
    ----------
    graph : networkx.Graph
        Graph representing street network.
        Ideally generated from GeoDataFrame using :func:`momepy.gdf_to_nx`
    radius: int
        radius defining the extent of subgraph
    distance : str, optional
        Use specified edge data key as distance.
        For example, setting ``distance=’weight’`` will use the edge ``weight`` to
        measure the distance from the node n.
    meshedness : bool, default True
        Calculate meshedness (True/False)
    cds_length : bool, default True
        Calculate cul-de-sac length (True/False)
    mode : str (defualt 'sum')
        if ``'sum'``, calculate total cds_length, if ``'mean'`` calculate mean cds_length
    degree : str
        name of attribute of node degree (:py:func:`momepy.node_degree`)
    length : str, default `mm_len`
        name of attribute of segment length (geographical)
    mean_node_degree : bool, default True
        Calculate mean node degree (True/False)
    proportion : dict, default {3: True, 4: True, 0: True}
        Calculate proportion {3: True/False, 4: True/False, 0: True/False}
    cyclomatic : bool, default True
        Calculate cyclomatic complexity (True/False)
    edge_node_ratio : bool, default True
        Calculate edge node ratio (True/False)
    gamma : bool, default True
        Calculate gamma index (True/False)
    local_closeness : bool, default True
        Calculate local closeness centrality (True/False)
    closeness_weight : str, optional
      Use the specified edge attribute as the edge distance in shortest
      path calculations in closeness centrality algorithm
    verbose : bool (default True)
        if True, shows progress bars in loops and indication of steps


    Returns
    -------
    Graph
        networkx.Graph

    Examples
    --------
    >>> network_graph = mm.subgraph(network_graph)
    """

    netx = graph.copy()

    for n in tqdm(netx, total=len(netx), disable=not verbose):
        sub = nx.ego_graph(
            netx, n, radius=radius, distance=distance
        )  # define subgraph of steps=radius

        if meshedness:
            netx.nodes[n]["meshedness"] = _meshedness(sub)

        if cds_length:
            for u, v, k in netx.edges(keys=True):
                if netx.nodes[u][degree] == 1 or netx.nodes[v][degree] == 1:
                    netx[u][v][k]["cdsbool"] = True
                else:
                    netx[u][v][k]["cdsbool"] = False

            netx.nodes[n]["cds_length"] = _cds_length(sub, mode=mode, length=length)

        if mean_node_degree:
            netx.nodes[n]["mean_node_degree"] = _mean_node_degree(sub, degree=degree)

        if proportion:
            counts = _proportion(sub, degree=degree)
            if proportion[3]:
                netx.nodes[n]["proportion_3"] = counts[3] / len(sub)
            if proportion[4]:
                netx.nodes[n]["proportion_4"] = counts[4] / len(sub)
            if proportion[0]:
                netx.nodes[n]["proportion_0"] = counts[1] / len(sub)

        if cyclomatic:
            netx.nodes[n]["cyclomatic"] = _cyclomatic(sub)

        if edge_node_ratio:
            netx.nodes[n]["edge_node_ratio"] = _edge_node_ratio(sub)

        if gamma:
            netx.nodes[n]["gamma"] = _gamma(sub)

        if local_closeness:
            lengraph = len(netx)
            netx.nodes[n]["local_closeness"] = _closeness_centrality(
                sub, n, length=closeness_weight, len_graph=lengraph
            )

    return netx


def mean_nodes(G, attr):
    """
    Calculates mean value of nodes attr for each edge.
    """
    for u, v, k in G.edges(keys=True):
        mean = (G.nodes[u][attr] + G.nodes[v][attr]) / 2
        G[u][v][k][attr] = mean
