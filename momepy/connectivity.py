#!/usr/bin/env python
# -*- coding: utf-8 -*-

# connectivity.py
# definitons of connectivity characters
import networkx as nx
from tqdm import tqdm
import numpy as np


def meshedness(graph, radius=5, name='meshedness', distance=None):
    """
    Calculates meshedness for subgraph around each node.

    Subgraph is generated around each node within set radius. If distance=None,
    radius will define topological distance, otherwise it uses values in distance
    attribute.

    .. math::


    Parameters
    ----------
    graph : networkx.Graph
        Graph representing street network.
        Ideally genereated from GeoDataFrame using :py:func:`momepy.gdf_to_nx`
    radius: int
        number of topological steps defining the extent of subgraph
    name : str, optional
        calculated attribute name
    distance : str, optional
        Use specified edge data key as distance.
        For example, setting distance=’weight’ will use the edge weight to
        measure the distance from the node n.

    Returns
    -------
    Graph
        networkx.Graph

    References
    ----------
    Ale, Song et al 2013

    Examples
    --------

    """
    netx = graph

    for n in tqdm(netx, total=len(netx)):
        sub = nx.ego_graph(netx, n, radius=radius, undirected=True, distance=distance)  # define subgraph of steps=radius
        e = sub.number_of_edges()
        v = sub.number_of_nodes()
        netx.nodes[n][name] = (e - v + 1) / (2 * v - 5)  # save value calulated for subgraph to node

    return netx


def mean_node_dist(graph, name='meanlen', length='mm_len'):
    """
    Calculates mean distance to neighbouring nodes.

    .. math::


    Parameters
    ----------
    graph : networkx.Graph
        Graph representing street network.
        Ideally genereated from GeoDataFrame using :py:func:`momepy.gdf_to_nx`
    name : str, optional
        calculated attribute name
    length : str, optional
        name of attribute of segment length (geographical)

    Returns
    -------
    Graph
        networkx.Graph

    References
    ----------

    Examples
    --------

    """
    netx = graph

    for n, nbrs in tqdm(netx.adj.items(), total=len(netx)):
        lengths = []
        for nbr, eattr in nbrs.items():
            lengths.append(eattr[length])
        netx.nodes[n][name] = np.mean(lengths)

    return netx


def cds_length(graph=None, radius=5, mode='sum', name='cds_len', degree='degree', length='mm_len', distance=None):
    """
    Calculates length of cul-de-sacs for subgraph around each node.

    Subgraph is generated around each node within set radius. If distance=None,
    radius will define topological distance, otherwise it uses values in distance
    attribute.

    .. math::


    Parameters
    ----------
    graph : networkx.Graph
        Graph representing street network.
        Ideally genereated from GeoDataFrame using :py:func:`momepy.gdf_to_nx`
    radius : int
        number of topological steps defining the extent of subgraph
    name : str, optional
        calculated attribute name
    degree : str
        name of attribute of node degree (:py:func:`momepy.node_degree`)
    length : str, optional
        name of attribute of segment length (geographical)
    distance : str, optional
        Use specified edge data key as distance.
        For example, setting distance=’weight’ will use the edge weight to
        measure the distance from the node n.


    Returns
    -------
    Graph
        networkx.Graph

    References
    ----------

    Examples
    --------

    """
    # node degree needed beforehand
    netx = graph

    for u, v in netx.edges():
        if netx.nodes[u][degree] == 1 or netx.nodes[v][degree] == 1:
            netx[u][v]['cdsbool'] = True
        else:
            netx[u][v]['cdsbool'] = False

    for n in tqdm(netx, total=len(netx)):
        sub = nx.ego_graph(netx, n, radius=radius, undirected=True, distance=distance)  # define subgraph of steps=radius
        lens = []
        for u, v, cds in sub.edges.data('cdsbool'):
            if cds:
                lens.append(sub[u][v][length])
        if mode == 'sum':
            netx.nodes[n][name] = sum(lens)  # save value calulated for subgraph to node
        elif mode == 'mean':
            netx.nodes[n][name] = np.mean(lens)  # save value calulated for subgraph to node

    return netx


def mean_node_degree(graph, radius=5, name='mean_nd', degree='degree', distance=None):
    """
    Calculates meshedness for subgraph around each node.

    Subgraph is generated around each node within set radius. If distance=None,
    radius will define topological distance, otherwise it uses values in distance
    attribute.

    .. math::


    Parameters
    ----------
    graph : networkx.Graph
        Graph representing street network.
        Ideally genereated from GeoDataFrame using :py:func:`momepy.gdf_to_nx`
    radius: int
        number of topological steps defining the extent of subgraph
    name : str, optional
        calculated attribute name
    degree : str
        name of attribute of node degree (:py:func:`momepy.node_degree`)
    distance : str, optional
        Use specified edge data key as distance.
        For example, setting distance=’weight’ will use the edge weight to
        measure the distance from the node n.

    Returns
    -------
    Graph
        networkx.Graph

    References
    ----------

    Examples
    --------

    """
    netx = graph

    for n in tqdm(netx, total=len(netx)):
        sub = nx.ego_graph(netx, n, radius=radius, undirected=True, distance=distance)  # define subgraph of steps=radius
        netx.nodes[n][name] = np.mean(list(dict(sub.nodes('degree')).values()))

    return netx


def proportion(graph, radius=5, three=None, four=None, dead=None, degree='degree', distance=None):
    """
    Calculates the proportion of intersection types for subgraph around each node.

    Subgraph is generated around each node within set radius. If distance=None,
    radius will define topological distance, otherwise it uses values in distance
    attribute.

    .. math::


    Parameters
    ----------
    graph : networkx.Graph
        Graph representing street network.
        Ideally genereated from GeoDataFrame using :py:func:`momepy.gdf_to_nx`
    radius: int
        number of topological steps defining the extent of subgraph
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
        For example, setting distance=’weight’ will use the edge weight to
        measure the distance from the node n.

    Returns
    -------
    Graph
        networkx.Graph

    References
    ----------

    Examples
    --------

    """
    if not three and not four and not dead:
        raise ValueError('Nothing to calculate. Define names for at least one proportion to be calculated: three, four, dead.')
    netx = graph
    import collections

    for n in tqdm(netx, total=len(netx)):
        sub = nx.ego_graph(netx, n, radius=radius, undirected=True, distance=distance)  # define subgraph of steps=radius
        values = list(dict(sub.nodes('degree')).values())
        counts = collections.Counter(values)
        if three:
            netx.nodes[n][three] = counts[3] / len(counts)
        if four:
            netx.nodes[n][four] = counts[4] / len(counts)
        if dead:
            netx.nodes[n][dead] = counts[1] / len(counts)
    return netx


def cyclomatic(graph, radius=5, name='cyclomatic', distance=None):
    """
    Calculates cyclomatic compelxity for subgraph around each node.

    Subgraph is generated around each node within set radius. If distance=None,
    radius will define topological distance, otherwise it uses values in distance
    attribute.

    .. math::


    Parameters
    ----------
    graph : networkx.Graph
        Graph representing street network.
        Ideally genereated from GeoDataFrame using :py:func:`momepy.gdf_to_nx`
    radius: int
        number of topological steps defining the extent of subgraph
    name : str, optional
        calculated attribute name
    distance : str, optional
        Use specified edge data key as distance.
        For example, setting distance=’weight’ will use the edge weight to
        measure the distance from the node n.

    Returns
    -------
    Graph
        networkx.Graph

    References
    ----------
    Bourdic, Salat, Nowacki 2012 Assessing cities

    Examples
    --------

    """
    netx = graph

    for n in tqdm(netx, total=len(netx)):
        sub = nx.ego_graph(netx, n, radius=radius, undirected=True, distance=distance)  # define subgraph of steps=radius
        e = sub.number_of_edges()
        v = sub.number_of_nodes()
        netx.nodes[n][name] = (e - v + 1)  # save value calulated for subgraph to node

    return netx


def edge_node_ratio(graph, radius=5, name='edge_node_ratio', distance=None):
    """
    Calculates edge / node ratio for subgraph around each node.

    Subgraph is generated around each node within set radius. If distance=None,
    radius will define topological distance, otherwise it uses values in distance
    attribute.

    .. math::


    Parameters
    ----------
    graph : networkx.Graph
        Graph representing street network.
        Ideally genereated from GeoDataFrame using :py:func:`momepy.gdf_to_nx`
    radius: int
        number of topological steps defining the extent of subgraph
    name : str, optional
        calculated attribute name
    distance : str, optional
        Use specified edge data key as distance.
        For example, setting distance=’weight’ will use the edge weight to
        measure the distance from the node n.

    Returns
    -------
    Graph
        networkx.Graph

    References
    ----------
    Dibble

    Examples
    --------

    """
    netx = graph

    for n in tqdm(netx, total=len(netx)):
        sub = nx.ego_graph(netx, n, radius=radius, undirected=True, distance=distance)  # define subgraph of steps=radius
        e = sub.number_of_edges()
        v = sub.number_of_nodes()
        netx.nodes[n][name] = e / v  # save value calulated for subgraph to node

    return netx


def gamma(graph, radius=5, name='gamma', distance=None):
    """
    Calculates connectivity gamma index for subgraph around each node.

    Subgraph is generated around each node within set radius. If distance=None,
    radius will define topological distance, otherwise it uses values in distance
    attribute.

    .. math::


    Parameters
    ----------
    graph : networkx.Graph
        Graph representing street network.
        Ideally genereated from GeoDataFrame using :py:func:`momepy.gdf_to_nx`
    radius: int
        number of topological steps defining the extent of subgraph
    name : str, optional
        calculated attribute name
    distance : str, optional
        Use specified edge data key as distance.
        For example, setting distance=’weight’ will use the edge weight to
        measure the distance from the node n.

    Returns
    -------
    Graph
        networkx.Graph

    References
    ----------
    Dibble

    Examples
    --------

    """
    netx = graph

    for n in tqdm(netx, total=len(netx)):
        sub = nx.ego_graph(netx, n, radius=radius, undirected=True, distance=distance)  # define subgraph of steps=radius
        e = sub.number_of_edges()
        v = sub.number_of_nodes()
        if v == 2:
            netx.nodes[n][name] = np.nan
        else:
            netx.nodes[n][name] = e / (3 * (v - 2))  # save value calulated for subgraph to node

    return netx


def _closeness_centrality(G, u=None, distance=None, wf_improved=True, len_graph=None):
    r"""Compute closeness centrality for nodes. Slight adaptation of networkx
    `closeness_centrality` to allow normalisation for local closeness.

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

    See Also
    --------
    betweenness_centrality, load_centrality, eigenvector_centrality,
    degree_centrality

    Notes
    -----
    The closeness centrality is normalized to `(n-1)/(|G|-1)` where
    `n` is the number of nodes in the connected part of graph
    containing the node.  If the graph is not completely connected,
    this algorithm computes the closeness centrality for each
    connected part separately scaled by that parts size.

    If the 'distance' keyword is set to an edge attribute key then the
    shortest-path length will be computed using Dijkstra's algorithm with
    that edge attribute as the edge weight.

    In NetworkX 2.2 and earlier a bug caused Dijkstra's algorithm to use the
    outward distance rather than the inward distance. If you use a 'distance'
    keyword and a DiGraph, your results will change between v2.2 and v2.3.

    References
    ----------
    .. [1] Linton C. Freeman: Centrality in networks: I.
       Conceptual clarification. Social Networks 1:215-239, 1979.
       http://leonidzhukov.ru/hse/2013/socialnetworks/papers/freeman79-centrality.pdf
    .. [2] pg. 201 of Wasserman, S. and Faust, K.,
       Social Network Analysis: Methods and Applications, 1994,
       Cambridge University Press.
    """

    if distance is not None:
        import functools
        # use Dijkstra's algorithm with specified attribute as edge weight
        path_length = functools.partial(nx.single_source_dijkstra_path_length,
                                        weight=distance)
    else:
        path_length = nx.single_source_shortest_path_length

    if u is None:
        nodes = G.nodes
    else:
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
    if u is not None:
        return closeness_centrality[u]
    else:
        return closeness_centrality


def local_closeness(graph, radius=5, name='closeness', distance=None, closeness_distance=None):
    """
    Calculates local closeness for each node based on the defined distance.

    Subgraph is generated around each node within set radius. If distance=None,
    radius will define topological distance, otherwise it uses values in distance
    attribute.

    .. math::


    Parameters
    ----------
    graph : networkx.Graph
        Graph representing street network.
        Ideally genereated from GeoDataFrame using :py:func:`momepy.gdf_to_nx`
    radius: int
        number of topological steps defining the extent of subgraph
    name : str, optional
        calculated attribute name
    distance : str, optional
        Use specified edge data key as distance.
        For example, setting distance=’weight’ will use the edge weight to
        measure the distance from the node n.
    closeness_distance : str, optional
      Use the specified edge attribute as the edge distance in shortest
      path calculations in closeness centrality algorithm

    Returns
    -------
    Graph
        networkx.Graph

    References
    ----------
    Porta

    Examples
    --------

    """
    netx = graph
    lengraph = len(netx)
    for n in tqdm(netx, total=len(netx)):
        sub = nx.ego_graph(netx, n, radius=radius, undirected=True, distance=distance)  # define subgraph of steps=radius
        netx.nodes[n][name] = _closeness_centrality(sub, n, distance=closeness_distance, len_graph=lengraph)

    return netx


def eigenvector(graph, name='eigen', **kwargs):
    """
    Calculates mean distance to neighbouring nodes.

    .. math::


    Parameters
    ----------
    graph : networkx.Graph
        Graph representing street network.
        Ideally genereated from GeoDataFrame using :py:func:`momepy.gdf_to_nx`
    name : str, optional
        calculated attribute name
    **kwargs : keyword arguments
        kwargs for `nx.eigenvector_centrality`

    Returns
    -------
    Graph
        networkx.Graph

    References
    ----------

    Examples
    --------

    """
    netx = graph

    vals = nx.eigenvector_centrality(netx, **kwargs)
    nx.set_node_attributes(netx, vals, name)

    return netx


def clustering(graph, name='cluster'):
    """
    Calculates the squares clustering coefficient for nodes.

    .. math::


    Parameters
    ----------
    graph : networkx.Graph
        Graph representing street network.
        Ideally genereated from GeoDataFrame using :py:func:`momepy.gdf_to_nx`
    name : str, optional
        calculated attribute name

    Returns
    -------
    Graph
        networkx.Graph

    References
    ----------

    Examples
    --------

    """
    netx = graph

    vals = nx.square_clustering(netx)
    nx.set_node_attributes(netx, vals, name)

    return netx
