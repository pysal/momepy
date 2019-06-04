#!/usr/bin/env python
# -*- coding: utf-8 -*-

# connectivity.py
# definitons of connectivity characters
import networkx as nx
from tqdm import tqdm
import numpy as np


def node_degree(graph, name='degree'):
    """
    Calculates node degree for each node.

    Wrapper around `networkx.degree()`

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

    degree = dict(nx.degree(netx))
    nx.set_node_attributes(netx, degree, name)

    return netx


def meshedness(graph, radius=5, name='meshedness'):
    """
    Calculates meshedness for subgraph around each node.

    Subgraph is generated around each node within set topological radius.

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
        sub = nx.ego_graph(netx, n, radius=radius, undirected=True)  # define subgraph of steps=radius
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


def cds_length(graph=None, radius=5, mode='sum', name='cds_len', degree='degree', length='mm_len'):
    """
    Calculates length of cul-de-sacs for subgraph around each node.

    Subgraph is generated around each node within set topological radius.

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
        sub = nx.ego_graph(netx, n, radius=radius, undirected=True)  # define subgraph of steps=radius
        lens = []
        for u, v, cds in sub.edges.data('cdsbool'):
            if cds:
                lens.append(sub[u][v][length])
        if mode == 'sum':
            netx.nodes[n][name] = sum(lens)  # save value calulated for subgraph to node
        elif mode == 'mean':
            netx.nodes[n][name] = np.mean(lens)  # save value calulated for subgraph to node

    return netx
