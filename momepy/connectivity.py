#!/usr/bin/env python
# -*- coding: utf-8 -*-

# connectivity.py
# definitons of connectivity characters
import networkx as nx
from tqdm import tqdm

from .utils import gdf_to_nx, nx_to_gdf


# edge-based example
def betweenness(graph=None, geodataframe=None, target='series', unique_id=None):
    if graph is not None and geodataframe is not None:
        import warnings
        warnings.war('Both graph and geodataframe were passed. Graph will be used for calculation.')

    netx = None
    if graph is not None:
        netx = graph
    elif geodataframe is not None:
        if netx is None:
            netx = gdf_to_nx(geodataframe)
    else:
        raise Warning('Either graph or geodataframe must be passed as an argument.')

    betweennesss = nx.edge_betweenness_centrality(netx)  # calculate
    nx.set_edge_attributes(netx, betweennesss, 'mm_between')

    if target == 'series':
        edges = nx_to_gdf(netx, nodes=False)
        joined = geodataframe.join(edges[[unique_id, 'mm_between']], rsuffix='r')
        series = joined['mm_between']
        return series
    elif target == 'graph':
        return netx
    raise Warning('Target {} is not supported. Use "series" for pandas.Series or "graph" for networkx.Graph.'.format(target))


# node based example
def node_degree(graph=None, geodataframe=None, target='gdf'):
    if graph is not None and geodataframe is not None:
        import warnings
        warnings.war('Both graph and geodataframe were passed. Graph will be used for calculation.')

    netx = None
    if graph is not None:
        netx = graph
    elif geodataframe is not None:
        if netx is None:
            netx = gdf_to_nx(geodataframe)
    else:
        raise Warning('Either graph or geodataframe must be passed as an argument.')

    degree = dict(nx.degree(netx))
    nx.set_node_attributes(netx, degree, 'degree')

    if target == 'gdf':
        nodes = nx_to_gdf(netx, edges=False)
        return nodes
    elif target == 'graph':
        return netx
    raise Warning('Target {} is not supported. Use "gdf" for geopandas.GeoDataFrame or "graph" for networkx.Graph.'.format(target))


def meshedness(graph=None, geodataframe=None, radius=5, target='gdf'):
    if graph is not None and geodataframe is not None:
        import warnings
        warnings.war('Both graph and geodataframe were passed. Graph will be used for calculation.')

    netx = None
    if graph is not None:
        netx = graph
    elif geodataframe is not None:
        if netx is None:
            netx = gdf_to_nx(geodataframe)
    else:
        raise Warning('Either graph or geodataframe must be passed as an argument.')

    for n in tqdm(netx, total=len(netx)):
        sub = nx.ego_graph(netx, n, radius=radius, undirected=True)  # define subgraph of steps=radius
        e = sub.number_of_edges()
        v = sub.number_of_nodes()
        netx.nodes[n]['meshedness'] = (e - v + 1) / (2 * v - 5)  # save value calulated for subgraph to node

    if target == 'gdf':
        nodes = nx_to_gdf(netx, edges=False)
        return nodes
    elif target == 'graph':
        return netx
    raise Warning('Target {} is not supported. Use "gdf" for geopandas.GeoDataFrame or "graph" for networkx.Graph.'.format(target))
