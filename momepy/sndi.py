"""
This Python script calculates the Street-Network Disconnectedness index (SNDi).

Journal article: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0223078

Authors: Christopher Barrington-Leigh and Adam Millard-Ball
Date: November 26, 2019

Adapted for momepy by: Andres Morfin Veytia
Date: September 25, 2021
"""

'''
TODO:
Step 1: copy over hardcoded values

Step 2: calculate all metrics
    a) merge all degree-2 edges (can currently do it with osmnx gdfs but perhaps it should be done with nx?)
    b) calculate nodal degree metrics (mean negative nodal degree)
    c) calculate circuity metrics
    d) calculate  bridge metrics
    e) calculate non-cycle metrics
    f) calculate sinuosity metrics

Step 3: calculate SNDI
'''


import numpy as np
from shapely import ops
import geopandas as gpd
import pandas as pd

from .utils import nx_to_gdf, gdf_to_nx
from .graph import mean_node_degree, node_degree
    

def SNdi(street_graph):

    # process streetgraph
    street_graph = street_graph

    # # convert to gdf
    # nodes, edges = nx_to_gdf(street_graph)
    
    '''
    PART 1
    mean, standard deviation, and PCA1 values for GADM dataset https://gitlab.com/cpbl/global-sprawl-2020
    '''

    # nodal degree (negative)
    mean_nodal_degree = -2.8303675378032223
    std_nodal_degree = 0.3429680685039067
    pca1_nodal_degree = 0.3493185

    # dead ends
    mean_frc_dead_ends = 0.1938995209875757
    std_frc_dead_ends = 0.12799681159868687
    pca1_frc_dead_ends = 0.34891059999999996

    # log circuity 0-0.5 km
    mean_log_circuity_1 = 0.17186301207993082
    std_log_circuity_1 = 0.08438691033187842
    pca1_log_circuity_1 = 0.2702526

    # log circuity 0.5-1 km
    mean_log_circuity_2 = 0.15163308503586118
    std_log_circuity_2 = 0.06012702604377113
    pca1_log_circuity_2 = 0.2956021

    # log circuity 1-1.5 km
    mean_log_circuity_3 = 0.1326839886337719
    std_log_circuity_3 = 0.04275427261228078
    pca1_log_circuity_3 = 0.2868906

    # log circuity 1.5-2 km
    mean_log_circuity_4 = 0.10899568877464477
    std_log_circuity_4 = 0.02797831394141562
    pca1_log_circuity_4 = 0.25418209999999997

    # log circuity 2-2.5 km
    mean_log_circuity_5 = 0.07702766891890014
    std_log_circuity_5 = 0.01748327463656427
    pca1_log_circuity_5 = 0.1882174

    # log circuity 2.5-3 km
    mean_log_circuity_6 = 0.03958868880572528
    std_log_circuity_6 = 0.011484248546405224
    pca1_log_circuity_6 = 0.11011389999999999

    # fraction bridges (length)
    mean_frc_bridges_length= 0.04576044830113072
    std_frc_bridges_length = 0.09259988437312076
    pca1_frc_bridges_length = 0.22149539999999998

    # fraction bridges (N edges)
    mean_frc_bridges_N = 0.1831914893399157
    std_frc_bridges_N = 0.14273719661948095
    pca1_frc_bridges_N = 0.28478059999999994

    # fraction non-cycle (length)
    mean_frc_non_cycle_length = 0.1683625923750888
    std_frc_non_cycle_length = 0.15402234858926675
    pca1_frc_non_cycle_length = 0.3198946

    # fraction non-cycle (N edges)
    mean_frc_non_cycle_N = 0.1831914893399157
    std_frc_non_cycle_N = 0.14273719661948095
    pca1_frc_non_cycle_N = 0.3792382

    # log sinuosity
    mean_log_sinuosity = 0.034455048569152755
    std_log_sinuosity = 0.026418496738454776
    pca1_log_sinuosity = 0.1582431

    '''
    calculate metrics
    '''

    # remove and merge all degree-2 nodes
    street_graph = anneal_degree2_nodes(street_graph)

    # add nodal degree to graph attributes
    street_graph = node_degree(street_graph)
    
    # get negative mean nodal degree. cannot use momepy function because we need to set any node with degree > 4 equal to 4
    array_deg = np.array(list(dict(street_graph.nodes('degree')).values()))

    # change all values greater than 4 to 4 and get 
    array_deg[array_deg > 4] = 4
    neg_mean_nodal_degree = -np.mean(array_deg)

    # get fraction of dead ends
    frc_dead_ends = np.count_nonzero(array_deg == 1)/array_deg.size



def anneal_degree2_nodes(G):
    """
    Remove degree2 nodes from a graph, since we ignore them for all our connectivity metrics.

    When we find a degree-2 node, we want to do something smart with the attributes of the two edges. 
    Pass a function merge_attributes to generate a dict of attributes given two dicts of attributes.

    """

    d2nodes=[nn for nn,dd in list(G.degree) if dd==2]

    attribute_merger=lambda a,b: a

    
    for node in  d2nodes:
        twoedges=G.edges(node,data=True)

        if len(twoedges)==1: continue # We're not removing self-loops (It would need to be an isolated node anyway.)
        n1,n2=twoedges[0][1],twoedges[1][1]
        a1,a2=twoedges[0][2],twoedges[1][2]
        assert isinstance(a1,dict)
        assert isinstance(a2,dict)
        attr=attribute_merger(a1,a2)
        if n2 in G.edge[n1]:
            # If the new edge already exists, we have a multi-node self-loop. The add_edge will replace the extant one, leaving a dead-end rather than self-loop or multipaths.
            # Keep track of the attributes by merging in what we're about to overwrite:
            attr=attribute_merger(attr, G.edge[n1][n2])
        G.add_edge(n1,n2,attr)
        G.remove_node(node)
    return

