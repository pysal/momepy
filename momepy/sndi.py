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
Step 1: copy over hardcoded values (DONE!)

Step 2: calculate all metrics
    a) merge all degree-2 edges (DONE! Thanks to momepy magic)
    b) calculate nodal degree metrics (DONE!)
    c) calculate circuity metrics
    d) calculate bridge metrics (TODO: get length. Adapt to multigraphs)
    e) calculate non-cycle metrics (TODO: get length. Adapt to multigraphs)
    f) calculate sinuosity metrics

Step 3: calculate SNDI
'''


from networkx.classes.function import edges
from networkx.generators import line
import numpy as np
from shapely import ops
from shapely.geometry import MultiLineString
import geopandas as gpd
import pandas as pd
import networkx as nx

from .utils import nx_to_gdf, gdf_to_nx
from .graph import mean_node_degree, node_degree
from .preprocessing import remove_false_nodes
    

def SNDi(street_graph):

    '''
    PART 1
    process graph 
    TODO: future input either gdf or graph
    '''
    # convert to gdf, only need lines
    lines = nx_to_gdf(street_graph, points=False)
    
    '''
    PART 2
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
    PART 3
    calculate NODAL DEGREE METRICS
    '''

    # remove and merge all degree-2 nodes from gdf and convert to graph (Note not working for Multigraph)
    lines = remove_false_nodes(lines)
    street_graph = gdf_to_nx(lines, multigraph=False)

    # add nodal degree to graph attributes
    street_graph = node_degree(street_graph)
    
    # get array of node degrees. 
    array_deg = np.array(list(dict(street_graph.nodes('degree')).values()))

    # change all values > 4 to 4
    array_deg[array_deg > 4] = 4

    # get fraction of type of nodes. There is only 1,3,4
    frc_node_deg_1 = np.count_nonzero(array_deg == 1)/array_deg.size
    frc_node_deg_3 = np.count_nonzero(array_deg == 3)/array_deg.size
    frc_node_deg_4 = np.count_nonzero(array_deg == 4)/array_deg.size
    # TESTING: frc_node_deg_1 + frc_node_deg_3 + frc_node_deg_4 should equal 1

    # fraction of degree 1 or 3
    frc_node_deg_1_3 = frc_node_deg_1 + frc_node_deg_3

    # get fraction of dead ends (I think number of node dead ends is equal to number of edge dead ends?)
    N_node_dead_ends = np.count_nonzero(array_deg == 1)
    frc_node_dead_ends = N_node_dead_ends/array_deg.size

    # Negative nodal degree
    # cannot use momepy function (mean_node_degree) because we need to set any node with degree > 4 equal to 4
    neg_mean_nodal_degree = -np.mean(array_deg)

    '''
    PART 4
    calculate DENDRICITY METRICS
    '''
    # get total number of edges
    N_edges = lines.shape[0]

    # get fraction of edge dead ends (I think number of node dead ends is equal to number of edge dead ends?)
    N_dead_ends = np.count_nonzero(array_deg == 1)
    frc_edge_dead_ends = N_dead_ends/N_edges
    
    # get number/fraction of bridges TODO: only working with Graph and not Multigraphs. Subtract number of dead-ends
    bridges = list(nx.bridges(street_graph))
    N_bridges = len(bridges) - N_dead_ends
    frc_edge_bridges = N_bridges/N_edges

    # get number/fraction of self loops TODO: only working with Graph and not Multigraphs
    self_loops = list(nx.selfloop_edges(street_graph))
    N_self_loops = len(self_loops)
    frc_edge_self_loops = N_self_loops/N_edges

    # get number/fraction of cycles and non_cycles
    N_cycles = N_edges - N_dead_ends - N_bridges - N_self_loops
    frc_edge_cylces = N_cycles/N_edges
    frc_edge_non_cycles = 1 - frc_edge_cylces

