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
    a) merge all degree-2 edges (DONE!)
    b) calculate nodal degree metrics (DONE!)
    c) calculate circuity metrics
    d) calculate bridge metrics (TODO: Adapt to multigraphs)
    e) calculate non-cycle metrics (TODO: Adapt to multigraphs)
    f) calculate sinuosity metrics (DONE!)

Step 3: calculate SNDI
'''

import numpy as np
from numpy import Inf
import math
import geopandas as gpd
import pandas as pd
import networkx as nx
import libpysal

from .utils import nx_to_gdf, gdf_to_nx
from .graph import mean_node_degree, node_degree
from .preprocessing import remove_false_nodes
from .shape import Linearity

import time

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

    # create dictionary to assign edge attributes. Make everything a cycle to start
    edge_dendricity = dict([(tuple(edge),'C') for edge in street_graph.edges()])

    # get bridges
    bridges = list(nx.bridges(street_graph))
    
    # Assign B value to bridge edges
    for edge in bridges:
        edge_dendricity[tuple(edge)]='B'

    # find degree 1 nodes for dead ends
    deg_1_nodes = [node for node, attr in street_graph.nodes(data=True) if attr['degree']==1]

    # Assign D value to dead end edges
    for node in deg_1_nodes:
        edge = list(street_graph.edges(node))[0]
        edge_dendricity[tuple(edge)]='D'

    # get fraction of edge dead ends
    N_dead_ends = len(deg_1_nodes)
    frc_edge_dead_ends = N_dead_ends/N_edges

    # get number/fraction of bridges TODO: only working with Graph and not Multigraphs. Subtract number of dead-ends
    N_bridges = len(bridges) - N_dead_ends
    frc_edge_bridges = N_bridges/N_edges

    # get number/fraction of self loops TODO: only working with Graph and not Multigraphs
    self_loops = list(nx.selfloop_edges(street_graph))

    # Assign B value to bridge edges
    for edge in self_loops:
        edge_dendricity[tuple(edge)]='S'

    N_self_loops = len(self_loops)
    frc_edge_self_loops = N_self_loops/N_edges

    # get number/fraction of cycles and non_cycles
    N_cycles = N_edges - N_dead_ends - N_bridges - N_self_loops
    frc_edge_cylces = N_cycles/N_edges
    frc_edge_non_cycles = 1 - frc_edge_cylces

    # set dendricity attributes to graph
    nx.set_edge_attributes(street_graph, edge_dendricity, 'dendricity')

    # convert back to gdf for edges
    lines = nx_to_gdf(street_graph, points=False)

    # length of all edges
    total_length = np.sum(lines.geometry.length)

    # length of cycles
    cycle_length = np.sum(lines[lines['dendricity']=='C'].geometry.length)
    frc_length_cycle = cycle_length/total_length

    # length of self loops
    self_loop_length = np.sum(lines[lines['dendricity']=='S'].geometry.length)
    frc_length_self_loops = self_loop_length/total_length

    # length of dead ends
    dead_ends_length = np.sum(lines[lines['dendricity']=='D'].geometry.length)
    frc_length_dead_ends = dead_ends_length/total_length

    # length of bridges
    bridges_length = np.sum(lines[lines['dendricity']=='B'].geometry.length)
    frc_length_bridges = bridges_length/total_length

    # get cycle/non-cycle lenghs
    frc_length_non_cycle = frc_length_self_loops + frc_length_dead_ends + frc_length_bridges

    # TEST this should always equal 1 :frc_length_non_cycle + frc_length_cycle

    '''
    PART 5
    calculate sinuosity
    '''
    # sinuosity is quite similar to mm.linearity. It is just the inverse of it and summed
    log_sinuosity = np.log(1 / Linearity(lines, aggregate=True).aggregated)

    '''
    PART 6
    calculate circuity metrics
    TODO: jsut need one function
    '''
    # get node list
    node_list = street_graph.nodes

    # euclidean distance matrix from node to node
    eucl_dist_matrix = libpysal.cg.distance_matrix(np.asarray(node_list), p=2.0)

    # set anything farher away than 3km to zero
    eucl_dist_matrix[eucl_dist_matrix > 3000] = 0.0

    # get shortest path distance matrix from all nodes to all nodes (TODO: THIS IS BOTTLENECK)
    df = pd.DataFrame.from_dict(dict(nx.all_pairs_dijkstra_path_length(street_graph, weight='mm_len')))
    path_dist_matrix = df.reindex(index=node_list, columns=node_list).to_numpy()
    path_dist_matrix = np.nan_to_num(path_dist_matrix, posinf=0, neginf=0)

    # 0-500 meter circuity
    eucl_0_500 = eucl_dist_matrix.copy()
    path_0_500 = path_dist_matrix.copy()

    bool_0_500 = np.array(np.where(eucl_0_500 <= 500, 1, 0), dtype=bool)
    
    eucl_0_500_sum = np.sum(eucl_0_500[bool_0_500])
    path_0_500_sum = np.sum(path_0_500[bool_0_500])

    # replace all infinity with zero
    if round(eucl_0_500_sum, 2):
        circuity_0_500 = path_0_500_sum/eucl_0_500_sum
        log_circuity_0_500 = np.log(circuity_0_500)
    else:
        log_circuity_0_500 = 0
        print('WARNING: No nodes within 0-500 meters of each other')

    # 500-1000 meter circuity
    eucl_500_1000 = eucl_dist_matrix.copy()
    path_500_1000 = path_dist_matrix.copy()

    bool_500_1000 = np.array(np.where((eucl_500_1000 > 500) & (eucl_500_1000 <= 1000), 1, 0), dtype=bool)
    
    eucl_500_1000_sum = np.sum(eucl_500_1000[bool_500_1000])
    path_500_1000_sum = np.sum(path_500_1000[bool_500_1000])

    if round(eucl_500_1000_sum, 2):
        circuity_500_1000 = path_500_1000_sum/eucl_500_1000_sum
        log_circuity_500_1000 = np.log(circuity_500_1000)
    else:
        log_circuity_500_1000 = 0
        print('WARNING: No nodes within 500-1000 meters of each other')

    # 1000-1500 meter circuity
    eucl_1000_1500 = eucl_dist_matrix.copy()
    path_1000_1500 = path_dist_matrix.copy()

    bool_1000_1500 = np.array(np.where((eucl_1000_1500 > 1000) & (eucl_1000_1500 <= 1500), 1, 0), dtype=bool)
    
    eucl_1000_1500_sum = np.sum(eucl_1000_1500[bool_1000_1500])
    path_1000_1500_sum = np.sum(path_1000_1500[bool_1000_1500])

    if round(eucl_1000_1500_sum, 2):
        circuity_1000_1500 = path_1000_1500_sum/eucl_1000_1500_sum
        log_circuity_1000_1500 = np.log(circuity_1000_1500)
    else:
        log_circuity_1000_1500 = 0
        print('WARNING: No nodes within 1000-1500 meters of each other')

    # 1500-2000 meter circuity
    eucl_1500_2000 = eucl_dist_matrix.copy()
    path_1500_2000 = path_dist_matrix.copy()

    bool_1500_2000 = np.array(np.where((eucl_1500_2000 > 1500) & (eucl_1500_2000 <= 2000), 1, 0), dtype=bool)
    
    eucl_1500_2000_sum = np.sum(eucl_1500_2000[bool_1500_2000])
    path_1500_2000_sum = np.sum(path_1500_2000[bool_1500_2000])
    
    if round(eucl_1500_2000_sum, 2):
        circuity_1500_2000 = path_1500_2000_sum/eucl_1500_2000_sum
        log_circuity_1500_2000 = np.log(circuity_1500_2000)
    else:
        log_circuity_1500_2000 = 0
        print('WARNING: No nodes within 1500-2000 meters of each other')

    # 2000-2500 meter circuity
    eucl_2000_2500 = eucl_dist_matrix.copy()
    path_2000_2500 = path_dist_matrix.copy()

    bool_2000_2500 = np.array(np.where((eucl_2000_2500 > 2000) & (eucl_2000_2500 <= 2500), 1, 0), dtype=bool)
    
    eucl_2000_2500_sum = np.sum(eucl_2000_2500[bool_2000_2500])
    path_2000_2500_sum = np.sum(path_2000_2500[bool_2000_2500])

    if round(eucl_2000_2500_sum, 2):
        circuity_2000_2500 = path_2000_2500_sum/eucl_2000_2500_sum
        log_circuity_2000_2500 = np.log(circuity_2000_2500)
    else:
        log_circuity_2000_2500 = 0
        print('WARNING: No nodes within 2000-2500 meters of each other')
    
    # 2500-3000 meter circuity
    eucl_2500_3000 = eucl_dist_matrix.copy()
    path_2500_3000 = path_dist_matrix.copy()

    bool_2500_3000 = np.array(np.where((eucl_2500_3000 > 2500) & (eucl_2500_3000 <= 3000), 1, 0), dtype=bool)
    
    eucl_2500_3000_sum = np.sum(eucl_2500_3000[bool_2500_3000])
    path_2500_3000_sum = np.sum(path_2500_3000[bool_2500_3000])

    if eucl_2500_3000_sum:
        circuity_2500_3000 = path_2500_3000_sum/eucl_2500_3000_sum
        log_circuity_2500_3000 = np.log(circuity_2500_3000)
    else:
        log_circuity_2500_3000 = 0
        print('WARNING: No nodes within 2500-3000 meters of each other')

    '''
    PART 7
    calculate PCA1
    '''
    PCA1_1 = pca1_nodal_degree*(neg_mean_nodal_degree - mean_nodal_degree)/std_nodal_degree
    PCA1_2 = pca1_frc_dead_ends*(frc_node_dead_ends - mean_frc_dead_ends)/std_frc_dead_ends
    PCA1_3 = pca1_log_circuity_1*(log_circuity_0_500 - mean_log_circuity_1)/std_log_circuity_1
    PCA1_4 = pca1_log_circuity_2*(log_circuity_500_1000 - mean_log_circuity_2)/std_log_circuity_2
    PCA1_5 = pca1_log_circuity_3*(log_circuity_1000_1500 - mean_log_circuity_3)/std_log_circuity_3
    PCA1_6 = pca1_log_circuity_4*(log_circuity_1500_2000 - mean_log_circuity_4)/std_log_circuity_4
    PCA1_7 = pca1_log_circuity_5*(log_circuity_2000_2500 - mean_log_circuity_5)/std_log_circuity_5
    PCA1_8 = pca1_log_circuity_6*(log_circuity_2500_3000 - mean_log_circuity_6)/std_log_circuity_6
    PCA1_9 = pca1_frc_bridges_length*(frc_length_bridges - mean_frc_bridges_length)/std_frc_bridges_length
    PCA1_10 = pca1_frc_bridges_N*(frc_edge_bridges - mean_frc_bridges_N)/std_frc_bridges_N
    PCA1_11 = pca1_frc_non_cycle_length*(frc_length_non_cycle - mean_frc_non_cycle_length)/std_frc_non_cycle_length
    PCA1_12 = pca1_frc_non_cycle_N*(frc_edge_non_cycles - mean_frc_non_cycle_N)/std_frc_non_cycle_N
    PCA1_13 = pca1_log_sinuosity*(log_sinuosity - mean_log_sinuosity)/std_log_sinuosity

    PCA1 = PCA1_1 + PCA1_2 + PCA1_3 + PCA1_4 + PCA1_5 + PCA1_6 + PCA1_7 + PCA1_8 + PCA1_9 + PCA1_10 + PCA1_11 + PCA1_12 + PCA1_13
    '''
    PART 8
    calculate SNDi
    '''

    SNDi = PCA1 + 3.0

    print(f'Mean nodal degree (-) is: {neg_mean_nodal_degree}')
    print('-------------------------------------------------')
    print(f'Fraction of node dead ends is: {frc_node_dead_ends}')
    print('-------------------------------------------------')
    print(f'Fraction of bridges length is: {frc_length_bridges}')
    print('-------------------------------------------------')
    print(f'Fraction of bridges N edges is: {frc_edge_bridges}')
    print('-------------------------------------------------')
    print(f'Fraction of non-cylce length is: {frc_length_non_cycle}')
    print('-------------------------------------------------')
    print(f'Fraction of bridges N edges is: {frc_edge_non_cycles}')
    print('-------------------------------------------------')
    print(f'Log of sinuosity is: {log_sinuosity}')
    print('-------------------------------------------------')
    print(f'Log of circuity 0-500 is: {log_circuity_0_500}')
    print('-------------------------------------------------')
    print(f'Log of circuity 500-1000 is: {log_circuity_500_1000}')
    print('-------------------------------------------------')
    print(f'Log of circuity 1000-1500 is: {log_circuity_1000_1500}')
    print('-------------------------------------------------')
    print(f'Log of circuity 1500-2000 is: {log_circuity_1500_2000}')
    print('-------------------------------------------------')
    print(f'Log of circuity 2000-2500 is: {log_circuity_2000_2500}')
    print('-------------------------------------------------')
    print(f'Log of circuity 2500-3000 is: {log_circuity_2500_3000}')
    return SNDi


def circuity(street_graph, distance_band=[]):

    # get node list
    node_list = street_graph.nodes
    
    # get shortest path distance matrix from all nodes to all nodes
    path_dist_matrix = nx.floyd_warshall_numpy(street_graph, nodelist=node_list, weight='mm_len')

    # euclidean distance matrix from node to node
    eucl_dist_matrix = libpysal.cg.distance_matrix(np.asarray(node_list), p=2.0)

    # get circuity for band
    circuity = circuity_band(path_dist_matrix, eucl_dist_matrix, distance_band)

    return circuity


def circuity_band(path_dist_matrix, eucl_dist_matrix, band=[0, 500]):

    # Make copies
    eucl_dist_band = eucl_dist_matrix.copy()
    path_dist_band = path_dist_matrix.copy()

    # get boolean array within using euclidean  
    bool_band = np.array(np.where((eucl_dist_band > band[0]) & (eucl_dist_band <= band[1]), 1, 0), dtype=bool)

    # get only values within eucldidean and sum
    circuity_band = np.sum(path_dist_band[bool_band])/np.sum(eucl_dist_band[bool_band])
    
    return circuity_band

