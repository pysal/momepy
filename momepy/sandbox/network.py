# import networkx as nx
# import matplotlib.pyplot as plt
# from shapely.geometry import Point, LineString, Polygon
# import geopandas as gpd
#
# path = "/Users/martin/Strathcloud/Personal Folders/Test data/Royston/sub.shp"
#
# graph = nx.read_shp(path)
# gpd_df = ''
# radius = 400
#
# for n in graph.nodes:
#     subgraph = nx.ego_graph(graph, n)
#     sub_points = [Point((data['x'], data['y'])) for node, data in subgraph.nodes(data=True)]
#     bounding_hull = gpd.GeoSeries(sub_points).unary_union.convex_hull
#
#     # Spatial index
#     sindex = gpd_df.sindex
#
#     # Potential neighbours
#     good = []
#     for m in sindex.intersection(bounding_hull):
#         dist = n.distance(gpd_df['geometry'][n])
#         if dist < radius:
#             good.append((dist, m))
#     # Sort list in ascending order by `dist`, then `n`
#     good.sort()
#     # Return only the neighbour indices, sorted by distance in ascending order
#     value = len([x[1] for x in good])
#     graph.add
#
#
# import numpy as np
# import networkx as nx
# import pandas as pd
#
# # create some test graph
# graph = nx.erdos_renyi_graph(1000, 0.005)
#
# # create an ego-graph for some node
# node = (536313.636, 241712.168)
# ego_graph = nx.ego_graph(graph, node, radius=100, undirected=True, distance='length')
#
# # plot to check
# nx.draw(ego_graph, with_labels=True)
# plt.show()
# graph.node()
# nx.write_shp(ego_graph, "/Users/martin/Strathcloud/Personal Folders/Test data/Royston/subego.shp")

from shapely.geometry import Point, LineString
import shapely.geometry
line = LineString([(0, 0), (1, 1), (2, 2)])
point = Point(536314.636, 241711.168)
att = line.interpolate(line.project(point))
att.xy


import geopandas as gpd
path = "/Users/martin/Strathcloud/Personal Folders/Test data/Royston/itn.shp"
network = gpd.read_file(path)
attached = network.interpolate(network.project(point))
network
for index, row in network.iterrows():
    line = row['geometry']
    att = line.interpolate(line.project(point))
    if index == 0:
        goal = att
    else:
        if point.distance(att) <= point.distance(goal):
            goal = att
        else:
            continue
goal.xy


def GeoMM(traj, gdfn, gdfe):
    """
    performs map matching on a given sequence of points

    Parameters
    ----------

    Returns
    -------
    list of tuples each containing timestamp, projected point to the line, the edge to which GPS point has been projected, the geometry of the edge))

    """

    traj = pd.DataFrame(traj, columns=['timestamp', 'xy'])
    traj['geom'] = traj.apply(lambda row: Point(row.xy), axis=1)
    traj = gpd.GeoDataFrame(traj, geometry=traj['geom'], crs=EPSG3740)
    traj.drop('geom', axis=1, inplace=True)

    n_sindex = gdfn.sindex

    res = []
    for gps in traj.itertuples():
        tm = gps[1]
        p = gps[3]
        circle = p.buffer(150)
        possible_matches_index = list(n_sindex.intersection(circle.bounds))
        possible_matches = gdfn.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(circle)]
        candidate_nodes = list(precise_matches.index)

        candidate_edges = []
        for nid in candidate_nodes:
            candidate_edges.append(G.in_edges(nid))
            candidate_edges.append(G.out_edges(nid))

        candidate_edges = [item for sublist in candidate_edges for item in sublist]
        dist = []
        for edge in candidate_edges:
            # get the geometry
            ls = gdfe[(gdfe.u == edge[0]) & (gdfe.v == edge[1])].geometry
            dist.append([ls.distance(p), edge, ls])

        dist.sort()
        true_edge = dist[0][1]
        true_edge_geom = dist[0][2].item()
        pp = true_edge_geom.interpolate(true_edge_geom.project(p)) # projected point
        res.append((tm, pp, true_edge, true_edge_geom))


        return res
