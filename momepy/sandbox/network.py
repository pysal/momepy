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
