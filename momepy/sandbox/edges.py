import geopandas as gpd
import pandas as pd
from tqdm import tqdm  # progress bar
import math
from rtree import index
from osgeo import ogr
from shapely.wkt import loads
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import *
import shapely.ops
import osmnx as ox
import operator

buildings = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Tests/181115/blg_oldtown.shp")
streets = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Tests/181115/street.shp")
tesselation = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Tests/181115/tess_oldtown.shp")
unique_id_column = 'uID'
network_id_column = 'nID'
tesselation_to = "/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Tests/181122/tess_oldtown_nid.shp"
buildings_to = "/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Tests/181122/blg_oldtown_nid.shp"


def get_network_id(buildings, streets, tesselation, unique_id_column, network_id_column, buildings_to, tesselation_to):
    INFTY = 1000000000000
    MIN_SIZE = 100
    # MIN_SIZE should be a vaule such that if you build a box centered in each
    # point with edges of size 2*MIN_SIZE, you know a priori that at least one
    # segment is intersected with the box. Otherwise, you could get an inexact
    # solution, there is an exception checking this, though.

    def distance(a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def get_distance(apoint, segment):
        a = apoint
        b, c = segment
        # t = <a-b, c-b>/|c-b|**2
        # because p(a) = t*(c-b)+b is the ortogonal projection of vector a
        # over the rectline that includes the points b and c.
        t = (a[0] - b[0]) * (c[0] - b[0]) + (a[1] - b[1]) * (c[1] - b[1])
        t = t / ((c[0] - b[0]) ** 2 + (c[1] - b[1]) ** 2)
        # Only if t 0 <= t <= 1 the projection is in the interior of
        # segment b-c, and it is the point that minimize the distance
        # (by pythagoras theorem).
        if 0 < t < 1:
            pcoords = (t * (c[0] - b[0]) + b[0], t * (c[1] - b[1]) + b[1])
            dmin = distance(a, pcoords)
            return pcoords, dmin
        elif t <= 0:
            return b, distance(a, b)
        elif 1 <= t:
            return c, distance(a, c)

    def get_rtree(lines):
        def generate_items():
            sindx = 0
            for nid, l in tqdm(lines, total=len(lines)):
                for i in range(len(l) - 1):
                    a, b = l[i]
                    c, d = l[i + 1]
                    segment = ((a, b), (c, d))
                    box = (min(a, c), min(b, d), max(a, c), max(b, d))
                    # box = left, bottom, right, top
                    yield (sindx, box, (segment, nid))
                    sindx += 1
        return index.Index(generate_items())

    def get_solution(idx, points):
        result = {}
        for p in tqdm(points, total=len(points)):
            pbox = (p[0] - MIN_SIZE, p[1] - MIN_SIZE, p[0] + MIN_SIZE, p[1] + MIN_SIZE)
            hits = idx.intersection(pbox, objects='raw')
            d = INFTY
            s = None
            for h in hits:
                nearest_p, new_d = get_distance(p, h[0])
                if d >= new_d:
                    d = new_d
                    # s = (h[0], h[1], nearest_p, new_d)
                    s = (h[0], h[1], h[-1])
            result[p] = s
            if s is None:
                result[p] = (0, 0)

        return result

    print('Generating centroids...')
    buildings_c = buildings.copy()
    buildings_c['geometry'] = buildings_c.centroid  # make centroids

    print('Generating list of points...')
    # make points list for input
    centroid_list = []
    for idx, row in tqdm(buildings_c.iterrows(), total=buildings_c.shape[0]):
        centroid_list = centroid_list + list(row['geometry'].coords)

    print('Generating list of lines...')
    # make streets list for input
    street_list = []
    for idx, row in tqdm(streets.iterrows(), total=streets.shape[0]):
        street_list.append((row[network_id_column], list(row['geometry'].coords)))
    print('Generating rtree...')
    idx = get_rtree(street_list)

    print('Snapping...')
    solutions = get_solution(idx, centroid_list)

    print('Forming DataFrame...')
    df = pd.DataFrame.from_dict(solutions, orient='index', columns=['unused', 'unused', network_id_column])  # solutions dict to df
    df['point'] = df.index  # point to column
    df = df.reset_index()
    df['idx'] = df.index
    buildings_c['idx'] = buildings_c.index

    print('Joining DataFrames...')
    joined = buildings_c.merge(df, on='idx')
    print('Cleaning DataFrames...')
    cleaned = joined[[unique_id_column, network_id_column]]

    print('Merging with tesselation...')
    tesselation = tesselation.merge(cleaned, on=unique_id_column)

    print('Saving tesselation to', tesselation_to)
    tesselation.to_file(tesselation_to)

    print('Buildings attribute join...')
    # attribute join cell -> building
    tess_nid_eid = tesselation[['uID', 'nID']]

    buildings = buildings.merge(tess_nid_eid, on='uID')

    print('Saving buildings to', buildings_to)
    buildings.to_file(buildings_to)

    print('Done.')
