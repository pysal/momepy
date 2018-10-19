import math
from rtree import index
from shapely.geometry import Polygon, LineString
from tqdm import tqdm  # progress bar

import geopandas as gpd
import pandas as pd

buildings = gpd.read_file('/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Vinohrady/blg.shp')
streets = gpd.read_file('/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Vinohrady/str.shp')
tesselation = gpd.read_file('/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Vinohrady/tess.shp')
street_name_column = 'MKN_1'
unique_id_column = 'uID'
block_id_column = 'bID'
network_id_column = 'nID'
save_to = '/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Vinohrady/edges.shp'
tesselation_to = '/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Vinohrady/tess.shp'
tesselation
tesselation = tesselation.drop(['street'], axis=1)

INFTY = 1000000000000
MIN_SIZE = 500
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
        for lid, nid, l in tqdm(lines, total=len(lines)):
            for i in range(len(l) - 1):
                a, b = l[i]
                c, d = l[i + 1]
                segment = ((a, b), (c, d))
                box = (min(a, c), min(b, d), max(a, c), max(b, d))
                # box = left, bottom, right, top
                yield (sindx, box, (lid, segment, nid))
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
            nearest_p, new_d = get_distance(p, h[1])
            if d >= new_d:
                d = new_d
                # s = (h[0], h[1], nearest_p, new_d)
                s = (h[0], h[1], h[-1])
        result[p] = s
        if s is None:
            result[p] = (0, 0)

        # some checking you could remove after you adjust the constants
        # if s is None:
        #     raise Warning("It seems INFTY is not big enough. Point was not attached to street. It might be too far.", p)

        # pboxpol = ((pbox[0], pbox[1]), (pbox[2], pbox[1]),
        #            (pbox[2], pbox[3]), (pbox[0], pbox[3]))
        # if not Polygon(pboxpol).intersects(LineString(s[1])):
        #     msg = "It seems MIN_SIZE is not big enough. "
        #     msg += "You could get inexact solutions if remove this exception."
        #     raise Exception(msg)

    return result

print('Generating centroids...')
buildings['geometry'] = buildings.centroid  # make centroids

print('Generating list of points...')
# make points list for input
centroid_list = []
for idx, row in tqdm(buildings.iterrows(), total=buildings.shape[0]):
    centroid_list = centroid_list + list(row['geometry'].coords)

print('Generating list of lines...')
# make streets list for input
street_list = []
for idx, row in tqdm(streets.iterrows(), total=streets.shape[0]):
    street_list.append((row[street_name_column], row[network_id_column], list(row['geometry'].coords)))
print('Generating rtree...')
idx = get_rtree(street_list)

print('Snapping...')
solutions = get_solution(idx, centroid_list)
solutions

result = {}
for p in tqdm(centroid_list, total=len(centroid_list)):
    pbox = (p[0] - MIN_SIZE, p[1] - MIN_SIZE, p[0] + MIN_SIZE, p[1] + MIN_SIZE)
    hits = idx.intersection(pbox, objects='raw')
    list(hits)
    d = INFTY
    s = None
    for h in hits:
        nearest_p, new_d = get_distance(p, h[1])
        if d >= new_d:
            d = new_d
            # s = (h[0], h[1], nearest_p, new_d)
            s = (h[0], h[1])
    result[p] = s
    if s is None:
        result[p] = (0, 0)

print('Forming DataFrame...')
df = pd.DataFrame.from_dict(solutions, orient='index', columns=['street', 'unused', network_id_column])  # solutions dict to df
df['point'] = df.index  # point to column
df = df.reset_index()
df['idx'] = df.index
buildings['idx'] = buildings.index

print('Joining DataFrames...')
joined = buildings.merge(df, on='idx')
print('Cleaning DataFrames...')
cleaned = joined[[unique_id_column, 'street', network_id_column]]

print('Merging with tesselation...')
tesselation = tesselation.merge(cleaned, on=unique_id_column)


print('Defining merge ID...')
for idx, row in tqdm(tesselation.iterrows(), total=tesselation.shape[0]):
    tesselation.loc[idx, 'mergeID'] = str(row['street']) + str(row[block_id_column])

print('Dissolving...')
edges = tesselation.dissolve(by='mergeID')
edges.columns
print('Generating unique edge ID...')
id = 1
for idx, row in tqdm(edges.iterrows(), total=edges.shape[0]):
    edges.loc[idx, 'eID'] = id
    id = id + 1

print('Cleaning edges...')
edges_clean = edges[['geometry', 'eID', block_id_column, network_id_column]]

print('Saving street edges to', save_to)
edges_clean.to_file(save_to)

print('Cleaning tesselation...')
tesselation = tesselation.drop(['street', 'mergeID'], axis=1)

print('Saving tesselation to', tesselation_to)
tesselation.to_file(tesselation_to)

print('Done.')
