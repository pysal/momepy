# intensity.py
# definitons of intensity characters
#!/usr/bin/env python3

import numpy as np
import numpy.random
import geopandas as gpd
import shapely.geometry
import operator
from tqdm import tqdm  # progress bar


# add description
def radius(gpd_df, cpt, radius):
    """
    :param gpd_df: Geopandas dataframe in which to search for points
    :param cpt:    Point about which to search for neighbouring points
    :param radius: Radius about which to search for neighbours
    :return:       List of point indices around the central point, sorted by
                 distance in ascending order
    """
    # Spatial index
    sindex = gpd_df.sindex
    # Bounding box of rtree search (West, South, East, North)
    bbox = (cpt.x - radius, cpt.y - radius, cpt.x + radius, cpt.y + radius)
    # Potential neighbours
    good = []
    for n in sindex.intersection(bbox):
        dist = cpt.distance(gpd_df['geometry'][n])
        if dist < radius:
            good.append((dist, n))
    # Sort list in ascending order by `dist`, then `n`
    good.sort()
    # Return only the neighbour indices, sorted by distance in ascending order
    return [x[1] for x in good]


def frequency(objects, look_for, column_name):
    # define new column
    objects[column_name] = None
    objects[column_name] = objects[column_name].astype('float')
    print('Calculating form factor.')

    def sort_NoneType(objects, look_for):
        if objects is not None and :
            return(geom.centroid)
        else:
            return(0)
    # SOLVE NONE TYPES AND CENTROIDS
    for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
        neighbours = radius(look_for.centroid, row['geometry'].centroid, 400)
        objects.loc[index, column_name] = len(neighbours)
