# diversity.py
# definitons of diversity characters

from tqdm import tqdm  # progress bar
import numpy as np
from .intensity import radius
import powerlaw
import pysal


'''
Gini index calculation. Gini script by https://github.com/oliviaguest/gini.
'''


def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1, array.shape[0] + 1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))


def gini_index(objects, look_for, source, column_name, id_column='uID'):
    # define new column

    print('Calculating Gini index...')

    objects_centroids = objects.copy()
    objects_centroids['geometry'] = objects_centroids.centroid

    look_for_centroids = objects.copy()
    look_for_centroids['geometry'] = look_for_centroids.centroid

    objects_centroids[column_name] = None
    objects_centroids[column_name] = objects_centroids[column_name].astype('float')

    for index, row in tqdm(objects_centroids.iterrows(), total=objects_centroids.shape[0]):
        neighbours = radius(look_for_centroids, row['geometry'], 400)
        rows = objects.iloc[neighbours]
        values = rows[[source]].values
        objects.loc[index, column_name] = gini(values)

    print('Gini index calculated.')
'''
Power law calculation.
'''


def power_law(objects, look_for, source, id_column='uID'):
    # define new column

    print('Calculating Power law...')

    objects_centroids = objects.copy()
    objects_centroids['geometry'] = objects_centroids.centroid

    look_for_centroids = objects.copy()
    look_for_centroids['geometry'] = look_for_centroids.centroid

    for index, row in tqdm(objects_centroids.iterrows(), total=objects_centroids.shape[0]):
        neighbours = radius(look_for_centroids, row['geometry'], 400)
        rows = objects.iloc[neighbours]
        values = rows[[source]]
        values_list = values[source].tolist()
        results = powerlaw.Fit(values_list)
        objects.loc[index, 'pw_alpha'] = results.power_law.alpha
        objects.loc[index, 'pw_xmin'] = results.power_law.xmin
        # R, p = results.distribution_compare('power_law', 'lognormal')
        # objects.loc[index, 'pw_r'] = R
        # objects.loc[index, 'pw_p'] = p

'''
Spatial autocorrelation.
'''


def moran_i_local(objects, source, column_name):
    print("Calculating local Moran's I...")
    print('Calculating weight matrix (rook)... (It might take a while.)')
    Wrook = pysal.weights.Rook.from_dataframe(objects, silent_island_warning=True)  # weight matrix 'rook'
    y = objects[[source]]
    print('Calculating...')
    lm = pysal.Moran_Local(y, Wrook)
    objects[column_name] = lm.Is
    print("Local Moran's I calculated.")

'''
import geopandas as gpd


path = "/Users/martin/Strathcloud/Personal Folders/Test data/Prague/p7_voro_single.shp"
test = gpd.read_file(path)
testdrop = test.drop(test.index[[1998, 2070, 4216]])  # ignore islands
W = pysal.weights.Queen.from_dataframe(test)  # weight matrix 'queen'
Wrook = pysal.weights.Rook.from_dataframe(testdrop)  # weight matrix 'rook'
y = testdrop[['ptcAre']]  # column to measure LISA
# y.drop(y.index[[1998, 2070, 4216]])
lm = pysal.Moran_Local(y, Wrook)  # calculate LISA Local Moran's I

# save LISA values (need to figure out which are useful)
testdrop['local'] = lm.Is
testdrop['sig'] = lm.p_sim
testdrop['lm_q'] = lm.q
testdrop['lm_sim'] = lm.z_sim

gg = pysal.G_Local(y, Wrook)  # calculate LISA Gettis Ord G


# save LISA values (need to figure out which are useful)
testdrop['g_Gs'] = gg.Gs
testdrop['g_EGs'] = gg.EGs
testdrop['g_Zs'] = gg.Zs
testdrop['g_psim'] = gg.p_sim


path2 = "/Users/martin/Strathcloud/Personal Folders/Test data/Prague/p7_voro_single2.shp"
testdrop.to_file(path2)
'''

'''
HDBSCAN clustering. Should be in different file than diversity.
'''

'''
import hdbscan
testdrop.columns
# select columns for analysis
clustering = testdrop[['pdcLAL', 'vicFre', 'ptcAre', 'vvcGAr', 'pw_alpha', 'local', 'g_Gs']]


# make clusterer
clusterer = hdbscan.HDBSCAN(min_cluster_size=13)

# cluster data
clusterer.fit(clustering)
# get number of clusters
clusterer.labels_.max()
# save cluster labels to geoDataFrame
testdrop['cluster'] = clusterer.labels_
'''
