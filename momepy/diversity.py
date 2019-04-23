#!/usr/bin/env python
# -*- coding: utf-8 -*-

# diversity.py
# definitons of diversity characters

from tqdm import tqdm  # progress bar
import numpy as np
from .intensity import radius
import pandas as pd
# import powerlaw


'''
Gini index calculation. Gini script by https://github.com/oliviaguest/gini.

There should be docs.
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

    print('Calculating Gini index...')

    objects_centroids = objects.copy()
    objects_centroids['geometry'] = objects_centroids.centroid

    look_for_centroids = objects.copy()
    look_for_centroids['geometry'] = look_for_centroids.centroid

    # define empty list for results
    results_list = []

    for index, row in tqdm(objects_centroids.iterrows(), total=objects_centroids.shape[0]):
        neighbours = radius(look_for_centroids, row['geometry'], 400)
        rows = objects.iloc[neighbours]
        values = rows[[source]].values
        results_list.append(gini(values))

    series = pd.Series(results_list)

    print('Gini index calculated.')
    return series


'''
Power law calculation.

Docs missing.
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
    return objects


'''
Spatial autocorrelation.

Docs missing.
'''
# from libpysal.weights import DistanceBand
# import esda
#
#
# def moran_i_local(objects, source, column_name):
#     print("Calculating local Moran's I...")
#     print('Calculating weight matrix... (It might take a while.)')
#     W = DistanceBand.from_dataframe(objects, 400)  # weight matrix
#     y = objects[[source]]
#     print('Calculating...')
#     lm = esda.Moran_Local(y, W)
#     objects[column_name] = lm.Is
#     print("Local Moran's I calculated.")


# import geopandas as gpd
# import esda
# from pysal.lib import weights, examples
#
#
#
# path = "/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/P7/p7_voro_single2.shp"
# test = gpd.read_file(path)
# test.columns
# # testdrop = test.drop(test.index[[1998, 2070, 4216]])  # ignore islands
# W = pysal.lib.weights.Distance.DistanceBand.from_dataframe(test, 400)  # weight matrix 'queen'
# Wqueen
# Wqueen = Queen.from_dataframe(test)  # weight matrix 'rook'
# y = test[['ptcAre']]  # column to measure LISA
# # y.drop(y.index[[1998, 2070, 4216]])
# lm = esda.Moran_Local(y, Wqueen)  # calculate LISA Local Moran's I
#
# # save LISA values (need to figure out which are useful)
# test['400local'] = lm.Is
# test['sig'] = lm.p_sim
# test['lm_q'] = lm.q
# test['lm_sim'] = lm.z_sim
#
# gg = pysal.G_Local(y, Wrook)  # calculate LISA Gettis Ord G
#
# test['5local']
# # save LISA values (need to figure out which are useful)
# testdrop['g_Gs'] = gg.Gs
# testdrop['g_EGs'] = gg.EGs
# testdrop['g_Zs'] = gg.Zs
# testdrop['g_psim'] = gg.p_sim
#
#
# path2 = "/Users/martin/Strathcloud/Personal Folders/Test data/Prague/p7_voro_single2.shp"
# test.to_file(path)


'''
HDBSCAN clustering. Should be in different file than diversity.

Remove.
'''

# import pandas as pd
# values = pd.read_csv('/Users/martin/Strathcloud/Personal Folders/Test data/Prague/Prague/CSV/prg.csv')
#
# import hdbscan
# values.columns
#
# # select columns for analysis
# clustering = values[['pdbHei', 'pdbAre', 'uID',
#                      'pdbFlA', 'pdbVol', 'pdbPer', 'pdbCoA', 'pdbLAL', 'psbFoF', 'psbFra',
#                      'psbVFR', 'psbCom', 'psbCom2', 'psbCon', 'psbCoI', 'psbCor', 'psbShI',
#                      'psbSqu', 'psbERI', 'psbElo', 'ptbOri', 'pdcLAL', 'ptcAre',
#                      'pscCom', 'pscElo', 'pivCAR', 'vicFre', 'vvcGAr', 'vvvGCA', 'uvvMCA',
#                      'pivFAR', 'bID', 'bdkAre', 'bdkPer', 'bdkSec', 'bskElo', 'bskFra',
#                      'bskCom', 'bskCom2', 'bskCon', 'bskShI', 'bskERI', 'btkOri']]
# clustering[['pivFAR', 'bID', 'bdkAre', 'bdkPer', 'bdkSec', 'bskElo', 'bskFra']].isnull().sum().sum()
# clustering.isnull().sum().sum()
# nan_rows = clustering[clustering['bskFra'].isnull()]
# nan_rows
# # make clusterer
# clusterer = hdbscan.HDBSCAN(min_cluster_size=3)
#
# # cluster data
# clusterer.fit(clustering)
# # get number of clusters
# clusterer.labels_.max()
# # save cluster labels to geoDataFrame
# values['cluster'] = clusterer.labels_
# values['cluster']
# togis = values[['uID', 'cluster']]
# togis.to_csv('/Users/martin/Strathcloud/Personal Folders/Test data/Prague/Prague/CSV/prg_clustered.csv')
