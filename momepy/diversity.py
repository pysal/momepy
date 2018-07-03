# diversity.py
# definitons of diversity characters

from tqdm import tqdm  # progress bar
import numpy as np
from .intensity import radius
import powerlaw


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

    print('Calculating Gini index.')

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

# rows
# valu = objects[[source]]
# valulist = valu[source].tolist()
# valulist
# results = powerlaw.Fit(valulist)
# print(results.power_law.alpha)
# print(results.power_law.xmin)
# R, p = results.distribution_compare('power_law', 'lognormal')
# R
# p
# np.seterr(divide='ignore', invalid='ignore')
# plt.show()
# import matplotlib.pyplot as plt
#
# print(R, p)
# plt.show()


def power_law(objects, look_for, source, id_column='uID'):
    # define new column

    print('Calculating Power law.')

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
