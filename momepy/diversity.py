#!/usr/bin/env python
# -*- coding: utf-8 -*-

# diversity.py
# definitons of diversity characters

from tqdm import tqdm  # progress bar
import numpy as np
import pandas as pd
import scipy as sp


def rng(objects, values, spatial_weights=None, order=None, rng=(0, 100), **kwargs):
    """
    Calculates the range of values within k steps of morphological tessellation

    Uses `scipy.stats.iqr` under the hood.

    .. math::


    Parameters
    ----------
    objects : GeoDataFrame
        GeoDataFrame containing morphological tessellation
    values : str, list, np.array, pd.Series
        the name of the dataframe column, np.array, or pd.Series where is stored character value.
    spatial_weights : libpysal.weights, optional
        spatial weights matrix - If None, Queen contiguity matrix of set order will be calculated
        based on objects.
    order : int
        order of Queen contiguity
    rng : Two-element sequence containing floats in range of [0,100], optional
        Percentiles over which to compute the range. Each must be
        between 0 and 100, inclusive. The order of the elements is not important.
    **kwargs : keyword arguments
        optional arguments for `scipy.stats.iqr`

    Returns
    -------
    Series
        Series containing resulting values.

    References
    ----------

    Examples
    --------

    """
    # define empty list for results
    results_list = []

    print('Calculating range...')

    if spatial_weights is None:
        print('Generating weights matrix (Queen) of {} topological steps...'.format(order))
        from momepy import Queen_higher
        # matrix to define area of analysis (more steps)
        spatial_weights = Queen_higher(k=order, geodataframe=objects)
    else:
        if not all(objects.index == range(len(objects))):
            raise ValueError('Index is not consecutive range 0:x, spatial weights will not match objects.')

    if values is not None:
        if not isinstance(values, str):
            objects['mm_v'] = values
            values = 'mm_v'

    for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
        neighbours = spatial_weights.neighbors[index]
        values_list = objects.iloc[neighbours][values].tolist()
        if values_list:
            values_list.append(row[values])
        else:
            values_list = [row[values]]

        results_list.append(sp.stats.iqr(values_list, rng=rng, **kwargs))

    series = pd.Series(results_list)

    if 'mm_v' in objects.columns:
        objects.drop(columns=['mm_v'], inplace=True)

    print('Range calculated.')
    return series


def theil(objects, values, spatial_weights=None, order=None, rng=(0, 100)):
    """
    Calculates the Theil measure of inequality of values within k steps of morphological tessellation

    Uses `pysal.explore.inequality.theil.Theil` under the hood.

    .. math::


    Parameters
    ----------
    objects : GeoDataFrame
        GeoDataFrame containing morphological tessellation
    values : str, list, np.array, pd.Series
        the name of the dataframe column, np.array, or pd.Series where is stored character value.
    spatial_weights : libpysal.weights, optional
        spatial weights matrix - If None, Queen contiguity matrix of set order will be calculated
        based on objects.
    order : int
        order of Queen contiguity
    rng : Two-element sequence containing floats in range of [0,100], optional
        Percentiles over which to compute the range. Each must be
        between 0 and 100, inclusive. The order of the elements is not important.

    Returns
    -------
    Series
        Series containing resulting values.

    References
    ----------

    Examples
    --------

    """
    from pysal.explore.inequality.theil import Theil
    # define empty list for results
    results_list = []

    print('Calculating Theil index...')

    if spatial_weights is None:
        print('Generating weights matrix (Queen) of {} topological steps...'.format(order))
        from momepy import Queen_higher
        # matrix to define area of analysis (more steps)
        spatial_weights = Queen_higher(k=order, geodataframe=objects)
    else:
        if not all(objects.index == range(len(objects))):
            raise ValueError('Index is not consecutive range 0:x, spatial weights will not match objects.')

    if values is not None:
        if not isinstance(values, str):
            objects['mm_v'] = values
            values = 'mm_v'

    for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
        neighbours = spatial_weights.neighbors[index]
        values_list = objects.iloc[neighbours][values].tolist()
        if values_list:
            values_list.append(row[values])
        else:
            values_list = [row[values]]

        if rng:
            from momepy import limit_range
            values_list = limit_range(values_list, rng=rng)
        results_list.append(Theil(values_list).T)

    series = pd.Series(results_list)

    if 'mm_v' in objects.columns:
        objects.drop(columns=['mm_v'], inplace=True)

    print('Theil index calculated.')
    return series


def simpson(objects, values, spatial_weights=None, order=None, binning='HeadTail_Breaks', **classification_kwds):
    """
    Calculates the Simpson\'s diversity index of values within k steps of morphological tessellation

    Uses `mapclassify.classifiers` under the hood for binning.

    .. math::


    Parameters
    ----------
    objects : GeoDataFrame
        GeoDataFrame containing morphological tessellation
    values : str, list, np.array, pd.Series
        the name of the dataframe column, np.array, or pd.Series where is stored character value.
    spatial_weights : libpysal.weights, optional
        spatial weights matrix - If None, Queen contiguity matrix of set order will be calculated
        based on objects.
    order : int
        order of Queen contiguity
    binning : str
        One of mapclassify classification schemes
        Options are Box_Plot, Equal_Interval, Fisher_Jenks,
        Fisher_Jenks_Sampled, HeadTail_Breaks, Jenks_Caspall,
        Jenks_Caspall_Forced, Jenks_Caspall_Sampled, Max_P_Classifier,
        Maximum_Breaks, Natural_Breaks, Quantiles, Percentiles, Std_Mean,
        User_Defined
    **classification_kwds : dict
        Keyword arguments for classification scheme
        For details see mapclassify documentation:
        https://mapclassify.readthedocs.io/en/latest/api.html

    Returns
    -------
    Series
        Series containing resulting values.

    References
    ----------

    Examples
    --------

    """
    def simpson_di(data):

        """ Given a hash { 'species': count } , returns the Simpson Diversity Index

        >>> simpson_di({'a': 10, 'b': 20, 'c': 30,})
        0.3888888888888889

        https://gist.github.com/martinjc/f227b447791df8c90568
        """

        def p(n, N):
            """ Relative abundance """
            if n == 0:
                return 0
            else:
                return float(n) / N

        N = sum(data.values())

        return sum(p(n, N)**2 for n in data.values() if n != 0)

    try:
        import mapclassify.classifiers
    except ImportError:
        raise ImportError(
            "The 'mapclassify' package is required to use the 'scheme' "
            "keyword")

    schemes = {}
    for classifier in mapclassify.classifiers.CLASSIFIERS:
        schemes[classifier.lower()] = getattr(mapclassify.classifiers,
                                              classifier)
    binning = binning.lower()
    if binning not in schemes:
        raise ValueError("Invalid binning. Binning must be in the"
                         " set: %r" % schemes.keys())

    # define empty list for results
    results_list = []

    print('Calculating Simpson\'s diversity index...')

    if spatial_weights is None:
        print('Generating weights matrix (Queen) of {} topological steps...'.format(order))
        from momepy import Queen_higher
        # matrix to define area of analysis (more steps)
        spatial_weights = Queen_higher(k=order, geodataframe=objects)
    else:
        if not all(objects.index == range(len(objects))):
            raise ValueError('Index is not consecutive range 0:x, spatial weights will not match objects.')

    if values is not None:
        if not isinstance(values, str):
            objects['mm_v'] = values
            values = 'mm_v'

    bins = schemes[binning](objects[values], **classification_kwds).bins

    for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
        neighbours = spatial_weights.neighbors[index]
        values_list = objects.iloc[neighbours][values].tolist()
        if values_list:
            values_list.append(row[values])
        else:
            values_list = [row[values]]

        sample_bins = mapclassify.classifiers.User_Defined(values_list, bins)
        counts = dict(zip(bins, sample_bins.counts))
        results_list.append(simpson_di(counts))

    series = pd.Series(results_list)

    if 'mm_v' in objects.columns:
        objects.drop(columns=['mm_v'], inplace=True)

    print('Simpson\'s diversity index calculated.')
    return series
