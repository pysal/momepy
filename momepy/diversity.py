#!/usr/bin/env python
# -*- coding: utf-8 -*-

# diversity.py
# definitons of diversity characters

from tqdm import tqdm  # progress bar
import pandas as pd
import scipy as sp


def rng(gdf, values, spatial_weights, unique_id, rng=(0, 100), **kwargs):
    """
    Calculates the range of values within k steps of morphological tessellation

    Uses `scipy.stats.iqr` under the hood.

    .. math::


    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing morphological tessellation
    values : str, list, np.array, pd.Series
        the name of the dataframe column, np.array, or pd.Series where is stored character value.
    spatial_weights : libpysal.weights
        spatial weights matrix
    unique_id : str
        name of the column with unique id used as spatial_weights index
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
    gdf = gdf.copy()
    print('Calculating range...')

    if values is not None:
        if not isinstance(values, str):
            gdf['mm_v'] = values
            values = 'mm_v'

    for index, row in tqdm(gdf.iterrows(), total=gdf.shape[0]):
        neighbours = spatial_weights.neighbors[row[unique_id]]
        if neighbours:
            neighbours.append(row[unique_id])
        else:
            neighbours = row[unique_id]
        values_list = gdf.loc[gdf[unique_id].isin(neighbours)][values]

        results_list.append(sp.stats.iqr(values_list, rng=rng, **kwargs))

    series = pd.Series(results_list)

    print('Range calculated.')
    return series


def theil(gdf, values, spatial_weights, unique_id, rng=(0, 100)):
    """
    Calculates the Theil measure of inequality of values within k steps of morphological tessellation

    Uses `pysal.explore.inequality.theil.Theil` under the hood.

    .. math::


    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing morphological tessellation
    values : str, list, np.array, pd.Series
        the name of the dataframe column, np.array, or pd.Series where is stored character value.
    spatial_weights : libpysal.weights
        spatial weights matrix
    unique_id : str
        name of the column with unique id used as spatial_weights index
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
    gdf = gdf.copy()

    print('Calculating Theil index...')

    if values is not None:
        if not isinstance(values, str):
            gdf['mm_v'] = values
            values = 'mm_v'

    for index, row in tqdm(gdf.iterrows(), total=gdf.shape[0]):
        neighbours = spatial_weights.neighbors[row[unique_id]]
        if neighbours:
            neighbours.append(row[unique_id])
        else:
            neighbours = row[unique_id]
        values_list = gdf.loc[gdf[unique_id].isin(neighbours)][values]

        if rng:
            from momepy import limit_range
            values_list = limit_range(values_list.tolist(), rng=rng)
        results_list.append(Theil(values_list).T)

    series = pd.Series(results_list)

    print('Theil index calculated.')
    return series


def simpson(gdf, values, spatial_weights, unique_id, binning='HeadTailBreaks', **classification_kwds):
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
        Options are BoxPlot, EqualInterval, FisherJenks,
        FisherJenksSampled, HeadTailBreaks, JenksCaspall,
        JenksCaspallForced, JenksCaspallSampled, MaxPClassifier,
        MaximumBreaks, NaturalBreaks, Quantiles, Percentiles, StdMean,
        UserDefined
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
            "The 'mapclassify' package is required.")

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
    gdf = gdf.copy()
    print('Calculating Simpson\'s diversity index...')

    if values is not None:
        if not isinstance(values, str):
            gdf['mm_v'] = values
            values = 'mm_v'

    bins = schemes[binning](gdf[values], **classification_kwds).bins

    for index, row in tqdm(gdf.iterrows(), total=gdf.shape[0]):
        neighbours = spatial_weights.neighbors[row[unique_id]]
        if neighbours:
            neighbours.append(row[unique_id])
        else:
            neighbours = row[unique_id]
        values_list = gdf.loc[gdf[unique_id].isin(neighbours)][values]

        sample_bins = mapclassify.classifiers.User_Defined(values_list, bins)
        counts = dict(zip(bins, sample_bins.counts))
        results_list.append(simpson_di(counts))

    series = pd.Series(results_list)

    print("Simpson's diversity index calculated.")
    return series
