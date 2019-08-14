#!/usr/bin/env python
# -*- coding: utf-8 -*-

# diversity.py
# definitons of diversity characters

from tqdm import tqdm  # progress bar
import pandas as pd
import scipy as sp
import numpy as np


def rng(gdf, values, spatial_weights, unique_id, rng=(0, 100), **kwargs):
    """
    Calculates the range of values within neighbours defined in `spatial_weights`.

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
    Dibble J, Prelorendjos A, Romice O, et al. (2017) On the origin of spaces: Morphometric foundations of urban form evolution.
    Environment and Planning B: Urban Analytics and City Science 46(4): 707–730.

    Examples
    --------
    >>> sw = momepy.sw_high(k=3, gdf=tessellation_df, ids='uID')
    >>> tessellation_df['area_IQR_3steps'] = mm.rng(tessellation_df, 'area', sw, 'uID', rng=(25, 75))
    Calculating range...
    100%|██████████| 144/144 [00:00<00:00, 722.50it/s]
    Range calculated.


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
        neighbours = spatial_weights.neighbors[row[unique_id]].copy()
        if neighbours:
            neighbours.append(row[unique_id])
        else:
            neighbours = [row[unique_id]]
        values_list = gdf.loc[gdf[unique_id].isin(neighbours)][values]

        results_list.append(sp.stats.iqr(values_list, rng=rng, **kwargs))

    series = pd.Series(results_list, index=gdf.index)

    print('Range calculated.')
    return series


def theil(gdf, values, spatial_weights, unique_id, rng=None):
    """
    Calculates the Theil measure of inequality of values within neighbours defined in `spatial_weights`.

    Uses `pysal.explore.inequality.theil.Theil` under the hood. Requires 'inequality' or 'pysal' package.

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

    Examples
    --------
    >>> sw = momepy.sw_high(k=3, gdf=tessellation_df, ids='uID')
    >>> tessellation_df['area_Theil'] = mm.theil(tessellation_df, 'area', sw, 'uID')
    Calculating Theil index...
    100%|██████████| 144/144 [00:00<00:00, 597.37it/s]
    Theil index calculated.
    """
    try:
        from inequality.theil import Theil
    except ImportError:
        try:
            from pysal.explore.inequality.theil import Theil
        except ImportError:
            raise ImportError(
                "The 'inequality' or 'pysal' package is required.")

    # define empty list for results
    results_list = []
    gdf = gdf.copy()

    print('Calculating Theil index...')

    if values is not None:
        if not isinstance(values, str):
            gdf['mm_v'] = values
            values = 'mm_v'

    for index, row in tqdm(gdf.iterrows(), total=gdf.shape[0]):
        neighbours = spatial_weights.neighbors[row[unique_id]].copy()
        if neighbours:
            neighbours.append(row[unique_id])
        else:
            neighbours = [row[unique_id]]
        values_list = gdf.loc[gdf[unique_id].isin(neighbours)][values]

        if rng:
            from momepy import limit_range
            values_list = limit_range(values_list.tolist(), rng=rng)
        results_list.append(Theil(values_list).T)

    series = pd.Series(results_list, index=gdf.index)

    print('Theil index calculated.')
    return series


def simpson(gdf, values, spatial_weights, unique_id, binning='HeadTailBreaks', **classification_kwds):
    """
    Calculates the Simpson\'s diversity index of values within neighbours defined in `spatial_weights`.

    Uses `mapclassify.classifiers` under the hood for binning. Requires `mapclassify>=.2.1.0` dependency
    or `pysal`.

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
    Feliciotti A (2018) RESILIENCE AND URBAN DESIGN:A SYSTEMS APPROACH TO THE STUDY OF RESILIENCE
    IN URBAN FORM. LEARNING FROM THE CASE OF GORBALS. Glasgow.

    Examples
    --------
    >>> sw = momepy.sw_high(k=3, gdf=tessellation_df, ids='uID')
    >>> tessellation_df['area_Simpson'] = mm.simpson(tessellation_df, 'area', sw, 'uID')
    Calculating Simpson's diversity index...
    100%|██████████| 144/144 [00:00<00:00, 455.83it/s]
    Simpson's diversity index calculated.
    """
    def _simpson_di(data):

        """ Given a hash { 'species': count } , returns the Simpson Diversity Index

        >>> simpson_di({'a': 10, 'b': 20, 'c': 30,})
        0.3888888888888889

        https://gist.github.com/martinjc/f227b447791df8c90568
        """

        def p(n, N):
            """ Relative abundance """
            if n == 0:
                return 0
            return float(n) / N

        N = sum(data.values())

        return sum(p(n, N)**2 for n in data.values() if n != 0)

    try:
        import mapclassify.classifiers as classifiers
    except ImportError:
        try:
            import pysal.viz.mapclassify.classifiers as classifiers
        except ImportError:
            raise ImportError("The 'mapclassify' or 'pysal' package is required")

    schemes = {}
    for classifier in classifiers.CLASSIFIERS:
        schemes[classifier.lower()] = getattr(classifiers,
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
        neighbours = spatial_weights.neighbors[row[unique_id]].copy()
        if neighbours:
            neighbours.append(row[unique_id])
        else:
            neighbours = [row[unique_id]]
        values_list = gdf.loc[gdf[unique_id].isin(neighbours)][values]

        sample_bins = classifiers.UserDefined(values_list, bins)
        counts = dict(zip(bins, sample_bins.counts))
        results_list.append(_simpson_di(counts))

    series = pd.Series(results_list, index=gdf.index)

    print("Simpson's diversity index calculated.")
    return series


def gini(gdf, values, spatial_weights, unique_id, rng=None):
    """
    Calculates the Gini index of values within neighbours defined in `spatial_weights`.

    Uses `inequality.gini.Gini` under the hood. Requires 'inequality' or 'pysal' package.

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

    Examples
    --------
    >>> sw = momepy.sw_high(k=3, gdf=tessellation_df, ids='uID')
    >>> tessellation_df['area_Gini'] = mm.gini(tessellation_df, 'area', sw, 'uID')
    Calculating Gini index...
    100%|██████████| 144/144 [00:00<00:00, 597.37it/s]
    Gini index calculated.
    """
    try:
        from inequality.gini import Gini
    except ImportError:
        try:
            from pysal.explore.inequality.gini import Gini
        except ImportError:
            raise ImportError(
                "The 'inequality' or 'pysal' package is required.")
    # define empty list for results
    results_list = []
    gdf = gdf.copy()
    if gdf[values].min() < 0:
        raise ValueError("Values contain negative numbers. Normalise data before"
                         "using momepy.gini.")
    print('Calculating Gini index...')

    for index, row in tqdm(gdf.iterrows(), total=gdf.shape[0]):
        neighbours = spatial_weights.neighbors[row[unique_id]].copy()
        if neighbours:
            neighbours.append(row[unique_id])

            values_list = gdf.loc[gdf[unique_id].isin(neighbours)][values].values

            if rng:
                from momepy import limit_range
                values_list = np.array(limit_range(values_list, rng=rng))
            results_list.append(Gini(values_list).g)
        else:
            results_list.append(0)
    series = pd.Series(results_list, index=gdf.index)

    print('Gini index calculated.')
    return series
