#!/usr/bin/env python
# -*- coding: utf-8 -*-

# diversity.py
# definitons of diversity characters

import numpy as np
import pandas as pd
import scipy as sp
from tqdm import tqdm  # progress bar

__all__ = ["Range", "Theil", "Simpson", "Gini"]


class Range:
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

    Attributes
    ----------
    r : Series
        Series containing resulting values
    gdf : GeoDataFrame
        original GeoDataFrame
    values : Series
        Series containing used values
    sw : libpysal.weights
        spatial weights matrix
    id : Series
        Series containing used unique ID
    rng : tuple
        range
    kwargs : dict
        kwargs

    References
    ----------
    Dibble J, Prelorendjos A, Romice O, et al. (2017) On the origin of spaces: Morphometric foundations of urban form evolution.
    Environment and Planning B: Urban Analytics and City Science 46(4): 707–730.

    Examples
    --------
    >>> sw = momepy.sw_high(k=3, gdf=tessellation_df, ids='uID')
    >>> tessellation_df['area_IQR_3steps'] = mm.Range(tessellation_df, 'area', sw, 'uID', rng=(25, 75)).r
    100%|██████████| 144/144 [00:00<00:00, 722.50it/s]


    """

    def __init__(self, gdf, values, spatial_weights, unique_id, rng=(0, 100), **kwargs):
        self.gdf = gdf
        self.sw = spatial_weights
        self.id = gdf[unique_id]
        self.rng = rng
        self.kwargs = kwargs

        data = gdf.copy()
        if values is not None:
            if not isinstance(values, str):
                data["mm_v"] = values
                values = "mm_v"
        self.values = data[values]

        data = data.set_index(unique_id)

        results_list = []
        for index, row in tqdm(data.iterrows(), total=data.shape[0]):
            neighbours = spatial_weights.neighbors[index].copy()
            if neighbours:
                neighbours.append(index)
            else:
                neighbours = [index]

            values_list = data.loc[neighbours][values]
            results_list.append(sp.stats.iqr(values_list, rng=rng, **kwargs))

        self.r = pd.Series(results_list, index=gdf.index)


class Theil:
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

    Attributes
    ----------
    t : Series
        Series containing resulting values
    gdf : GeoDataFrame
        original GeoDataFrame
    values : Series
        Series containing used values
    sw : libpysal.weights
        spatial weights matrix
    id : Series
        Series containing used unique ID
    rng : tuple, optional
        range

    Examples
    --------
    >>> sw = momepy.sw_high(k=3, gdf=tessellation_df, ids='uID')
    >>> tessellation_df['area_Theil'] = mm.Theil(tessellation_df, 'area', sw, 'uID').t
    100%|██████████| 144/144 [00:00<00:00, 597.37it/s]
    """

    def __init__(self, gdf, values, spatial_weights, unique_id, rng=None):
        try:
            from inequality.theil import Theil
        except ImportError:
            try:
                from pysal.explore.inequality.theil import Theil
            except ImportError:
                raise ImportError("The 'inequality' or 'pysal' package is required.")

        self.gdf = gdf
        self.sw = spatial_weights
        self.id = gdf[unique_id]
        self.rng = rng

        data = gdf.copy()
        if values is not None:
            if not isinstance(values, str):
                data["mm_v"] = values
                values = "mm_v"
        self.values = data[values]

        data = data.set_index(unique_id)

        results_list = []
        for index, row in tqdm(data.iterrows(), total=data.shape[0]):
            neighbours = spatial_weights.neighbors[index].copy()
            if neighbours:
                neighbours.append(index)
            else:
                neighbours = [index]

            values_list = data.loc[neighbours][values]

            if rng:
                from momepy import limit_range

                values_list = limit_range(values_list, rng=rng)
            results_list.append(Theil(values_list).T)

        self.t = pd.Series(results_list, index=gdf.index)


class Simpson:
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
        https://pysal.org/mapclassify

    Attributes
    ----------
    s : Series
        Series containing resulting values
    gdf : GeoDataFrame
        original GeoDataFrame
    values : Series
        Series containing used values
    sw : libpysal.weights
        spatial weights matrix
    id : Series
        Series containing used unique ID
    binning : str
        binning
    classification_kwds : dict
        classification_kwds

    References
    ----------
    Feliciotti A (2018) RESILIENCE AND URBAN DESIGN:A SYSTEMS APPROACH TO THE STUDY OF RESILIENCE
    IN URBAN FORM. LEARNING FROM THE CASE OF GORBALS. Glasgow.

    Examples
    --------
    >>> sw = momepy.sw_high(k=3, gdf=tessellation_df, ids='uID')
    >>> tessellation_df['area_Simpson'] = mm.Simpson(tessellation_df, 'area', sw, 'uID').s
    100%|██████████| 144/144 [00:00<00:00, 455.83it/s]
    """

    def __init__(
        self,
        gdf,
        values,
        spatial_weights,
        unique_id,
        binning="HeadTailBreaks",
        **classification_kwds
    ):
        try:
            import mapclassify.classifiers as classifiers
        except ImportError:
            try:
                import pysal.viz.mapclassify.classifiers as classifiers
            except ImportError:
                raise ImportError("The 'mapclassify' or 'pysal' package is required")

        schemes = {}
        for classifier in classifiers.CLASSIFIERS:
            schemes[classifier.lower()] = getattr(classifiers, classifier)
        binning = binning.lower()
        if binning not in schemes:
            raise ValueError(
                "Invalid binning. Binning must be in the" " set: %r" % schemes.keys()
            )

        self.gdf = gdf
        self.sw = spatial_weights
        self.id = gdf[unique_id]
        self.binning = binning
        self.classification_kwds = classification_kwds

        data = gdf.copy()
        if values is not None:
            if not isinstance(values, str):
                data["mm_v"] = values
                values = "mm_v"
        self.values = data[values]

        bins = schemes[binning](data[values], **classification_kwds).bins
        data = data.set_index(unique_id)
        results_list = []
        for index, row in tqdm(data.iterrows(), total=data.shape[0]):
            neighbours = spatial_weights.neighbors[index].copy()
            if neighbours:
                neighbours.append(index)
            else:
                neighbours = [index]
            values_list = data.loc[neighbours][values]

            sample_bins = classifiers.UserDefined(values_list, bins)
            counts = dict(zip(bins, sample_bins.counts))
            results_list.append(self._simpson_di(counts))

        self.s = pd.Series(results_list, index=gdf.index)

    def _simpson_di(self, data):

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

        return sum(p(n, N) ** 2 for n in data.values() if n != 0)


class Gini:
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

    Attributes
    ----------
    g : Series
        Series containing resulting values
    gdf : GeoDataFrame
        original GeoDataFrame
    values : Series
        Series containing used values
    sw : libpysal.weights
        spatial weights matrix
    id : Series
        Series containing used unique ID
    rng : tuple
        range

    Examples
    --------
    >>> sw = momepy.sw_high(k=3, gdf=tessellation_df, ids='uID')
    >>> tessellation_df['area_Gini'] = mm.Gini(tessellation_df, 'area', sw, 'uID').g
    100%|██████████| 144/144 [00:00<00:00, 597.37it/s]
    """

    def __init__(self, gdf, values, spatial_weights, unique_id, rng=None):
        try:
            from inequality.gini import Gini
        except ImportError:
            try:
                from pysal.explore.inequality.gini import Gini
            except ImportError:
                raise ImportError("The 'inequality' or 'pysal' package is required.")

        self.gdf = gdf
        self.sw = spatial_weights
        self.id = gdf[unique_id]
        self.rng = rng

        data = gdf.copy()
        if values is not None:
            if not isinstance(values, str):
                data["mm_v"] = values
                values = "mm_v"
        self.values = data[values]

        if self.values.min() < 0:
            raise ValueError(
                "Values contain negative numbers. Normalise data before"
                "using momepy.Gini."
            )

        data = data.set_index(unique_id)

        results_list = []
        for index, row in tqdm(data.iterrows(), total=data.shape[0]):
            neighbours = spatial_weights.neighbors[index].copy()
            if neighbours:
                neighbours.append(index)

                values_list = data.loc[neighbours][values].values

                if rng:
                    from momepy import limit_range

                    values_list = np.array(limit_range(values_list, rng=rng))
                results_list.append(Gini(values_list).g)
            else:
                results_list.append(0)

        self.g = pd.Series(results_list, index=gdf.index)
