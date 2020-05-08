#!/usr/bin/env python
# -*- coding: utf-8 -*-

# diversity.py
# definitions of diversity characters

import numpy as np
import pandas as pd
import scipy as sp
from tqdm import tqdm  # progress bar

__all__ = ["Range", "Theil", "Simpson", "Gini", "Shannon", "Unique"]


class Range:
    """
    Calculates the range of values within neighbours defined in ``spatial_weights``.

    Uses ``scipy.stats.iqr`` under the hood.

    Adapted from :cite:`dibble2017`.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing morphological tessellation
    values : str, list, np.array, pd.Series
        the name of the dataframe column, ``np.array``, or ``pd.Series`` where is stored character value.
    spatial_weights : libpysal.weights
        spatial weights matrix
    unique_id : str
        name of the column with unique id used as ``spatial_weights`` index
    rng : Two-element sequence containing floats in range of [0,100], optional
        Percentiles over which to compute the range. Each must be
        between 0 and 100, inclusive. The order of the elements is not important.
    **kwargs : keyword arguments
        optional arguments for ``scipy.stats.iqr``

    Attributes
    ----------
    series : Series
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

    Examples
    --------
    >>> sw = momepy.sw_high(k=3, gdf=tessellation_df, ids='uID')
    >>> tessellation_df['area_IQR_3steps'] = mm.Range(tessellation_df, 'area', sw, 'uID', rng=(25, 75)).series
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

        data = data.set_index(unique_id)[values]

        results_list = []
        for index in tqdm(data.index, total=data.shape[0]):
            if index in spatial_weights.neighbors.keys():
                neighbours = spatial_weights.neighbors[index].copy()
                if neighbours:
                    neighbours.append(index)
                else:
                    neighbours = [index]

                values_list = data.loc[neighbours]
                results_list.append(sp.stats.iqr(values_list, rng=rng, **kwargs))
            else:
                results_list.append(np.nan)

        self.series = pd.Series(results_list, index=gdf.index)


class Theil:
    """
    Calculates the Theil measure of inequality of values within neighbours defined in ``spatial_weights``.

    Uses ``inequality.theil.Theil`` under the hood. Requires '`inequality`' package.

    .. math::

        T = \sum_{i=1}^n \left( \\frac{y_i}{\sum_{i=1}^n y_i} \ln \left[ N \\frac{y_i}{\sum_{i=1}^n y_i}\\right] \\right)

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing morphological tessellation
    values : str, list, np.array, pd.Series
        the name of the dataframe column, ``np.array``, or ``pd.Series`` where is stored character value.
    spatial_weights : libpysal.weights
        spatial weights matrix
    unique_id : str
        name of the column with unique id used as ``spatial_weights`` index
    rng : Two-element sequence containing floats in range of [0,100], optional
        Percentiles over which to compute the range. Each must be
        between 0 and 100, inclusive. The order of the elements is not important.

    Attributes
    ----------
    series : Series
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
    >>> tessellation_df['area_Theil'] = mm.Theil(tessellation_df, 'area', sw, 'uID').series
    100%|██████████| 144/144 [00:00<00:00, 597.37it/s]
    """

    def __init__(self, gdf, values, spatial_weights, unique_id, rng=None):
        try:
            from inequality.theil import Theil
        except ImportError:
            raise ImportError("The 'inequality' package is required.")

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

        data = data.set_index(unique_id)[values]

        if rng:
            from momepy import limit_range

        results_list = []
        for index in tqdm(data.index, total=data.shape[0]):
            if index in spatial_weights.neighbors.keys():
                neighbours = spatial_weights.neighbors[index].copy()
                if neighbours:
                    neighbours.append(index)
                else:
                    neighbours = [index]

                values_list = data.loc[neighbours]

                if rng:
                    values_list = limit_range(values_list, rng=rng)
                results_list.append(Theil(values_list).T)
            else:
                results_list.append(np.nan)

        self.series = pd.Series(results_list, index=gdf.index)


class Simpson:
    """
    Calculates the Simpson\'s diversity index of values within neighbours defined in ``spatial_weights``.

    Uses ``mapclassify.classifiers`` under the hood for binning. Requires ``mapclassify>=.2.1.0`` dependency.

    .. math::

        \\lambda=\\sum_{i=1}^{R} p_{i}^{2}

    Adapted from :cite:`feliciotti2018`.

    Parameters
    ----------
    objects : GeoDataFrame
        GeoDataFrame containing morphological tessellation
    values : str, list, np.array, pd.Series
        the name of the dataframe column, ``np.array``, or ``pd.Series`` where is stored character value.
    spatial_weights : libpysal.weights, optional
        spatial weights matrix - If None, Queen contiguity matrix of set order will be calculated
        based on objects.
    order : int
        order of Queen contiguity
    binning : str
        One of mapclassify classification schemes
        Options are ``BoxPlot``, ``EqualInterval``, ``FisherJenks``,
        ``FisherJenksSampled``, ``HeadTailBreaks``, ``JenksCaspall``,
        ``JenksCaspallForced``, ``JenksCaspallSampled``, ``MaxPClassifier``,
        ``MaximumBreaks``, ``NaturalBreaks``, ``Quantiles``, ``Percentiles``, ``StdMean``,
        ``UserDefined``
    gini_simpson : bool (default False)
        return Gini-Simpson index instead of Simpson index (``1 - λ``)
    inverse : bool (default False)
        return Inverse Simpson index instead of Simpson index (``1 / λ``)
    categorical : bool (default False)
        treat values as categories (will not use ``binning``)
    categories : list-like (default None)
        list of categories. If None ``values.unique()`` is used.
    **classification_kwds : dict
        Keyword arguments for classification scheme
        For details see `mapclassify documentation <https://pysal.org/mapclassify>`_.

    Attributes
    ----------
    series : Series
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
        binning method
    bins : mapclassify.classifiers.Classifier
        generated bins
    classification_kwds : dict
        classification_kwds

    Examples
    --------
    >>> sw = momepy.sw_high(k=3, gdf=tessellation_df, ids='uID')
    >>> tessellation_df['area_Simpson'] = mm.Simpson(tessellation_df, 'area', sw, 'uID').series
    100%|██████████| 144/144 [00:00<00:00, 455.83it/s]
    """

    def __init__(
        self,
        gdf,
        values,
        spatial_weights,
        unique_id,
        binning="HeadTailBreaks",
        gini_simpson=False,
        inverse=False,
        categorical=False,
        categories=None,
        **classification_kwds
    ):
        if not categorical:
            try:
                import mapclassify.classifiers as classifiers
            except ImportError:
                raise ImportError("The 'mapclassify' package is required")

            schemes = {}
            for classifier in classifiers.CLASSIFIERS:
                schemes[classifier.lower()] = getattr(classifiers, classifier)
            binning = binning.lower()
            if binning not in schemes:
                raise ValueError(
                    "Invalid binning. Binning must be in the"
                    " set: %r" % schemes.keys()
                )

        self.gdf = gdf
        self.sw = spatial_weights
        self.id = gdf[unique_id]
        self.binning = binning
        self.gini_simpson = gini_simpson
        self.inverse = inverse
        self.categorical = categorical
        self.categories = categories
        self.classification_kwds = classification_kwds

        data = gdf.copy()
        if values is not None:
            if not isinstance(values, str):
                data["mm_v"] = values
                values = "mm_v"
        self.values = data[values]

        if not categorical:
            self.bins = schemes[binning](data[values], **classification_kwds).bins

        data = data.set_index(unique_id)[values]

        if not categories:
            categories = data.unique()

        results_list = []
        for index in tqdm(data.index, total=data.shape[0]):
            if index in spatial_weights.neighbors.keys():
                neighbours = spatial_weights.neighbors[index].copy()
                if neighbours:
                    neighbours.append(index)
                else:
                    neighbours = [index]
                values_list = data.loc[neighbours]

                if categorical:
                    counts = values_list.value_counts().to_dict()
                    for c in categories:
                        if c not in counts.keys():
                            counts[c] = 0
                else:
                    sample_bins = classifiers.UserDefined(values_list, self.bins)
                    counts = dict(zip(self.bins, sample_bins.counts))

                results_list.append(self._simpson_di(counts))
            else:
                results_list.append(np.nan)

        if gini_simpson:
            self.series = 1 - pd.Series(results_list, index=gdf.index)
        elif inverse:
            self.series = 1 / pd.Series(results_list, index=gdf.index)
        else:
            self.series = pd.Series(results_list, index=gdf.index)

    def _simpson_di(self, data):

        """ Given a hash { 'species': count } , returns the Simpson Diversity Index

        >>> _simpson_di({'a': 10, 'b': 20, 'c': 30,})
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
    Calculates the Gini index of values within neighbours defined in ``spatial_weights``.

    Uses ``inequality.gini.Gini`` under the hood. Requires '`inequality`' package.

    .. math::

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing morphological tessellation
    values : str, list, np.array, pd.Series
        the name of the dataframe column, ``np.array``, or ``pd.Series`` where is stored character value.
    spatial_weights : libpysal.weights
        spatial weights matrix
    unique_id : str
        name of the column with unique id used as ``spatial_weights`` index
    rng : Two-element sequence containing floats in range of [0,100], optional
        Percentiles over which to compute the range. Each must be
        between 0 and 100, inclusive. The order of the elements is not important.

    Attributes
    ----------
    series : Series
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
    >>> tessellation_df['area_Gini'] = mm.Gini(tessellation_df, 'area', sw, 'uID').series
    100%|██████████| 144/144 [00:00<00:00, 597.37it/s]
    """

    def __init__(self, gdf, values, spatial_weights, unique_id, rng=None):
        try:
            from inequality.gini import Gini
        except ImportError:
            raise ImportError("The 'inequality' package is required.")

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

        data = data.set_index(unique_id)[values]

        if rng:
            from momepy import limit_range

        results_list = []
        for index in tqdm(data.index, total=data.shape[0]):
            if index in spatial_weights.neighbors.keys():
                neighbours = spatial_weights.neighbors[index].copy()
                if neighbours:
                    neighbours.append(index)

                    values_list = data.loc[neighbours].values

                    if rng:
                        values_list = limit_range(values_list, rng=rng)
                    results_list.append(Gini(values_list).g)
                else:
                    results_list.append(0)
            else:
                results_list.append(np.nan)

        self.series = pd.Series(results_list, index=gdf.index)


class Shannon:
    """
    Calculates the Shannon index of values within neighbours defined in ``spatial_weights``.

    Uses ``mapclassify.classifiers`` under the hood for binning. Requires ``mapclassify>=.2.1.0`` dependency.

    .. math::

        H^{\\prime}=-\\sum_{i=1}^{R} p_{i} \\ln p_{i}

    Parameters
    ----------
    objects : GeoDataFrame
        GeoDataFrame containing morphological tessellation
    values : str, list, np.array, pd.Series
        the name of the dataframe column, ``np.array``, or ``pd.Series`` where is stored character value.
    spatial_weights : libpysal.weights, optional
        spatial weights matrix - If None, Queen contiguity matrix of set order will be calculated
        based on objects.
    order : int
        order of Queen contiguity
    binning : str
        One of mapclassify classification schemes
        Options are ``BoxPlot``, ``EqualInterval``, ``FisherJenks``,
        ``FisherJenksSampled``, ``HeadTailBreaks``, ``JenksCaspall``,
        ``JenksCaspallForced``, ``JenksCaspallSampled``, ``MaxPClassifier``,
        ``MaximumBreaks``, ``NaturalBreaks``, ``Quantiles``, ``Percentiles``, ``StdMean``,
        ``UserDefined``
    categorical : bool (default False)
        treat values as categories (will not use binning)
    categories : list-like (default None)
        list of categories. If None values.unique() is used.
    **classification_kwds : dict
        Keyword arguments for classification scheme
        For details see `mapclassify documentation <https://pysal.org/mapclassify>`_.

    Attributes
    ----------
    series : Series
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
        binning method
    bins : mapclassify.classifiers.Classifier
        generated bins
    classification_kwds : dict
        classification_kwds

    Examples
    --------
    >>> sw = momepy.sw_high(k=3, gdf=tessellation_df, ids='uID')
    >>> tessellation_df['area_Shannon'] = mm.Shannon(tessellation_df, 'area', sw, 'uID').series
    100%|██████████| 144/144 [00:00<00:00, 455.83it/s]
    """

    def __init__(
        self,
        gdf,
        values,
        spatial_weights,
        unique_id,
        binning="HeadTailBreaks",
        categorical=False,
        categories=None,
        **classification_kwds
    ):
        if not categorical:
            try:
                import mapclassify.classifiers as classifiers
            except ImportError:
                raise ImportError("The 'mapclassify' package is required")

            schemes = {}
            for classifier in classifiers.CLASSIFIERS:
                schemes[classifier.lower()] = getattr(classifiers, classifier)
            binning = binning.lower()
            if binning not in schemes:
                raise ValueError(
                    "Invalid binning. Binning must be in the"
                    " set: %r" % schemes.keys()
                )

        self.gdf = gdf
        self.sw = spatial_weights
        self.id = gdf[unique_id]
        self.binning = binning
        self.categorical = categorical
        self.categories = categories
        self.classification_kwds = classification_kwds

        data = gdf.copy()
        if values is not None:
            if not isinstance(values, str):
                data["mm_v"] = values
                values = "mm_v"
        self.values = data[values]

        if not categorical:
            self.bins = schemes[binning](data[values], **classification_kwds).bins

        data = data.set_index(unique_id)[values]

        if not categories:
            categories = data.unique()

        results_list = []
        for index in tqdm(data.index, total=data.shape[0]):
            if index in spatial_weights.neighbors.keys():
                neighbours = spatial_weights.neighbors[index].copy()
                if neighbours:
                    neighbours.append(index)
                else:
                    neighbours = [index]
                values_list = data.loc[neighbours]

                if categorical:
                    counts = values_list.value_counts().to_dict()
                    for c in categories:
                        if c not in counts.keys():
                            counts[c] = 0
                else:
                    sample_bins = classifiers.UserDefined(values_list, self.bins)
                    counts = dict(zip(self.bins, sample_bins.counts))

                results_list.append(self._shannon(counts))
            else:
                results_list.append(np.nan)

        self.series = pd.Series(results_list, index=gdf.index)

    def _shannon(self, data):
        """ Given a hash { 'species': count } , returns the SDI

        >>> _shannon({'a': 10, 'b': 20, 'c': 30,})
        1.0114042647073518

        https://gist.github.com/audy/783125
        """

        from math import log as ln

        def p(n, N):
            """ Relative abundance """
            if n == 0:
                return 0
            return (float(n) / N) * ln(float(n) / N)

        N = sum(data.values())

        return -sum(p(n, N) for n in data.values() if n != 0)


class Unique:
    """
    Calculates the number of unique values within neighbours defined in ``spatial_weights``.

    .. math::


    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing morphological tessellation
    values : str, list, np.array, pd.Series
        the name of the dataframe column, ``np.array``, or ``pd.Series`` where is stored character value.
    spatial_weights : libpysal.weights
        spatial weights matrix
    unique_id : str
        name of the column with unique id used as ``spatial_weights`` index

    Attributes
    ----------
    series : Series
        Series containing resulting values
    gdf : GeoDataFrame
        original GeoDataFrame
    values : Series
        Series containing used values
    sw : libpysal.weights
        spatial weights matrix
    id : Series
        Series containing used unique ID

    Examples
    --------
    >>> sw = momepy.sw_high(k=3, gdf=tessellation_df, ids='uID')
    >>> tessellation_df['cluster_unique'] = mm.Unique(tessellation_df, 'cluster', sw, 'uID').series
    100%|██████████| 144/144 [00:00<00:00, 722.50it/s]


    """

    def __init__(self, gdf, values, spatial_weights, unique_id):
        self.gdf = gdf
        self.sw = spatial_weights
        self.id = gdf[unique_id]

        data = gdf.copy()
        if values is not None:
            if not isinstance(values, str):
                data["mm_v"] = values
                values = "mm_v"
        self.values = data[values]

        data = data.set_index(unique_id)[values]

        results_list = []
        for index in tqdm(data.index, total=data.shape[0]):
            if index in spatial_weights.neighbors.keys():
                neighbours = spatial_weights.neighbors[index].copy()
                if neighbours:
                    neighbours.append(index)
                else:
                    neighbours = [index]

                values_list = data.loc[neighbours]
                results_list.append(len(values_list.unique()))
            else:
                results_list.append(np.nan)

        self.series = pd.Series(results_list, index=gdf.index)
