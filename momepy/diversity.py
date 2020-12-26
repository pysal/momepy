#!/usr/bin/env python
# -*- coding: utf-8 -*-

# diversity.py
# definitions of diversity characters

import numpy as np
import pandas as pd
import scipy as sp
from tqdm import tqdm  # progress bar

__all__ = [
    "Range",
    "Theil",
    "Simpson",
    "Gini",
    "Shannon",
    "Unique",
    "simpson_diversity",
    "shannon_diversity",
    "Percentiles",
]


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
    verbose : bool (default True)
        if True, shows progress bars in loops and indication of steps

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

    def __init__(
        self,
        gdf,
        values,
        spatial_weights,
        unique_id,
        rng=(0, 100),
        verbose=True,
        **kwargs
    ):
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
        for index in tqdm(data.index, total=data.shape[0], disable=not verbose):
            if index in spatial_weights.neighbors.keys():
                neighbours = [index]
                neighbours += spatial_weights.neighbors[index]

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
    verbose : bool (default True)
        if True, shows progress bars in loops and indication of steps

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

    def __init__(self, gdf, values, spatial_weights, unique_id, rng=None, verbose=True):
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
        for index in tqdm(data.index, total=data.shape[0], disable=not verbose):
            if index in spatial_weights.neighbors.keys():
                neighbours = [index]
                neighbours += spatial_weights.neighbors[index]

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
    gdf : GeoDataFrame
        GeoDataFrame containing morphological tessellation
    values : str, list, np.array, pd.Series
        the name of the dataframe column, ``np.array``, or ``pd.Series`` where is stored character value.
    spatial_weights : libpysal.weights, optional
        spatial weights matrix - If None, Queen contiguity matrix of set order will be calculated
        based on objects.
    unique_id : str
        name of the column with unique id used as ``spatial_weights`` index
    binning : str (default 'HeadTailBreaks')
        One of mapclassify classification schemes. For details see
        `mapclassify API documentation <http://pysal.org/mapclassify/api.html>`_.
    gini_simpson : bool (default False)
        return Gini-Simpson index instead of Simpson index (``1 - λ``)
    inverse : bool (default False)
        return Inverse Simpson index instead of Simpson index (``1 / λ``)
    categorical : bool (default False)
        treat values as categories (will not use ``binning``)
    categories : list-like (default None)
        list of categories. If None ``values.unique()`` is used.
    verbose : bool (default True)
        if True, shows progress bars in loops and indication of steps
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

    See also
    --------
    momepy.simpson_diversity : Calculates the Simpson\'s diversity index of data
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
        verbose=True,
        **classification_kwds
    ):
        if not categorical:
            try:
                from mapclassify import classify
            except ImportError:
                raise ImportError("The 'mapclassify >= 2.4.2` package is required.")

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

        data = data.set_index(unique_id)[values]

        if not categories:
            categories = data.unique()

        if not categorical:
            self.bins = classify(data, scheme=binning, **classification_kwds).bins
        else:
            self.bins = categories

        results_list = []
        for index in tqdm(data.index, total=data.shape[0], disable=not verbose):
            if index in spatial_weights.neighbors.keys():
                neighbours = [index]
                neighbours += spatial_weights.neighbors[index]
                values_list = data.loc[neighbours]

                results_list.append(
                    simpson_diversity(
                        values_list,
                        self.bins,
                        categorical=categorical,
                        categories=categories,
                    )
                )
            else:
                results_list.append(np.nan)

        if gini_simpson:
            self.series = 1 - pd.Series(results_list, index=gdf.index)
        elif inverse:
            self.series = 1 / pd.Series(results_list, index=gdf.index)
        else:
            self.series = pd.Series(results_list, index=gdf.index)


def simpson_diversity(data, bins=None, categorical=False, categories=None):
    """
    Calculates the Simpson\'s diversity index of data. Helper function for
    :py:class:`momepy.Simpson`.

    .. math::

        \\lambda=\\sum_{i=1}^{R} p_{i}^{2}

    Formula adapted from https://gist.github.com/martinjc/f227b447791df8c90568.

    Parameters
    ----------
    data : GeoDataFrame
        GeoDataFrame containing morphological tessellation
    bins : array, optional
        array of top edges of classification bins. Result of binnng.bins.
    categorical : bool (default False)
        treat values as categories (will not use ``bins``)
    categories : list-like (default None)
        list of categories

    Returns
    -------
    float
        Simpson's diversity index

    See also
    --------
    momepy.Simpson : Calculates the Simpson\'s diversity index of values within neighbours
    """
    if not categorical:
        try:
            import mapclassify as mc
        except ImportError:
            raise ImportError("The 'mapclassify' package is required")

    def p(n, N):
        """ Relative abundance """
        if n == 0:
            return 0
        return float(n) / N

    if categorical:
        counts = data.value_counts().to_dict()
        for c in categories:
            if c not in counts.keys():
                counts[c] = 0
    else:
        sample_bins = mc.UserDefined(data, bins)
        counts = dict(zip(bins, sample_bins.counts))

    N = sum(counts.values())

    return sum(p(n, N) ** 2 for n in counts.values() if n != 0)


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
    verbose : bool (default True)
        if True, shows progress bars in loops and indication of steps

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

    def __init__(self, gdf, values, spatial_weights, unique_id, rng=None, verbose=True):
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
        for index in tqdm(data.index, total=data.shape[0], disable=not verbose):
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
    gdf : GeoDataFrame
        GeoDataFrame containing morphological tessellation
    values : str, list, np.array, pd.Series
        the name of the dataframe column, ``np.array``, or ``pd.Series`` where is stored character value.
    spatial_weights : libpysal.weights, optional
        spatial weights matrix - If None, Queen contiguity matrix of set order will be calculated
        based on objects.
    unique_id : str
        name of the column with unique id used as ``spatial_weights`` index
    binning : str
        One of mapclassify classification schemes. For details see
        `mapclassify API documentation <http://pysal.org/mapclassify/api.html>`_.
    categorical : bool (default False)
        treat values as categories (will not use binning)
    categories : list-like (default None)
        list of categories. If None values.unique() is used.
    verbose : bool (default True)
        if True, shows progress bars in loops and indication of steps
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
        verbose=True,
        **classification_kwds
    ):
        if not categorical:
            try:
                from mapclassify import classify
            except ImportError:
                raise ImportError("The 'mapclassify >= 2.4.2` package is required.")

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

        data = data.set_index(unique_id)[values]

        if not categories:
            categories = data.unique()

        if not categorical:
            self.bins = classify(data, scheme=binning, **classification_kwds).bins
        else:
            self.bins = categories

        results_list = []
        for index in tqdm(data.index, total=data.shape[0], disable=not verbose):
            if index in spatial_weights.neighbors.keys():
                neighbours = [index]
                neighbours += spatial_weights.neighbors[index]
                values_list = data.loc[neighbours]

                results_list.append(
                    shannon_diversity(
                        values_list,
                        self.bins,
                        categorical=categorical,
                        categories=categories,
                    )
                )
            else:
                results_list.append(np.nan)

        self.series = pd.Series(results_list, index=gdf.index)


def shannon_diversity(data, bins=None, categorical=False, categories=None):
    """
    Calculates the Shannon\'s diversity index of data. Helper function for
    :py:class:`momepy.Shannon`.

    .. math::

        \\lambda=\\sum_{i=1}^{R} p_{i}^{2}

    Formula adapted from https://gist.github.com/audy/783125

    Parameters
    ----------
    data : GeoDataFrame
        GeoDataFrame containing morphological tessellation
    bins : array, optional
        array of top edges of classification bins. Result of binnng.bins.
    categorical : bool (default False)
        treat values as categories (will not use ``bins``)
    categories : list-like (default None)
        list of categories

    Returns
    -------
    float
        Shannon's diversity index

    See also
    --------
    momepy.Shannon : Calculates the Shannon's diversity index of values within neighbours
    momepy.Simpson : Calculates the Simpson's diversity index of values within neighbours
    momepy.simpson_diversity : Calculates the Simpson's diversity index of data
    """
    from math import log as ln

    if not categorical:
        try:
            import mapclassify as mc
        except ImportError:
            raise ImportError("The 'mapclassify' package is required")

    def p(n, N):
        """ Relative abundance """
        if n == 0:
            return 0
        return (float(n) / N) * ln(float(n) / N)

    if categorical:
        counts = data.value_counts().to_dict()
        for c in categories:
            if c not in counts.keys():
                counts[c] = 0
    else:
        sample_bins = mc.UserDefined(data, bins)
        counts = dict(zip(bins, sample_bins.counts))

    N = sum(counts.values())

    return -sum(p(n, N) for n in counts.values() if n != 0)


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
    verbose : bool (default True)
        if True, shows progress bars in loops and indication of steps

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

    def __init__(self, gdf, values, spatial_weights, unique_id, verbose=True):
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
        for index in tqdm(data.index, total=data.shape[0], disable=not verbose):
            if index in spatial_weights.neighbors.keys():
                neighbours = [index]
                neighbours += spatial_weights.neighbors[index]

                values_list = data.loc[neighbours]
                results_list.append(len(values_list.unique()))
            else:
                results_list.append(np.nan)

        self.series = pd.Series(results_list, index=gdf.index)


class Percentiles:
    """
    Calculates the percentiles of values within neighbours defined in ``spatial_weights``.

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
    percentiles : array-like (default [25, 50, 75])
        percentiles to return
    interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
        This optional parameter specifies the interpolation method to
        use when the desired percentile lies between two data points
        ``i < j``:

        * ``'linear'``
        * ``'lower'``
        * ``'higher'``
        * ``'nearest'``
        * ``'midpoint'``

        See the documentation of ``numpy.percentile`` for details.
   
    verbose : bool (default True)
        if True, shows progress bars in loops and indication of steps

    Attributes
    ----------
    frame : DataFrame
        DataFrame containing resulting values
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
    >>> tessellation_df['cluster_unique'] = mm.Percentiles(tessellation_df, 'cluster', sw, 'uID').frame
    100%|██████████| 144/144 [00:00<00:00, 722.50it/s]
    """

    def __init__(
        self,
        gdf,
        values,
        spatial_weights,
        unique_id,
        percentiles=[25, 50, 75],
        interpolation="midpoint",
        verbose=True,
    ):
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
        for index in tqdm(data.index, total=data.shape[0], disable=not verbose):
            if index in spatial_weights.neighbors.keys():
                neighbours = [index]
                neighbours += spatial_weights.neighbors[index]

                values_list = data.loc[neighbours]
                results_list.append(
                    np.nanpercentile(
                        values_list, percentiles, interpolation=interpolation
                    )
                )
            else:
                results_list.append(np.nan)

        self.frame = pd.DataFrame(results_list, columns=percentiles, index=gdf.index)
