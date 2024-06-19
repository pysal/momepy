#!/usr/bin/env python

# diversity.py
# definitions of diversity characters

import warnings

import numpy as np
import pandas as pd
import scipy as sp
from numpy.lib import NumpyVersion
from tqdm.auto import tqdm  # progress bar

from .utils import deprecated, removed

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


@deprecated("values_range")
class Range:
    """
    Calculates the range of values within neighbours defined in ``spatial_weights``.
    Uses ``scipy.stats.iqr`` under the hood.

    Adapted from :cite:`dibble2017`.

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing morphological tessellation.
    values : str, list, np.array, pd.Series
        The name of the dataframe column, ``np.array``, or ``pd.Series``
        where character values are stored.
    spatial_weights : libpysal.weights
        A spatial weights matrix.
    unique_id : str
        The name of the column with unique IDs used as the ``spatial_weights`` index.
    rng : tuple, list, optional (default None)
        A two-element sequence containing floats between 0 and 100 (inclusive)
        that are the percentiles over which to compute the range.
        The order of the elements is not important.
    **kwargs : dict
        Optional arguments for ``scipy.stats.iqr``.
    verbose : bool (default True)
        If ``True``, shows progress bars in loops and indication of steps.

    Attributes
    ----------
    series : Series
        A Series containing resulting values.
    gdf : GeoDataFrame
        The original GeoDataFrame.
    values : Series
        A Series containing used values.
    sw : libpysal.weights
        The spatial weights matrix.
    id : Series
        A Series containing used unique ID.
    rng : tuple, list, optional (default None)
        A two-element sequence containing floats between 0 and 100 (inclusive)
        that are the percentiles over which to compute the range.
        The order of the elements is not important.
    kwargs : dict
        Optional arguments for ``scipy.stats.iqr``.

    Examples
    --------
    >>> sw = momepy.sw_high(k=3, gdf=tessellation_df, ids='uID')
    >>> tessellation_df['area_IQR_3steps'] = mm.Range(tessellation_df,
    ...                                               'area',
    ...                                               sw,
    ...                                               'uID',
    ...                                               rng=(25, 75)).series
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
        **kwargs,
    ):
        self.gdf = gdf
        self.sw = spatial_weights
        self.id = gdf[unique_id]
        self.rng = rng
        self.kwargs = kwargs

        data = gdf.copy()
        if values is not None and not isinstance(values, str):
            data["mm_v"] = values
            values = "mm_v"
        self.values = data[values]

        data = data.set_index(unique_id)[values]

        results_list = []
        for index in tqdm(data.index, total=data.shape[0], disable=not verbose):
            if index in spatial_weights.neighbors:
                neighbours = [index]
                neighbours += spatial_weights.neighbors[index]

                values_list = data.loc[neighbours]
                results_list.append(sp.stats.iqr(values_list, rng=rng, **kwargs))
            else:
                results_list.append(np.nan)

        self.series = pd.Series(results_list, index=gdf.index)


@deprecated("theil")
class Theil:
    """
    Calculates the Theil measure of inequality of values within neighbours defined in
    ``spatial_weights``. Uses ``inequality.theil.Theil`` under the hood.
    Requires '`inequality`' package.

    .. math::

        T = \\sum_{i=1}^n \\left(
            \\frac{y_i}{\\sum_{i=1}^n y_i} \\ln \\left[
                N \\frac{y_i} {\\sum_{i=1}^n y_i}
            \\right]
        \\right)

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing morphological tessellation.
    values : str, list, np.array, pd.Series
        The name of the dataframe column, ``np.array``, or ``pd.Series``
        where character values are stored.
    spatial_weights : libpysal.weights
        A spatial weights matrix.
    unique_id : str
        The name of the column with unique IDs used as the ``spatial_weights`` index.
    rng : tuple, list, optional (default None)
        A two-element sequence containing floats between 0 and 100 (inclusive)
        that are the percentiles over which to compute the range.
        The order of the elements is not important.
    verbose : bool (default True)
        If ``True``, shows progress bars in loops and indication of steps.

    Attributes
    ----------
    series : Series
        A Series containing resulting values.
    gdf : GeoDataFrame
        The original GeoDataFrame.
    values : Series
        A Series containing used values.
    sw : libpysal.weights
        The spatial weights matrix.
    id : Series
        A Series containing used unique ID.
    rng : tuple, list, optional (default None)
        A two-element sequence containing floats between 0 and 100 (inclusive)
        that are the percentiles over which to compute the range.
        The order of the elements is not important.

    Examples
    --------
    >>> sw = momepy.sw_high(k=3, gdf=tessellation_df, ids='uID')
    >>> tessellation_df['area_Theil'] = mm.Theil(tessellation_df,
    ...                                          'area',
    ...                                          sw,
    ...                                          'uID').series
    100%|██████████| 144/144 [00:00<00:00, 597.37it/s]
    """

    def __init__(self, gdf, values, spatial_weights, unique_id, rng=None, verbose=True):
        try:
            from inequality.theil import Theil
        except ImportError as err:
            raise ImportError("The 'inequality' package is required.") from err

        self.gdf = gdf
        self.sw = spatial_weights
        self.id = gdf[unique_id]
        self.rng = rng

        data = gdf.copy()
        if values is not None and not isinstance(values, str):
            data["mm_v"] = values
            values = "mm_v"
        self.values = data[values]

        data = data.set_index(unique_id)[values]

        if rng:
            from momepy import limit_range

        results_list = []
        for index in tqdm(data.index, total=data.shape[0], disable=not verbose):
            if index in spatial_weights.neighbors:
                neighbours = [index]
                neighbours += spatial_weights.neighbors[index]

                values_list = data.loc[neighbours]

                if rng:
                    values_list = limit_range(values_list.values, rng=rng)
                results_list.append(Theil(values_list).T)
            else:
                results_list.append(np.nan)

        self.series = pd.Series(results_list, index=gdf.index)


@deprecated("simpson")
class Simpson:
    """
    Calculates the Simpson's diversity index of values within neighbours defined in
    ``spatial_weights``. Uses ``mapclassify.classifiers`` under the hood for binning.
    Requires ``mapclassify>=.2.1.0`` dependency.

    .. math::

        \\lambda=\\sum_{i=1}^{R} p_{i}^{2}

    Adapted from :cite:`feliciotti2018`.

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing morphological tessellation.
    values : str, list, np.array, pd.Series
        The name of the dataframe column, ``np.array``, or ``pd.Series``
        where character values are stored.
    spatial_weights : libpysal.weights, optional
        A spatial weights matrix. If ``None``, Queen contiguity
        matrix of set order will be calculated based on objects.
    unique_id : str
        The name of the column with unique IDs used as the ``spatial_weights`` index.
    binning : str (default 'HeadTailBreaks')
        One of mapclassify classification schemes. For details see
        `mapclassify API documentation <http://pysal.org/mapclassify/api.html>`_.
    gini_simpson : bool (default False)
        Return Gini-Simpson index instead of Simpson index (``1 - λ``).
    inverse : bool (default False)
        Return Inverse Simpson index instead of Simpson index (``1 / λ``).
    categorical : bool (default False)
        Treat values as categories (will not use ``binning``).
    verbose : bool (default True)
        If ``True``, shows progress bars in loops and indication of steps.
    **classification_kwds : dict
        Keyword arguments for the classification scheme.
        For details see `mapclassify documentation <https://pysal.org/mapclassify>`_.

    Attributes
    ----------
    series : Series
        A Series containing resulting values.
    gdf : GeoDataFrame
        The original GeoDataFrame.
    values : Series
        A Series containing used values.
    sw : libpysal.weights
        The spatial weights matrix.
    id : Series
        A Series containing used unique ID.
    binning : str
        The binning method used.
    bins : mapclassify.classifiers.Classifier
        The generated bins.
    classification_kwds : dict
        Keyword arguments for the classification scheme.
        For details see `mapclassify documentation <https://pysal.org/mapclassify>`_.

    Examples
    --------
    >>> sw = momepy.sw_high(k=3, gdf=tessellation_df, ids='uID')
    >>> tessellation_df['area_Simpson'] = mm.Simpson(tessellation_df,
    ...                                              'area',
    ...                                              sw,
    ...                                              'uID').series
    100%|██████████| 144/144 [00:00<00:00, 455.83it/s]

    See also
    --------
    momepy.simpson_diversity : Calculates the Simpson's diversity index of data.
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
        verbose=True,
        **classification_kwds,
    ):
        if not categorical:
            try:
                from mapclassify import classify
            except ImportError as err:
                raise ImportError(
                    "The 'mapclassify >= 2.4.2` package is required."
                ) from err

        self.gdf = gdf
        self.sw = spatial_weights
        self.id = gdf[unique_id]
        self.binning = binning
        self.gini_simpson = gini_simpson
        self.inverse = inverse
        self.categorical = categorical
        self.classification_kwds = classification_kwds

        data = gdf.copy()
        if values is not None and not isinstance(values, str):
            data["mm_v"] = values
            values = "mm_v"
        self.values = data[values]

        data = data.set_index(unique_id)[values]

        if not categorical:
            self.bins = classify(data, scheme=binning, **classification_kwds).bins
        else:
            self.bins = None

        results_list = []
        for index in tqdm(data.index, total=data.shape[0], disable=not verbose):
            if index in spatial_weights.neighbors:
                neighbours = [index]
                neighbours += spatial_weights.neighbors[index]
                values_list = data.loc[neighbours]

                results_list.append(
                    simpson_diversity(
                        values_list,
                        self.bins,
                        categorical=categorical,
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


def simpson_diversity(values, bins=None, categorical=False):
    """
    Calculates the Simpson's diversity index of data. Helper function for
    :py:class:`momepy.Simpson`.

    .. math::

        \\lambda=\\sum_{i=1}^{R} p_{i}^{2}

    Parameters
    ----------
    values : pandas.Series
        A list of values.
    bins : array, optional
        An array of top edges of classification bins.
        Should be equal to the result of ``binning.bins``.
    categorical : bool (default False)
        Treat values as categories (will not use ``bins``).

    Returns
    -------
    float
        Simpson's diversity index.

    See also
    --------
    momepy.Simpson : Calculates the Simpson's diversity index.
    """
    if not categorical:
        try:
            import mapclassify as mc
        except ImportError as err:
            raise ImportError("The 'mapclassify' package is required") from err

    if categorical:
        counts = values.value_counts()

    else:
        sample_bins = mc.UserDefined(values, bins)
        counts = sample_bins.counts

    return sum((n / sum(counts)) ** 2 for n in counts if n != 0)


@deprecated("gini")
class Gini:
    """
    Calculates the Gini index of values within neighbours defined in
    ``spatial_weights``. Uses ``inequality.gini.Gini`` under the hood.
    Requires '`inequality`' package.

    .. math::

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing morphological tessellation.
    values : str, list, np.array, pd.Series
        The name of the dataframe column, ``np.array``, or ``pd.Series``
        where character values are stored.
    spatial_weights : libpysal.weights
        A spatial weights matrix.
    unique_id : str
        The name of the column with unique IDs used as the ``spatial_weights`` index.
    rng : tuple, list, optional (default None)
        A two-element sequence containing floats between 0 and 100 (inclusive)
        that are the percentiles over which to compute the range.
        The order of the elements is not important.
    verbose : bool (default True)
        If ``True``, shows progress bars in loops and indication of steps.

    Attributes
    ----------
    series : Series
        A Series containing resulting values.
    gdf : GeoDataFrame
        The original GeoDataFrame.
    values : Series
        A Series containing used values.
    sw : libpysal.weights
        The spatial weights matrix.
    id : Series
        A Series containing used unique ID.
    rng : tuple, list, optional (default None)
        A two-element sequence containing floats between 0 and 100 (inclusive)
        that are the percentiles over which to compute the range.
        The order of the elements is not important.

    Examples
    --------
    >>> sw = momepy.sw_high(k=3, gdf=tessellation_df, ids='uID')
    >>> tessellation_df['area_Gini'] = mm.Gini(tessellation_df,
    ...                                        'area',
    ...                                        sw,
    ...                                        'uID').series
    100%|██████████| 144/144 [00:00<00:00, 597.37it/s]
    """

    def __init__(self, gdf, values, spatial_weights, unique_id, rng=None, verbose=True):
        try:
            from inequality.gini import Gini
        except ImportError as err:
            raise ImportError("The 'inequality' package is required.") from err

        self.gdf = gdf
        self.sw = spatial_weights
        self.id = gdf[unique_id]
        self.rng = rng

        data = gdf.copy()
        if values is not None and not isinstance(values, str):
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
            if index in spatial_weights.neighbors:
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


@deprecated("shannon")
class Shannon:
    """
    Calculates the Shannon index of values within neighbours defined in
    ``spatial_weights``. Uses ``mapclassify.classifiers`` under the hood
    for binning. Requires ``mapclassify>=.2.1.0`` dependency.

    .. math::

        H^{\\prime}=-\\sum_{i=1}^{R} p_{i} \\ln p_{i}

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing morphological tessellation.
    values : str, list, np.array, pd.Series
        The name of the dataframe column, ``np.array``, or ``pd.Series``
        where character values are stored.
    spatial_weights : libpysal.weights, optional
        A spatial weights matrix. If ``None``, Queen contiguity
        matrix of set order will be calculated based on objects.
    unique_id : str
        The name of the column with unique IDs used as the ``spatial_weights`` index.
    binning : str
        One of mapclassify classification schemes. For details see
        `mapclassify API documentation <http://pysal.org/mapclassify/api.html>`_.
    categorical : bool (default False)
        Treat values as categories (will not use binning).
    categories : list-like (default None)
        A list of categories. If ``None``, ``values.unique()`` is used.
    verbose : bool (default True)
        If ``True``, shows progress bars in loops and indication of steps.
    **classification_kwds : dict
        Keyword arguments for classification scheme
        For details see `mapclassify documentation <https://pysal.org/mapclassify>`_.

    Attributes
    ----------
    series : Series
        A Series containing resulting values.
    gdf : GeoDataFrame
        The original GeoDataFrame.
    values : Series
        A Series containing used values.
    sw : libpysal.weights
        The spatial weights matrix.
    id : Series
        A Series containing used unique ID.
    binning : str
        The binning method used.
    bins : mapclassify.classifiers.Classifier
        The generated bins.
    classification_kwds : dict
        Keyword arguments for classification scheme
        For details see `mapclassify documentation <https://pysal.org/mapclassify>`_.

    Examples
    --------
    >>> sw = momepy.sw_high(k=3, gdf=tessellation_df, ids='uID')
    >>> tessellation_df['area_Shannon'] = mm.Shannon(tessellation_df,
    ...                                              'area',
    ...                                              sw,
    ...                                              'uID').series
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
        **classification_kwds,
    ):
        if not categorical:
            try:
                from mapclassify import classify
            except ImportError as err:
                raise ImportError(
                    "The 'mapclassify >= 2.4.2` package is required."
                ) from err

        self.gdf = gdf
        self.sw = spatial_weights
        self.id = gdf[unique_id]
        self.binning = binning
        self.categorical = categorical
        self.categories = categories
        self.classification_kwds = classification_kwds

        data = gdf.copy()
        if values is not None and not isinstance(values, str):
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
            if index in spatial_weights.neighbors:
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
    Calculates the Shannon's diversity index of data. Helper function for
    :py:class:`momepy.Shannon`.

    .. math::

        \\lambda=\\sum_{i=1}^{R} p_{i}^{2}

    Formula adapted from https://gist.github.com/audy/783125.

    Parameters
    ----------
    data : GeoDataFrame
        A GeoDataFrame containing morphological tessellation.
    bins : array, optional
        An array of top edges of classification bins. Result of ``binning.bins``.
    categorical : bool (default False)
        tTeat values as categories (will not use ``bins``).
    categories : list-like (default None)
        A list of categories.

    Returns
    -------
    float
        Shannon's diversity index.

    See also
    --------
    momepy.Shannon : Calculates the Shannon's diversity index.
    momepy.Simpson : Calculates the Simpson's diversity index.
    momepy.simpson_diversity : Calculates the Simpson's diversity index.
    """
    from math import log as ln

    if not categorical:
        try:
            import mapclassify as mc
        except ImportError as err:
            raise ImportError("The 'mapclassify' package is required") from err

    def p(n, sum_n):
        """Relative abundance"""
        if n == 0:
            return 0
        return (n / sum_n) * ln(n / sum_n)

    if categorical:
        counts = dict.fromkeys(categories, 0)
        counts.update(data.value_counts())
    else:
        sample_bins = mc.UserDefined(data, bins)
        counts = dict(zip(bins, sample_bins.counts, strict=True))

    return -sum(p(n, sum(counts.values())) for n in counts.values() if n != 0)


@removed("`.describe()` method of libpysal.graph.Graph")
class Unique:
    """
    Calculates the number of unique values within neighbours defined in
    ``spatial_weights``.

    .. math::


    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing morphological tessellation.
    values : str, list, np.array, pd.Series
        The name of the dataframe column, ``np.array``, or ``pd.Series``
        where character values are stored.
    spatial_weights : libpysal.weights
        A spatial weights matrix.
    unique_id : str
        The name of the column with unique IDs used as the ``spatial_weights`` index.
    dropna : bool (default True)
        Don’t include ``NaN`` in the counts of unique values.
    verbose : bool (default True)
        If ``True``, shows progress bars in loops and indication of steps.

    Attributes
    ----------
    series : Series
        A Series containing resulting values.
    gdf : GeoDataFrame
        The original GeoDataFrame.
    values : Series
        A Series containing used values.
    sw : libpysal.weights
        The spatial weights matrix.
    id : Series
        A Series containing used unique ID.

    Examples
    --------
    >>> sw = momepy.sw_high(k=3, gdf=tessellation_df, ids='uID')
    >>> tessellation_df['cluster_unique'] = mm.Unique(tessellation_df,
    ...                                              'cluster',
    ...                                              sw,
    ...                                              'uID').series
    100%|██████████| 144/144 [00:00<00:00, 722.50it/s]
    """

    def __init__(
        self, gdf, values, spatial_weights, unique_id, dropna=True, verbose=True
    ):
        self.gdf = gdf
        self.sw = spatial_weights
        self.id = gdf[unique_id]

        data = gdf.copy()
        if values is not None and not isinstance(values, str):
            data["mm_v"] = values
            values = "mm_v"
        self.values = data[values]

        data = data.set_index(unique_id)[values]

        results_list = []
        for index in tqdm(data.index, total=data.shape[0], disable=not verbose):
            if index in spatial_weights.neighbors:
                neighbours = [index]
                neighbours += spatial_weights.neighbors[index]

                values_list = data.loc[neighbours]
                results_list.append(values_list.nunique(dropna=dropna))
            else:
                results_list.append(np.nan)

        self.series = pd.Series(results_list, index=gdf.index)


@deprecated("percentile")
class Percentiles:
    """
    Calculates the percentiles of values within
    neighbours defined in ``spatial_weights``.

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing source geometry.
    values : str, list, np.array, pd.Series
        The name of the dataframe column, ``np.array``, or ``pd.Series``
        where character values are stored.
    spatial_weights : libpysal.weights
        A spatial weights matrix.
    unique_id : str
        The name of the column with unique IDs used as the ``spatial_weights`` index.
    percentiles : array-like (default [25, 50, 75])
        The percentiles to return.
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
        If ``True``, shows progress bars in loops and indication of steps.
    weighted : {'linear', None} (default None)
        Distance decay weighting. If ``None``, each neighbor within
        ``spatial_weights`` has equal weight. If ``'linear'``, linear
        inverse distance between centroids is used as a weight.


    Attributes
    ----------
    frame : DataFrame
        A DataFrame containing resulting values.
    gdf : GeoDataFrame
        The original GeoDataFrame.
    values : Series
        A Series containing used values.
    sw : libpysal.weights
        The spatial weights matrix.
    id : Series
        A Series containing used unique ID.

    Examples
    --------
    >>> sw = momepy.sw_high(k=3, gdf=tessellation_df, ids='uID')
    >>> percentiles_df = mm.Percentiles(tessellation_df,
    ...                                 'area',
    ...                                 sw,
    ...                                 'uID').frame
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
        weighted=None,
    ):
        self.gdf = gdf
        self.sw = spatial_weights
        self.id = gdf[unique_id]

        data = gdf.copy()

        if values is not None and not isinstance(values, str):
            data["mm_v"] = values
            values = "mm_v"
        self.values = data[values]

        results_list = []

        if weighted == "linear":
            data = data.set_index(unique_id)[[values, data.geometry.name]]
            data.geometry = data.centroid

            for i, geom in tqdm(
                data.geometry.items(), total=data.shape[0], disable=not verbose
            ):
                if i in spatial_weights.neighbors:
                    neighbours = spatial_weights.neighbors[i]

                    vicinity = data.loc[neighbours]
                    distance = vicinity.distance(geom)
                    distance_decay = 1 / distance
                    vals = vicinity[values].values
                    sorter = np.argsort(vals)
                    vals = vals[sorter]
                    nan_mask = np.isnan(vals)
                    if nan_mask.all():
                        results_list.append(np.array([np.nan] * len(percentiles)))
                    else:
                        sample_weight = distance_decay.values[sorter][~nan_mask]
                        weighted_quantiles = (
                            np.cumsum(sample_weight) - 0.5 * sample_weight
                        )
                        weighted_quantiles /= np.sum(sample_weight)
                        interpolate = np.interp(
                            [x / 100 for x in percentiles],
                            weighted_quantiles,
                            vals[~nan_mask],
                        )
                        results_list.append(interpolate)
                else:
                    results_list.append(np.array([np.nan] * len(percentiles)))

            self.frame = pd.DataFrame(
                results_list, columns=percentiles, index=gdf.index
            )

        elif weighted is None:
            data = data.set_index(unique_id)[values]

            if NumpyVersion(np.__version__) >= "1.22.0":
                method = {"method": interpolation}
            else:
                method = {"interpolation": interpolation}

            for index in tqdm(data.index, total=data.shape[0], disable=not verbose):
                if index in spatial_weights.neighbors:
                    neighbours = [index]
                    neighbours += spatial_weights.neighbors[index]
                    values_list = data.loc[neighbours]
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore", message="All-NaN slice encountered"
                        )
                        results_list.append(
                            np.nanpercentile(values_list, percentiles, **method)
                        )
                else:
                    results_list.append(np.array([np.nan] * len(percentiles)))

            self.frame = pd.DataFrame(
                results_list, columns=percentiles, index=gdf.index
            )

        else:
            raise ValueError(f"'{weighted}' is not a valid option.")
