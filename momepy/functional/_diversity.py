import warnings

import numpy as np
import scipy as sp
from libpysal.graph import Graph
from numpy.typing import NDArray
from pandas import DataFrame, Series

import momepy as mm

__all__ = ["describe", "values_range", "theil", "simpson", "shannon", "gini", "unique"]


def describe(
    y: NDArray[np.float_] | Series,
    graph: Graph,
    q: tuple[float, float] | None = None,
    include_mode: bool = False,
) -> DataFrame:
    """Describe the distribution of values within a set neighbourhood.

    Given the graph, computes the descriptive statisitcs of values within the
    neighbourhood of each node. Optionally, the values can be limited to a certain
    quantile range before computing the statistics.

    Notes
    -----
    The index of ``values`` must match the index along which the ``graph`` is
    built.

    The numba package is used extensively in this function to accelerate the computation
    of statistics. Without numba, these computations may become slow on large data.

    Parameters
    ----------
    y : NDArray[np.float_] | Series
        An 1D array of numeric values to be described.
    graph : libpysal.graph.Graph
        Graph representing spatial relationships between elements.
    q : tuple[float, float] | None, optional
        Tuple of percentages for the percentiles to compute. Values must be between 0
        and 100 inclusive. When set, values below and above the percentiles will be
        discarded before computation of the average. The percentiles are computed for
        each neighborhood. By default None.
    include_mode : False
        Compute mode along with other statistics. Default is False. Mode is
        computationally expensive and not useful for continous variables.

    Returns
    -------
    DataFrame
        A DataFrame with descriptive statistics.
    """
    try:
        from numba import njit
    except (ModuleNotFoundError, ImportError):
        warnings.warn(
            "The numba package is used extensively in this function to accelerate the"
            " computation of statistics but it is not installed or  cannot be imported."
            " Without numba, these computations may become slow on large data.",
            UserWarning,
            stacklevel=2,
        )
        from libpysal.common import jit as njit

    @njit
    def _mode(array):
        """Custom mode function for numba."""
        array = np.sort(array.ravel())
        mask = np.empty(array.shape, dtype=np.bool_)
        mask[:1] = True
        mask[1:] = array[1:] != array[:-1]
        unique = array[mask]
        idx = np.nonzero(mask)[0]
        idx = np.append(idx, mask.size)
        counts = np.diff(idx)
        return unique[np.argmax(counts)]

    @njit
    def _describe(values, q, include_mode=False):
        """Helper function to calculate average."""
        nan_tracker = np.isnan(values)

        if (len(values) > 2) and (not nan_tracker.all()):
            if nan_tracker.any():
                lower, higher = np.nanpercentile(values, q)
            else:
                lower, higher = np.percentile(values, q)
            values = values[(lower <= values) & (values <= higher)]

        results = [
            np.mean(values),
            np.median(values),
            np.std(values),
            np.min(values),
            np.max(values),
            np.sum(values),
        ]
        if include_mode:
            results.append(_mode(values))
        return results

    if not isinstance(y, Series):
        y = Series(y)

    grouper = y.take(graph._adjacency.index.codes[1]).groupby(
        graph._adjacency.index.codes[0]
    )

    if q is None:
        stat_ = grouper.agg(["mean", "median", "std", "min", "max", "sum"])
        if include_mode:
            stat_["mode"] = grouper.agg(lambda x: _mode(x.values))
    else:
        agg = grouper.agg(lambda x: _describe(x.values, q=q, include_mode=include_mode))
        stat_ = DataFrame(zip(*agg, strict=True)).T
        cols = ["mean", "median", "std", "min", "max", "sum"]
        if include_mode:
            cols.append("mode")
        stat_.columns = cols

    return stat_


def values_range(
    data: DataFrame | Series, graph: Graph, rng: tuple | list = (0, 100), **kwargs
):
    """
    Calculates the range of values within neighbours defined in ``graph``.
    Uses ``scipy.stats.iqr`` under the hood.

    Adapted from :cite:`dibble2017`.

    Parameters
    ----------
    data : DataFrame | Series
        A DataFrame or Series containing the values to be analysed.
    graph : libpysal.graph.Graph
        A spatial weights matrix for the data.
    rng : tuple, list, optional (default (0,100)))
        A two-element sequence containing floats between 0 and 100 (inclusive)
        that are the percentiles over which to compute the range.
        The order of the elements is not important.
    **kwargs : dict
        Optional arguments for ``scipy.stats.iqr``.

    Returns
    ----------
    Series
        A Series containing resulting values.

    Examples
    --------
    >>> tessellation_df['area_IQR_3steps'] = mm.range(tessellation_df['area'],
    ...                                               graph,
    ...                                               rng=(25, 75))
    """

    def _apply_range(values):
        return sp.stats.iqr(values, rng=rng, **kwargs)

    return graph.apply(data, _apply_range)


def theil(data: DataFrame | Series, graph: Graph, rng: tuple | list = None):
    """
    Calculates the Theil measure of inequality of values within neighbours defined in
    ``graph``. Uses ``inequality.theil.Theil`` under the hood.
    Requires '`inequality`' package.

    .. math::

        T = \\sum_{i=1}^n \\left(
            \\frac{y_i}{\\sum_{i=1}^n y_i} \\ln \\left[
                N \\frac{y_i} {\\sum_{i=1}^n y_i}
            \\right]
        \\right)

    Parameters
    ----------
    data : DataFrame | Series
        A DataFrame or Series containing the values to be analysed.
    graph : libpysal.graph.Graph
        A spatial weights matrix for the data.
    rng : tuple, list, optional (default (0,100)))
        A two-element sequence containing floats between 0 and 100 (inclusive)
        that are the percentiles over which to compute the range.
        The order of the elements is not important.

    Returns
    ----------
    Series
        A Series containing resulting values.

    Examples
    --------
    >>> tessellation_df['area_Theil'] = mm.theil(tessellation_df['area'],
    ...                                          graph)
    """

    try:
        from inequality.theil import Theil
    except ImportError as err:
        raise ImportError("The 'inequality' package is required.") from err
    if rng:
        from momepy import limit_range

    def _apply_theil(values):
        if rng:
            values = limit_range(values, rng=rng)
        return Theil(values).T

    return graph.apply(data, _apply_theil)


def simpson(
    data: DataFrame | Series,
    graph: Graph,
    binning: str = "HeadTailBreaks",
    gini_simpson: bool = False,
    inverse: bool = False,
    categorical: bool = False,
    **classification_kwds,
):
    """
    Calculates the Simpson's diversity index of values within neighbours defined in
    ``graph``. Uses ``mapclassify.classifiers`` under the hood for binning.
    Requires ``mapclassify>=.2.1.0`` dependency.

    .. math::

        \\lambda=\\sum_{i=1}^{R} p_{i}^{2}

    Adapted from :cite:`feliciotti2018`.

    Parameters
    ----------
    data : DataFrame | Series
        A DataFrame or Series containing the values to be analysed.
    graph : libpysal.graph.Graph
        A spatial weights matrix for the data.
    binning : str (default 'HeadTailBreaks')
        One of mapclassify classification schemes. For details see
        `mapclassify API documentation <http://pysal.org/mapclassify/api.html>`_.
    gini_simpson : bool (default False)
        Return Gini-Simpson index instead of Simpson index (``1 - λ``).
    inverse : bool (default False)
        Return Inverse Simpson index instead of Simpson index (``1 / λ``).
    categorical : bool (default False)
        Treat values as categories (will not use ``binning``).
    **classification_kwds : dict
        Keyword arguments for the classification scheme.
        For details see `mapclassify documentation <https://pysal.org/mapclassify>`_.

    Returns
    -------
    Series
        A Series containing resulting values.

    Examples
    --------
    >>> tessellation_df['area_Simpson'] = mm.simpson(tessellation_df['area'],
    ...                                              graph)

    See also
    --------
    momepy.simpson_diversity : Calculates the Simpson's diversity index of data.
    """
    if not categorical:
        try:
            from mapclassify import classify
        except ImportError as err:
            raise ImportError(
                "The 'mapclassify >= 2.4.2` package is required."
            ) from err
        bins = classify(data, scheme=binning, **classification_kwds).bins
    else:
        bins = None

    def _apply_simpson_diversity(values):
        return mm.simpson_diversity(
            values,
            bins,
            categorical=categorical,
        )

    result = graph.apply(data, _apply_simpson_diversity)

    if gini_simpson:
        result = 1 - result
    elif inverse:
        result = 1 / result
    return result


def shannon(
    data: DataFrame | Series,
    graph: Graph,
    binning: str = "HeadTailBreaks",
    categorical: bool = False,
    categories: list = None,
    **classification_kwds,
):
    """
    Calculates the Shannon index of values within neighbours defined in
    ``graph``. Uses ``mapclassify.classifiers`` under the hood
    for binning. Requires ``mapclassify>=.2.1.0`` dependency.

    .. math::

        H^{\\prime}=-\\sum_{i=1}^{R} p_{i} \\ln p_{i}

    Parameters
    ----------
    data : DataFrame | Series
        A DataFrame or Series containing the values to be analysed.
    graph : libpysal.graph.Graph
        A spatial weights matrix for the data.
    binning : str (default 'HeadTailBreaks')
        One of mapclassify classification schemes. For details see
        `mapclassify API documentation <http://pysal.org/mapclassify/api.html>`_.
    categorical : bool (default False)
        Treat values as categories (will not use binning).
    categories : list-like (default None)
        A list of categories. If ``None``, ``values.unique()`` is used.
    **classification_kwds : dict
        Keyword arguments for classification scheme
        For details see `mapclassify documentation <https://pysal.org/mapclassify>`_.

    Returns
    ----------
    Series
        A Series containing resulting values.

    Examples
    --------
    >>> tessellation_df['area_Shannon'] = mm.shannon(tessellation_df['area'],
    ...                                              graph)
    """

    if not categories:
        categories = data.unique()

    if not categorical:
        try:
            from mapclassify import classify
        except ImportError as err:
            raise ImportError(
                "The 'mapclassify >= 2.4.2` package is required."
            ) from err
        bins = classify(data, scheme=binning, **classification_kwds).bins
    else:
        bins = categories

    def _apply_shannon(values):
        return mm.shannon_diversity(values, bins, categorical, categories)

    return graph.apply(data, _apply_shannon)


def gini(data: DataFrame | Series, graph: Graph, rng: tuple | list = None):
    """
    Calculates the Gini index of values within neighbours defined in
    ``graph``. Uses ``inequality.gini.Gini`` under the hood.
    Requires '`inequality`' package.

    .. math::

    Parameters
    ----------
    data : DataFrame | Series
        A DataFrame or Series containing the values to be analysed.
    graph : libpysal.graph.Graph
        A spatial weights matrix for the data.
    rng : tuple, list, optional (default (0,100)))
        A two-element sequence containing floats between 0 and 100 (inclusive)
        that are the percentiles over which to compute the range.
        The order of the elements is not important.

    Returns
    ----------
    Series
        A Series containing resulting values.

    Examples
    --------
    >>> tessellation_df['area_Gini'] = mm.gini(tessellation_df['area'],
    ...                                              graph)
    """
    try:
        from inequality.gini import Gini
    except ImportError as err:
        raise ImportError("The 'inequality' package is required.") from err

    if data.min() < 0:
        raise ValueError(
            "Values contain negative numbers. Normalise data before"
            "using momepy.Gini."
        )
    if rng:
        from momepy import limit_range

    def _apply_gini(values):
        if isinstance(values, Series):
            values = values.values
        if rng:
            values = limit_range(values, rng=rng)
        return Gini(values).g

    return graph.apply(data, _apply_gini)


def unique(data: DataFrame | Series, graph: Graph, dropna: bool = True):
    """
    Calculates the number of unique values within neighbours defined in
    ``graph``.

    .. math::


    Parameters
    ----------
    data : DataFrame | Series
        A DataFrame or Series containing the values to be analysed.
    graph : libpysal.graph.Graph
        A spatial weights matrix for the data.
    dropna : bool (default True)
        Don’t include ``NaN`` in the counts of unique values.

    Returns
    ----------
    Series
        A Series containing resulting values.

    Examples
    --------
    >>> tessellation_df['cluster_unique'] = mm.Unique(tessellation_df['cluster'],
    ...                                              graph)
    """

    def _apply_range(values):
        return values.nunique(dropna=dropna)

    return graph.apply(data, _apply_range)
