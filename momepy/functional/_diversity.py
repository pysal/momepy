import warnings

import numpy as np
import pandas as pd
from libpysal.graph import Graph
from libpysal.graph._utils import _percentile_filtration_grouper
from numpy.typing import NDArray
from packaging.version import Version
from pandas import DataFrame, Series

from ..diversity import shannon_diversity, simpson_diversity

try:
    from numba import njit

    HAS_NUMBA = True
except (ModuleNotFoundError, ImportError):
    HAS_NUMBA = False
    from libpysal.common import jit as njit

__all__ = [
    "describe",
    "describe_reached",
    "values_range",
    "theil",
    "simpson",
    "shannon",
    "gini",
    "percentile",
]


def _get_grouper(y, graph):
    return y.take(graph._adjacency.index.codes[1]).groupby(
        graph._adjacency.index.codes[0]
    )


@njit
def _interpolate(values, q):
    weights = values[:, 0]
    group = values[:, 1]
    nan_tracker = np.isnan(group)
    if nan_tracker.all():
        return np.array([float(np.nan) for _ in q])
    group = group[~nan_tracker]
    sorter = np.argsort(group)
    group = group[sorter]
    weights = weights[~nan_tracker][sorter]

    xs = np.cumsum(weights) - 0.5 * weights
    xs = xs / weights.sum()
    ys = group
    interpolate = np.interp(
        [x / 100 for x in q],
        xs,
        ys,
    )
    return interpolate


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

    if nan_tracker.all():
        return [
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ]

    else:
        values = values[~np.isnan(values)]

    if len(values) > 2:
        lower, higher = np.percentile(values, q)
        values = values[(lower <= values) & (values <= higher)]

    mean = np.mean(values)
    n = values.shape[0]
    if n == 1:
        std = np.nan
    else:
        # for pandas compatability
        std = np.sqrt(np.sum(np.abs(values - mean) ** 2) / (n - 1))

    results = [
        n,
        mean,
        np.median(values),
        std,
        np.min(values),
        np.max(values),
        np.sum(values),
    ]

    if include_mode:
        results.append(_mode(values))

    return results


def _compute_stats(grouper, q=None, include_mode=False):
    """
    Fast compute of "count", "mean", "median", "std", "min", "max", "sum" statistics,
    with optional mode and quartile limits.

    Parameters
    ----------
    grouper : pandas.GroupBy
        Groupby Object which specifies the aggregations to be performed.
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
    """
    if q is None:
        stat_ = grouper.agg(["count", "mean", "median", "std", "min", "max", "sum"])
        if include_mode:
            stat_["mode"] = grouper.agg(lambda x: _mode(x.values))
    else:
        agg = grouper.agg(lambda x: _describe(x.values, q=q, include_mode=include_mode))
        stat_ = DataFrame(np.stack(agg.values), index=agg.index)
        cols = ["count", "mean", "median", "std", "min", "max", "sum"]
        if include_mode:
            cols.append("mode")
        stat_.columns = cols

    return stat_


def describe(
    y: NDArray[np.float_] | Series,
    graph: Graph,
    q: tuple[float, float] | None = None,
    include_mode: bool = False,
) -> DataFrame:
    """Describe the distribution of values within a set neighbourhood.

    Given the graph, computes the descriptive statistics of values within the
    neighbourhood of each node. Optionally, the values can be limited to a certain
    quantile range before computing the statistics.

    Notes
    -----
    The index of ``y`` must match the index along which the ``graph`` is
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

    if not HAS_NUMBA:
        warnings.warn(
            "The numba package is used extensively in this function to accelerate the"
            " computation of statistics but it is not installed or  cannot be imported."
            " Without numba, these computations may become slow on large data.",
            UserWarning,
            stacklevel=2,
        )

    if not isinstance(y, Series):
        y = Series(y)

    grouper = y.take(graph._adjacency.index.codes[1]).groupby(
        graph._adjacency.index.codes[0]
    )

    return _compute_stats(grouper, q, include_mode)


def values_range(
    y: Series | NDArray[np.float64], graph: Graph, q: tuple | list = (0, 100)
) -> Series:
    """Calculates the range of values within neighbours defined in ``graph``.

    Adapted from :cite:`dibble2017`.

    Notes
    -----
    The index of ``y`` must match the index along which the ``graph`` is
    built.

    Parameters
    ----------
    data : Series
        A DataFrame or Series containing the values to be analysed.
    graph : libpysal.graph.Graph
        A spatial weights matrix for the data.
    q : tuple, list, optional (default (0,100)))
        A two-element sequence containing floats between 0 and 100 (inclusive)
        that are the percentiles over which to compute the range.
        The order of the elements is not important.

    Returns
    ----------
    Series
        A Series containing resulting values.

    Examples
    --------
    >>> tessellation_df['area_IQR_3steps'] = mm.range(tessellation_df['area'],
    ...                                               graph,
    ...                                               q=(25, 75))
    """

    stats = percentile(y, graph, q=q)
    return stats[max(q)] - stats[min(q)]


def theil(y: Series, graph: Graph, q: tuple | list | None = None) -> Series:
    """Calculates the Theil measure of inequality of values within neighbours defined in
    ``graph``.

    Uses ``inequality.theil.Theil`` under the hood. Requires '`inequality`' package.

    .. math::

        T = \\sum_{i=1}^n \\left(
            \\frac{y_i}{\\sum_{i=1}^n y_i} \\ln \\left[
                N \\frac{y_i} {\\sum_{i=1}^n y_i}
            \\right]
        \\right)

    Notes
    -----
    The index of ``y`` must match the index along which the ``graph`` is
    built.

    Parameters
    ----------
    y : Series
        A DataFrame or Series containing the values to be analysed.
    graph : libpysal.graph.Graph
        A spatial weights matrix for the data.
    q : tuple, list, optional (default (0,100)))
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

    if q:
        grouper = _percentile_filtration_grouper(y, graph._adjacency.index, q=q)
    else:
        grouper = _get_grouper(y, graph)

    result = grouper.apply(lambda x: Theil(x.values).T)
    result.index = graph.unique_ids
    return result


def simpson(
    y: Series,
    graph: Graph,
    binning: str = "HeadTailBreaks",
    gini_simpson: bool = False,
    inverse: bool = False,
    categorical: bool = False,
    **classification_kwds,
) -> Series:
    """Calculates the Simpson's diversity index of values within neighbours defined in
    ``graph``.
    Uses ``mapclassify.classifiers`` under the hood for binning.
    Requires ``mapclassify>=.2.1.0`` dependency.

    .. math::

        \\lambda=\\sum_{i=1}^{R} p_{i}^{2}

    Adapted from :cite:`feliciotti2018`.

    Notes
    -----
    The index of ``y`` must match the index along which the ``graph`` is
    built.

    Parameters
    ----------
    y : Series
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
        bins = classify(y, scheme=binning, **classification_kwds).bins
    else:
        bins = None

    def _apply_simpson_diversity(values):
        return simpson_diversity(
            values,
            bins,
            categorical=categorical,
        )

    result = graph.apply(y, _apply_simpson_diversity)

    if gini_simpson:
        result = 1 - result
    elif inverse:
        result = 1 / result
    return result


def shannon(
    y: Series,
    graph: Graph,
    binning: str = "HeadTailBreaks",
    categorical: bool = False,
    categories: list | None = None,
    **classification_kwds,
) -> Series:
    """Calculates the Shannon index of values within neighbours defined in
    ``graph``.
    Uses ``mapclassify.classifiers`` under the hood
    for binning. Requires ``mapclassify>=.2.1.0`` dependency.

    .. math::

        H^{\\prime}=-\\sum_{i=1}^{R} p_{i} \\ln p_{i}

    Notes
    -----
    The index of ``y`` must match the index along which the ``graph`` is
    built.

    Parameters
    ----------
    y : Series
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
        categories = y.unique()

    if not categorical:
        try:
            from mapclassify import classify
        except ImportError as err:
            raise ImportError(
                "The 'mapclassify >= 2.4.2` package is required."
            ) from err
        bins = classify(y, scheme=binning, **classification_kwds).bins
    else:
        bins = categories

    def _apply_shannon(values):
        return shannon_diversity(values, bins, categorical, categories)

    return graph.apply(y, _apply_shannon)


def gini(y: Series, graph: Graph, q: tuple | list | None = None) -> Series:
    """Calculates the Gini index of values within neighbours defined in ``graph``.
    Uses ``inequality.gini.Gini`` under the hood. Requires '`inequality`' package.

    Notes
    -----
    The index of ``y`` must match the index along which the ``graph`` is
    built.

    Parameters
    ----------
    y :  Series
        A DataFrame or Series containing the values to be analysed.
    graph : libpysal.graph.Graph
        A spatial weights matrix for the data.
    q : tuple, list, optional (default (0,100)))
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

    if y.min() < 0:
        raise ValueError(
            "Values contain negative numbers. Normalise data before"
            "using momepy.Gini."
        )
    if q:
        grouper = _percentile_filtration_grouper(y, graph._adjacency.index, q=q)
    else:
        grouper = _get_grouper(y, graph)

    result = grouper.apply(lambda x: Gini(x.values).g)
    result.index = graph.unique_ids
    return result


def describe_reached(
    y: np.ndarray | Series,
    graph_index: np.ndarray | Series,
    result_index: pd.Index = None,
    graph: Graph = None,
    q: tuple | list | None = None,
    include_mode: bool = False,
) -> DataFrame:
    """Describe the distribution of values reached on a neighbourhood graph.

    Given a neighborhood graph or a grouping, computes the descriptive statistics
    of values reached. Optionally, the values can be limited to a certain
    quantile range before computing the statistics.

    The statistics calculated are count, sum, mean, median, std.
    Optionally, mode can be calculated, or the statistics can be calculated in
    quantiles ``q``.

    The neighbourhood is defined in ``graph``. If ``graph`` is ``None``,
    the function will assume topological distance ``0`` (element itself)
    and ``result_index`` is required in order to arrange the results.
    If ``graph``, the results are arranged according to the spatial weights ordering.

    Adapted from :cite:`hermosilla2012` and :cite:`feliciotti2018`.

    Notes
    -----
    The numba package is used extensively in this function to accelerate the computation
    of statistics. Without numba, these computations may become slow on large data.

    Parameters
    ----------
    y : Series | numpy.array
        A Series or numpy.array containing values to analyse.
    graph_index : Series | numpy.array
        The unique ID that specifies the aggregation
        of ``y`` objects to ``graph`` groups.
    result_index : pd.Index (default None)
        An index that specifies how to order the results when ``graph`` is None.
        When ``graph`` is given, the index is derived from its unique IDs.
    graph : libpysal.graph.Graph (default None)
        A spatial weights matrix of the element ``y`` is grouped into.
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

    Examples
    --------
    >>> res = mm.describe_reached(
    ...         tessellation['area'], tessellation['nID'] , graph=streets_q1
    ...     )
    >>> streets["tessalations_reached"] = res['count']
    >>> streets["tessalations_reached_area"] = res['sum']

    """

    if Version(pd.__version__) <= Version("2.1.0"):
        raise ImportError("pandas 2.1.0 or newer is required to use this function.")

    if not HAS_NUMBA:
        warnings.warn(
            "The numba package is used extensively in this function to accelerate the"
            " computation of statistics but it is not installed or  cannot be imported."
            " Without numba, these computations may become slow on large data.",
            UserWarning,
            stacklevel=2,
        )

    param_err = ValueError(
        "One of result_index or graph has to be specified, but not both."
    )
    # case where params are none
    if (result_index is None) and (graph is None):
        raise param_err
    elif result_index is None:
        result_index = graph.unique_ids
    elif graph is None:
        result_index = result_index
    # case where both params are passed
    else:
        raise param_err

    # series indice needs renaming, since multiindices
    # without explicit names cannot be joined
    if isinstance(y, np.ndarray):
        y = pd.Series(y, name="obs_index")
    else:
        y = y.rename_axis("obs_index")

    if isinstance(graph_index, np.ndarray):
        graph_index = pd.Series(graph_index, name="neighbor")
    else:
        graph_index = graph_index.rename("neighbor")

    # aggregate data
    if graph is None:
        grouper = y.groupby(graph_index)

    else:
        df_multi_index = y.to_frame().set_index(graph_index, append=True).swaplevel()
        combined_index = graph.adjacency.index.join(df_multi_index.index).dropna()
        grouper = y.loc[combined_index.get_level_values(-1)].groupby(
            combined_index.get_level_values(0)
        )

    stats = _compute_stats(grouper, q, include_mode)

    result = pd.DataFrame(
        np.full((result_index.shape[0], stats.shape[1]), np.nan), index=result_index
    )
    result.loc[stats.index.values] = stats.values
    result.columns = stats.columns
    # fill only counts with zeros, other stats are NA
    result.loc[:, "count"] = result.loc[:, "count"].fillna(0)
    result.index.names = result_index.names

    return result


def percentile(
    y: Series,
    graph: Graph,
    q: tuple | list = [25, 50, 75],
):
    """Calculates linearly weighted percentiles of ``y`` values using
    the neighbourhoods and weights defined in ``graph``.

    The specific interpolation method implemented is "hazen".

    Parameters
    ----------
    y : Series
        A Series containing the values to be analysed.
    graph : libpysal.graph.Graph
        A spatial weights matrix for the data.
    q : array-like (default [25, 50, 75])
        The percentiles to return.

    Returns
    -------
    Dataframe
        A Dataframe with columns as the results for each percentile

    Examples
    --------
    >>> percentiles_df = mm.percentile(tessellation_df['area'],
    ...                                 graph)
    """

    weights = graph._adjacency.values
    vals = y.loc[graph._adjacency.index.get_level_values(1)]
    vals = np.vstack((weights, vals)).T
    vals = DataFrame(vals, columns=["weights", "values"])
    grouper = vals.groupby(graph._adjacency.index.get_level_values(0))
    q = tuple(q)
    stats = grouper.apply(lambda x: _interpolate(x.values, q))
    result = DataFrame(np.stack(stats), columns=q, index=stats.index)
    result.loc[graph.isolates] = np.nan
    return result
