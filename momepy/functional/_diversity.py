import warnings

import numpy as np
import pandas as pd
from libpysal.graph import Graph
from numpy.typing import NDArray
from packaging.version import Version
from pandas import DataFrame, Series

try:
    from numba import njit

    HAS_NUMBA = True
except (ModuleNotFoundError, ImportError):
    HAS_NUMBA = False
    from libpysal.common import jit as njit

__all__ = ["describe", "describe_reached"]


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
def _describe(values, q, include_mode=False, include_nunique=False):
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

    if include_nunique:
        results.append(np.unique(values).shape[0])

    return results


def _compute_stats(grouper, q=None, include_mode=False, include_nunique=False):
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
        if include_nunique:
            stat_["nunique"] = grouper.agg("nunique")
    else:
        agg = grouper.agg(
            lambda x: _describe(
                x.values,
                q=q,
                include_mode=include_mode,
                include_nunique=include_nunique,
            )
        )
        stat_ = DataFrame(np.stack(agg.values), index=agg.index)
        cols = ["count", "mean", "median", "std", "min", "max", "sum"]
        if include_mode:
            cols.append("mode")
        if include_nunique:
            cols.append("nunique")
        stat_.columns = cols

    return stat_


def describe(
    y: NDArray[np.float_] | Series,
    graph: Graph,
    q: tuple[float, float] | None = None,
    include_mode: bool = False,
    include_nunique: bool = False,
) -> DataFrame:
    """Describe the distribution of values within a set neighbourhood.

    Given the graph, computes the descriptive statistics of values within the
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

    return _compute_stats(grouper, q, include_mode, include_nunique)


def describe_reached(
    y: np.ndarray | Series,
    graph_index: np.ndarray | Series,
    result_index: pd.Index = None,
    graph: Graph = None,
    q: tuple | list | None = None,
    include_mode: bool = False,
    include_nunique: bool = False,
) -> DataFrame:
    """Describe the distribution of values reached on a neighbourhood graph.

    Given a neighborhood graph or a grouping, computes the descriptive statistics
    of values reached. Optionally, the values can be limited to a certain
    quantile range before computing the statistics.

    The statistics calculated are count, sum, mean, median, std.
    Optionally, mode and number of unique elements can be calculated,
    or the statistics can be calculated in quantiles ``q``.

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
    include_nunique : False
        Compute number of uniue elements along with other statistics. Default is False.

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

    stats = _compute_stats(grouper, q, include_mode, include_nunique)

    result = pd.DataFrame(
        np.full((result_index.shape[0], stats.shape[1]), np.nan), index=result_index
    )
    result.loc[stats.index.values] = stats.values
    result.columns = stats.columns
    # fill only counts with zeros, other stats are NA
    result.loc[:, "count"] = result.loc[:, "count"].fillna(0)
    result.index.names = result_index.names

    return result
