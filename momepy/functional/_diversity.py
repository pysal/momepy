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
def _mode(values, index):  # noqa: ARG001
    """Custom mode function for numba."""
    array = np.sort(values.ravel())
    mask = np.empty(array.shape, dtype=np.bool_)
    mask[:1] = True
    mask[1:] = array[1:] != array[:-1]
    unique = array[mask]
    idx = np.nonzero(mask)[0]
    idx = np.append(idx, mask.size)
    counts = np.diff(idx)
    return unique[np.argmax(counts)]


@njit
def _limit_range(values, index, low, high):  # noqa: ARG001
    nan_tracker = np.isnan(values)

    if (not nan_tracker.all()) & (len(values[~nan_tracker]) > 2):
        lower, higher = np.percentile(values, (low, high))
    else:
        return ~nan_tracker

    return (lower <= values) & (values <= higher)


def percentile_limited_graph_grouper(y, graph_adjacency_index, q=(25, 75)):
    """Carry out a filtration of graph neighbours based on quantiles, \\
        specified in ``q``"""
    grouper = (
        y.take(graph_adjacency_index.codes[-1])
        .reset_index(drop=True)
        .groupby(graph_adjacency_index.codes[0])
    )
    to_keep = grouper.transform(_limit_range, q[0], q[1], engine="numba").values.astype(
        bool
    )
    filtered_grouper = y.take(graph_adjacency_index.codes[-1][to_keep]).groupby(
        graph_adjacency_index.codes[0][to_keep]
    )
    return filtered_grouper


def percentile_limited_group_grouper(y, group_index, q=(25, 75)):
    """Carry out a filtration of group members based on \\
    quantiles, specified in ``q``"""
    grouper = y.groupby(group_index)
    to_keep = grouper.transform(_limit_range, q[0], q[1], engine="numba").values.astype(
        bool
    )
    filtered_grouper = y[to_keep].groupby(group_index[to_keep])
    return filtered_grouper


def _compute_stats(grouper, to_compute: list[str] | None = None):
    """Fast compute of "count", "mean", "median", "std", "min", "max", \\
    "sum", "nunique" and "mode" within a grouper object. Using numba.

    Parameters
    ----------
    grouper : pandas.GroupBy
        Groupby Object which specifies the aggregations to be performed.
    to_compute : List[str]
        A list of stats functions to pass to groupby.agg

    Returns
    -------
    DataFrame
    """
    if to_compute is None:
        to_compute = [
            "count",
            "mean",
            "median",
            "std",
            "min",
            "max",
            "sum",
            "nunique",
            "mode",
        ]

    agg_to_compute = [f for f in to_compute if f != "mode"]
    stat_ = grouper.agg(agg_to_compute)
    if "mode" in to_compute:
        stat_["mode"] = grouper.agg(_mode, engine="numba")

    return stat_


def describe(
    y: NDArray[np.float_] | Series,
    graph: Graph,
    q: tuple[float, float] | None = None,
    to_compute: list[str] | None = None,
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
    to_compute : List[str] | None
        A list of stats functions to compute. If None, compute all
        available functions - "count", "mean", "median",
        "std", "min", "max", "sum", "nunique", "mode". By default None.

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

    if q is None:
        grouper = y.take(graph._adjacency.index.codes[1]).groupby(
            graph._adjacency.index.codes[0]
        )
    else:
        grouper = percentile_limited_graph_grouper(y, graph._adjacency.index, q=q)

    return _compute_stats(grouper, to_compute)


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

    The statistics calculated are count, sum, mean, median, std, nunique, mode.
    The desired statistics to compute can be passed to ``to_compute``

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
        Compute the number of unique elements along with other statistics.
        Default is False.

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
        if q is None:
            grouper = y.groupby(graph_index)
        else:
            grouper = percentile_limited_group_grouper(y, graph_index, q=q)

    else:
        df_multi_index = y.to_frame().set_index(graph_index, append=True).swaplevel()
        combined_index = graph.adjacency.index.join(df_multi_index.index).dropna()

        if q is None:
            grouper = y.loc[combined_index.get_level_values(-1)].groupby(
                combined_index.get_level_values(0)
            )
        else:
            grouper = percentile_limited_graph_grouper(y, combined_index, q=q)

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
