import numpy as np
import pandas as pd
from libpysal.graph import Graph
from packaging.version import Version
from pandas import DataFrame, Series

try:
    from numba import njit

    HAS_NUMBA = True
except (ModuleNotFoundError, ImportError):
    HAS_NUMBA = False
    from libpysal.common import jit as njit

from libpysal.graph._utils import (
    _compute_stats,
    _limit_range,
    _percentile_filtration_grouper,
)

__all__ = [
    "describe_agg",
    "describe_reached_agg",
    "percentile",
]


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


def _percentile_limited_group_grouper(y, group_index, q=(25, 75)):
    """Carry out a filtration of group members based on \\
    quantiles, specified in ``q``"""
    grouper = y.reset_index(drop=True).groupby(group_index)
    to_keep = grouper.transform(_limit_range, q[0], q[1], engine="numba").values.astype(
        bool
    )
    filtered_grouper = y[to_keep].groupby(group_index[to_keep])
    return filtered_grouper


def describe_agg(
    y: np.ndarray | Series,
    aggregation_key: np.ndarray | Series,
    result_index: pd.Index = None,
    q: tuple[float, float] | list[float] | None = None,
    to_compute: list[str] | None = None,
) -> DataFrame:
    """Describe the distribution of values within the groups of an aggregation.

    The desired statistics to compute can be passed to ``to_compute``.
    By default the statistics calculated are count,
    sum, mean, median, std, nunique, mode.

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
        An index that specifies how to order the results.
    q : tuple[float, float] | None, optional
        Tuple of percentages for the percentiles to compute. Values must be between 0
        and 100 inclusive. When set, values below and above the percentiles will be
        discarded before computation of the average. The percentiles are computed for
        each neighborhood. By default None.
    to_compute : list[str]
        A list of stats functions to pass to groupby.agg.

    Returns
    -------
    DataFrame

    Examples
    --------
    >>> res = mm.describe_agg(
    ...         tessellation['area'], tessellation['nID'] ,
    ...         result_index=df_streets.index
    ...     )
    >>> streets["tessalations_reached"] = res['count']
    >>> streets["tessalations_reached_area"] = res['sum']

    """

    if Version(pd.__version__) <= Version("2.1.0"):
        raise ImportError("pandas 2.1.0 or newer is required to use this function.")

    # series indice needs renaming, since multiindices
    # without explicit names cannot be joined
    if isinstance(y, np.ndarray):
        y = pd.Series(y)

    if isinstance(aggregation_key, np.ndarray):
        aggregation_key = pd.Series(aggregation_key)

    # aggregate data
    if q is None:
        grouper = y.groupby(aggregation_key)
    else:
        grouper = _percentile_limited_group_grouper(y, aggregation_key, q=q)

    stats = _compute_stats(grouper, to_compute=to_compute)

    if result_index is None:
        result_index = stats.index

    # post processing to have the same behaviour as describe_reached_agg
    result = pd.DataFrame(
        np.full((result_index.shape[0], stats.shape[1]), np.nan), index=result_index
    )
    result.loc[stats.index.values] = stats.values
    result.columns = stats.columns
    # fill only counts with zeros, other stats are NA
    result.loc[:, "count"] = result.loc[:, "count"].fillna(0)
    result.index.names = result_index.names

    return result


def describe_reached_agg(
    y: np.ndarray | Series,
    graph_index: np.ndarray | Series,
    graph: Graph,
    q: tuple[float, float] | list[float] | None = None,
    to_compute: list[str] | None = None,
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
    graph : libpysal.graph.Graph (default None)
        A spatial weights matrix of the element ``y`` is grouped into.
    q : tuple[float, float] | None, optional
        Tuple of percentages for the percentiles to compute. Values must be between 0
        and 100 inclusive. When set, values below and above the percentiles will be
        discarded before computation of the average. The percentiles are computed for
        each neighborhood. By default None.
    to_compute : list[str]
        A list of stats functions to pass to groupby.agg.
    Returns
    -------
    DataFrame

    Examples
    --------
    >>> res = mm.describe_reached_agg(
    ...         tessellation['area'], tessellation['nID'] , graph=streets_q1
    ...     )
    >>> streets["tessalations_reached"] = res['count']
    >>> streets["tessalations_reached_area"] = res['sum']

    """

    if Version(pd.__version__) <= Version("2.1.0"):
        raise ImportError("pandas 2.1.0 or newer is required to use this function.")

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
    df_multi_index = y.to_frame().set_index(graph_index, append=True).swaplevel()
    combined_index = graph.adjacency.index.join(df_multi_index.index).dropna()

    if q is None:
        grouper = y.loc[combined_index.get_level_values(-1)].groupby(
            combined_index.get_level_values(0)
        )
    else:
        grouper = _percentile_filtration_grouper(y, combined_index, q=q)

    result = _compute_stats(grouper, to_compute)

    # fill only counts with zeros, other stats are NA
    result.loc[:, "count"] = result.loc[:, "count"].fillna(0)

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
