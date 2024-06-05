import warnings

import numpy as np
import pandas as pd
from libpysal.graph import Graph
from packaging.version import Version
from pandas import DataFrame, Series

try:
    # from numba import njit

    HAS_NUMBA = True
except (ModuleNotFoundError, ImportError):
    HAS_NUMBA = False
    # from libpysal.common import jit as njit

from libpysal.utils import _compute_stats, _limit_range, _percentile_filtration_grouper

__all__ = ["describe_agg", "describe_reached_agg"]


def percentile_limited_group_grouper(y, group_index, q=(25, 75)):
    """Carry out a filtration of group members based on \\
    quantiles, specified in ``q``"""
    grouper = y.groupby(group_index)
    to_keep = grouper.transform(_limit_range, q[0], q[1], engine="numba").values.astype(
        bool
    )
    filtered_grouper = y[to_keep].groupby(group_index[to_keep])
    return filtered_grouper


def describe_agg():
    pass


def describe_reached_agg(
    y: np.ndarray | Series,
    graph_index: np.ndarray | Series,
    result_index: pd.Index = None,
    graph: Graph = None,
    q: tuple | list | None = None,
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
            grouper = _percentile_filtration_grouper(y, combined_index, q=q)

    stats = _compute_stats(grouper, q)

    result = pd.DataFrame(
        np.full((result_index.shape[0], stats.shape[1]), np.nan), index=result_index
    )
    result.loc[stats.index.values] = stats.values
    result.columns = stats.columns
    # fill only counts with zeros, other stats are NA
    result.loc[:, "count"] = result.loc[:, "count"].fillna(0)
    result.index.names = result_index.names

    return result
