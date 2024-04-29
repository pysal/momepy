import warnings

import numpy as np
import pandas as pd
from libpysal.graph import Graph
from numpy.typing import NDArray
from pandas import DataFrame, Series

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
        values.shape[0],
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


def _compute_stats(grouper, q=None, include_mode=False):
    """
    Fast compute of "count", "mean", "median", "std", "min", "max", "sum" statistics,
    with optional mode and quartile limits.

    Parameters
    ----------
    grouper : pandas.GroupBy
        Groupby Object which specifies the aggregations to be performed
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
        stat_ = DataFrame(zip(*agg, strict=True)).T
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
    if not isinstance(y, Series):
        y = Series(y)

    grouper = y.take(graph._adjacency.index.codes[1]).groupby(
        graph._adjacency.index.codes[0]
    )

    return _compute_stats(grouper, q, include_mode)


def describe_reached(
    y, y_group, result_index=None, graph=None, q=None, include_mode=False
) -> DataFrame:
    """
    Calculates statistics of ``y`` objects reached on a street network.
    Requires a ``y_group`` that links the ``y`` objects to streets (or ``graph``)
    assigned beforehand (e.g. using :py:func:`momepy.get_network_id`).
    The number of elements within neighbourhood defined in ``graph``. If
    ``graph`` is ``None``, it will assume topological distance ``0`` (element itself)
    and ``result_index`` is required in order to arrange the results.
    If ``graph``, the results are arranged according to the spatial weights ordering.
    The statistics calculated are count, sum, mean, median, std.
    Optionally, mode can be calculated, or the statistics can be calculated in
    quantiles ``q``.

    Adapted from :cite:`hermosilla2012` and :cite:`feliciotti2018`.

    Parameters
    ----------
    y : DataFrame | Series | numpy.array
        A GeoDataFrame containing objects to analyse.
    y_group : Series | numpy.array
        The unique ID that specifies the aggregation
        of ``y`` objects to ``graph`` groups.
    result_index : pd.Index (default None)
        An index that specifies how to order the results.
    graph : libpysal.graph.Graph (default None)
        A spatial weights matrix of the streets.
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

    if (result_index is None) and (graph is None):
        raise ValueError(
            "One of result_index or graph has to be specified, but not both."
        )
    elif result_index is None:
        result_index = graph.unique_ids
    elif graph is None:
        result_index = result_index
    else:
        raise ValueError(
            "One of result_index or graph has to be specified, but not both."
        )

    if isinstance(y, np.ndarray):
        y = pd.Series(y, name="obs_index")
    elif isinstance(y, Series):
        y = y.to_frame()

    if isinstance(y_group, np.ndarray):
        y_group = pd.Series(y_group, name="neighbor")
    else:
        y_group = y_group.rename("neighbor")

    # aggregate data
    if graph is None:
        grouper = y.groupby(y_group)

    else:
        # y.index needs renaming, since multiindixes
        # without explicit names cannot be joined
        df_multi_index = (
            y.rename_axis("obs_index").set_index(y_group, append=True).swaplevel()
        )
        combined_index = graph.adjacency.index.join(df_multi_index.index).dropna()
        grouper = y.loc[combined_index.get_level_values(-1)].groupby(
            combined_index.get_level_values(0)
        )

    stats = _compute_stats(grouper, q, include_mode)

    result = pd.DataFrame(
        np.zeros((result_index.shape[0], stats.shape[1])), index=result_index
    )
    result.loc[stats.index.values] = stats.values
    result.columns = stats.columns
    result = result.fillna(0)

    return result
