import numpy as np
from libpysal.graph import Graph
from numpy.typing import NDArray
from pandas import DataFrame, Series
from scipy import stats

from ..utils import limit_range

__all__ = ["describe"]


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

    def _describe(values, q, include_mode=False):
        """Helper function to calculate average."""
        values = limit_range(values.values, q)

        results = [
            np.mean(values),
            np.median(values),
            np.std(values),
            np.min(values),
            np.max(values),
            np.sum(values),
        ]
        if include_mode:
            results.append(stats.mode(values, keepdims=False)[0])
        return results

    if not isinstance(y, Series):
        y = Series(y)

    grouper = y.take(graph._adjacency.index.codes[1]).groupby(
        graph._adjacency.index.codes[0]
    )

    if q is None:
        stat_ = grouper.agg(["mean", "median", "std", "min", "max", "sum"])
        if include_mode:
            stat_["mode"] = grouper.agg(lambda x: stats.mode(x, keepdims=False)[0])
    else:
        agg = grouper.agg(_describe, q=q, include_mode=include_mode)
        stat_ = DataFrame(zip(*agg, strict=True)).T
        cols = ["mean", "median", "std", "min", "max", "sum"]
        if include_mode:
            cols.append("mode")
        stat_.columns = cols

    return stat_
