import warnings

import geopandas as gpd
import numpy as np
from libpysal.graph import Graph
from numpy.typing import NDArray
from pandas import DataFrame, Series

__all__ = ["describe", "unweighted_percentile", "linearly_weighted_percentiles"]


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


def unweighted_percentile(
    data: DataFrame | Series,
    graph: Graph,
    percentiles: tuple | list = [25, 50, 75],
    interpolation: str = "midpoint",
):
    """
    Calculates the percentiles of values within
    neighbours defined in ``graph``.

    Parameters
    ----------
    data : DataFrame | Series
        A DataFrame or Series containing the values to be analysed.
    graph : libpysal.graph.Graph
        A spatial weights matrix for the data.
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

    Returns
    --------
    Dataframe
        A Dataframe with columns as the results for each percentile

    Examples
    --------
    >>> percentiles_df = mm.unweighted_percentile(tessellation_df['area'],
    ...                                 graph)
    """

    from numpy.lib import NumpyVersion

    if NumpyVersion(np.__version__) >= "1.22.0":
        method = {"method": interpolation}
    else:
        method = {"interpolation": interpolation}

    def _apply_percentile(values):
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            return np.nanpercentile(values, percentiles, **method)

    stats = graph.apply(data, _apply_percentile)
    result = DataFrame(np.stack(stats), columns=percentiles, index=stats.index)
    return result


def linearly_weighted_percentiles(
    data: DataFrame | Series,
    geometry: gpd.GeoDataFrame,
    graph: Graph,
    percentiles: tuple | list = [25, 50, 75],
):
    """
    Calculates the percentiles of values within
    neighbours defined in ``graph`` using weighted distance decay. Linear
    inverse distance between the centroids in ``geometry`` is used as a weight.

    Parameters
    ----------
    data : DataFrame | Series
        A DataFrame or Series containing the values to be analysed.
    graph : libpysal.graph.Graph
        A spatial weights matrix for the data.
    geometry : GeoDataFrame
        The original GeoDataFrame.
    percentiles : array-like (default [25, 50, 75])
        The percentiles to return.

    Returns
    --------
    Dataframe
        A Dataframe with columns as the results for each percentile

    Examples
    --------
    >>> percentiles_df = mm.linearly_weighted_percentiles(
    ...                                 tessellation_df['area'],
    ...                                 tessellation_df.geometry,
    ...                                 graph)
    """

    from scipy.spatial.distance import cdist

    centroids = geometry.centroid
    xys = np.vstack((centroids.x, centroids.y)).T
    data = np.hstack((xys, data.values.reshape(data.shape[0], -1)))

    def _apply_weights(group):
        focal = np.where(group.index.values == group.name)[0]
        neighbours = np.where(group.index.values != group.name)[0]

        distances = cdist(
            group.iloc[focal, [0, 1]], group.iloc[neighbours, [0, 1]].values
        ).flatten()
        distance_decay = 1 / distances

        vals = group.iloc[neighbours, 2:].values.flatten()
        sorter = np.argsort(vals)
        vals = vals[sorter]

        nan_mask = np.isnan(vals)
        if nan_mask.all():
            return np.array([np.nan] * len(percentiles))

        sample_weight = distance_decay[sorter][~nan_mask]
        weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
        weighted_quantiles /= np.sum(sample_weight)
        interpolate = np.interp(
            [x / 100 for x in percentiles],
            weighted_quantiles,
            vals[~nan_mask],
        )
        return interpolate

    stats = graph.apply(data, _apply_weights).values
    result = DataFrame(np.stack(stats), columns=percentiles, index=graph.unique_ids)
    return result
