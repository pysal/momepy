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

from libpysal.graph._utils import (
    _compute_stats,
    _limit_range,
)

__all__ = [
    "mean_deviation",
    "describe_agg",
    "describe_reached_agg",
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


def _percentile_limited_group_grouper(y, group_index, q=(25, 75)):
    """Carry out a filtration of group members based on \\
    quantiles, specified in ``q``"""
    grouper = y.reset_index(drop=True).groupby(group_index)
    if HAS_NUMBA:
        to_keep = grouper.transform(
            _limit_range, q[0], q[1], engine="numba"
        ).values.astype(bool)
    else:
        to_keep = grouper.transform(
            lambda x: _limit_range(x.values, x.index, q[0], q[1])
        ).values.astype(bool)
    filtered_grouper = y[to_keep].groupby(group_index[to_keep])
    return filtered_grouper


def describe_agg(
    y: NDArray[np.float64] | Series,
    aggregation_key: NDArray[np.float64] | Series,
    q: tuple[float, float] | list[float] | None = None,
    statistics: list[str] | None = None,
) -> DataFrame:
    """Describe the distribution of values within the groups of an aggregation.

    The desired statistics to compute can be passed to ``statistics``.
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
    aggregation_key : Series | numpy.array
        The unique ID that specifies the aggregation
        of ``y`` objects to groups.
    q : tuple[float, float] | None, optional
        Tuple of percentages for the percentiles to compute. Values must be between 0
        and 100 inclusive. When set, values below and above the percentiles will be
        discarded before computation of the average. The percentiles are computed for
        each neighborhood. By default None.
    statistics : list[str]
        A list of stats functions to pass to groupby.agg.

    Returns
    -------
    DataFrame

    Examples
    --------
    >>> path = momepy.datasets.get_path("bubenec")
    >>> buildings = geopandas.read_file(path, layer="buildings")
    >>> streets = geopandas.read_file(path, layer="streets")
    >>> buildings["street_index"] = momepy.get_nearest_street(buildings, streets)
    >>> buildings.head()
       uID                                           geometry  street_index
    0    1  POLYGON ((1603599.221 6464369.816, 1603602.984...           0.0
    1    2  POLYGON ((1603042.88 6464261.498, 1603038.961 ...          33.0
    2    3  POLYGON ((1603044.65 6464178.035, 1603049.192 ...          10.0
    3    4  POLYGON ((1603036.557 6464141.467, 1603036.969...           8.0
    4    5  POLYGON ((1603082.387 6464142.022, 1603081.574...           8.0

    >>> momepy.describe_agg(buildings.area, buildings["street_index"]).head()   # doctest: +SKIP
                  count         mean       median         std         min          max          sum  nunique        mode
    street_index
    0.0             9.0   366.827019   339.636871  266.747247   68.336193   800.045495  3301.443174      9.0   68.336193
    1.0             1.0   618.447036   618.447036         NaN  618.447036   618.447036   618.447036      1.0  618.447036
    2.0            12.0   504.523575   535.973108  318.660691   92.280807  1057.998520  6054.282903     12.0   92.280807
    5.0             5.0  1150.865099  1032.693716  580.660030  673.015192  2127.752228  5754.325496      5.0  673.015192
    6.0             7.0   662.179187   662.192603  291.397747  184.798661  1188.294675  4635.254306      7.0  184.798661

    The result can be directly assigned a columns of the ``streets`` GeoDataFrame.

    To eliminate the effect of outliers, you can take into account only values within a
    specified percentile range (``q``). At the same time, you can specify only a subset
    of statistics to compute:

    >>> momepy.describe_agg(
    ...     buildings.area,
    ...     buildings["street_index"],
    ...     q=(10, 90),
    ...     statistics=["mean", "std"],
    ... ).head()
                        mean         std
    street_index
    0.0           347.580212  219.797123
    1.0           618.447036         NaN
    2.0           476.592190  206.011102
    5.0           984.519359  203.718644
    6.0           652.432194   32.829824
    """  # noqa: E501

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

    stats = _compute_stats(grouper, to_compute=statistics)

    # fill only counts with zeros, other stats are NA
    if "count" in stats.columns:
        stats.loc[:, "count"] = stats.loc[:, "count"].fillna(0)

    return stats


def describe_reached_agg(
    y: NDArray[np.float64] | Series,
    graph_index: NDArray[np.float64] | Series,
    graph: Graph,
    q: tuple[float, float] | list[float] | None = None,
    statistics: list[str] | None = None,
) -> DataFrame:
    """Describe the distribution of values reached on a neighbourhood graph.

    Given a neighborhood graph or a grouping, computes the descriptive statistics
    of values reached. Optionally, the values can be limited to a certain
    quantile range before computing the statistics.

    The statistics calculated are count, sum, mean, median, std, nunique, mode.
    The desired statistics to compute can be passed to ``statistics``

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
    statistics : list[str]
        A list of stats functions to pass to groupby.agg.
    Returns
    -------
    DataFrame

    Examples
    --------
    >>> from libpysal import graph
    >>> path = momepy.datasets.get_path("bubenec")
    >>> buildings = geopandas.read_file(path, layer="buildings")
    >>> streets = geopandas.read_file(path, layer="streets")
    >>> buildings["street_index"] = momepy.get_nearest_street(buildings, streets)
    >>> buildings.head()
       uID                                           geometry  street_index
    0    1  POLYGON ((1603599.221 6464369.816, 1603602.984...           0.0
    1    2  POLYGON ((1603042.88 6464261.498, 1603038.961 ...          33.0
    2    3  POLYGON ((1603044.65 6464178.035, 1603049.192 ...          10.0
    3    4  POLYGON ((1603036.557 6464141.467, 1603036.969...           8.0
    4    5  POLYGON ((1603082.387 6464142.022, 1603081.574...           8.0

    >>> queen_contig = graph.Graph.build_contiguity(streets, rook=False)
    >>> queen_contig
    <Graph of 35 nodes and 148 nonzero edges indexed by
     [0, 1, 2, 3, 4, ...]>

    >>> momepy.describe_reached_agg(
    ...     buildings.area,
    ...     buildings["street_index"],
    ...     queen_contig,
    ... ).head()  # doctest: +SKIP
       count        mean      median         std         min          max           sum  nunique        mode
    0   43.0  643.595418  633.692589  412.563790   53.851509  2127.752228  27674.602973     43.0   53.851509
    1   41.0  735.058515  662.921280  381.827737   51.246377  2127.752228  30137.399128     41.0   51.246377
    2   50.0  636.304006  625.190488  450.182157   53.851509  2127.752228  31815.200298     50.0   53.851509
    3    6.0  405.782514  370.352071  334.848563   57.138700   863.828420   2434.695086      6.0   57.138700
    4    1.0  683.514930  683.514930         NaN  683.514930   683.514930    683.514930      1.0  683.514930

    The result can be directly assigned a columns of the ``streets`` GeoDataFrame.

    To eliminate the effect of outliers, you can take into account only values within a
    specified percentile range (``q``). At the same time, you can specify only a subset
    of statistics to compute:

    >>> momepy.describe_reached_agg(
    ...     buildings.area,
    ...     buildings["street_index"],
    ...     queen_contig,
    ...     q=(10, 90),
    ...     statistics=["mean", "std"],
    ... ).head()
            mean         std
    0  619.104840  250.369496
    1  721.441808  216.516469
    2  597.379925  297.213321
    3  378.431992  274.631290
    4  683.514930         NaN
    """  # noqa: E501

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

    stats = _compute_stats(grouper, statistics)

    result = pd.DataFrame(
        np.full((graph.unique_ids.shape[0], stats.shape[1]), np.nan),
        index=graph.unique_ids,
    )
    result.loc[stats.index.values] = stats.values
    result.columns = stats.columns
    # fill only counts with zeros, other stats are NA
    if "count" in result.columns:
        result.loc[:, "count"] = result.loc[:, "count"].fillna(0)
    result.index.name = None

    return result


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
    y : Series
        A DataFrame or Series containing the values to be analysed.
    graph : libpysal.graph.Graph
        A spatial weights matrix for the data.
    q : tuple, list, optional (default (0,100)))
        A two-element sequence containing floats between 0 and 100 (inclusive)
        that are the percentiles over which to compute the range.
        The order of the elements is not important.

    Returns
    -------
    Series
        A Series containing resulting values.

    Examples
    --------
    >>> from libpysal import graph
    >>> path = momepy.datasets.get_path("bubenec")
    >>> buildings = geopandas.read_file(path, layer="buildings")
    >>> buildings.head()
       uID                                           geometry
    0    1  POLYGON ((1603599.221 6464369.816, 1603602.984...
    1    2  POLYGON ((1603042.88 6464261.498, 1603038.961 ...
    2    3  POLYGON ((1603044.65 6464178.035, 1603049.192 ...
    3    4  POLYGON ((1603036.557 6464141.467, 1603036.969...
    4    5  POLYGON ((1603082.387 6464142.022, 1603081.574...

    Define spatial graph:

    >>> knn5 = graph.Graph.build_knn(buildings.centroid, k=5)
    >>> knn5
    <Graph of 144 nodes and 720 nonzero edges indexed by
     [0, 1, 2, 3, 4, ...]>

    Range of building area within 5 nearest neighbors:

    >>> momepy.values_range(buildings.area, knn5)
    focal
    0        559.745602
    1        444.997770
    2      10651.932677
    3        365.239452
    4        339.585788
            ...
    139      769.179096
    140      721.444718
    141      996.921755
    142      119.708607
    143      798.344284
    Length: 144, dtype: float64

    To eliminate the effect of outliers, you can take into account only values within a
    specified percentile range (``q``).

    >>> momepy.values_range(buildings.area, knn5, q=(25, 75))
    focal
    0       258.656230
    1       113.990829
    2      2878.811586
    3        92.005635
    4        87.637833
            ...
    139     587.139513
    140     325.726611
    141     621.315615
    142      34.446110
    143     488.967863
    Length: 144, dtype: float64
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
    -------
    Series
        A Series containing resulting values.

    Examples
    --------
    >>> from libpysal import graph
    >>> path = momepy.datasets.get_path("bubenec")
    >>> buildings = geopandas.read_file(path, layer="buildings")
    >>> buildings.head()
       uID                                           geometry
    0    1  POLYGON ((1603599.221 6464369.816, 1603602.984...
    1    2  POLYGON ((1603042.88 6464261.498, 1603038.961 ...
    2    3  POLYGON ((1603044.65 6464178.035, 1603049.192 ...
    3    4  POLYGON ((1603036.557 6464141.467, 1603036.969...
    4    5  POLYGON ((1603082.387 6464142.022, 1603081.574...

    Define spatial graph:

    >>> knn5 = graph.Graph.build_knn(buildings.centroid, k=5)
    >>> knn5
    <Graph of 144 nodes and 720 nonzero edges indexed by
     [0, 1, 2, 3, 4, ...]>

    Theil index of building area within 5 nearest neighbors:

    >>> momepy.theil(buildings.area, knn5)
    focal
    0      0.106079
    1      0.023256
    2      0.800522
    3      0.016015
    4      0.013829
            ...
    139    0.547522
    140    0.041755
    141    0.098827
    142    0.159690
    143    0.068131
    Length: 144, dtype: float64

    To eliminate the effect of outliers, you can take into account only values within a
    specified percentile range (``q``).

    >>> momepy.theil(buildings.area, knn5, q=(25, 75))
    focal
    0      1.144550e-02
    1      3.121656e-06
    2      1.295882e-02
    3      1.772730e-07
    4      2.913017e-06
            ...
    139    5.402548e-01
    140    6.658486e-03
    141    3.330720e-02
    142    1.433583e-03
    143    2.096421e-02
    Length: 144, dtype: float64
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
    >>> from libpysal import graph
    >>> path = momepy.datasets.get_path("bubenec")
    >>> buildings = geopandas.read_file(path, layer="buildings")
    >>> buildings.head()
       uID                                           geometry
    0    1  POLYGON ((1603599.221 6464369.816, 1603602.984...
    1    2  POLYGON ((1603042.88 6464261.498, 1603038.961 ...
    2    3  POLYGON ((1603044.65 6464178.035, 1603049.192 ...
    3    4  POLYGON ((1603036.557 6464141.467, 1603036.969...
    4    5  POLYGON ((1603082.387 6464142.022, 1603081.574...

    Define spatial graph:

    >>> knn5 = graph.Graph.build_knn(buildings.centroid, k=5)
    >>> knn5
    <Graph of 144 nodes and 720 nonzero edges indexed by
     [0, 1, 2, 3, 4, ...]>

    Simpson index of building area within 5 nearest neighbors:

    >>> momepy.simpson(buildings.area, knn5)
    focal
    0      1.00
    1      0.68
    2      0.36
    3      0.68
    4      0.68
        ...
    139    0.68
    140    0.44
    141    0.44
    142    1.00
    143    0.52
    Length: 144, dtype: float64

    In some occasions, you may want to override the binning method:

    >>> momepy.simpson(buildings.area, knn5, binning="fisher_jenks", k=8)
    focal
    0      0.28
    1      0.68
    2      0.36
    3      0.68
    4      0.68
        ...
    139    0.44
    140    0.28
    141    0.28
    142    1.00
    143    0.20
    Length: 144, dtype: float64

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
    -------
    Series
        A Series containing resulting values.

    Examples
    --------
    >>> from libpysal import graph
    >>> path = momepy.datasets.get_path("bubenec")
    >>> buildings = geopandas.read_file(path, layer="buildings")
    >>> buildings.head()
       uID                                           geometry
    0    1  POLYGON ((1603599.221 6464369.816, 1603602.984...
    1    2  POLYGON ((1603042.88 6464261.498, 1603038.961 ...
    2    3  POLYGON ((1603044.65 6464178.035, 1603049.192 ...
    3    4  POLYGON ((1603036.557 6464141.467, 1603036.969...
    4    5  POLYGON ((1603082.387 6464142.022, 1603081.574...

    Define spatial graph:

    >>> knn5 = graph.Graph.build_knn(buildings.centroid, k=5)
    >>> knn5
    <Graph of 144 nodes and 720 nonzero edges indexed by
     [0, 1, 2, 3, 4, ...]>

    Shannon index of building area within 5 nearest neighbors:

    >>> momepy.shannon(buildings.area, knn5)
    focal
    0     -0.000000
    1      0.500402
    2      1.054920
    3      0.500402
    4      0.500402
            ...
    139    0.500402
    140    0.950271
    141    0.950271
    142   -0.000000
    143    0.673012
    Length: 144, dtype: float64

    In some occasions, you may want to override the binning method:

    >>> momepy.shannon(buildings.area, knn5, binning="fisher_jenks", k=8)
    focal
    0      1.332179
    1      0.500402
    2      1.054920
    3      0.500402
    4      0.500402
            ...
    139    0.950271
    140    1.332179
    141    1.332179
    142   -0.000000
    143    1.609438
    Length: 144, dtype: float64
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
    -------
    Series
        A Series containing resulting values.

    Examples
    --------
    >>> from libpysal import graph
    >>> path = momepy.datasets.get_path("bubenec")
    >>> buildings = geopandas.read_file(path, layer="buildings")
    >>> buildings.head()
       uID                                           geometry
    0    1  POLYGON ((1603599.221 6464369.816, 1603602.984...
    1    2  POLYGON ((1603042.88 6464261.498, 1603038.961 ...
    2    3  POLYGON ((1603044.65 6464178.035, 1603049.192 ...
    3    4  POLYGON ((1603036.557 6464141.467, 1603036.969...
    4    5  POLYGON ((1603082.387 6464142.022, 1603081.574...

    Define spatial graph:

    >>> knn5 = graph.Graph.build_knn(buildings.centroid, k=5)
    >>> knn5
    <Graph of 144 nodes and 720 nonzero edges indexed by
     [0, 1, 2, 3, 4, ...]>

    Gini index of building area within 5 nearest neighbors:

    >>> momepy.gini(buildings.area, knn5)
    focal
    0      0.228493
    1      0.102110
    2      0.605867
    3      0.085589
    4      0.080435
            ...
    139    0.525724
    140    0.156737
    141    0.239009
    142    0.259808
    143    0.204820
    Length: 144, dtype: float64

    To eliminate the effect of outliers, you can take into account only values within a
    specified percentile range (``q``).

    >>> momepy.gini(buildings.area, knn5, q=(25, 75))
    focal
    0      0.073817
    1      0.001264
    2      0.077521
    3      0.000321
    4      0.001264
            ...
    139    0.505618
    140    0.055096
    141    0.130501
    142    0.025522
    143    0.110987
    Length: 144, dtype: float64
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


def percentile(
    y: Series,
    graph: Graph,
    q: tuple | list = [25, 50, 75],
) -> DataFrame:
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
    >>> from libpysal import graph
    >>> path = momepy.datasets.get_path("bubenec")
    >>> buildings = geopandas.read_file(path, layer="buildings")
    >>> buildings.head()
       uID                                           geometry
    0    1  POLYGON ((1603599.221 6464369.816, 1603602.984...
    1    2  POLYGON ((1603042.88 6464261.498, 1603038.961 ...
    2    3  POLYGON ((1603044.65 6464178.035, 1603049.192 ...
    3    4  POLYGON ((1603036.557 6464141.467, 1603036.969...
    4    5  POLYGON ((1603082.387 6464142.022, 1603081.574...

    Define spatial graph:

    >>> knn5 = graph.Graph.build_knn(buildings.centroid, k=5)
    >>> knn5
    <Graph of 144 nodes and 720 nonzero edges indexed by
     [0, 1, 2, 3, 4, ...]>

    Percentiles of building area within 5 nearest neighbors:

    >>> momepy.percentile(buildings.area, knn5).head()
                   25          50           75
    focal
    0      347.252959  427.819360   605.909188
    1      621.834862  641.629131   735.825691
    2      622.262074  903.746689  3501.073660
    3      621.834862  641.629131   713.840496
    4      621.834862  641.987211   709.472695

    Optionally, you can specify which percentile values shall be computed.

    >>> momepy.percentile(buildings.area, knn5, q=[10, 90]).head()
                   10            90
    focal
    0      123.769329    683.514930
    1      564.160901   1009.158671
    2      564.160901  11216.093578
    3      564.160901    929.400353
    4      564.160901    903.746689
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


def mean_deviation(y: Series, graph: Graph) -> Series:
    """Calculate the mean deviation of each ``y`` value and its graph neighbours.

    .. math::
        \\frac{1}{n}\\sum_{i=1}^n dev_i=\\frac{dev_1+dev_2+\\cdots+dev_n}{n}

    Parameters
    ----------
    y : Series
        A Series containing the values to be analysed.
    graph : libpysal.graph.Graph
        Graph representing spatial relationships between elements.

    Returns
    -------
    Series

    Examples
    --------
    >>> from libpysal import graph
    >>> path = momepy.datasets.get_path("bubenec")
    >>> buildings = geopandas.read_file(path, layer="buildings")
    >>> buildings.head()
       uID                                           geometry
    0    1  POLYGON ((1603599.221 6464369.816, 1603602.984...
    1    2  POLYGON ((1603042.88 6464261.498, 1603038.961 ...
    2    3  POLYGON ((1603044.65 6464178.035, 1603049.192 ...
    3    4  POLYGON ((1603036.557 6464141.467, 1603036.969...
    4    5  POLYGON ((1603082.387 6464142.022, 1603081.574...

    Define spatial graph:

    >>> knn5 = graph.Graph.build_knn(buildings.centroid, k=5)
    >>> knn5
    <Graph of 144 nodes and 720 nonzero edges indexed by
     [0, 1, 2, 3, 4, ...]>

    Mean deviation of building area and area of 5 nearest neighbors:

    >>> momepy.mean_deviation(buildings.area, knn5)
    0        281.179149
    1      10515.948995
    2       2240.706061
    3        230.360732
    4         68.719810
            ...
    139      259.180720
    140      517.496703
    141      331.849751
    142       25.297225
    143      654.691897
    Length: 144, dtype: float64
    """

    inp = graph._adjacency.index.get_level_values(0)
    res = graph._adjacency.index.get_level_values(1)

    itself = inp == res
    inp = inp[~itself]
    res = res[~itself]

    left = y.loc[inp].reset_index(drop=True)
    right = y.loc[res].reset_index(drop=True)
    deviations = (left - right).abs()

    vals = deviations.groupby(inp).mean()

    result = Series(np.nan, index=y.index)
    result.loc[vals.index] = vals.values
    return result
