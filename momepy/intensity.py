import shapely
from geopandas import GeoDataFrame, GeoSeries
from libpysal.graph import Graph
from pandas import Series

__all__ = ["courtyards"]


def courtyards(
    geometry: GeoDataFrame | GeoSeries, graph: Graph, buffer: float = 0.01
) -> Series:
    """Calculate the number of courtyards within the joined structure.

    Adapted from :cite:`schirmer2015`.

    Parameters
    ----------
    geometry : GeoDataFrame
        A GeoDataFrame containing objects to analyse.
    graph : libpysal.graph.Graph
        A spatial weights matrix for the geodataframe,
        it is used to denote adjacent buildings.
    buffer: float
        Buffer value for the geometry. It can be used
        to account for topological problems.

    Returns
    -------
    Series
        A Series containing the resulting values.

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

    >>> contiguity = graph.Graph.build_contiguity(buildings)
    >>> contiguity
    <Graph of 144 nodes and 248 nonzero edges indexed by
     [0, 1, 2, 3, 4, ...]>

    >>> momepy.courtyards(buildings, contiguity)
    0      0
    1      1
    2      1
    3      1
    4      1
        ..
    139    0
    140    0
    141    0
    142    0
    143    1
    Length: 144, dtype: int32
    """

    def _calculate_courtyards(group):
        """helper function to carry out the per group calculations"""
        return shapely.get_num_interior_rings(
            shapely.union_all(shapely.buffer(group.values, buffer))
        )

    # calculate per group courtyards
    temp_result = (
        geometry["geometry"]
        .groupby(graph.component_labels.values)
        .apply(_calculate_courtyards)
    )
    # assign group courtyard values to each building in the group
    ## note: this is faster than using transform directly
    result = Series(
        temp_result.loc[graph.component_labels.values].values, index=geometry.index
    )

    return result
