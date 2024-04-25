import geopandas as gpd
from geopandas import GeoDataFrame, GeoSeries
from libpysal.graph import Graph
from packaging.version import Version
from pandas import Series
from shapely import buffer, get_num_interior_rings, unary_union

__all__ = ["courtyards"]

GPD_013 = Version(gpd.__version__) >= Version("0.13")


def courtyards(geometry: GeoDataFrame | GeoSeries, graph: Graph) -> Series:
    """Calculate the number of courtyards within the joined structure.

    Adapted from :cite:`schirmer2015`.

    Parameters
    ----------
    geometry : GeoDataFrame
        A GeoDataFrame containing objects to analyse.
    graph : libpysal.graph.Graph
        A spatial weights matrix for the geodataframe,
        it is used to denote adjacent buildings.
    Returns
    ----------
    series : Series
        A Series containing the resulting values.

    Examples
    --------
    >>> courtyards = mm.calculate_courtyards(buildings_df, graph)"""

    # helper function to carry out the per group calculations
    def _calculate_courtyards(group):
        return get_num_interior_rings(unary_union(buffer(group.values, 0.01)))

    ## calculate per group courtyards
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
