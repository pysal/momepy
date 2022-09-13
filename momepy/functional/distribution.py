def shared_walls(gdf):
    """
    Calculate the length of shared walls of adjacent elements (typically buildings)

    .. math::
        \\textit{length of shared walls}

    Note that data needs to be topologically correct. Overlapping polygons will lead to
    incorrect results.

    Adapted from :cite:`hamaina2012a`.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing gdf to analyse

    Returns
    -------
    pandas.Series
        Series containing resulting values

    Examples
    --------
    >>> buildings_df['swr'] = momepy.shared_walls(buildings_df)
    """
    inp, res = gdf.sindex.query_bulk(gdf.geometry, predicate="intersects")
    left = gdf.geometry.take(inp).reset_index(drop=True)
    right = gdf.geometry.take(res).reset_index(drop=True)
    intersections = left.intersection(right).length
    results = intersections.groupby(inp).sum().reset_index(
        drop=True
    ) - gdf.geometry.length.reset_index(drop=True)
    results.index = gdf.index

    return results


def shared_walls_ratio(gdf, perimeters=None):
    """
    Calculate shared walls ratio of adjacent elements (typically buildings)

    .. math::
        \\textit{length of shared walls} \\over perimeter

    Note that data needs to be topologically correct. Overlapping polygons will lead to
    incorrect results.

    Adapted from :cite:`hamaina2012a`.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing gdf to analyse
    perimeters : str, list, np.array, pd.Series (default None, optional)
        the name of the dataframe column, ``np.array``, or ``pd.Series`` where is
        stored perimeter value

    Returns
    -------
    pandas.Series
        Series containing resulting values

    Examples
    --------
    >>> buildings_df['swr'] = momepy.shared_walls_ratio(buildings_df)
    >>> buildings_df['swr'][10]
    0.3424804411228673
    """
    if perimeters is None:
        perimeters = gdf.geometry.length
    elif isinstance(perimeters, str):
        perimeters = gdf[perimeters]

    return shared_walls(gdf) / perimeters
