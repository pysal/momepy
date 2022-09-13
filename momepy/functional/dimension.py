import pandas as pd


def perimeter(gdf, geom_col=None):
    """
    Calculates perimeter of each object in given GeoDataFrame. It can be used for any
    suitable element (building footprint, plot, tessellation, block).

    It is a simple wrapper for GeoPandas ``.length`` for the consistency of momepy.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects to analyse
    geom_col : str, default None
        name of the geometry column to measure if it is not an active one

    Returns
    -------
    pandas.Series
        Series containing resulting values


    Examples
    --------
    >>> buildings = gpd.read_file(momepy.datasets.get_path('bubenec'),
    ...                           layer='buildings')
    >>> buildings['perimeter'] = momepy.perimeter(buildings)
    >>> buildings.perimeter[0]
    137.18630991119903
    """
    if geom_col:
        return [gdf[geom_col]].length

    return gdf.length


def volume(gdf, heights, areas=None):
    """
    Calculates volume of each object in given GeoDataFrame based on its height and area.

    .. math::
        area * height

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects to analyse
    heights : str, list, np.array, pd.Series
        the name of the dataframe column, ``np.array``, or ``pd.Series``
        where is stored height value
    areas : str, list, np.array, pd.Series (default None)
        the name of the dataframe column, ``np.array``, or ``pd.Series``
        where is stored area value. If set to None, function will calculate areas
        during the process without saving them separately.

    Returns
    -------
    pandas.Series
        Series containing resulting values

    Examples
    --------
    >>> buildings['volume'] = momepy.volume(buildings, heights='height_col')
    >>> buildings.volume[0]
    7285.5749470443625

    >>> buildings['volume'] = momepy.volume(buildings, heights='height_col',
    ...                                     areas='area_col')
    >>> buildings.volume[0]
    7285.5749470443625
    """
    if isinstance(heights, str):
        heights = gdf[heights]

    if isinstance(areas, str):
        areas = gdf[areas]

    else:
        areas = gdf.area

    return areas * heights


def perimeter_wall(gdf, w=None):
    """
    Calculate the perimeter wall length the joined structure.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects to analyse
    w : libpysal.weights, optional
        spatial weights matrix based in positional index
        If None, Queen contiguity matrix will be calculated based on gdf.
        It is to denote adjacent buildings.

    Returns
    -------
    pandas.Series
        Series containing resulting values


    Examples
    --------
    >>> buildings_df['wall_length'] = mm.perimeter_wall(buildings_df)

    Notes
    -----
    It might take a while to compute this character.
    """

    if w is None:
        from libpysal.weights import Queen

        w = Queen.from_dataframe(gdf, silence_warnings=True)

    # dict to store walls for each uID
    walls = {}
    components = pd.Series(w.component_labels, index=range(len(gdf)))
    geom = gdf.geometry

    for i in range(gdf.shape[0]):
        # if the id is already present in walls, continue (avoid repetition)
        if i in walls:
            continue
        else:
            comp = w.component_labels[i]
            to_join = components[components == comp].index
            joined = geom.iloc[to_join]
            # buffer to avoid multipolygons where buildings touch by corners only
            dissolved = joined.buffer(0.01).unary_union
            for b in to_join:
                walls[b] = dissolved.exterior.length

    results_list = [walls[i] for i in range(gdf.shape[0])]
    return pd.Series(results_list, index=gdf.index)
