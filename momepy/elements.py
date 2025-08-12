import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from packaging.version import Version
from shapely.geometry.base import BaseGeometry
from shapely.ops import polygonize

__all__ = [
    "enclosures",
    "get_network_ratio",
]

GPD_GE_10 = Version(gpd.__version__) >= Version("1.0dev")


def get_network_ratio(df, edges, initial_buffer=500):
    """
    Link polygons to network edges based on the proportion of overlap (if a cell
    intersects more than one edge). Useful if you need to link enclosed tessellation to
    street network. Ratios can be used as weights when linking network-based values
    to cells. For a purely distance-based link use :func:`momepy.get_network_id`.
    Links are based on the integer position of edge (``iloc``).

    Parameters
    ----------
    df : GeoDataFrame
        A GeoDataFrame containing objects to snap (typically enclosed tessellation).
    edges : GeoDataFrame
        A GeoDataFrame containing a street network.
    initial_buffer : float
        The initial buffer used to link non-intersecting cells.

    Returns
    -------
    result : DataFrame
        The resultant DataFrame.

    See also
    --------
    momepy.get_network_id
    momepy.get_node_id

    Examples
    --------
    >>> links = mm.get_network_ratio(enclosed_tessellation, streets)
    >>> links.head()
      edgeID_keys                              edgeID_values
    0        [34]                                      [1.0]
    1     [0, 34]  [0.38508998545027145, 0.6149100145497285]
    2        [32]                                        [1]
    3         [0]                                      [1.0]
    4        [26]                                        [1]
    """

    (df_ix, edg_ix), dist = edges.sindex.nearest(
        df.geometry, max_distance=initial_buffer, return_distance=True
    )

    touching = dist < 0.1

    intersections = (
        df.iloc[df_ix[touching]]
        .intersection(edges.buffer(0.0001).iloc[edg_ix[touching]], align=False)
        .reset_index()
    )

    mask = intersections.area > 0.0001

    df_ix_touching = df_ix[touching][mask]
    lengths = intersections[mask].area
    grouped = lengths.groupby(df_ix_touching)
    totals = grouped.sum()
    ints_vect = []
    for name, group in grouped:
        ratios = group / totals.loc[name]
        ints_vect.append(
            {edg_ix[touching][item[0]]: item[1] for item in ratios.items()}
        )

    ratios = pd.Series(ints_vect, index=df.index[list(grouped.groups.keys())])

    near = []
    df_ix_non = df_ix[~touching]
    grouped = pd.Series(dist[~touching]).groupby(df_ix_non)
    for _, group in grouped:
        near.append({edg_ix[~touching][group.idxmin()]: 1.0})

    near = pd.Series(near, index=df.index[list(grouped.groups.keys())])

    ratios = pd.concat([ratios, near])

    nans = df[~df.index.isin(ratios.index)]
    if not nans.empty:
        df_ix, edg_ix = edges.sindex.nearest(
            nans.geometry, return_all=False, max_distance=None
        )
        additional = pd.Series([{i: 1.0} for i in edg_ix], index=nans.index)

        ratios = pd.concat([ratios, additional])

    result = pd.DataFrame()
    result["edgeID_keys"] = ratios.apply(lambda d: list(d.keys()))
    result["edgeID_values"] = ratios.apply(lambda d: list(d.values()))

    return result


def enclosures(
    primary_barriers,
    limit=None,
    additional_barriers=None,
    enclosure_id="eID",
    clip=False,
):
    """
    Generate enclosures based on passed barriers. Enclosures are areas enclosed from
    all sides by at least one type of a barrier. Barriers are typically roads,
    railways, natural features like rivers and other water bodies or coastline.
    Enclosures are a result of polygonization of the  ``primary_barrier`` and ``limit``
    and its subdivision based on additional_barriers.

    Parameters
    ----------
    primary_barriers : GeoDataFrame, GeoSeries
        A GeoDataFrame or GeoSeries containing primary barriers.
        (Multi)LineString geometry is expected.
    limit : GeoDataFrame, GeoSeries, shapely geometry (default None)
        A GeoDataFrame, GeoSeries or shapely geometry containing external limit
        of enclosures, i.e. the area which gets partitioned. If ``None`` is passed,
        the internal area of ``primary_barriers`` will be used.
    additional_barriers : GeoDataFrame
        A GeoDataFrame or GeoSeries containing additional barriers.
        (Multi)LineString geometry is expected.
    enclosure_id : str (default 'eID')
        The name of the ``enclosure_id`` (to be created).
    clip : bool (default False)
        If ``True``, returns enclosures with representative point within the limit
        (if given). Requires ``limit`` composed of Polygon or MultiPolygon geometries.

    Returns
    -------
    enclosures : GeoDataFrame
       A GeoDataFrame containing enclosure geometries and ``enclosure_id``.

    Examples
    --------
    >>> enclosures = mm.enclosures(streets, admin_boundary, [railway, rivers])
    """
    if limit is not None:
        if isinstance(limit, BaseGeometry):
            limit = gpd.GeoSeries([limit], crs=primary_barriers.crs)
        if limit.geom_type.isin(["Polygon", "MultiPolygon"]).any():
            limit_b = limit.boundary
        else:
            limit_b = limit
        barriers = pd.concat([primary_barriers.geometry, limit_b.geometry])
    else:
        barriers = primary_barriers
    unioned = barriers.union_all() if GPD_GE_10 else barriers.unary_union
    polygons = polygonize(unioned)
    enclosures = gpd.GeoSeries(list(polygons), crs=primary_barriers.crs)

    if additional_barriers is not None:
        if not isinstance(additional_barriers, list):
            raise TypeError(
                "`additional_barriers` expects a list of GeoDataFrames "
                f"or GeoSeries. Got {type(additional_barriers)}."
            )
        additional = pd.concat([gdf.geometry for gdf in additional_barriers])

        inp, res = enclosures.sindex.query(additional.geometry, predicate="intersects")
        unique = np.unique(res)

        new = []

        for i in unique:
            poly = enclosures.array[i]  # get enclosure polygon
            crossing = inp[res == i]  # get relevant additional barriers
            buf = shapely.buffer(poly, 0.01)  # to avoid floating point errors
            crossing_ins = shapely.intersection(
                buf, additional.array[crossing]
            )  # keeping only parts of additional barriers within polygon
            union = shapely.union_all(
                np.append(crossing_ins, shapely.boundary(poly))
            )  # union
            polygons = shapely.get_parts(shapely.polygonize([union]))  # polygonize
            within = shapely.covered_by(
                polygons, buf
            )  # keep only those within original polygon
            new += list(polygons[within])

        final_enclosures = pd.concat(
            [
                gpd.GeoSeries(enclosures).drop(unique),
                gpd.GeoSeries(new, crs=primary_barriers.crs),
            ]
        ).reset_index(drop=True)

        final_enclosures = gpd.GeoDataFrame(
            {enclosure_id: range(len(final_enclosures))}, geometry=final_enclosures
        )

    else:
        final_enclosures = gpd.GeoDataFrame(
            {enclosure_id: range(len(enclosures))}, geometry=enclosures
        )

    if clip and limit is not None:
        if not limit.geom_type.isin(["Polygon", "MultiPolygon"]).all():
            raise TypeError(
                "`limit` requires a GeoDataFrame or GeoSeries with Polygon or "
                "MultiPolygon geometry to be used with `clip=True`."
            )
        _, encl_index = final_enclosures.representative_point().sindex.query(
            limit.geometry, predicate="contains"
        )
        keep = np.unique(encl_index)
        return final_enclosures.iloc[keep]

    return final_enclosures
