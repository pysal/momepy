import geopandas as gpd
import shapely


def tessellation(
    gdf,
    limit=None,
    shrink=0.4,
    segment=0.5,
):
    if gdf.crs and gdf.crs.is_geographic:
        raise ValueError(
            "Geometry is in a geographic CRS. "
            "Use 'GeoDataFrame.to_crs()' to re-project geometries to a "
            "projected CRS before using Tessellation.",
        )

    objects = shapely.set_precision(gdf.geometry.copy(), 0.00001)
    geom_types = objects.geom_type
    mask_poly = geom_types.isin(["Polygon", "MultiPolygon"])
    mask_line = objects.geom_type.isin(["LineString", "MultiLineString"])

    if shrink != 0 and mask_poly.any():
        objects[mask_poly] = (
            objects[mask_poly]
            .buffer(-shrink, cap_style=2, join_style=2)
            .segmentize(segment)
        )

    # exclude conincident points
    if mask_line.any():
        objects[mask_line] = (
            objects.loc[mask_line]
            .segmentize(segment)
            .get_coordinates(index_parts=True)
            .drop_duplicates(keep=False)
            .groupby(level=0)
            .apply(shapely.multipoints)
        )

    voronoi = shapely.voronoi_polygons(
        shapely.GeometryCollection(objects.values), extend_to=limit
    )
    geoms = gpd.GeoSeries(shapely.get_parts(voronoi), crs=gdf.crs)
    ids_objects, ids_geoms = geoms.sindex.query(objects, predicate="intersects")
    polygons = (
        geoms.iloc[ids_geoms]
        .groupby(objects.index.take(ids_objects))
        .agg(shapely.coverage_union_all)
    ).set_crs(gdf.crs)

    if limit is not None:
        to_be_clipped = polygons.sindex.query(limit.boundary, "intersects")
        polygons.iloc[to_be_clipped] = polygons.iloc[to_be_clipped].intersection(limit)

    return polygons
