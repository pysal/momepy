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

    objects = gdf.geometry.copy()

    if shrink != 0:
        mask = objects.geom_type.isin(["Polygon", "MultiPolygon"])
        objects.loc[mask] = objects[mask].buffer(-shrink, cap_style=2, join_style=2)
    objects = objects.segmentize(segment)
    voronoi = shapely.voronoi_polygons(
        shapely.GeometryCollection(objects.values), extend_to=limit
    )
    geoms = gpd.GeoSeries(shapely.get_parts(voronoi), crs=gdf.crs)
    ids_objects, ids_geoms = geoms.sindex.query(objects, predicate="intersects")
    polygons = (
        geoms.iloc[ids_geoms]
        .groupby(objects.index.take(ids_objects))
        .agg(shapely.coverage_union_all)
    )

    if limit is not None:
        return polygons.clip(limit)

    return polygons
