import geopandas as gpd
import pandas as pd
from tqdm import tqdm  # progress bar
from osgeo import ogr
from shapely.wkt import loads
import shapely.geometry
from shapely.geometry import MultiPoint, Point, Polygon, LineString, MultiPolygon
import shapely.ops
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import shapefile


path = "/Users/martin/Strathcloud/Personal Folders/Test data/Royston/Tess/bul.shp"
objects_used = "/Users/martin/Strathcloud/Personal Folders/Test data/Royston/Tess/for_convert.shp"
points_used = "/Users/martin/Strathcloud/Personal Folders/Test data/Royston/Tess/points.shp"
voronoi_network = "/Users/martin/Strathcloud/Personal Folders/Test data/Royston/Tess/voronoi.shp"
print('Loading file.')
objects = gpd.read_file(path)  # load file into geopandas
print('Shapefile loaded.')

objects['geometry'] = tqdm(objects.buffer(-0.1, resolution=1))  # change original geometry with buffered one
print('Buffered.')


# densify geometry beforo Voronoi tesselation
def densify(geom):
    wkt = geom.wkt  # shapely Polygon to wkt
    geom = ogr.CreateGeometryFromWkt(wkt)  # create ogr geometry
    geom.Segmentize(2)  # densify geometry by 2 metres
    wkt2 = geom.ExportToWkt()  # ogr geometry to wkt
    new = loads(wkt2)  # wkt to shapely Polygon
    return new

objects['geometry'] = tqdm(objects['geometry'].map(densify))
print('Densified.')
# Voronoi

# define new numpy.array
voronoi_points = np.empty([1, 2])

# fill array with all points from densified geometry
for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
    poly_ext = row['geometry'].exterior
    point_coords = poly_ext.coords
    row_array = np.array(point_coords)
    voronoi_points = np.concatenate((voronoi_points, row_array))

# delete initial row of array to keep only points from geometry
voronoi_points = voronoi_points[1:]
# make voronoi diagram
voronoi_diagram = Voronoi(voronoi_points)
# generate lines from scipy voronoi output
lines = [LineString(voronoi_diagram.vertices[line]) for line in voronoi_diagram.ridge_vertices if -1 not in line]
# generate convex hull around points to resolve the edge
convex_hull = MultiPoint([Point(i) for i in voronoi_points]).convex_hull.buffer(20)
# generate dataframe with polygons clipped by convex hull
result = pd.DataFrame({'geometry':
                      [poly.intersection(convex_hull) for poly in shapely.ops.polygonize(lines)]})
# generate geoDataFrame of Voronoi polygons
voronoi_polygons = gpd.GeoDataFrame(result, geometry='geometry')
print('Voronoi polygons ready')
voronoi_polygons.crs = {'init': 'epsg:102065'}
#
# df = pd.DataFrame({'id':
#                   [row['OBJECTID'] for index, row in objects.iterrows()], 'geometry':
#                   [row['geometry'].exterior.coords for index, row in objects.iterrows()]})
# gdf = gpd.GeoDataFrame(df, geometry='geometry')
#
# voronoi_with_id = gpd.sjoin(voronoi_polygons, objects, how='inner', op='intersects')
# voronoi_with_id.crs = {'proj': 'tmerc', 'lat_0': 49, 'lon_0': -2, 'k': 0.9996012717, 'x_0':
#                        400000, 'y_0': -100000, 'datum': 'OSGB36', 'units': 'm', 'no_defs': True}
#

# voronoi_plots.crs = {'proj': 'tmerc', 'lat_0': 49, 'lon_0': -2, 'k': 0.9996012717, 'x_0':
#                      400000, 'y_0': -100000, 'datum': 'OSGB36', 'units': 'm', 'no_defs': True}

#
#
# voronoi_polygons.to_file("/Users/martin/Strathcloud/Personal Folders/Test data/Royston/Tess/test_vor2.shp")
#
#

objects.to_file(objects_used)

sf = shapefile.Reader(objects_used)

newType = shapefile.MULTIPOINT
w = shapefile.Writer(newType)
w._shapes.extend(sf.shapes())
for s in w.shapes():
    s.shapeType = newType
w.fields = list(sf.fields)
w.records.extend(sf.records())
w.save(points_used)
print('Points ready.')
points = gpd.read_file(points_used)
points.crs = {'init': 'epsg:102065'}
voronoi_with_id = gpd.sjoin(voronoi_polygons, points, how='inner', op='intersects')
voronoi_with_id.crs = {'init': 'epsg:102065'}
# voronoi_with_id.to_file(voronoi_network)
# objects = gpd.read_file(path)
# objects.crs = {'init': 'epsg:27700'}
# # vor_w_holes = gpd.overlay(voronoi_with_id, objects, how='difference')
# df = gpd.GeoDataFrame(pd.concat([objects, voronoi_with_id], ignore_index=True))


# print('Holes cutted.')
# merged = gpd.overlay(vor_w_holes, objects, how='union')
print('Joined.')
voronoi_plots = voronoi_with_id.dissolve(by='OBJECTID')
print('Dissolved.')

# usedpoints = MultiPoint(voronoi_points)
# df = pd.DataFrame({'geometry': [usedpoints]})
# gdf = gpd.GeoDataFrame(df, geometry='geometry')
voronoi_plots.to_file("/Users/martin/Strathcloud/Personal Folders/Test data/Royston/Tess/uaaaa.shp")
print('Saved.')
#
# merged = voronoi_with_id.append(objects)
# voronoi_plots.to_file("/Users/martin/Strathcloud/Personal Folders/Test data/Royston/Tess/vorplots.shp")
# voronoi_plots = merged.dissolve(by='OBJECTID')


'''
Voronoi tesselation as a substitute of morphological plot

input layer: building footprints
'''

path = "/Users/martin/Strathcloud/Personal Folders/Test data/Royston/Tess/bul.shp"
print('Loading file.')
objects = gpd.read_file(path)  # load file into geopandas
print('Shapefile loaded.')

objects['geometry'] = tqdm(objects.buffer(-0.1))  # change original geometry with buffered one
print('Buffered.')

# densify geometry beforo Voronoi tesselation
def densify(geom):
    wkt = geom.wkt  # shapely Polygon to wkt
    geom = ogr.CreateGeometryFromWkt(wkt)  # create ogr geometry
    geom.Segmentize(2)  # densify geometry by 2 metres
    wkt2 = geom.ExportToWkt()  # ogr geometry to wkt
    new = loads(wkt2)  # wkt to shapely Polygon
    return new

objects['geometry'] = tqdm(objects['geometry'].map(densify))
print('Densified.')
# Voronoi

voronoi_points = np.empty([1, 2])
voronoi_points

for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
    poly_ext = row['geometry'].exterior
    point_coords = poly_ext.coords
    row_array = np.array(point_coords)
    voronoi_points = np.concatenate((voronoi_points, row_array))

voronoi_diagram = Voronoi(voronoi_points[1:])

voronoi_plot_2d(voronoi_diagram)
# NEXT STEP = SAVE VORONOI TESSELATION TO SHAPEFILE
# https://stackoverflow.com/questions/27548363/from-voronoi-tessellation-to-shapely-polygons


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

regions, vertices = voronoi_finite_polygons_2d(voronoi_diagram)

pts = MultiPoint([Point(i) for i in voronoi_points[1:]])
mask = pts.convex_hull.buffer(10, resolution=5, cap_style=1)
new_vertices = []
for region in regions:
    polygon = vertices[region]
    shape = list(polygon.shape)
    shape[0] += 1
    p = Polygon(np.append(polygon, polygon[0]).reshape(*shape)).intersection(mask)
    poly = np.array(list(zip(p.boundary.coords.xy[0][:-1], p.boundary.coords.xy[1][:-1])))
    new_vertices.append(poly)
    plt.fill(*zip(*poly), alpha=0.4)
plt.plot(points[:, 0], points[:, 1], 'ko')
plt.title("Clipped Voronois")
plt.show()
