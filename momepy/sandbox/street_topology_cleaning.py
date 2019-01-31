import geopandas as gpd
from shapely.geometry import Point
import shapely
from tqdm import tqdm

streets = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Character sets/s_street_shape.shp")


def network_false_nodes(streets):

    for i in range(2):
        print('Iteration {} out of 2'.format(i + 1))
        sindex = streets.sindex

        false_points = []
        print('Identifying false points...')
        for idx, row in tqdm(streets.iterrows(), total=streets.shape[0]):
            line = row['geometry']
            l_coords = list(line.coords)
            # network_w = network.drop(idx, axis=0)['geometry']  # ensure that it wont intersect itself
            start = Point(l_coords[0])
            end = Point(l_coords[-1])

            # find out whether ends of the line are connected or not
            possible_first_index = list(sindex.intersection(start.bounds))
            possible_first_matches = streets.iloc[possible_first_index]
            possible_first_matches_clean = possible_first_matches.drop(idx, axis=0)
            real_first_matches = possible_first_matches_clean[possible_first_matches_clean.intersects(start)]

            possible_second_index = list(sindex.intersection(end.bounds))
            possible_second_matches = streets.iloc[possible_second_index]
            possible_second_matches_clean = possible_second_matches.drop(idx, axis=0)
            real_second_matches = possible_second_matches_clean[possible_second_matches_clean.intersects(end)]

            if len(real_first_matches) == 1:
                false_points.append(start)
            if len(real_second_matches) == 1:
                false_points.append(end)

        false_xy = []
        for p in false_points:
            false_xy.append([p.x, p.y])

        false_xy_unique = [list(x) for x in set(tuple(x) for x in false_xy)]

        false_unique = []
        for p in false_xy_unique:
            false_unique.append(Point(p[0], p[1]))

        geoms = streets.geometry

        print('Merging segments...')
        for point in tqdm(false_unique):
            matches = list(geoms[geoms.intersects(point)].index)
            idx = max(geoms.index) + 1
            try:
                multiline = geoms[matches[0]].union(geoms[matches[1]])
                linestring = shapely.ops.linemerge(multiline)
                geoms = geoms.append(gpd.GeoSeries(linestring, index=[idx]))
                geoms = geoms.drop(matches)
            except IndexError:
                import warnings
                warnings.warn('An exception during merging occured. Lines at point [{x}, {y}] were not merged.'.format(x=point.x, y=point.y))

        geoms_gdf = gpd.GeoDataFrame(geometry=geoms)
        geoms_gdf.crs = streets.crs
        streets = geoms_gdf
    return streets

s = network_false_nodes(streets)
streets.to_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Tests/190124/topo_cl.shp")
mp = gpd.read_file('/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Tests/190124/mp.shp')
