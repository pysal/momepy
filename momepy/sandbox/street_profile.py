import geopandas as gpd
from tqdm import tqdm

buildings = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Tests/190123/blg.shp")
streets = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Tests/190123/str.shp")


from shapely.geometry import LineString, Point
import math
import pandas as pd
import numpy as np
import operator

height_column = 'pdbHei'

print('Calculating street profile widths...')

# http://wikicode.wikidot.com/get-angle-of-line-between-two-points
# https://glenbambrick.com/tag/perpendicular/
# angle between two points
def _getAngle(pt1, pt2):
    x_diff = pt2.x - pt1.x
    y_diff = pt2.y - pt1.y
    return math.degrees(math.atan2(y_diff, x_diff))

# start and end points of chainage tick
# get the first end point of a tick
def _getPoint1(pt, bearing, dist):
    angle = bearing + 90
    bearing = math.radians(angle)
    x = pt.x + dist * math.cos(bearing)
    y = pt.y + dist * math.sin(bearing)
    return Point(x, y)

# get the second end point of a tick
def _getPoint2(pt, bearing, dist):
    bearing = math.radians(bearing)
    x = pt.x + dist * math.cos(bearing)
    y = pt.y + dist * math.sin(bearing)
    return Point(x, y)

# distance between each points
distance = 10
# the length of each tick
tick_length = 50
sindex = buildings.sindex

results_list = []
heights_list = []

for idx, row in tqdm(streets.iterrows(), total=streets.shape[0]):
    # list to hold all the point coords
    list_points = []
    # set the current distance to place the point
    current_dist = distance
    # make shapely MultiLineString object
    shapely_line = row.geometry
    # get the total length of the line
    line_length = shapely_line.length
    # append the starting coordinate to the list
    list_points.append(Point(list(shapely_line.coords)[0]))
    # https://nathanw.net/2012/08/05/generating-chainage-distance-nodes-in-qgis/
    # while the current cumulative distance is less than the total length of the line
    while current_dist < line_length:
        # use interpolate and increase the current distance
        list_points.append(shapely_line.interpolate(current_dist))
        current_dist += distance
    # append end coordinate to the list
    list_points.append(Point(list(shapely_line.coords)[-1]))

    ticks = []
    for num, pt in enumerate(list_points, 1):
        # start chainage 0
        if num == 1:
            angle = _getAngle(pt, list_points[num])
            line_end_1 = _getPoint1(pt, angle, tick_length / 2)
            angle = _getAngle(line_end_1, pt)
            line_end_2 = _getPoint2(line_end_1, angle, tick_length)
            tick1 = LineString([(line_end_1.x, line_end_1.y), (pt.x, pt.y)])
            tick2 = LineString([(line_end_2.x, line_end_2.y), (pt.x, pt.y)])
            ticks.append([tick1, tick2])

        # everything in between
        if num < len(list_points) - 1:
            angle = _getAngle(pt, list_points[num])
            line_end_1 = _getPoint1(list_points[num], angle, tick_length / 2)
            angle = _getAngle(line_end_1, list_points[num])
            line_end_2 = _getPoint2(line_end_1, angle, tick_length)
            tick1 = LineString([(line_end_1.x, line_end_1.y), (list_points[num].x, list_points[num].y)])
            tick2 = LineString([(line_end_2.x, line_end_2.y), (list_points[num].x, list_points[num].y)])
            ticks.append([tick1, tick2])

        # end chainage
        if num == len(list_points):
            angle = _getAngle(list_points[num - 2], pt)
            line_end_1 = _getPoint1(pt, angle, tick_length / 2)
            angle = _getAngle(line_end_1, pt)
            line_end_2 = _getPoint2(line_end_1, angle, tick_length)
            tick1 = LineString([(line_end_1.x, line_end_1.y), (pt.x, pt.y)])
            tick2 = LineString([(line_end_2.x, line_end_2.y), (pt.x, pt.y)])
            ticks.append([tick1, tick2])
    widths = []
    heights = []
    for duo in ticks:
        width = []
        for tick in duo:
            possible_intersections_index = list(sindex.intersection(tick.bounds))
            possible_intersections = buildings.iloc[possible_intersections_index]
            real_intersections = possible_intersections.intersects(tick)
            get_height = buildings.iloc[list(real_intersections.index)]
            possible_int = get_height.exterior.intersection(tick)

            if possible_int.any():
                true_int = []
                for one in list(possible_int.index):
                    if possible_int[one].type == 'Point':
                        true_int.append(possible_int[one])
                    elif possible_int[one].type == 'MultiPoint':
                        true_int.append(possible_int[one][0])
                        true_int.append(possible_int[one][1])

                if len(true_int) > 1:
                    distances = []
                    ix = 0
                    for p in true_int:
                        distance = p.distance(Point(tick.coords[-1]))
                        distances.append(distance)
                        ix = ix + 1
                    minimal = min(distances)
                    width.append(minimal)
                else:
                    width.append(true_int[0].distance(Point(tick.coords[-1])))

                indices = {}
                for idx, row in get_height.iterrows():
                    dist = row.geometry.distance(Point(tick.coords[-1]))
                    indices[idx] = dist
                minim = min(indices, key=indices.get)
                heights.append(buildings.iloc[minim][height_column])
            else:
                width.append(np.nan)
        widths.append(width[0] + width[1])

    results_list.append(np.nanmean(widths))
    heights_list.append(np.mean(heights))

widths_series = pd.Series(results_list)
heights_series = pd.Series(heights_list)
profile_ratio = heights_series / widths_series
print('Street profile calculated.')


streets['profile'] = series
streets.to_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Tests/190123/str.shp")
ticks_ser = []
for t in range(len(ticks)):
    ticks_ser = ticks_ser + ticks[t]

ser = gpd.GeoSeries(ticks_ser)
toqgis = gpd.GeoDataFrame(geometry=ser)

toqgis.to_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Tests/190123/ticks.shp")
