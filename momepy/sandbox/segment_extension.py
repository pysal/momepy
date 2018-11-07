from shapely.geometry import *
import math
import operator
from tqdm import tqdm
import geopandas as gpd


def snap_street_network(network, tolerance=20):
    # extrapolating function - makes line as a extrapolation of existing with set length (tolerance)
    def getExtrapoledLine(p1, p2):
        'Creates a line extrapoled in p1->p2 direction'
        EXTRAPOL_RATIO = tolerance  # length of a line
        a = p2

        # defining new point based on the vector between existing points
        if p1[0] >= p2[0] and p1[1] >= p2[1]:
            b = (p2[0] - EXTRAPOL_RATIO * math.cos(math.atan(math.fabs(p1[1] - p2[1] + 0.000001) / math.fabs(p1[0] - p2[0] + 0.000001))),
                 p2[1] - EXTRAPOL_RATIO * math.sin(math.atan(math.fabs(p1[1] - p2[1] + 0.000001) / math.fabs(p1[0] - p2[0] + 0.000001))))
        elif p1[0] <= p2[0] and p1[1] >= p2[1]:
            b = (p2[0] + EXTRAPOL_RATIO * math.cos(math.atan(math.fabs(p1[1] - p2[1] + 0.000001) / math.fabs(p1[0] - p2[0] + 0.000001))),
                 p2[1] - EXTRAPOL_RATIO * math.sin(math.atan(math.fabs(p1[1] - p2[1] + 0.000001) / math.fabs(p1[0] - p2[0] + 0.000001))))
        elif p1[0] <= p2[0] and p1[1] <= p2[1]:
            b = (p2[0] + EXTRAPOL_RATIO * math.cos(math.atan(math.fabs(p1[1] - p2[1] + 0.000001) / math.fabs(p1[0] - p2[0] + 0.000001))),
                 p2[1] + EXTRAPOL_RATIO * math.sin(math.atan(math.fabs(p1[1] - p2[1] + 0.000001) / math.fabs(p1[0] - p2[0] + 0.000001))))
        else:
            b = (p2[0] - EXTRAPOL_RATIO * math.cos(math.atan(math.fabs(p1[1] - p2[1] + 0.000001) / math.fabs(p1[0] - p2[0] + 0.000001))),
                 p2[1] + EXTRAPOL_RATIO * math.sin(math.atan(math.fabs(p1[1] - p2[1] + 0.000001) / math.fabs(p1[0] - p2[0] + 0.000001))))
        return LineString([a, b])

    # function extending line to closest object within set distance
    def extend_line():
        if Point(l_coords[-2]).distance(Point(l_coords[-1])) <= 0.001:
            extra = l_coords[-3:-1]
        else:
            extra = l_coords[-2:]
        extrapolation = getExtrapoledLine(*extra)  # we use the last two points

        possible_intersections_index = list(sindex.intersection(extrapolation.bounds))
        possible_intersections_lines = file.iloc[possible_intersections_index]
        possible_intersections_clean = possible_intersections_lines.drop(idx, axis=0)
        possible_intersections = possible_intersections_clean.intersection(extrapolation)

        if possible_intersections.any():

            true_int = []
            for one in list(possible_intersections.index):
                if possible_intersections[one].type == 'Point':
                    true_int.append(possible_intersections[one])
                elif possible_intersections[one].type == 'MultiPoint':
                    true_int.append(possible_intersections[one][0])
                    true_int.append(possible_intersections[one][1])

            if len(true_int) > 1:
                distances = {}
                ix = 0
                for p in true_int:
                    distance = p.distance(Point(l_coords[-1]))
                    distances[ix] = distance
                    ix = ix + 1
                minimal = min(distances.items(), key=operator.itemgetter(1))[0]
                new_point_coords = true_int[minimal].coords[0]
            else:
                new_point_coords = true_int[0].coords[0]

            l_coords.append(new_point_coords)
            new_extended_line = LineString(l_coords)
            file.loc[idx, 'geometry'] = new_extended_line

    file = network

    # generating spatial index (rtree)
    sindex = file.sindex

    # iterating over each street segment
    for idx, row in tqdm(file.iterrows(), total=file.shape[0]):

        line = row['geometry']
        l_coords = list(line.coords)
        network = file.drop(idx, axis=0)['geometry']  # ensure that it wont intersect itself
        start = Point(l_coords[0])
        end = Point(l_coords[-1])

        # find out whether ends of the line are connected or not
        possible_first_index = list(sindex.intersection(start.bounds))
        possible_first_matches = file.iloc[possible_first_index]
        possible_first_matches_clean = possible_first_matches.drop(idx, axis=0)
        first = possible_first_matches_clean.intersects(start).any()

        possible_second_index = list(sindex.intersection(end.bounds))
        possible_second_matches = file.iloc[possible_second_index]
        possible_second_matches_clean = possible_second_matches.drop(idx, axis=0)
        second = possible_second_matches_clean.intersects(end).any()

        # both ends connected, do nothing
        if first == True and second == True:
            continue
        # start connected, extend  end
        elif first == True and second == False:
            extend_line()
        # end connected, extend start
        elif first == False and second == True:
            l_coords.reverse()
            extend_line()
        # unconnected, extend both ends
        elif first == False and second == False:
            extend_line()
            l_coords.reverse()
            extend_line()
        else:
            print('Something went wrong.')

    return file


fixed = snap_street_network(gpd.read_file('/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/snap/5.shp'))


fixed.plot()
fixed.to_file('/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/snap/5fn2.shp')
