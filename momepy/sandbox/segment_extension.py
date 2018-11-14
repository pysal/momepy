from shapely.geometry import *
import math
import operator
from tqdm import tqdm
import geopandas as gpd
import osmnx as ox


def snap_street_network(network, buildings, tolerance=20):
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
        possible_intersections_lines = network.iloc[possible_intersections_index]
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

            # check whether the line goes through buildings. if so, ignore it
            possible_buildings_index = list(bindex.intersection(new_extended_line.bounds))
            possible_buildings = buildings.iloc[possible_buildings_index]
            possible_intersections = possible_buildings.intersection(new_extended_line)

            if possible_intersections.any():
                pass
            else:
                network.loc[idx, 'geometry'] = new_extended_line

    # generating spatial index (rtree)
    print('Building R-tree for network...')
    sindex = network.sindex
    print('Building R-tree for buildings...')
    bindex = buildings.sindex
    # iterating over each street segment
    for idx, row in tqdm(network.iterrows(), total=network.shape[0]):

        line = row['geometry']
        l_coords = list(line.coords)
        # network_w = network.drop(idx, axis=0)['geometry']  # ensure that it wont intersect itself
        start = Point(l_coords[0])
        end = Point(l_coords[-1])

        # find out whether ends of the line are connected or not
        possible_first_index = list(sindex.intersection(start.bounds))
        possible_first_matches = network.iloc[possible_first_index]
        possible_first_matches_clean = possible_first_matches.drop(idx, axis=0)
        first = possible_first_matches_clean.intersects(start).any()

        possible_second_index = list(sindex.intersection(end.bounds))
        possible_second_matches = network.iloc[possible_second_index]
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

    return network


def snap_street_network_edge(network, buildings, tessellation, tolerance=20):
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

        possible_intersections_index = list(qindex.intersection(extrapolation.bounds))
        possible_intersections_lines = geometry_cut.iloc[possible_intersections_index]
        possible_intersections = possible_intersections_lines.intersection(extrapolation)

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

            # check whether the line goes through buildings. if so, ignore it
            possible_buildings_index = list(bindex.intersection(new_extended_line.bounds))
            possible_buildings = buildings.iloc[possible_buildings_index]
            possible_intersections = possible_buildings.intersection(new_extended_line)

            if possible_intersections.any():
                pass
            else:
                network.loc[idx, 'geometry'] = new_extended_line

    # generating spatial index (rtree)
    print('Building R-tree for network...')
    sindex = network.sindex
    print('Building R-tree for buildings...')
    bindex = buildings.sindex
    print('Dissolving tesselation...')
    cells['diss'] = 0
    built_up = cells.dissolve(by='diss')
    print('Preparing buffer zone for edge resolving (quadrat cut)...')
    geometry = built_up['geometry'].iloc[0].boundary
    # quadrat_width is in the units the geometry is in
    geometry_cut = ox.quadrat_cut_geometry(geometry, quadrat_width=500)
    qindex = geometry_cut.sindex

    # iterating over each street segment
    for idx, row in tqdm(network.iterrows(), total=network.shape[0]):

        line = row['geometry']
        l_coords = list(line.coords)
        # network_w = network.drop(idx, axis=0)['geometry']  # ensure that it wont intersect itself
        start = Point(l_coords[0])
        end = Point(l_coords[-1])

        # find out whether ends of the line are connected or not
        possible_first_index = list(sindex.intersection(start.bounds))
        possible_first_matches = network.iloc[possible_first_index]
        possible_first_matches_clean = possible_first_matches.drop(idx, axis=0)
        first = possible_first_matches_clean.intersects(start).any()

        possible_second_index = list(sindex.intersection(end.bounds))
        possible_second_matches = network.iloc[possible_second_index]
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

    return network

net = gpd.read_file('/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/snap/6.shp')
blg = gpd.read_file('/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/snap/6b.shp')
fixed = snap_street_network(net, blg)


fixed.plot()
fixed.to_file('/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/snap/6bl.shp')

# MISSING - RESOLVING THE EDGE OF THE URBAN AREA. EXTEND LINES TO THE EDGE, EVEN THOUGH IT IS LONGER THAN NORMAL TOLERANCE
#           EXTEND TOLERANCE, FIND INTERSECTION WITH AREA.BOUNDARY INSTEAD OF NETWORK
