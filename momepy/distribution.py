#!/usr/bin/env python
# -*- coding: utf-8 -*-

# distribution.py
# definitons of spatial distribution characters

from tqdm import tqdm  # progress bar
from shapely.geometry import LineString, Point
import numpy as np
import pandas as pd
import statistics


def orientation(gdf):
    """
    Calculate the orientation of object

    Defined as an orientation of the longext axis of bounding rectangle in range 0 - 45.
    It captures the deviation of orientation from cardinal directions.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects to analyse

    Returns
    -------
    Series
        Series containing resulting values.

    References
    ---------
    Schirmer PM and Axhausen KW (2015) A multiscale classiﬁcation of urban morphology.
    Journal of Transport and Land Use 9(1): 101–130. (adapted)

    Examples
    --------
    >>> buildings_df['orientation'] = momepy.orientation(buildings_df)
    Calculating orientations...
    100%|██████████| 144/144 [00:00<00:00, 630.54it/s]
    Orientations calculated.
    >>> buildings_df['orientation'][0]
    41.05146788287027
    """
    # define empty list for results
    results_list = []

    print('Calculating orientations...')

    def _azimuth(point1, point2):
        '''azimuth between 2 shapely points (interval 0 - 180)'''
        angle = np.arctan2(point2.x - point1.x, point2.y - point1.y)
        return np.degrees(angle)if angle > 0 else np.degrees(angle) + 180

    # iterating over rows one by one
    for index, row in tqdm(gdf.iterrows(), total=gdf.shape[0]):
        bbox = list(row['geometry'].minimum_rotated_rectangle.exterior.coords)
        centroid_ab = LineString([bbox[0], bbox[1]]).centroid
        centroid_cd = LineString([bbox[2], bbox[3]]).centroid
        axis1 = centroid_ab.distance(centroid_cd)

        centroid_bc = LineString([bbox[1], bbox[2]]).centroid
        centroid_da = LineString([bbox[3], bbox[0]]).centroid
        axis2 = centroid_bc.distance(centroid_da)

        if axis1 <= axis2:
            az = _azimuth(centroid_bc, centroid_da)
            if 90 > az >= 45:
                diff = az - 45
                az = az - 2 * diff
            elif 135 > az >= 90:
                diff = az - 90
                az = az - 2 * diff
                diff = az - 45
                az = az - 2 * diff
            elif 181 > az >= 135:
                diff = az - 135
                az = az - 2 * diff
                diff = az - 90
                az = az - 2 * diff
                diff = az - 45
                az = az - 2 * diff
            results_list.append(az)
        else:
            az = 170
            az = _azimuth(centroid_ab, centroid_cd)
            if 90 > az >= 45:
                diff = az - 45
                az = az - 2 * diff
            elif 135 > az >= 90:
                diff = az - 90
                az = az - 2 * diff
                diff = az - 45
                az = az - 2 * diff
            elif 181 > az >= 135:
                diff = az - 135
                az = az - 2 * diff
                diff = az - 90
                az = az - 2 * diff
                diff = az - 45
                az = az - 2 * diff
            results_list.append(az)

    series = pd.Series(results_list, index=gdf.index)
    print('Orientations calculated.')
    return series


def shared_walls_ratio(gdf, unique_id, perimeters=None):
    """
    Calculate shared walls ratio

    .. math::
        \\textit{length of shared walls} \\over perimeter

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing gdf to analyse
    unique_id : str, list, np.array, pd.Series
        the name of the dataframe column, np.array, or pd.Series with unique id
    perimeters : str, list, np.array, pd.Series (default None)
        the name of the dataframe column, np.array, or pd.Series where is stored perimeter value

    Returns
    -------
    Series
        Series containing resulting values.

    References
    ---------
    Hamaina R, Leduc T and Moreau G (2012) Towards Urban Fabrics Characterization
    Based on Buildings Footprints. In: Lecture Notes in Geoinformation and Cartography,
    Berlin, Heidelberg: Springer Berlin Heidelberg, pp. 327–346. Available from:
    https://link.springer.com/chapter/10.1007/978-3-642-29063-3_18.

    Examples
    --------
    >>> buildings_df['swr'] = momepy.shared_walls_ratio(buildings_df, 'uID')
    Generating spatial index...
    Calculating shared walls ratio...
    100%|██████████| 144/144 [00:00<00:00, 648.72it/s]
    Shared walls ratio calculated.
    >>> buildings_df['swr'][10]
    0.3424804411228673
    """
    print('Generating spatial index...')
    gdf = gdf.copy()
    sindex = gdf.sindex  # define rtree index
    # define empty list for results
    results_list = []

    print('Calculating shared walls ratio...')

    if perimeters is None:
        gdf['mm_p'] = gdf.geometry.length
        perimeters = 'mm_p'
    else:
        if not isinstance(perimeters, str):
            gdf['mm_p'] = perimeters
            perimeters = 'mm_p'
    if not isinstance(unique_id, str):
        gdf['mm_uid'] = unique_id
        unique_id = 'mm_uid'

    for index, row in tqdm(gdf.iterrows(), total=gdf.shape[0]):
        neighbors = list(sindex.intersection(row.geometry.bounds))
        neighbors.remove(index)

        # if no neighbour exists
        length = 0
        if not neighbors:
            results_list.append(0)
        else:
            for i in neighbors:
                subset = gdf.loc[i]['geometry']
                length = length + row.geometry.intersection(subset).length
            results_list.append(length / row[perimeters])
    series = pd.Series(results_list, index=gdf.index)
    print('Shared walls ratio calculated.')

    return series


def street_alignment(left, right, orientations, left_network_id, right_network_id):
    """
    Calculate the difference between street orientation and orientation of object in degrees

    Orientation of street segment is represented by the orientation of line
    connecting first and last point of the segment. Network ID linking each object
    to specific street segment is needed. Can be generated by :py:func:`momepy.get_network_id`.

    .. math::
        \\left|{\\textit{building orientation} - \\textit{street orientation}}\\right|

    Parameters
    ----------
    left : GeoDataFrame
        GeoDataFrame containing objects to analyse
    right : GeoDataFrame
        GeoDataFrame containing street network
    orientations : str, list, np.array, pd.Series
        the name of the dataframe column, np.array, or pd.Series where is stored object orientation value
        (can be calculated using :py:func:`momepy.orientation`)
    left_network_id : str, list, np.array, pd.Series
        the name of the left dataframe column, np.array, or pd.Series where is stored object network ID
    right_network_id : str, list, np.array, pd.Series
        the name of the right dataframe column, np.array, or pd.Series of streets with unique network id (has to be defined beforehand)
        (can be defined using :py:func:`momepy.elements.unique_id`)

    Returns
    -------
    Series
        Series containing resulting values.

    Examples
    --------
    >>> buildings_df['street_alignment'] = momepy.street_alignment(buildings_df, streets_df, 'orientation', 'nID', 'nID')
    Calculating street alignments...
    100%|██████████| 144/144 [00:00<00:00, 529.94it/s]
    Street alignments calculated.
    >>> buildings_df['street_alignment'][0]
    0.29073888476702336
    """
    # define empty list for results
    results_list = []

    print('Calculating street alignments...')
    left = left.copy()
    right = right.copy()
    if not isinstance(orientations, str):
        left['mm_o'] = orientations
        orientations = 'mm_o'
    if not isinstance(left_network_id, str):
        left['mm_nid'] = left_network_id
        left_network_id = 'mm_nid'
    if not isinstance(right_network_id, str):
        right['mm_nis'] = right_network_id
        right_network_id = 'mm_nis'

    def azimuth(point1, point2):
        '''azimuth between 2 shapely points (interval 0 - 180)'''
        angle = np.arctan2(point2.x - point1.x, point2.y - point1.y)
        return np.degrees(angle)if angle > 0 else np.degrees(angle) + 180

    # iterating over rows one by one
    for index, row in tqdm(left.iterrows(), total=left.shape[0]):
        if pd.isnull(row[left_network_id]):
            results_list.append(0)
        else:
            network_id = row[left_network_id]
            streetssub = right.loc[right[right_network_id] == network_id]
            start = Point(streetssub.iloc[0]['geometry'].coords[0])
            end = Point(streetssub.iloc[0]['geometry'].coords[-1])
            az = azimuth(start, end)
            if 90 > az >= 45:
                diff = az - 45
                az = az - 2 * diff
            elif 135 > az >= 90:
                diff = az - 90
                az = az - 2 * diff
                diff = az - 45
                az = az - 2 * diff
            elif 181 > az >= 135:
                diff = az - 135
                az = az - 2 * diff
                diff = az - 90
                az = az - 2 * diff
                diff = az - 45
                az = az - 2 * diff
            results_list.append(abs(row[orientations] - az))
    series = pd.Series(results_list, index=left.index)

    print('Street alignments calculated.')
    return series


def cell_alignment(left, right, left_orientations, right_orientations, left_unique_id, right_unique_id):
    """
    Calculate the difference between cell orientation and orientation of object

    .. math::
        \\left|{\\textit{building orientation} - \\textit{cell orientation}}\\right|

    Parameters
    ----------
    left : GeoDataFrame
        GeoDataFrame containing objects to analyse
    right : GeoDataFrame
        GeoDataFrame containing street network
    left_orientations : str, list, np.array, pd.Series
        the name of the left dataframe column, np.array, or pd.Series where is stored object orientation value
        (can be calculated using :py:func:`momepy.orientation`)
    right_orientations : str, list, np.array, pd.Series
        the name of the dataframe column, np.array, or pd.Series where is stored object orientation value
        (can be calculated using :py:func:`momepy.orientation`)
    left_unique_id : str
        the name of the left dataframe column with unique id shared between left and right gdf
    right_unique_id : str
        the name of the right dataframe column with unique id shared between left and right gdf

    Returns
    -------
    Series
        Series containing resulting values.

    Examples
    --------
    >>> buildings_df['cell_alignment'] = momepy.cell_alignment(buildings_df, tessellation_df, 'bl_orient', 'tes_orient', 'uID', 'uID')
    Calculating cell alignments...
    100%|██████████| 144/144 [00:00<00:00, 799.09it/s]
    Cell alignments calculated.
    >>> buildings_df['cell_alignment'][0]
    0.8795123936951939

    """
    print('Calculating cell alignments...')
    left = left.copy()
    right = right.copy()
    if not isinstance(left_orientations, str):
        left['mm_o'] = left_orientations
        left_orientations = 'mm_o'
    if not isinstance(right_orientations, str):
        right['mm_o'] = right_orientations
        right_orientations = 'mm_o'

    # define empty list for results
    results_list = []

    for index, row in tqdm(left.iterrows(), total=left.shape[0]):

        results_list.append(abs(row[left_orientations] - right[right[right_unique_id] == row[left_unique_id]][right_orientations].iloc[0]))

    series = pd.Series(results_list, index=left.index)

    print('Cell alignments calculated.')
    return series


def alignment(gdf, spatial_weights, unique_id, orientations):

    """
    Calculate the mean deviation of solar orientation of objects on adjacent cells from an object

    .. math::
        \\frac{1}{n}\\sum_{i=1}^n dev_i=\\frac{dev_1+dev_2+\\cdots+dev_n}{n}

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects to analyse
    spatial_weights : libpysal.weights, optional
        spatial weights matrix
    orientations : str, list, np.array, pd.Series
        the name of the left dataframe column, np.array, or pd.Series where is stored object orientation value
        (can be calculated using :py:func:`momepy.orientation`)
    unique_id : str
        name of the column with unique id used as spatial_weights index.

    Returns
    -------
    Series
        Series containing resulting values.

    Examples
    --------
    >>> buildings_df['alignment'] = momepy.alignment(buildings_df, sw, 'uID', bl_orient)
    Calculating alignments...
    Calculating spatial weights...
    Spatial weights ready...
    100%|██████████| 144/144 [00:01<00:00, 140.84it/s]
    Alignments calculated.
    >>> buildings_df['alignment'][0]
    18.299481296455237
    """
    # define empty list for results
    results_list = []
    gdf = gdf.copy()
    if not isinstance(orientations, str):
        gdf['mm_o'] = orientations
        orientations = 'mm_o'

    print('Calculating alignments...')

    # iterating over rows one by one
    for index, row in tqdm(gdf.iterrows(), total=gdf.shape[0]):

        neighbours = spatial_weights.neighbors[row[unique_id]]
        if neighbours:
            orientation = gdf.loc[gdf[unique_id].isin(neighbours)][orientations]
            deviations = abs(orientation - row[orientations])

            results_list.append(statistics.mean(deviations))
        else:
            results_list.append(0)

    series = pd.Series(results_list, index=gdf.index)

    print('Alignments calculated.')
    return series


def neighbour_distance(gdf, spatial_weights, unique_id):
    """
    Calculate the mean distance to adjacent buildings (based on spatial_weights)

    .. math::
        \\frac{1}{n}\\sum_{i=1}^n dist_i=\\frac{dist_1+dist_2+\\cdots+dist_n}{n}

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects to analyse
    spatial_weights : libpysal.weights, optional
        spatial weights matrix
    unique_id : str
        name of the column with unique id used as spatial_weights index.

    Returns
    -------
    Series
        Series containing resulting values.

    References
    ---------
    Schirmer PM and Axhausen KW (2015) A multiscale classiﬁcation of urban morphology.
    Journal of Transport and Land Use 9(1): 101–130. (adapted)

    Examples
    --------
    >>> buildings_df['neighbour_distance'] = momepy.neighbour_distance(buildings_df, sw, 'uID')
    Calculating distances...
    100%|██████████| 144/144 [00:00<00:00, 345.78it/s]
    Distances calculated.
    >>> buildings_df['neighbour_distance'][0]
    29.18589019096464
    """
    # define empty list for results
    results_list = []

    print('Calculating distances...')

    # iterating over rows one by one
    for index, row in tqdm(gdf.iterrows(), total=gdf.shape[0]):
        neighbours = spatial_weights.neighbors[row[unique_id]]
        building_neighbours = gdf.loc[gdf[unique_id].isin(neighbours)]
        if len(building_neighbours) > 0:
            results_list.append(np.mean(building_neighbours.geometry.distance(row['geometry'])))
        else:
            results_list.append(0)

    series = pd.Series(results_list, index=gdf.index)

    print('Distances calculated.')
    return series


def mean_interbuilding_distance(gdf, spatial_weights, unique_id, spatial_weights_higher=None, order=3):
    """
    Calculate the mean interbuilding distance within x topological steps

    Interbuilding distances are calculated between buildings on adjacent cells based on `spatial_weights`,
    while the extend is defined in `spatial_weights_higher`.

    .. math::


    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects to analyse
    unique_id : str
        name of the column with unique id used as spatial_weights index
    spatial_weights : libpysal.weights
        spatial weights matrix (order 1)
    spatial_weights_higher : libpysal.weights, optional
        spatial weights matrix - If None, Queen contiguity of higher order will be calculated
        based on spatial_weights
    order : int
        Order of Queen contiguity

    Returns
    -------
    Series
        Series containing resulting values.

    References
    ---------
    ADD, but it is adapted quite a lot.

    Notes
    -----
    Fix UserWarning.

    Examples
    --------
    >>> buildings_df['mean_interbuilding_distance'] = momepy.mean_interbuilding_distance(buildings_df, sw, 'uID')
    Calculating mean interbuilding distances...
    Generating weights matrix (Queen) of 3 topological steps...
    Generating adjacency matrix based on weights matrix...
    Computing interbuilding distances...
    100%|██████████| 746/746 [00:03<00:00, 200.14it/s]
    Computing mean interbuilding distances...
    100%|██████████| 144/144 [00:00<00:00, 317.42it/s]
    Mean interbuilding distances calculated.
    >>> buildings_df['mean_interbuilding_distance'][0]
    29.305457092042744
    """

    print('Calculating mean interbuilding distances...')
    if spatial_weights_higher is None:
        print('Generating weights matrix (Queen) of {} topological steps...'.format(order))
        from momepy import Queen_higher
        # matrix to define area of analysis (more steps)
        spatial_weights_higher = Queen_higher(k=order, weights=spatial_weights)

    # define empty list for results
    results_list = []

    print('Generating adjacency matrix based on weights matrix...')
    # define adjacency list from lipysal
    adj_list = spatial_weights.to_adjlist()
    adj_list['distance'] = -1

    print('Computing interbuilding distances...')
    # measure each interbuilding distance of neighbours and save them to adjacency list
    for index, row in tqdm(adj_list.iterrows(), total=adj_list.shape[0]):
        inverted = adj_list[(adj_list.focal == row.neighbor)][(adj_list.neighbor == row.focal)].iloc[0]['distance']
        if inverted == -1:
            building_object = gdf.loc[gdf[unique_id] == row.focal.astype(int)]

            building_neighbour = gdf.loc[gdf[unique_id] == row.neighbor.astype(int)]
            adj_list.loc[index, 'distance'] = building_neighbour.iloc[0].geometry.distance(building_object.iloc[0].geometry)
        else:
            adj_list.at[index, 'distance'] = inverted

    print('Computing mean interbuilding distances...')
    # iterate over objects to get the final values
    for index, row in tqdm(gdf.iterrows(), total=gdf.shape[0]):
        # id to match spatial weights
        uid = row[unique_id]
        # define neighbours based on weights matrix defining analysis area
        neighbours = spatial_weights_higher.neighbors[uid]
        neighbours.append(uid)
        if neighbours:
            selection = adj_list[adj_list.focal.isin(neighbours)][adj_list.neighbor.isin(neighbours)]
            results_list.append(np.nanmean(selection.distance))

    series = pd.Series(results_list, index=gdf.index)
    print('Mean interbuilding distances calculated.')
    return series


def neighbouring_street_orientation_deviation(gdf):
    """
    Calculate the mean deviation of solar orientation of adjacent streets

    Orientation of street segment is represented by the orientation of line
    connecting first and last point of the segment.

    .. math::
        \\frac{1}{n}\\sum_{i=1}^n dev_i=\\frac{dev_1+dev_2+\\cdots+dev_n}{n}

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing street network to analyse

    Returns
    -------
    Series
        Series containing resulting values.

    Examples
    --------
    >>> streets_df['orient_dev'] = momepy.neighbouring_street_orientation_deviation(streets_df)
    Calculating street alignments...
    Preparing street orientations...
    Generating spatial index...
    100%|██████████| 33/33 [00:00<00:00, 249.02it/s]
    Street alignments calculated.
    >>> streets_df['orient_dev'][6]
    7.043096518688273
    """
    # define empty list for results
    results_list = []
    gdf = gdf.copy()
    print('Calculating street alignments...')

    def azimuth(point1, point2):
        '''azimuth between 2 shapely points (interval 0 - 180)'''
        angle = np.arctan2(point2.x - point1.x, point2.y - point1.y)
        return np.degrees(angle)if angle > 0 else np.degrees(angle) + 180

    # iterating over rows one by one
    print(' Preparing street orientations...')
    for index, row in tqdm(gdf.iterrows(), total=gdf.shape[0]):

        start = Point(row['geometry'].coords[0])
        end = Point(row['geometry'].coords[-1])
        az = azimuth(start, end)
        if 90 > az >= 45:
            diff = az - 45
            az = az - 2 * diff
        elif 135 > az >= 90:
            diff = az - 90
            az = az - 2 * diff
            diff = az - 45
            az = az - 2 * diff
        elif 181 > az >= 135:
            diff = az - 135
            az = az - 2 * diff
            diff = az - 90
            az = az - 2 * diff
            diff = az - 45
            az = az - 2 * diff
        results_list.append(az)
    series = pd.Series(results_list, index=gdf.index)

    gdf['tmporient'] = series

    print(' Generating spatial index...')
    sindex = gdf.sindex
    results_list = []

    for index, row in tqdm(gdf.iterrows(), total=gdf.shape[0]):
        possible_neighbors_idx = list(sindex.intersection(row.geometry.bounds))
        possible_neighbours = gdf.iloc[possible_neighbors_idx]
        neighbors = possible_neighbours[possible_neighbours.intersects(row.geometry)]
        neighbors.drop([index])

        orientations = []
        for idx, r in neighbors.iterrows():
            orientations.append(r.tmporient)

        deviations = []
        for o in orientations:
            dev = abs(o - row.tmporient)
            deviations.append(dev)

        if deviations:
            results_list.append(np.mean(deviations))
        else:
            results_list.append(0)

    series = pd.Series(results_list, index=gdf.index)
    print('Street alignments calculated.')
    return series


def building_adjacency(gdf, spatial_weights_higher, unique_id, spatial_weights=None):
    """
    Calculate the level of building adjacency

    Building adjacency reflects how much buildings tend to join together into larger structures.
    It is calculated as a ratio of joined built-up structures and buildings within k topological steps.

    .. math::


    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects to analyse
    spatial_weights_higher : libpysal.weights, optional
        spatial weights matrix
    unique_id : str
        name of the column with unique id used as spatial_weights index
    spatial_weights : libpysal.weights, optional
        spatial weights matrix - If None, Queen contiguity matrix will be calculated
        based on gdf. It is to denote adjacent buildings (note: based on index, not ID).

    Returns
    -------
    Series
        Series containing resulting values.

    References
    ---------
    Vanderhaegen S and Canters F (2017) Mapping urban form and function at city
    block level using spatial metrics. Landscape and Urban Planning 167: 399–409.

    Examples
    --------
    >>> buildings_df['adjacency'] = momepy.building_adjacency(buildings_df, swh, unique_id='uID')
    Calculating adjacency...
    Calculating spatial weights...
    Spatial weights ready...
    Generating weights matrix (Queen) of 3 topological steps...
    Generating dictionary of built-up patches...
    100%|██████████| 144/144 [00:00<00:00, 9301.73it/s]
    Calculating adjacency within k steps...
    100%|██████████| 144/144 [00:00<00:00, 335.55it/s]
    Adjacency calculated.
    >>> buildings_df['adjacency'][10]
    0.23809523809523808
    """
    # define empty list for results
    results_list = []

    print('Calculating adjacency...')

    # if weights matrix is not passed, generate it from gdf
    if spatial_weights is None:
        print('Calculating spatial weights...')
        from libpysal.weights import Queen
        spatial_weights = Queen.from_dataframe(gdf, silence_warnings=True, ids=unique_id)
        print('Spatial weights ready...')

    print('Generating dictionary of built-up patches...')
    # dict to store nr of courtyards for each uID
    patches = {}
    jID = 1
    for index, row in tqdm(gdf.iterrows(), total=gdf.shape[0]):

        # if the id is already present in courtyards, continue (avoid repetition)
        if row[unique_id] in patches:
            continue
        else:
            to_join = [row[unique_id]]  # list of indices which should be joined together
            neighbours = []  # list of neighbours
            weights = spatial_weights.neighbors[row[unique_id]]  # neighbours from spatial weights
            for w in weights:
                neighbours.append(w)  # make a list from weigths
            neighbours = spatial_weights.neighbors[row[unique_id]]

            for n in neighbours:
                while n not in to_join:  # until there is some neighbour which is not in to_join
                    to_join.append(n)
                    weights = spatial_weights.neighbors[n]
                    for w in weights:
                        neighbours.append(w)  # extend neighbours by neighbours of neighbours :)
            for b in to_join:
                patches[b] = jID  # fill dict with values
            jID = jID + 1

    print('Calculating adjacency within k steps...')
    for index, row in tqdm(gdf.iterrows(), total=gdf.shape[0]):
        neighbours = spatial_weights_higher.neighbors[row[unique_id]]
        if neighbours:
            neighbours.append(row[unique_id])

            patches_sub = [patches[x] for x in neighbours]
            patches_nr = len(set(patches_sub))

            results_list.append(patches_nr / len(neighbours))
        else:
            results_list.append(0)

    series = pd.Series(results_list, index=gdf.index)

    print('Adjacency calculated.')
    return series


def neighbours(gdf, spatial_weights, unique_id, weighted=False):
    """
    Calculate the number of topological neighbours of each object.

    Topological neighbours are defined by queen adjacency. If weighted=True, number of neighbours
    will be divided by the perimeter of object, to return relative value.

    .. math::


    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects to analyse
    spatial_weights : libpysal.weights (default None)
        spatial weights matrix
    unique_id : str
        name of the column with unique id used as spatial_weights index
    weighted : bool (default False)
        if weighted=True, number of neighbours will be divided by the perimeter of object, to return relative value

    Returns
    -------
    Series
        Series containing resulting values.

    References
    ---------
    Hermosilla T, Ruiz LA, Recio JA, et al. (2012) Assessing contextual descriptive features
    for plot-based classification of urban areas. Landscape and Urban Planning, Elsevier B.V.
    106(1): 124–137.

    Examples
    --------
    >>> sw = libpysal.weights.contiguity.Queen.from_dataframe(tessellation_df, ids='uID')
    >>> tessellation_df['neighbours'] = momepy.neighbours(tessellation_df, sw, 'uID')
    Calculating neighbours...
    100%|██████████| 144/144 [00:00<00:00, 6909.50it/s]
    Neighbours calculated.
    >>> tessellation_df['neighbours'][0]
    4
    """

    print('Calculating neighbours...')
    neighbours = []
    for index, row in tqdm(gdf.iterrows(), total=gdf.shape[0]):
        if weighted is True:
            neighbours.append(spatial_weights.cardinalities[row[unique_id]] / row.geometry.length)
        else:
            neighbours.append(spatial_weights.cardinalities[row[unique_id]])

    series = pd.Series(neighbours, index=gdf.index)
    print('Neighbours calculated.')
    return series
