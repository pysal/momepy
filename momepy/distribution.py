# distribution.py
# definitons of spatial distribution characters

from tqdm import tqdm  # progress bar
from shapely.geometry import LineString
import numpy as np
import geopandas as gpd
import math

'''
orientation():
    Calculate orientation (azimuth) of object

    Formula: orientation of the longext axis of bounding rectangle expressed by sin(x)

    Reference: Schirmer PM and Axhausen KW (2015) A multiscale classiﬁcation of urban morphology.
               Journal of Transport and Land Use 9(1): 101–130.

    Attributes: objects = geoDataFrame with objects
                column_name = name of the column to save calculated values


'''


def orientation(objects, column_name):
    # define new column
    objects[column_name] = None
    objects[column_name] = objects[column_name].astype('float')

    print('Calculating orientations...')

    def azimuth(point1, point2):
        '''azimuth between 2 shapely points (interval 0 - 180)'''
        angle = np.arctan2(point2.x - point1.x, point2.y - point1.y)
        return np.degrees(angle)if angle > 0 else np.degrees(angle) + 180

    # iterating over rows one by one
    for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
        bbox = list(row['geometry'].minimum_rotated_rectangle.exterior.coords)
        centroid_ab = LineString([bbox[0], bbox[1]]).centroid
        centroid_cd = LineString([bbox[2], bbox[3]]).centroid
        axis1 = centroid_ab.distance(centroid_cd)

        centroid_bc = LineString([bbox[1], bbox[2]]).centroid
        centroid_da = LineString([bbox[3], bbox[0]]).centroid
        axis2 = centroid_bc.distance(centroid_da)

        if axis1 <= axis2:
            objects.loc[index, column_name] = math.sin(math.radians(azimuth(centroid_bc, centroid_da)))
        else:
            objects.loc[index, column_name] = math.sin(math.radians(azimuth(centroid_ab, centroid_cd)))

    print('Orientations calculated.')

'''
shared_walls_ratio():
    Calculate shared walls ratio

    Formula: length of shared walls / perimeter

    Reference: Hamaina R, Leduc T and Moreau G (2012) Towards Urban Fabrics Characterization
               Based on Buildings Footprints. In: Lecture Notes in Geoinformation and Cartography,
               Berlin, Heidelberg: Springer Berlin Heidelberg, pp. 327–346. Available from:
               https://link.springer.com/chapter/10.1007/978-3-642-29063-3_18.

    Attributes: objects = geoDataFrame with objects
                column_name = name of the column to save calculated values
                perimeter_column = name of the column where is stored perimeter value
                unique_id = name of the column with unique id
'''


def shared_walls_ratio(objects, column_name, perimeter_column, unique_id):

    # define new column
    objects[column_name] = None
    objects[column_name] = objects[column_name].astype('float')

    print('Calculating shared walls ratio...')

    for index, row in objects.iterrows():
        neighbors = objects[~objects.geometry.disjoint(row.geometry)][unique_id].tolist()
        neighbors = [name for name in neighbors if row[unique_id] != name]
        # if no neighbour exists
        global length
        length = 0
        if len(neighbors) is 0:
            objects.loc[index, column_name] = 0
        else:
            for i in neighbors:
                subset = objects.loc[objects[unique_id] == i]
                length = length + row.geometry.intersection(subset.iloc[0]['geometry']).length
                objects.loc[index, column_name] = length / row[perimeter_column]

# to be deleted, keep at the end
#
# path = "/Users/martin/Strathcloud/Personal Folders/Test data/Royston/buildings.shp"
# objects = gpd.read_file(path)
#
# orientation(objects, 'ptbOri')
# objects.to_file(path)
