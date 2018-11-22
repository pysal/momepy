import geopandas as gpd
import math
from tqdm import tqdm  # progress bar
import numpy as np
from shapely.geometry import Point, LineString
import momepy as mm

objects = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Tests/181122/blg_oldtown_nid.shp")
streets = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Tests/181115/street.shp")
column_name = 'stbAli'
orientation_column = 'stbOri'
network_id_column = 'nID'

mm.orientation(objects, orientation_column)


# define new column
objects[column_name] = None
objects[column_name] = objects[column_name].astype('float')

print('Calculating street alignments...')

def azimuth(point1, point2):
    '''azimuth between 2 shapely points (interval 0 - 180)'''
    angle = np.arctan2(point2.x - point1.x, point2.y - point1.y)
    return np.degrees(angle)if angle > 0 else np.degrees(angle) + 180

# iterating over rows one by one
for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
    network_id = row[network_id_column]
    streetssub = streets.loc[streets[network_id_column] == network_id]
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
    objects.loc[index, column_name] = abs(row[orientation_column] - az)
print('Street alignments calculated.')



objects.to_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Tests/181122/blg_ali.shp")


objects.plot(column=column_name, cmap='Spectral')
