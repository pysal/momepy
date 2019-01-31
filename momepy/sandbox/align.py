import geopandas as gpd
import math
from tqdm import tqdm  # progress bar
import numpy as np
from shapely.geometry import Point, LineString
import momepy as mm
import pandas as pd

objects = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Character sets/s_tessellation_shape.shp")
streets = gpd.read_file('/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Clean data (wID)/Street Network/prg_street_network.shp')
orientation_column = 'orient'
network_id_column = 'nID'

objects['orient'] = 0

results_list = []

print('Calculating street alignments...')

def azimuth(point1, point2):
    '''azimuth between 2 shapely points (interval 0 - 180)'''
    angle = np.arctan2(point2.x - point1.x, point2.y - point1.y)
    return np.degrees(angle)if angle > 0 else np.degrees(angle) + 180

# iterating over rows one by one
for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
    if pd.isnull(row[network_id_column]):
        results_list.append(0)
    else:
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
        results_list.append(abs(row[orientation_column] - az))
series = pd.Series(results_list)
print('Street alignments calculated.')

a = row[network_id_column] == np.NaN
a.isnull()
type(row[network_id_column])
pd.isnull(row[network_id_column])
