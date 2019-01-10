import geopandas as gpd
from shapely.geometry import Polygon
from tqdm import tqdm
import pandas as pd

objects = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Tests/190110/shared.shp")
perimeter_column = 'peri'

sindex = objects.sindex  # define rtree index
# define empty list for results
results_list = []

print('Calculating shared walls ratio...')

for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
    neighbors = list(sindex.intersection(row.geometry.bounds))
    neighbors.remove(index)

    # if no neighbour exists
    length = 0
    if len(neighbors) is 0:
        results_list.append(-1)
    else:
        for i in neighbors:
            subset = objects.loc[i]['geometry']
            length = length + row.geometry.intersection(subset).length
        results_list.append(length / row[perimeter_column])
series = pd.Series(results_list)
print('Shared walls ratio calculated.')
# return series










objects['nkjk'] = series2
objects.plot(column='nkjk', legend=True)

objects.to_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Tests/181217/blg_mmp.shp")

for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
    neighbors = list(sindex.intersection(row.geometry.bounds))
    neighbors.remove(index)
    possible_matches = objects.iloc[neighbors]
    precise_matches = possible_matches[possible_matches.intersects(row.geometry)]
    results_list.append(len(precise_matches))
