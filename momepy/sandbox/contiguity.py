import libpysal
import geopandas as gpd
from tqdm import tqdm
import statistics
import momepy as mm

tessellation = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Tests/181122/tess_oldtown_nid.shp")

weights_queen = libpysal.weights.Queen.from_dataframe(tessellation)

objects = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Tests/181122/blg_oldtown_nid.shp")
column_name = 'dist'
weights_matrix = weights_queen

objects[column_name] = None
objects[column_name] = objects[column_name].astype('float')

print('Calculating distances...')

if weights_matrix is None:
    print('Calculating spatial weights...')
    from libpysal.weights import Queen
    weights_matrix = Queen.from_dataframe(tessellation)
    print('Spatial weights ready...')

# iterating over rows one by one
for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
    id = tessellation.loc[tessellation['uID'] == row['uID']].index[0]
    neighbours = weights_matrix.neighbors[id]
    neighbours_ids = []

    for n in neighbours:
        uniq = tessellation.iloc[n]['uID']
        neighbours_ids.append(uniq)

    distances = []
    for i in neighbours_ids:
        dist = objects.loc[objects['uID'] == i].iloc[0]['geometry'].distance(row['geometry'])
        distances.append(dist)

    objects.loc[index, column_name] = statistics.mean(distances)
print('Distances calculated.')

objects.to_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Tests/181204/blg_oldtown_nid.shp")
