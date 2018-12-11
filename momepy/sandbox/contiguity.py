import libpysal
import geopandas as gpd

tessellation = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Tests/181122/tess_oldtown_nid.shp")

weights_queen = libpysal.weights.Queen.from_dataframe(tessellation)  # classic queen of 1st order
second_order = libpysal.weights.higher_order(weights_queen, k=2)  # second order neighbours (only)
joined_weights = libpysal.weights.w_union(weights_queen, second_order)  # joined together

weights_queen.neighbors[10]
second_order.neighbors[10]
joined_weights.neighbors[10]


def Queen_higher(dataframe, k):
    first_order = libpysal.weights.Queen.from_dataframe(dataframe)
    joined = first_order
    for i in list(range(2, k + 1)):
        i_order = libpysal.weights.higher_order(first_order, k=i)
        joined = libpysal.weights.w_union(joined, i_order)
    return joined

fourth = Queen_higher(tessellation, k=4)
fourth.mean_neighbors

first = libpysal.weights.Queen.from_dataframe(tessellation)
first.mean_neighbors

objects = tessellation
column_name = 'mesh3'
area_column = 'area'
from tqdm import tqdm
import momepy as mm
tessellation = mm.area(tessellation, 'area')
spatial_weights = fourth

# define new column
objects[column_name] = None
objects[column_name] = objects[column_name].astype('float')

print('Calculating effective mesh size...')

for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
    neighbours = spatial_weights.neighbors[index]
    neighbour_areas = [row[area_column] ** 2]
    total_area = row[area_column]
    for n in neighbours:
        n_area = objects.iloc[n][area_column]
        total_area = total_area + n_area
        neighbour_areas.append(n_area ** 2)

    objects.loc[index, column_name] = (1 / total_area) * sum(neighbour_areas)

objects.plot(column='mesh3', cmap='Spectral')
objects.to_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Tests/181205/tess_oldtown.shp")
