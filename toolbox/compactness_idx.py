'''
Calculate compactness index of each object in given shapefile.

Formula described by Dibble (2016).

compIdx = Area/(Area of the smallest circumscribed circle)
'''

import geopandas as gpd

# set path to shapefile
path = "/Users/martin/Strathcloud/Personal Folders/Test data/Royston/buildings.shp"
objects = gpd.read_file(path)  # load file into geopandas
print('Shapefile loaded.')

# define new column called 'area'
objects['compIdx'] = None
print('Column ready.')

# fill new column with the value of area, iterating over rows one by one
for index, row in objects.iterrows():
    # objects.loc[index, 'compIdx'] = (row['geometry'].area)/(row['geometry'].) - This line needs to be changed.

print('Compactness index calculated.')
# save dataframe back to file
objects.to_file(path)
print('File saved.')
