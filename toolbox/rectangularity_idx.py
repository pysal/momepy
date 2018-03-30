'''
Calculate rectangularity index of each object in given shapefile.

Formula described by Dibble (2016).

rectangularity_idx = Area/(Area of the rotated bounding recatngle)
'''

import geopandas as gpd

# set path to shapefile
path = "/Users/martin/Strathcloud/Personal Folders/Test data/Royston/buildings.shp"

def rectangularity_idx(path):
    objects = gpd.read_file(path)  # load file into geopandas
    print('Shapefile loaded.')

    # define new column called 'area'
    objects['rectIdx'] = None
    objects['rectIdx'] = objects['rectIdx'].astype('float')
    print('Column ready.')

    # fill new column with the value of area, iterating over rows one by one
    for index, row in objects.iterrows():
        objects.loc[index, 'rectIdx'] = (row['geometry'].area)/(row['geometry'].convex_hull.area)

    print('Rectangularity index calculated.')

    # save dataframe back to file
    objects.to_file(path)
    print('File saved.')
