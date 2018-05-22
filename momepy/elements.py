# elements.py
# generating derived elements (street edge, block)

import geopandas as gpd

'''
clean geometry

delete building with zero height (to avoid division by 0)
'''


def clean_buildings(path, height_column):
    print('Loading file.')
    objects = gpd.read_file(path)  # load file into geopandas
    print('Shapefile loaded.')

    objects = objects[objects[height_column] > 0]
    print('Zero height buildings ommited.')

    # save dataframe back to file
    print('Saving file.')
    objects.to_file(path)
    print('File saved.')
