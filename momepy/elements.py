# elements.py
# generating derived elements (street edge, block)

import geopandas as gpd
from tqdm import tqdm  # progress bar


'''
clean_buildings():

Clean building geometry

Delete building with zero height (to avoid division by 0)
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

'''
clean_null():

Clean null geometry

Delete rows of GeoDataFrame with null geometry.
'''


def clean_null(path):
    print('Loading file.')
    objects = gpd.read_file(path)
    print('Shapefile loaded.')

    objects_none = objects[objects['geometry'].notnull()]  # filter nulls
    # save dataframe back to file
    print('Saving file.')
    objects_none.to_file(path)
    print('File saved.')

'''
unique_id():

Add an attribute with unique ID to each row of GeoDataFrame.

Optional: Delete all the columns except ID and geometry (set clear to True)
          To keep some of the columns and delete rest, pass them to keep.
          Typically we want to keep building height and get rid of the rest.
'''


def unique_id(path, clear=False, keep=None):
    print('Loading file.')
    objects = gpd.read_file(path)
    print('Shapefile loaded.')

    objects['uID'] = None
    objects['uID'] = objects['uID'].astype('float')
    id = 1
    for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
        objects.loc[index, 'uID'] = id
        id = id + 1

    cols = objects.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    if clear is False:
        objects = objects[cols]
    else:
        if keep is None:
            objects = objects.iloc[:, [-2, -1]]
        else:
            keep_col = objects.columns.get_loc(keep)
            objects = objects.iloc[:, [-2, keep_col, -1]]

    # save dataframe back to file
    print('Saving file.')
    objects.to_file(path)
    print('File saved.')
