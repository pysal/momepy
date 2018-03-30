'''
Calculate area of each object in given shapefile.

Attributes: path = path to the shapefile
            column_name = name of the column to save the area values
'''

import geopandas as gpd


def object_area(path, column_name):
    objects = gpd.read_file(path)  # load file into geopandas
    print('Shapefile loaded.')

    # define new column called 'area'
    objects[column_name] = None
    objects[column_name] = objects[column_name].astype('float')
    print('Column ready.')

    # fill new column with the value of area, iterating over rows one by one
    for index, row in objects.iterrows():
        objects.loc[index, column_name] = row['geometry'].area

    print('Areas calculated.')
    # save dataframe back to file
    objects.to_file(path)
    print('File saved.')
