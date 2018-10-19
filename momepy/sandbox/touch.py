import geopandas as gpd
import matplotlib.pyplot as plt

blg = gpd.read_file('/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Royston/touch3.shp')
blg
blg.plot()

blg.loc[7, 'geometry'].intersection(x.loc[0, 'geometry']).length

blg['shared2'] = None
blg['shared2'] = blg['shared2'].astype('float')

for index, row in blg.iterrows():
    neighbors = blg[~blg.geometry.disjoint(row.geometry)].OBJECTID.tolist()
    neighbors = [name for name in neighbors if row.OBJECTID != name]
    if len(neighbors) is 0:
        blg.loc[index, 'shared2'] = 0
    else:
        max = len(neighbors) - 1
        if max is 0:
            global length
            subset = blg.loc[blg['OBJECTID'] == neighbors]
            length = row.geometry.intersection(subset.iloc[0, -2]).length
        else:
            for i in range(0, max):
                subset = blg.loc[blg['OBJECTID'] == neighbors[i]]
                blg.loc[blg['OBJECTID'] == neighbors[i]]
                if i == 0:
                    subset.iloc[0, -2]
                    neighbors[i]
                    length = row.geometry.intersection(subset.iloc[0, -2]).length
                    print(i, length)
                else:
                    length = length + row.geometry.intersection(subset.iloc[0, -2]).length
                    print(i, length)
        blg.loc[index, 'shared2'] = length
blg.to_file('/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Royston/touch3.shp')
blg


for index, row in blg.iterrows():
    neighbors = blg[~blg.geometry.disjoint(row.geometry)].OBJECTID.tolist()
    neighbors = [name for name in neighbors if row.OBJECTID != name]
    # if no neighbour exists
    global length
    length = 0
    if len(neighbors) is 0:
        blg.loc[index, 'shared2'] = 0
    else:
        max = len(neighbors) - 1
        # if there is only one neighbour
        if max is 0:
            subset = blg.loc[blg['OBJECTID'] == neighbors]
            blg.loc[index, 'shared2'] = row.geometry.intersection(subset.iloc[0]['geometry']).length / row['pdbPer']
        # if there is more neighbours
        else:
            for i in neighbors:
                subset = blg.loc[blg['OBJECTID'] == i]
                length = length + row.geometry.intersection(subset.iloc[0]['geometry']).length
                blg.loc[index, 'shared2'] = length / row['pdbPer']

blg


def shared_walls_ratio(objects, column_name, perimeter_column, unique_id):

    objects[column_name] = None
    objects[column_name] = objects[column_name].astype('float')

    for index, row in objects.iterrows():
        neighbors = objects[~objects.geometry.disjoint(row.geometry)][unique_id].tolist()
        neighbors = [name for name in neighbors if row[unique_id] != name]
        # if no neighbour exists
        global length
        length = 0
        if len(neighbors) is 0:
            objects.loc[index, column_name] = 0
        else:
            for i in neighbors:
                subset = objects.loc[objects[unique_id] == i]['geometry']
                length = length + row.geometry.intersection(subset.iloc[0]).length
                objects.loc[index, column_name] = length / row[perimeter_column]

shared_walls_ratio(blg, 'shr8', 'pdbPer', unique_id='OBJECTID')

blg
subset = blg.loc[blg['OBJECTID'] == 7644.0]['geometry']
type(subset)
subset.iloc[0]
a = subset.geometry
type(a)
