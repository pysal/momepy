# blocks
import geopandas as gpd
import pandas as pd
from tqdm import tqdm
from shapely.geometry import Polygon

cells = gpd.read_file('/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Tests/181206/tess.shp')
streets = gpd.read_file('/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Tests/181206/str.shp')
buildings = gpd.read_file('/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Tests/181206/blg.shp')
id_name = 'bID'
unique_id = 'uID'

cells = cells.drop(['bID'], axis=1)
buildings = buildings.drop(['bID'], axis=1)

cells['diss'] = 0
built_up = cells.dissolve(by='diss')

print('Buffering streets...')
street_buff = streets.copy()
street_buff['geometry'] = streets.buffer(0.1)

print('Dissolving streets...')
street_cut = street_buff.unary_union

print('Defining street-based blocks...')
street_blocks = built_up['geometry'].difference(street_cut)

blocks_gdf = gpd.GeoDataFrame(street_blocks)

blocks_gdf = blocks_gdf.rename(columns={0: 'geometry'}).set_geometry('geometry')

def multi2single(gpdf):
    gpdf_singlepoly = gpdf[gpdf.geometry.type == 'Polygon']
    gpdf_multipoly = gpdf[gpdf.geometry.type == 'MultiPolygon']

    for i, row in gpdf_multipoly.iterrows():
        Series_geometries = pd.Series(row.geometry)
        df = pd.concat([gpd.GeoDataFrame(row, crs=gpdf_multipoly.crs).T] * len(Series_geometries), ignore_index=True)
        df['geometry'] = Series_geometries
        gpdf_singlepoly = pd.concat([gpdf_singlepoly, df])

    gpdf_singlepoly.reset_index(inplace=True, drop=True)
    return gpdf_singlepoly

print('Multipart to singlepart...')
blocks_single = multi2single(blocks_gdf)

blocks_single['geometry'] = blocks_single.buffer(0.1)

print('Defining block ID...')  # street based
blocks_single[id_name] = None
blocks_single[id_name] = blocks_single[id_name].astype('float')
id = 1
for idx, row in tqdm(blocks_single.iterrows(), total=blocks_single.shape[0]):
    blocks_single.loc[idx, id_name] = id
    id = id + 1

print('Generating centroids...')
buildings_c = buildings.copy()
buildings_c['geometry'] = buildings_c.centroid  # make centroids
blocks_single.crs = buildings.crs

print('Spatial join...')
centroids_tempID = gpd.sjoin(buildings_c, blocks_single, how='left', op='intersects')

tempID_to_uID = centroids_tempID[[unique_id, id_name]]

print('Attribute join (tesselation)...')
cells = cells.merge(tempID_to_uID, on=unique_id)
cells = cells.drop(['diss'], axis=1)

print('Generating blocks...')
blocks = cells.dissolve(by=id_name)
cells = cells.drop([id_name], axis=1)

print('Multipart to singlepart...')
blocks = multi2single(blocks)

blocks['geometry'] = blocks.exterior

id = 1
for idx, row in tqdm(blocks.iterrows(), total=blocks.shape[0]):
    blocks.loc[idx, id_name] = id
    id = id + 1
    blocks.loc[idx, 'geometry'] = Polygon(row['geometry'])

# if polygon is within another one, delete it
sindex = blocks.sindex
for idx, row in tqdm(blocks.iterrows(), total=blocks.shape[0]):
    possible_matches = list(sindex.intersection(row.geometry.bounds))
    possible_matches.remove(idx)
    possible = blocks.iloc[possible_matches]

    for idx2, row2 in possible.iterrows():
        if row['geometry'].within(row2['geometry']):
            blocks.loc[idx, 'delete'] = 1

blocks = blocks.drop(list(blocks.loc[blocks['delete'] == 1].index))

blocks_save = blocks[[id_name, 'geometry']]
blocks_save['geometry'] = blocks_save.buffer(0.000000001)

centroids_w_bl_ID2 = gpd.sjoin(buildings_c, blocks_save, how='left', op='intersects')
bl_ID_to_uID = centroids_w_bl_ID2[[unique_id, id_name]]

print('Attribute join (buildings)...')
buildings = buildings.merge(bl_ID_to_uID, on=unique_id)

print('Attribute join (tesselation)...')
cells = cells.merge(bl_ID_to_uID, on=unique_id)

print('Done')
# return (buildings, cells, blocks_save)
blocks.to_file('/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Tests/181206/block.shp')
