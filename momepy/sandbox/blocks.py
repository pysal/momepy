# blocks
import geopandas as gpd
import pandas as pd
from tqdm import tqdm

cells = gpd.read_file('/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Vinohrady/tess.shp')
streets = gpd.read_file('/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Vinohrady/str.shp')
buildings = gpd.read_file('/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Vinohrady/blg.shp')


def blocks(cells, streets, buildings, id_name, unique_id, cells_to, buildings_to, blocks_to):
    cells['diss'] = 0
    built_up = cells.dissolve(by='diss')

    street_buff = streets.copy()
    street_buff['geometry'] = streets.buffer(0.1)

    street_cut = street_buff.unary_union

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

    blocks_single = multi2single(blocks_gdf)

    blocks_single[id_name] = None
    blocks_single[id_name] = blocks_single[id_name].astype('float')
    id = 1
    for idx, row in tqdm(blocks_single.iterrows(), total=blocks_single.shape[0]):
        blocks_single.loc[idx, id_name] = id
        id = id + 1

    print('Generating centroids...')
    buildings_c = buildings.copy()
    buildings_c['geometry'] = buildings_c.centroid  # make centroids

    centroids_w_bl_ID = gpd.sjoin(buildings_c, blocks_single, how='inner', op='intersects')

    bl_ID_to_uID = centroids_w_bl_ID[[unique_id, id_name]]

    buildings = buildings.merge(bl_ID_to_uID, on=unique_id)

    cells = cells.merge(bl_ID_to_uID, on=unique_id)

    blocks = cells.dissolve(by=id_name)
    blocks[id_name] = blocks.index

    buildings.to_file(buildings_to)
    cells.to_file(cells_to)
    blocks.to_file(blocks_to)
