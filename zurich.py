import momepy as mm
import geopandas as gpd
import multiprocessing

buildings = gpd.read_file('/Users/martin/Dropbox/StrathUni/PhD/Papers/Voronoi tesselation/Data/Prague/Jizak/buildings.shp')

# cadastre = gpd.read_file('files/zurich/cadastre_values.shp')
# cadastre = cadastre.rename(index=str, columns={"uID": 'cID'})
#
# cadastre = mm.covered_area_ratio(cadastre, buildings, column_name='car_new', area_column='area', look_for_area_column='b_area', id_column='cID')
# cadastre = mm.gini_index(cadastre, cadastre, source='car_new', column_name='gini_car', id_column='cID')
# cadastre.to_file('files/zurich/cadastre_values.shp')


def worker(buf):
    print('Calculating', buf)
    tessellation = mm.tessellation(buildings, cut_buffer=buf)
    tessellation.to_file('/Users/martin/Dropbox/StrathUni/PhD/Papers/Voronoi tesselation/Data/Prague/Jizak/tess_{}.shp'.format(buf))
    return

if __name__ == '__main__':
    jobs = []
    buffers = [10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300]
    for buf in buffers:
        p = multiprocessing.Process(target=worker, args=(buf,))
        jobs.append(p)
        p.start()

# buildings = mm.unique_id(buildings)
# tessellation = gpd.read_file('files/zurich/{0}_tessellation.shp'.format(buf))
# tessellation['geometry'] = tessellation.buffer(0)
# tessellation = tessellation.dissolve(by='cID')
# tessellation['cID'] = tessellation.index.astype('float')
#
# buildings = mm.area(buildings, 'b_area')
# tessellation = mm.frequency(tessellation, tessellation, column_name='frequency', id_column='cID')
#
# tessellation = mm.area(tessellation, column_name='area')
# tessellation = mm.longest_axis_length(tessellation, column_name='longest_ax')
# tessellation = mm.circular_compactness(tessellation, column_name='compactnes', area_column='area')
# tessellation = mm.shape_index(tessellation, column_name='shape_idx', area_column='area', longest_axis_column='longest_ax')
# tessellation = mm.rectangularity(tessellation, column_name='rectangula', area_column='area')
# tessellation = mm.perimeter(tessellation, 'per')
# tessellation = mm.fractal_dimension(tessellation, column_name='fractal', area_column='area', perimeter_column='per')
# tessellation = mm.orientation(tessellation, column_name='orient')
# tessellation = mm.covered_area_ratio(tessellation, buildings, column_name='car', area_column='area',
#                                      look_for_area_column='b_area', id_column='cID')
# tessellation = mm.gini_index(tessellation, tessellation, source='area', column_name='gini_area', id_column='cID')
# tessellation = mm.gini_index(tessellation, tessellation, source='car', column_name='gini_car', id_column='cID')
# tessellation.to_file('files/zurich/{0}_tessellation_values.shp'.format(buf))

# cadastre = gpd.read_file('files/zurich/Zurich_cadastre.shp')
#
# cadastre = mm.longest_axis_length(cadastre, column_name='longest_ax')
# cadastre = mm.circular_compactness(cadastre, column_name='compactnes', area_column='area')
# cadastre = mm.shape_index(cadastre, column_name='shape_idx', area_column='area', longest_axis_column='longest_ax')
# cadastre = mm.rectangularity(cadastre, column_name='rectangula', area_column='area')
# cadastre = mm.perimeter(cadastre, 'per')
# cadastre = mm.fractal_dimension(cadastre, column_name='fractal', area_column='area', perimeter_column='per')
# cadastre = mm.orientation(cadastre, column_name='orient')
# cadastre = mm.covered_area_ratio(cadastre, buildings, column_name='car', area_column='area', look_for_area_column='b_area')
# cadastre = mm.frequency(cadastre, cadastre, column_name='frequency')
# cadastre = mm.gini_index(cadastre, cadastre, source='area', column_name='gini_area')
# cadastre = mm.gini_index(cadastre, cadastre, source='car', column_name='gini_car')
# cadastre.to_file('files/zurich/cadastre_values.shp')

# remember - it is calculated by uID. before comparioson, you have to join it together by cID to have same areas as cadastre
#
# cadastre = gpd.read_file('/Users/martin/Dropbox/StrathUni/PhD/Papers/Voronoi tesselation/Data/Zurich/_momepy/values/cadastre_values.shp')
# cadastre['car'].max()
# cadastre = cadastre.rename(index=str, columns={"uID": 'cID'})
#
# buildings = gpd.read_file('/Users/martin/Dropbox/StrathUni/PhD/Papers/Voronoi tesselation/Data/Zurich/Final Data/Buildings/Buildings_clean_id.shp')
# buildings = buildings.dissolve(by='cID')
#
# buildings['cID'] = buildings.index.astype('float')
# buildings = mm.area(buildings, column_name='area2')
# buildings = buildings.reset_index(drop=True)
#
# cadastre2 = mm.covered_area_ratio(cadastre, buildings, column_name='car2', area_column='area', look_for_area_column='area2', id_column='cID')
# cadastre2['cID'][0]  buildings['cID'][0]
#
#
# look_for = buildings
# column_name = 'car2'
# area_column = 'area'
# look_for_area_column = 'area2'
# id_column = 'cID'
# objects = cadastre
# from tqdm import tqdm
#
# print('Merging DataFrames...')
# look_for = look_for[[id_column, look_for_area_column]]  # keeping only necessary columns
# objects_merged = objects.merge(look_for, on=id_column)  # merging dataframes together
#
# print('Calculating CAR...')
#
# # define new column
# objects_merged[column_name] = None
# objects_merged[column_name] = objects_merged[column_name].astype('float')
#
# # fill new column with the value of area, iterating over rows one by one
# for index, row in tqdm(objects_merged.iterrows(), total=objects_merged.shape[0]):
#         objects_merged.loc[index, column_name] = row[look_for_area_column] / row[area_column]
#
# # transfer data from merged df to original df
# print('Merging data...')
# tomerge = objects_merged[[id_column, column_name]]
# objects = objects.merge(tomerge, on=id_column)
#
# print('Covered area ratio calculated.')
#
# objects['car2'].max()
#
#
