import momepy as mm
import geopandas as gpd
import pandas as pd

buffers = pd.Series([10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300])
buildings = gpd.read_file('files/zurich/Buildings_clean_id.shp')
buildings = mm.area(buildings, 'b_area')
buildings = mm.unique_id(buildings)

for buf in buffers:
    tessellation = mm.tessellation(buildings, cut_buffer=buf)
    tessellation = tessellation.drop(['b_area'], axis=1)
    tessellation = mm.area(tessellation, column_name='area')
    tessellation = mm.longest_axis_length(tessellation, column_name='longest_ax')
    tessellation = mm.circular_compactness(tessellation, column_name='compactnes', area_column='area')
    tessellation = mm.shape_index(tessellation, column_name='shape_idx', area_column='area', longest_axis_column='longest_ax')
    tessellation = mm.rectangularity(tessellation, column_name='rectangula', area_column='area')
    tessellation = mm.perimeter(tessellation, 'per')
    tessellation = mm.fractal_dimension(tessellation, column_name='fractal', area_column='area', perimeter_column='per')
    tessellation = mm.orientation(tessellation, column_name='orient')
    tessellation = mm.covered_area_ratio(tessellation, buildings, column_name='car', area_column='area', look_for_area_column='b_area')
    tessellation = mm.frequency(tessellation, tessellation, column_name='frequency')
    tessellation = mm.gini_index(tessellation, tessellation, source='area', column_name='gini_area')
    tessellation = mm.gini_index(tessellation, tessellation, source='car', column_name='gini_car')
    tessellation.to_file('files/zurich/{0}_tessellation.shp'.format(buf))

cadastre = gpd.read_file('files/zurich/Zurich_cadastre.shp')

cadastre = mm.longest_axis_length(cadastre, column_name='longest_ax')
cadastre = mm.circular_compactness(cadastre, column_name='compactnes', area_column='area')
cadastre = mm.shape_index(cadastre, column_name='shape_idx', area_column='area', longest_axis_column='longest_ax')
cadastre = mm.rectangularity(cadastre, column_name='rectangula', area_column='area')
cadastre = mm.perimeter(cadastre, 'per')
cadastre = mm.fractal_dimension(cadastre, column_name='fractal', area_column='area', perimeter_column='per')
cadastre = mm.orientation(cadastre, column_name='orient')
cadastre = mm.covered_area_ratio(cadastre, buildings, column_name='car', area_column='area', look_for_area_column='b_area')
cadastre = mm.frequency(cadastre, cadastre, column_name='frequency')
cadastre = mm.gini_index(cadastre, cadastre, source='area', column_name='gini_area')
cadastre = mm.gini_index(cadastre, cadastre, source='car', column_name='gini_car')
cadastre.to_file('files/zurich/cadastre_values.shp')

# remember - it is calculated by uID. before comparioson, you have to join it together by cID to have same areas as cadastre
