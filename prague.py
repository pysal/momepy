import momepy as mm
import geopandas as gpd

# buildings = '/Users/martin/Strathcloud/Personal Folders/Test data/Prague/Prague/Buildings/prg_buildings_values_bID.shp'
cells = 'C:/Users/wlb17212/ShareFile/Personal Folders/Test data/Prague/Prague/Cells/prg_tesselation50_values_bID.shp'
# blocks = '/Users/martin/Strathcloud/Personal Folders/Test data/Prague/Prague/Blocks/05prg_blocks.shp'
# p7 = "/Users/martin/Strathcloud/Personal Folders/Test data/Prague/p7-6.shp"

# load
# print('Loading file', buildings)
# blg = gpd.read_file(buildings)  # load file into geopandas
# print('Shapefile loaded.')

print('Loading file', cells)
voro = gpd.read_file(cells)  # load file into geopandas
print('Shapefile loaded.')

# print('Loading file', blocks)
# blo = gpd.read_file(blocks)  # load file into geopandas
# print('Shapefile loaded.')

# # mm.object_height_prg(blg, column_name='pdbHei', floors_column='POCET_POD', floor_type='TYP')
# mm.object_area(objects=blg, column_name='pdbAre')
# mm.object_floor_area(blg, column_name='pdbFlA', area_column='pdbAre', height_column='pdbHei', area_calculated=True)
# mm.object_volume(blg, column_name='pdbVol', area_column='pdbAre', height_column='pdbHei', area_calculated=True)
# mm.object_perimeter(blg, column_name='pdbPer')
# mm.courtyard_area(objects=blg, column_name='pdbCoA', area_column='pdbAre', area_calculated=True)
#
# print('Saving file to', buildings_saveas)
# blg.to_file(buildings_saveas)
# print('File saved.')
#
# mm.longest_axis_length2(voro, column_name='pdcLAL')
# mm.longest_axis_length2(blg, column_name='pdbLAL')
# mm.form_factor(blg, column_name='psbFoF', area_column='pdbAre', volume_column='pdbVol')
# mm.fractal_dimension(blg, column_name='psbFra', area_column='pdbAre', perimeter_column='pdbPer')
# mm.volume_facade_ratio(blg, column_name='psbVFR', volume_column='pdbVol', perimeter_column='pdbPer', height_column='pdbHei')
#
# print('Saving file to', buildings_saveas)
# blg.to_file(buildings_saveas)
# print('File saved.')
#
# print('Saving file to', cells_saveas)
# voro.to_file(cells_saveas)
# print('File saved.')
#
# mm.compactness_index(blg, column_name='psbCom', area_column='pdbAre')
# mm.compactness_index2(blg, column_name='psbCom2', area_column='pdbAre', perimeter_column='pdbPer')
# mm.convexeity(blg, column_name='psbCon', area_column='pdbAre')
# mm.courtyard_index(blg, column_name='psbCoI', area_column='pdbAre', courtyard_column='pdbCoA')
# mm.corners(blg, column_name='psbCor')
#
# print('Saving file to', buildings_saveas)
# blg.to_file(buildings_saveas)
# print('File saved.')

# mm.shape_index(blg, column_name='psbShI', area_column='pdbAre', longest_axis_column='pdbLAL')
# mm.squareness(blg, column_name='psbSqu')
# mm.equivalent_rectangular_index(blg, column_name='psbERI', area_column='pdbAre', perimeter_column='pdbPer')

# mm.elongation(blg, column_name='psbElo')

# mm.centroid_corners(blg, column_name='psbCCD')
# mm.object_area(voro, column_name='ptcAre')

# mm.compactness_index(voro, column_name='pscCom', area_column='ptcAre')
#
# print('Saving file to', buildings_saveas)
# blg.to_file(buildings_saveas)
# print('File saved.')
#
# print('Saving file to', cells_saveas)
# voro.to_file(cells_saveas)
# print('File saved.')

# mm.elongation(voro, column_name='pscElo')
# mm.orientation(blg, column_name='ptbOri')
# mm.covered_area_ratio(voro, blg, column_name='pivCAR', area_column='ptcAre', look_for_area_column="pdbAre", id_column="uID")
# mm.frequency(voro, voro, column_name='vicFre')
#
# print('Saving file to', buildings_saveas)
# blg.to_file(buildings_saveas)
# print('File saved.')
#
# print('Saving file to', cells_saveas)
# voro.to_file(cells_saveas)
# print('File saved.')

# mm.gini_index(voro, voro, source='ptcAre', column_name='vvcGAr')
# print('Saving file to', cells_saveas)
# voro.to_file(cells_saveas)
# print('File saved.')
# # mm.gini_index(voro, blg, source='pdbAre', column_name='vvbGAr')
# mm.gini_index(voro, voro, source='pivCAR', column_name='vvvGCA')
# print('Saving file to', cells_saveas)
# voro.to_file(cells_saveas)
# print('File saved.')
# # join values to cells
# # mm.moran_i_local()
# mm.moran_i_local(voro, source='pivCAR', column_name='uvvMCA')
#
# print('Saving file to', cells_saveas)
# voro.to_file(cells_saveas)
# print('File saved.')
# mm.cell_floor_area_ratio(cells, buildings)

# mm.object_area(blo, column_name='bdkAre')
# mm.object_perimeter(blo, column_name='bdkPer')
# mm.longest_axis_length2(blo, column_name='bdkSec')
# mm.elongation(blo, column_name='bskElo')
# mm.fractal_dimension(blo, column_name='bskFra', area_column='bdkAre', perimeter_column='bdkPer')
# mm.compactness_index(blo, column_name='bskCom', area_column='bdkAre')
# mm.compactness_index2(blo, column_name='bskCom2', area_column='bdkAre', perimeter_column='bdkPer')
# mm.convexeity(blo, column_name='bskCon', area_column='bdkAre')
# mm.shape_index(blo, column_name='bskShI', area_column='bdkAre', longest_axis_column='bdkSec')
# mm.equivalent_rectangular_index(blo, column_name='bskERI', area_column='bdkAre', perimeter_column='bdkPer')
# mm.orientation(blo, column_name='btkOri')

mm.moran_i_local(voro, source='pivFAR', column_name='uvvMFA')

print('Saving file to', cells)
voro.to_file(cells)
print('File saved.')
