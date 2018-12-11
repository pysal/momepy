import momepy as mm
import geopandas as gpd
from tqdm import tqdm

buildings = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Momepy 181116/v2/mm_buildings_bID.shp")

granularity = mm.block_density(buildings, 'granula', blocks, 'bID', 'uID')

granularity.to_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Tests/181205/granularity.shp")
to_blocks = granularity[['bID', 'granula']]

cleaned = to_blocks.drop_duplicates(subset='bID')

for index, row in tqdm(blocks.iterrows(), total=blocks.shape[0]):
    value = cleaned[cleaned['bID'] == row['bID']].iloc[0]['granula']
    blocks.loc[index, 'granula'] = value

blocks.to_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Tests/181205/blocks_granularity.shp")
