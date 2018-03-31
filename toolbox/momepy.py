from area import object_area
from rectangularity_idx import object_rectangularity_idx
from compactness_idx import *
from convexity_idx import object_convexity_idx

path = "/Users/martin/Strathcloud/Personal Folders/Test data/Royston/buildings.shp"

object_rectangularity_idx(path, 'rectan')
object_area(path, 'area')
object_compactness_index(path, 'compact')
object_convexity_idx(path, 'convex')
