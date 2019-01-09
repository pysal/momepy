'''
Spatial autocorrelation.
'''
from libpysal.weights import DistanceBand
import esda

import matplotlib.pyplot as plt
import pysal as ps
from libpysal.weights.contiguity import Queen
from libpysal import examples
import numpy as np
import pandas as pd
import geopandas as gpd
import os
import splot

import libpysal
import esda
import geopandas as gpd
from splot.esda import lisa_cluster, moran_scatterplot, plot_local_autocorrelation
import matplotlib.pyplot as plt


def Queen_higher(dataframe, k):
    first_order = libpysal.weights.Queen.from_dataframe(dataframe)
    joined = first_order
    for i in list(range(2, k + 1)):
        i_order = libpysal.weights.higher_order(first_order, k=i)
        joined = libpysal.weights.w_union(joined, i_order)
    return joined

tess50 = gpd.read_file('/Users/martin/Dropbox/StrathUni/PhD/Papers/Voronoi tesselation/Data/Zurich/_momepy/values_/50_tessellation_values.shp')

weights = Queen_higher(tess50, 3)
lm = esda.Moran_Local(tess50[['gini_area']], weights)  # calculate LISA Local Moran's I

fig, ax = moran_scatterplot(lm)

lisa_cluster(lm, tess50, p=0.05, figsize=(10, 10))

plot_local_autocorrelation(lm, tess50, 'gini_area')
plt.savefig('/Users/martin/Dropbox/StrathUni/PhD/Papers/Voronoi tesselation/Analysis/Results_mm/moran.png', dpi=300)


tess50['moran'] = lm.q
tess50['moran_p'] = lm.p_sim

from distutils.version import LooseVersion

if LooseVersion(gpd.__version__) < LooseVersion('0.3.0'):
    print('yes')
type(gpd.__version__)
