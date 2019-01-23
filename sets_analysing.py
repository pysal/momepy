import geopandas as gpd
import momepy as mm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

buildings = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Character sets/s_building_dimension.shp")

# plotting correlation matrix
sns.set(style="white")

values = ['pdbHei', 'area', 'fl_area', 'volume', 'perimet',
          'courtyard', 'bboxarea', 'bboxper', 'bboxwid', 'bboxlen', 'circle_r',
          'ch_area', 'ch_perim']

corr = buildings[values].corr()
corr.to_csv('/Users/martin/Dropbox/StrathUni/PhD/00_Core Chapters/06 Identification of urban tissues/Characters subset testing/s_building_dimension_corr.csv')

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

new_labels = ['height', 'area', 'floor area', 'volume', 'perimeter',
              'courtyards', 'bounding rectangle area', 'bounding rectangle perimeter',
              'bounding rectangle width', 'bounding rectangle length',
              'enclosing circle radius', 'convex hull area', 'convex hull perimeter']

# Draw the heatmap with the mask and correct aspect ratio
g = sns.heatmap(corr, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True,
                xticklabels=new_labels, yticklabels=new_labels)

plt.savefig('/Users/martin/Dropbox/StrathUni/PhD/00_Core Chapters/06 Identification of urban tissues/Characters subset testing/s_building_dimension_corr.svg', dpi=300, bbox_inches='tight')


# Global Spatial Autocorrelation
tessellation = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Momepy190117/prg_g_tessellation_bID.shp")

w = mm.Queen_higher(tessellation, k=3)
w.n
import esda

moran = esda.Moran(buildings[['courtyard']], w)
moran.I
moran.z_sim
moran.p_sim
moran.EI
from splot.esda import plot_moran

plot_moran(moran, zstandard=True, figsize=(10,4))
plt.show()


tess_com = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/archive/Prague/Cells/prg_tesselation50_values_bID.shp")
w_com = mm.Queen_higher(tess_com, k=6)

moran2 = esda.Moran(tess_com[['pscCom']], w_com)
moran2.I
moran2.p_sim
moran2.z_sim

import numpy as np
tess_com['randNumCol'] = np.random.randint(1, 100, tess_com.shape[0])
moranrand = esda.Moran(tess_com[['randNumCol']], k1)
moranrand.I
moranrand.p_sim
moranrand.z_sim
plot_moran(moran2, zstandard=True, figsize=(10, 4))
lm = esda.Moran_Local(tess_com[['pscCom']], w_com)  # calculate LISA Local Moran's I
from splot.esda import lisa_cluster, moran_scatterplot, plot_local_autocorrelation
lisa_cluster(lm, tess_com, p=0.05, figsize=(10, 10))

import libpysal
k1 = libpysal.weights.Queen.from_dataframe(tess_com)
morank1 = esda.Moran(tess_com[['pscCom']], k1)
morank1.I
morank1.p_sim
morank1.z_sim
lmk1 = esda.Moran_Local(tess_com[['pscCom']], k1)  # calculate LISA Local Moran's I
lisa_cluster(lmk1, tess_com, p=0.05, figsize=(15, 15))
