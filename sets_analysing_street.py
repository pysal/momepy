import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

streets = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Character sets/s_street_shape.shp")

"""
street DIMENSION+SHAPE
correlation matrix
"""

sns.set(style="white")

values = ['length', 'linear', 'width', 'height', 'profile']

corr = streets[values].corr()
corr.to_csv('/Users/martin/Dropbox/StrathUni/PhD/00_Core Chapters/06 Identification '
            'of urban tissues/Characters subset testing/s_street_corr.csv')

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

new_labels = ['length', 'linearity', 'mean width', 'mean height', 'height / width ratio']

# Draw the heatmap with the mask and correct aspect ratio
g = sns.heatmap(corr, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True,
                xticklabels=new_labels, yticklabels=new_labels)

for ext in ['png', 'svg']:
    plt.savefig('/Users/martin/Dropbox/StrathUni/PhD/00_Core Chapters/06 Identification '
                'of urban tissues/Characters subset testing/s_street_corr.' + ext, dpi=300, bbox_inches='tight')
plt.gcf().clear()

"""
street DIMENSION+SHAPE
coefficient of variation
"""
sns.set(style="white")

df1 = pd.DataFrame(values, columns=['character'])
list = []
for v in values:
    list.append(np.nanstd(streets[v]) / np.nanmean(streets[v]))
df1['variation'] = list
df1.to_csv('/Users/martin/Dropbox/StrathUni/PhD/00_Core Chapters/06 Identification '
           'of urban tissues/Characters subset testing/s_street_var.csv')
sns.set(style="whitegrid")
sns.set_palette("hls")
sns.set_context(context='paper', font_scale=1, rc=None)
ax = sns.barplot(x="character", y="variation", data=df1)
sns.despine(bottom=True, left=True)
plt.ylabel("Coefficient of variation")
plt.xlabel("")
ax.set_xticklabels(new_labels, rotation=90)
# Text on the top of each barplot
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + 0.4,
            p.get_y() + p.get_height() + 0.02,
            '{:1.2f}'.format(height),
            ha="center", color='grey')
for ext in ['png', 'svg']:
    plt.savefig('/Users/martin/Dropbox/StrathUni/PhD/00_Core Chapters/06 Identification '
                'of urban tissues/Characters subset testing/s_street_var.' + ext, dpi=300, bbox_inches='tight')
plt.gcf().clear()
