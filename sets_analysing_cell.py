import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statistics
import pandas as pd

tessellation = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Character sets/s_tessellation_dimension.shp")

"""
tessellation DIMENSION
correlation matrix
"""

sns.set(style="white")

values_d = ['area', 'lal', 'perimet',
            'bboxarea', 'bboxper', 'bboxwid', 'bboxlen', 'circle_r',
            'ch_area', 'ch_perim']

corr = tessellation[values_d].corr()
corr.to_csv('/Users/martin/Dropbox/StrathUni/PhD/00_Core Chapters/06 Identification '
            'of urban tissues/Characters subset testing/s_tessellation_dimension_corr.csv')

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

new_labels_d = ['area', 'longest axis length', 'perimeter',
                'bounding rectangle area', 'bounding rectangle perimeter',
                'bounding rectangle width', 'bounding rectangle length',
                'enclosing circle radius', 'convex hull area', 'convex hull perimeter']

# Draw the heatmap with the mask and correct aspect ratio
g = sns.heatmap(corr, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True,
                xticklabels=new_labels_d, yticklabels=new_labels_d)

for ext in ['png', 'svg']:
    plt.savefig('/Users/martin/Dropbox/StrathUni/PhD/00_Core Chapters/06 Identification '
                'of urban tissues/Characters subset testing/s_tessellation_dimension_corr.' + ext, dpi=300, bbox_inches='tight')
plt.gcf().clear()

"""
tessellation DIMENSION
coefficient of variation
"""
sns.set(style="white")

df1 = pd.DataFrame(values_d, columns=['character'])
list = []
for v in values_d:
    list.append(statistics.stdev(tessellation[v]) / statistics.mean(tessellation[v]))
df1['variation'] = list
df1.to_csv('/Users/martin/Dropbox/StrathUni/PhD/00_Core Chapters/06 Identification '
           'of urban tissues/Characters subset testing/s_tessellation_dimension_var.csv')
sns.set(style="whitegrid")
sns.set_palette("hls")
sns.set_context(context='paper', font_scale=1, rc=None)
ax = sns.barplot(x="character", y="variation", data=df1)
sns.despine(bottom=True, left=True)
plt.ylabel("Coefficient of variation")
plt.xlabel("")
ax.set_xticklabels(new_labels_d, rotation=90)
# Text on the top of each barplot
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + 0.4,
            p.get_y() + p.get_height() + 0.02,
            '{:1.2f}'.format(height),
            ha="center", color='grey')
for ext in ['png', 'svg']:
    plt.savefig('/Users/martin/Dropbox/StrathUni/PhD/00_Core Chapters/06 Identification '
                'of urban tissues/Characters subset testing/s_tessellation_dimension_var.' + ext, dpi=300, bbox_inches='tight')
plt.gcf().clear()


"""
tessellation SHAPE
correlation matrix
"""
tessellation = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Character sets/s_tessellation_shape.shp")

# plotting correlation matrix
sns.set(style="white")

values_s = ['fractal', 'circom', 'squcom', 'convex', 'shape', 'eri', 'elo']

corr = tessellation[values_s].corr()
corr.to_csv('/Users/martin/Dropbox/StrathUni/PhD/00_Core Chapters/06 Identification '
            'of urban tissues/Characters subset testing/s_tessellation_shape_corr.csv')

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

new_labels_s = ['fractal dimension',
                'circular compactness', 'square compactness', 'convexeity',
                'shape index', 'equivalent rectangular index', 'elongation']

# Draw the heatmap with the mask and correct aspect ratio
g = sns.heatmap(corr, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True,
                xticklabels=new_labels_s, yticklabels=new_labels_s)
for ext in ['png', 'svg']:
    plt.savefig('/Users/martin/Dropbox/StrathUni/PhD/00_Core Chapters/06 Identification '
                'of urban tissues/Characters subset testing/s_tessellation_shape_corr.' + ext, dpi=300, bbox_inches='tight')
plt.gcf().clear()

"""
coefficient of variation
"""
sns.set(style="white")

df1 = pd.DataFrame(values_s, columns=['character'])
list = []
for v in values_s:
    list.append(statistics.stdev(tessellation[v]) / statistics.mean(tessellation[v]))
df1['variation'] = list
df1.to_csv('/Users/martin/Dropbox/StrathUni/PhD/00_Core Chapters/06 Identification '
           'of urban tissues/Characters subset testing/s_tessellation_shape_var.csv')
sns.set(style="whitegrid")
sns.set_palette("hls")
sns.set_context(context='paper', font_scale=1, rc=None)
ax = sns.barplot(x="character", y="variation", data=df1)
sns.despine(bottom=True, left=True)
plt.ylabel("Coefficient of variation")
plt.xlabel("")
ax.set_xticklabels(new_labels_s, rotation=90)
# Text on the top of each barplot
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + 0.4,
            p.get_y() + p.get_height() + 0.005,
            '{:1.2f}'.format(height),
            ha="center", color='grey')
for ext in ['png', 'svg']:
    plt.savefig('/Users/martin/Dropbox/StrathUni/PhD/00_Core Chapters/06 Identification '
                'of urban tissues/Characters subset testing/s_tessellation_shape_var.' + ext, dpi=300, bbox_inches='tight')
plt.gcf().clear()

"""
tessellation DISTRIBUTION
correlation matrix
"""
tessellation = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Character sets/s_tessellation_dist.shp")

# plotting correlation matrix
sns.set(style="white")

values_t = ['orient', 's_align']

corr = tessellation[values_t].corr()
corr.to_csv('/Users/martin/Dropbox/StrathUni/PhD/00_Core Chapters/06 Identification '
            'of urban tissues/Characters subset testing/s_tessellation_dist_corr.csv')

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

new_labels_t = ['orientation', 'street alignment']

# Draw the heatmap with the mask and correct aspect ratio
g = sns.heatmap(corr, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True,
                xticklabels=new_labels_t, yticklabels=new_labels_t)
for ext in ['png', 'svg']:
    plt.savefig('/Users/martin/Dropbox/StrathUni/PhD/00_Core Chapters/06 Identification '
                'of urban tissues/Characters subset testing/s_tessellation_dist_corr.' + ext, dpi=300, bbox_inches='tight')
plt.gcf().clear()

"""
coefficient of variation
"""
sns.set(style="white")

df1 = pd.DataFrame(values_t, columns=['character'])
list = []
for v in values_t:
    list.append(statistics.stdev(tessellation[v]) / statistics.mean(tessellation[v]))
df1['variation'] = list
df1.to_csv('/Users/martin/Dropbox/StrathUni/PhD/00_Core Chapters/06 Identification '
           'of urban tissues/Characters subset testing/s_tessellation_dist_var.csv')
sns.set(style="whitegrid")
sns.set_palette("hls")
sns.set_context(context='paper', font_scale=1, rc=None)
ax = sns.barplot(x="character", y="variation", data=df1)
sns.despine(bottom=True, left=True)
plt.ylabel("Coefficient of variation")
plt.xlabel("")
ax.set_xticklabels(new_labels_t, rotation=90)
# Text on the top of each barplot
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + 0.4,
            p.get_y() + p.get_height() + 0.005,
            '{:1.2f}'.format(height),
            ha="center", color='grey')
for ext in ['png', 'svg']:
    plt.savefig('/Users/martin/Dropbox/StrathUni/PhD/00_Core Chapters/06 Identification '
                'of urban tissues/Characters subset testing/s_tessellation_dist_var.' + ext, dpi=300, bbox_inches='tight')
plt.gcf().clear()
"""
tessellation INTENSITY
correlation matrix
"""
tessellation = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Character sets/s_tessellation_int.shp")

# plotting correlation matrix
sns.set(style="white")

values_i = ['car', 'far']

corr = tessellation[values_i].corr()
corr.to_csv('/Users/martin/Dropbox/StrathUni/PhD/00_Core Chapters/06 Identification '
            'of urban tissues/Characters subset testing/s_tessellation_int_corr.csv')

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

new_labels_i = ['coverage area ratio', 'floor area ratio']

# Draw the heatmap with the mask and correct aspect ratio
g = sns.heatmap(corr, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True,
                xticklabels=new_labels_i, yticklabels=new_labels_i)
for ext in ['png', 'svg']:
    plt.savefig('/Users/martin/Dropbox/StrathUni/PhD/00_Core Chapters/06 Identification '
                'of urban tissues/Characters subset testing/s_tessellation_int_corr.' + ext, dpi=300, bbox_inches='tight')
plt.gcf().clear()

"""
coefficient of variation
"""
sns.set(style="white")

df1 = pd.DataFrame(values_i, columns=['character'])
list = []
for v in values_i:
    list.append(statistics.stdev(tessellation[v]) / statistics.mean(tessellation[v]))
df1['variation'] = list
df1.to_csv('/Users/martin/Dropbox/StrathUni/PhD/00_Core Chapters/06 Identification '
           'of urban tissues/Characters subset testing/s_tessellation_int_var.csv')
sns.set(style="whitegrid")
sns.set_palette("hls")
sns.set_context(context='paper', font_scale=1, rc=None)
ax = sns.barplot(x="character", y="variation", data=df1)
sns.despine(bottom=True, left=True)
plt.ylabel("Coefficient of variation")
plt.xlabel("")
ax.set_xticklabels(new_labels_i, rotation=90)
# Text on the top of each barplot
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + 0.4,
            p.get_y() + p.get_height() + 0.005,
            '{:1.2f}'.format(height),
            ha="center", color='grey')
for ext in ['png', 'svg']:
    plt.savefig('/Users/martin/Dropbox/StrathUni/PhD/00_Core Chapters/06 Identification '
                'of urban tissues/Characters subset testing/s_tessellation_int_var.' + ext, dpi=300, bbox_inches='tight')
plt.gcf().clear()
"""
tessellation ALL
correlation matrix
"""
tessellation = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Character sets/s_tessellation_int.shp")

# plotting correlation matrix
sns.set(style="white")

values = values_d + values_s + values_t + values_i

corr = tessellation[values].corr()
corr.to_csv('/Users/martin/Dropbox/StrathUni/PhD/00_Core Chapters/06 Identification '
            'of urban tissues/Characters subset testing/s_tessellation_corr.csv')

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(22, 18))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

new_labels = new_labels_d + new_labels_s + new_labels_t + new_labels_i

# Draw the heatmap with the mask and correct aspect ratio
g = sns.heatmap(corr, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True,
                xticklabels=new_labels, yticklabels=new_labels)
for ext in ['png', 'svg']:
    plt.savefig('/Users/martin/Dropbox/StrathUni/PhD/00_Core Chapters/06 Identification '
                'of urban tissues/Characters subset testing/s_tessellation_corr.' + ext, dpi=300, bbox_inches='tight')
