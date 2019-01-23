import geopandas as gpd
import momepy as mm
import pandas as pd

buildings_clean = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Clean data (wID)/Buildings/prg_buildings.shp")
buildings_mm = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Momepy190117/prg_g_buildings_bID.shp")
tessellation_mm = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Momepy190117/prg_g_tessellation_bID.shp")
tessellation = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Momepy190117/prg_g_tessellation.shp")

len(buildings_clean.index)
len(buildings_mm.index)
len(tessellation.index)
len(tessellation_mm.index)
(buildings_clean['uID'] != buildings_mm['uID']).any()
buildings_mm['uID'] == tessellation_mm['uID']
ids = list(buildings_mm['uID'])
ids2 = list(tessellation['uID'])

len(ids)
len(ids2)
set(ids).difference(ids2)

duplicates = pd.concat(g for _, g in tessellation_mm.groupby("uID") if len(g) > 1)
duplicates.to_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Tests/190117/dup_tess.shp")

collapse = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Tests/190117/collapse.shp")
tess_collapse = mm.tessellation(collapse)

len(tess_collapse.index)
    import geopandas as gpd
    import shapely

    p1 = shapely.geometry.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    p2 = shapely.geometry.Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])

    g = gpd.GeoSeries([p1, p2])

    gdf = gpd.GeoDataFrame(geometry=g)
    gdf['angle'] = [25, 45]

    for index, row in gdf.iterrows():
        rotated = shapely.affinity.rotate(row['geometry'], row['angle'])
        gdf.loc[index, 'geometry'] = rotated

    gdf.plot()

file = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Tests/190117/dup_tess.shp")
file.to_csv("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Tests/190117/dup_tess.csv")
df = pd.read_csv("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Tests/190117/dup_tess.csv")
df.rename(index=str, columns={"geometry": "pols"}, inplace=True)
gdf = gpd.GeoDataFrame(df)
gdf.plot()
plt.show()

df = ['one', 'two']
gdf = gpd.GeoDataFrame(df)

from shapely.geometry import Point


def points_from_xy(x, y, z=None):
    """
    Generate list of shapely.Point geometries from x, y(, z) cordinates.

    Parameters
    ----------
    x, y, z : array

    Examples
    --------
    >>> geometry = geopandas.points_from_xy(x=[1, 0], y=[0, 1])
    >>> geometry = geopandas.points_from_xy(df['x'], df['y'], df['z'])
    >>> gdf = geopandas.GeoDataFrame(df,
                                     geometry=geopandas.points_from_xy(df['x'],
                                                                       df['y']))

    Returns
    -------
    list : list
    """
    if not len(x) == len(y):
        raise ValueError("x and y arrays must be equal length.")
    if z is not None:
        if not len(z) == len(x):
            raise ValueError("z array must be same length as x and y.")
        geom = [Point(i, j, k) for i, j, k in zip(x, y, z)]
    else:
        geom = [Point(i, j) for i, j in zip(x, y)]
    return geom


from geopandas import GeoSeries, GeoDataFrame

import numpy as np
a = np.array([[1, 2], [3, 4]])

a = {'a': 1, 'b': 5}
b = [2, 1]
c = [4, 6]

av = a.values()
df = pd.DataFrame({'x': b,'y': c})
g = points_from_xy(a, gdf['y'])
g
gdf = gdf.set_geometry(g)
gdf.plot()
geometry = points_from_xy(x=[1, 0], y=[0, 1])

level = [Point(i, j) for i, j in zip(X, Y)]
gdf = gpd.GeoDataFrame(df,
                       geometry=points_from_xy(df['x'],
                                               df['y']))
df = GeoDataFrame([{'other_geom': Point(x, x), 'x': x, 'y': x, 'z': x} for x in range(10)])
gs = [Point(x, x) for x in range(10)]
geometry1 = points_from_xy(df['x'], df['y'])
assert geometry1 == gs


df = pd.DataFrame(
    {'City': ['Buenos Aires', 'Brasilia', 'Santiago', 'Bogota', 'Caracas'],
     'Country': ['Argentina', 'Brazil', 'Chile', 'Colombia', 'Venezuela'],
     'Latitude': [-34.58, -15.78, -33.45, 4.60, 10.48],
     'Longitude': [-58.66, -47.91, -70.66, -74.08, -66.86]})

gdf = GeoDataFrame(df, geometry=points_from_xy(df['Longitude'], df['Latitude']))

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# We restrict to South America.
ax = world[world.continent == 'South America'].plot(
    color='white', edgecolor='black')

# We can now plot our GeoDataFrame.
gdf.plot(ax=ax, color='red')

plt.show()



import pandas as pd
import geopandas
import matplotlib.pyplot as plt

###############################################################################
# From longitudes and latitudes
# =============================
#
# First, let's consider a ``DataFrame`` containing cities and their respective
# longitudes and latitudes.

df = pd.DataFrame(
    {'City': ['Buenos Aires', 'Brasilia', 'Santiago', 'Bogota', 'Caracas'],
     'Country': ['Argentina', 'Brazil', 'Chile', 'Colombia', 'Venezuela'],
     'Latitude': [-34.58, -15.78, -33.45, 4.60, 10.48],
     'Longitude': [-58.66, -47.91, -70.66, -74.08, -66.86]})

###############################################################################
# A ``GeoDataFrame`` needs a ``shapely`` object. We use geopandas
# ``points_from_xy()`` to transform **Longitude** and **Latitude** into a list
# of ``shapely.Point`` objects and set it as a ``geometry`` while creating the
# ``GeoDataFrame``.

gdf = geopandas.GeoDataFrame(df,
                             geometry=points_from_xy(df.Longitude, df.Latitude))


###############################################################################
# ``gdf`` looks like this :

print(gdf.head())

###############################################################################
# Finally, we plot the coordinates over a country-level map.

world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

# We restrict to South America.
ax = world[world.continent == 'South America'].plot(
    color='white', edgecolor='black')

# We can now plot our GeoDataFrame.
gdf.plot(ax=ax, color='red')

plt.show()
