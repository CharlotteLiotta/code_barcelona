import geopandas as gpd # type: ignore
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from sklearn.cluster import DBSCAN # type: ignore

from import_data import *
from plotting_tools import *

path_data = "../data_barcelona/"
center = "0801901025"

#STEP 1: EMPLOYMENT DATA
gdf = import_data(path_data, center, option = "SECTION")

#Import data on employment
jobs = pd.read_csv(path_data + 'employment_distrib.csv') #from the INE API
jobs = jobs.loc[:,["SPERSONAS", "ID_LUGAR_TRAB_N3"]]
jobs["code_city"] = jobs["ID_LUGAR_TRAB_N3"].str[:5]
jobs = jobs.loc[:,["SPERSONAS", "code_city"]]

#STEP 2: DOWNSCALE USING LAND USE

land_cover = gpd.read_file(path_data + "cobertes-sol-v1r0-2023.gpkg", layer="cobertes_sol", engine="fiona")
layer_styles = gpd.read_file(path_data + "cobertes-sol-v1r0-2023.gpkg", layer="layer_styles", engine="fiona")
categories = gpd.read_file(path_data + "cobertes-sol-v1r0-2023.gpkg", layer="cobertes_sol_categories", engine="fiona")
gdf = import_data(path_data, center, option = "SECTION")
land_cover = land_cover.to_crs(gdf.crs)
intersection = gpd.overlay(land_cover, gdf, how='intersection')
intersection['area_lc'] = intersection.geometry.area
# Group by tract and land cover, summing areas
grouped = (
    intersection
    .groupby(['ID', 'nivell_2'])['area_lc']
    .sum()
    .reset_index()
)
pivot = grouped.pivot(index='ID', columns='nivell_2', values='area_lc').fillna(0)
pivot = pivot.reset_index()
land_use = gdf.loc[:,['ID', 'geometry', 'area']].merge(pivot, on = "ID", how = "left")

gdf = gdf.merge(land_use.loc[:,["ID", 347]], on = "ID", how = "left")

#Merge with gdf
gdf["code_city"] = gdf["ID"].str[:5]
gdf = gdf.merge(jobs, on = "code_city", how = "left")

size_city = gdf.loc[:,["code_city", 347]].groupby("code_city").sum(347)
size_city.columns = ["area_comm"]
gdf = gdf.merge(size_city, on = "code_city", how = "left")

gdf["density_employment"] = gdf["SPERSONAS"] / gdf["area_comm"]

gdf["employment"] = gdf["SPERSONAS"] * (gdf[347] / gdf["area_comm"])
gdf = gdf.drop(columns = ["SPERSONAS", "area_comm"])
gdf["density_employment"] = gdf["employment"] / gdf["area"]

gdf.loc[np.isnan(gdf.density_employment), "density_employment"] = 0

# Create decile bins with readable labels
bins = [0, 178, 398, 709, 1074, 1573, 2427, 3626, 6038, 11521, 45937]

gdf['decile'] = pd.cut(
    gdf['density_employment'],
    bins=bins,
    include_lowest=True,   # include the leftmost edge
    right=True,            # bins are right-closed: (a, b]
    labels=[f"{bins[i]}–{bins[i+1]}" for i in range(len(bins)-1)]           # gives integer codes 0–9 instead of intervals
)

# Plot with legend showing decile ranges
fig, ax = plt.subplots(figsize=(10, 10))

gdf.plot(
    column='decile',
    cmap='Reds',
    linewidth=0.2,
    edgecolor='black',
    legend=True,
    ax=ax,
    legend_kwds={
        "fontsize": 14,   # increase legend font size
        "loc": "lower right"  # position bottom-right
    },
    missing_kwds={
        "color": "lightgrey",      # fill color for missing values
        "edgecolor": "black",      # optional: keep outline for consistency
        "label": "Missing values"  # shows up in legend
    }
)

plt.axis('off')
#plt.title('Employment Density Deciles', fontsize=14)
plt.show()



### CLUSTERS

# Optional: filter out low-density tracts (e.g., bottom 20%)
gdf_here = gdf.loc[gdf['density_employment']>10,:].copy()

# --- Step 3: Extract spatial coordinates ---
# Use centroids in meters (projected CRS)
gdf_here = gdf_here.to_crs(epsg=3395)
gdf_here['x'] = gdf_here.centroid.x
gdf_here['y'] = gdf_here.centroid.y
coords = gdf_here[['x', 'y']].values

# --- Step 4: Run DBSCAN ---
# eps = max distance between tracts in meters (try values like 2000–5000)
# min_samples = minimum number of tracts to form a cluster


db = DBSCAN(eps=500, min_samples=3).fit(coords)
gdf_here['cluster'] = db.labels_

# Step 1: Ensure 'cluster' is categorical
#gdf_here['cluster'] = gdf_here['cluster'].astype('category')

clusters = gdf_here['cluster'].cat.categories if gdf_here['cluster'].dtype.name == 'category' else np.unique(gdf_here['cluster'])
non_noise_clusters = [c for c in clusters if c != -1]
n_clusters = len(non_noise_clusters)

base_cmap = plt.cm.get_cmap('tab20', 20)
colors_list = [mcolors.to_hex(base_cmap(i)) for i in range(20)]
colors_extended = (colors_list * (n_clusters // 20 + 1))[:n_clusters]

color_dict = {-1: '#B0B0B0'}  # grey for noise

cluster_colors = {c: colors_extended[i] for i, c in enumerate(non_noise_clusters)}
color_dict.update(cluster_colors)

# Map colors to GeoDataFrame
gdf_here['color'] = gdf_here['cluster'].map(color_dict).fillna('#cccccc')


# Create legend handles
handles = [
    mpatches.Patch(color=color, label=f"Cluster {cluster}")
    for cluster, color in sorted(color_dict.items())
]

fig, ax = plt.subplots(figsize=(12, 12))
gdf_here.plot(color=gdf_here['color'], ax=ax, linewidth=0.1, edgecolor='black')
plt.axis('off')
plt.title('Clusters (noise in grey)')
plt.legend(handles=handles, title="Clusters", loc='lower left', fontsize='small', frameon=False)
plt.show()





# Remove noise (label = -1)
gdf_here = gdf_here[gdf_here['cluster'] != -1]

# --- Step 5: Rank clusters by total jobs ---
cluster_jobs = gdf_here.groupby('cluster')['employment'].sum().sort_values(ascending=False)
cluster_jobs = pd.DataFrame(cluster_jobs)
np.nansum(cluster_jobs.employment)
cluster_jobs.columns = ["employment_cluster"]

gdf_here = gdf_here.merge(cluster_jobs, left_on ="cluster", right_index = True)
gdf_here = gdf_here.loc[gdf_here.employment_cluster > 500,:]

gdf = gdf.merge(gdf_here.loc[:,["cluster", "employment_cluster", "ID"]], on = "ID", how = "left")

plot_with_missing(gdf, gdf["cluster"])

cluster_geom = gdf.dissolve(by='cluster')

# Compute the centroid of each cluster geometry
cluster_geom['centroid'] = cluster_geom.geometry.centroid

# Plot base geometries
fig, ax = plt.subplots(figsize=(12, 12))
gdf.plot(ax=ax, color='lightgrey', edgecolor='white', linewidth=0.1)

cluster_geom["geometry"] = cluster_geom["centroid"]

# Plot centroids scaled by a variable (e.g. 'jobs')
cluster_geom.plot(
    ax=ax,
    markersize=cluster_geom['employment_cluster'] / 200,  # adjust scaling as needed
    color='red',
    marker='o',
    label='Cluster Centroids'
)

plt.legend()
plt.axis('off')
plt.title("Clusters and Centroids Sized by Job Count")
plt.show()

cluster_geom.employment_cluster = cluster_geom.employment_cluster * (np.nansum(gdf["pop"])/np.nansum(cluster_geom.employment_cluster))

#### CHECK
plot_with_missing(gdf, gdf["density_employment"])

gdf_city = gdf.loc[:,["code_city", "geometry", "employment"]].dissolve(by='code_city', aggfunc='sum')

# 2. Optional: reset index for plotting/merging
gdf_city = gdf_city.reset_index()
gdf_city.plot(column='employment', legend=True, cmap='OrRd', edgecolor='white', linewidth=0.3)
plt.title("Jobs per City")
plt.axis('off')
plt.show()

cluster_geom_city = cluster_geom.loc[:,["code_city", "employment_cluster"]].groupby("code_city").sum()


comparison = gdf_city.merge(cluster_geom_city, left_on = "code_city", right_index = True, how = "outer")

plot_with_missing(comparison, comparison["employment"])
plot_with_missing(comparison, comparison["employment_cluster"])

plot_with_missing(comparison, 100 * (comparison["employment_cluster"] - comparison["employment"]) / comparison["employment"])
comparison["error"] = (comparison["employment_cluster"] - comparison["employment"])

cluster_geom.loc[:,["geometry", "employment_cluster"]].to_file(path_data + "cluster_employment.shp")

#gdf = gdf.drop(columns = ["cluster", "employment_cluster"])



fig, ax = plt.subplots(figsize=(8, 8))

# Plot polygons
gdf.plot(ax=ax, color="white", edgecolor="grey")

# Plot points on top
employment_centers.plot(ax=ax, color="red", markersize=20)
for x, y, label in zip(employment_centers.geometry.x, employment_centers.geometry.y, employment_centers["cluster"]):
    ax.text(x, y, label, fontsize=12, ha="right", va="bottom", color="red")
plt.show()