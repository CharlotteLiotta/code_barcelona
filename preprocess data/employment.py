import geopandas as gpd  # type: ignore
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.neighbors import KernelDensity
from sklearn.cluster import DBSCAN
from shapely.geometry import Point

from import_data import *
from plotting_tools import *

path_data = "../data_barcelona/"
center = "0801901025"

# ============================================================
# STEP 1: Import employment and land use data (unchanged)
# ============================================================

gdf = import_data(path_data, center, option="SECTION")

jobs = pd.read_csv(path_data + 'employment_distrib.csv')
jobs = jobs.loc[:, ["SPERSONAS", "ID_LUGAR_TRAB_N3"]]
jobs["code_city"] = jobs["ID_LUGAR_TRAB_N3"].str[:5]
jobs = jobs.loc[:, ["SPERSONAS", "code_city"]]

land_cover = gpd.read_file(path_data + "cobertes-sol-v1r0-2023.gpkg", layer="cobertes_sol")
categories = gpd.read_file(path_data + "cobertes-sol-v1r0-2023.gpkg", layer="cobertes_sol_categories")
land_cover = land_cover.to_crs(gdf.crs)

intersection = gpd.overlay(land_cover, gdf, how='intersection')
intersection['area_lc'] = intersection.geometry.area

grouped = intersection.groupby(['ID', 'nivell_2'])['area_lc'].sum().reset_index()
pivot = grouped.pivot(index='ID', columns='nivell_2', values='area_lc').fillna(0).reset_index()

land_use = gdf.loc[:, ['ID', 'geometry', 'area']].merge(pivot, on="ID", how="left")
gdf = gdf.merge(land_use.loc[:, ["ID", 347]], on="ID", how="left")

gdf["code_city"] = gdf["ID"].str[:5]
gdf = gdf.merge(jobs, on="code_city", how="left")

size_city = gdf.loc[:, ["code_city", 347]].groupby("code_city").sum()
size_city.columns = ["area_comm"]
gdf = gdf.merge(size_city, on="code_city", how="left")

gdf["density_employment"] = gdf["SPERSONAS"] / gdf["area_comm"]
gdf["employment"] = gdf["SPERSONAS"] * (gdf[347] / gdf["area_comm"])
gdf = gdf.drop(columns=["SPERSONAS", "area_comm"])
gdf["density_employment"] = gdf["employment"] / gdf["area"]
gdf.loc[np.isnan(gdf.density_employment), "density_employment"] = 0

bins = [0, 178, 398, 709, 1074, 1573, 2427, 3626, 6038, 11521, np.inf]

labels = [
    "0–178",
    "178–398",
    "398–709",
    "709–1,074",
    "1,074–1,573",
    "1,573–2,427",
    "2,427–3,626",
    "3,626–6,038",
    "6,038–11,521",
    "> 11,521"
]

gdf["density_bin"] = pd.cut(
    gdf["density_employment"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

fig, ax = plt.subplots(1, 1, figsize=(6, 6))

gdf.plot(
    column="density_bin",
    cmap="Reds",
    categorical=True,
    legend=True,
    edgecolor="black",
    linewidth=0.1,
    ax=ax,
    legend_kwds={
        "loc": "lower right"
    }
)

#gpd.GeoSeries([CBD]).plot(ax=ax, color="blue", marker="*", markersize=100)

ax.axis("off")
plt.tight_layout()
plt.show()

# ============================================================
# STEP 2: Find CBD (unchanged)
# ============================================================

gdf["centroid"] = gdf.geometry.centroid
gdf["x"] = gdf.centroid.x
gdf["y"] = gdf.centroid.y
gdf.fillna({"employment": 0}, inplace=True)

#x_cbd = np.average(gdf["x"], weights=gdf["employment"])
#y_cbd = np.average(gdf["y"], weights=gdf["employment"])


#CBD = Point(x_cbd, y_cbd)
lon, lat = 2.170047, 41.387016 #placa catalunya
from pyproj import Transformer
transformer = Transformer.from_crs("EPSG:4326", "EPSG:25830", always_xy=True)
CBD = Point(transformer.transform(lon, lat))
#print(f"CBD coordinates: ({x_cbd:.3f}, {y_cbd:.3f})")

gdf["dist_cbd"] = gdf.centroid.distance(CBD)

# ============================================================
# STEP 3: McMillen Method (adapted)
# ============================================================

# --- Step 3.1: Compute density and log density
gdf["area_km2"] = gdf.geometry.area / 1e6
gdf["density"] = gdf["employment"] / gdf["area_km2"]
gdf["ln_density"] = np.log(gdf["density"] + 1)
gdf["dist_km"] = gdf["dist_cbd"] / 1000

# --- Step 3.2: Estimate monocentric gradient with LOESS (McMillen 2003)
# Replaces the simple OLS ln(density) = α + β * distance
gdf = gdf.sort_values("dist_km")
smoothed = lowess(gdf["ln_density"], gdf["dist_km"], frac=0.4, return_sorted=False)


gdf["fitted_ln_density"] = smoothed
gdf["residual"] = gdf["ln_density"] - gdf["fitted_ln_density"]

threshold = np.percentile(gdf["residual"], 85)   # try 85% or 90%
gdf["candidate"] = gdf["residual"] >= threshold
gdf.loc[gdf.dist_km < 1.5, "candidate"] = False #1.5

# --- Step 3.5: Cluster adjacent high-residual tracts (DBSCAN)
candidate_coords = np.vstack([gdf.loc[gdf["candidate"], "x"], gdf.loc[gdf["candidate"], "y"]]).T
clustering = DBSCAN(eps=500, min_samples=1).fit(candidate_coords)
gdf.loc[gdf["candidate"], "cluster"] = clustering.labels_


# --- Step 3.6: Extract subcenter centroids
subcenters = (
    gdf[gdf["candidate"] & (gdf["cluster"] != -1)]
    .groupby("cluster")[["x", "y", "residual"]]
    .mean()
    .reset_index()
)

# ============================================================
# STEP 4: Validation Regression (McMillen confirmatory step)
# ============================================================


gdf["dist_km_sq"] = gdf["dist_km"] ** 2

for idx, row in subcenters.iterrows():
    gdf[f"dist_{idx}"] = gdf.centroid.distance(Point(row["x"], row["y"])) / 1000
    gdf[f"inv_dist_{idx}"] = 1 / (gdf[f"dist_{idx}"] + 1e-9)  # avoid divide by zero

candidate_cols = []
for idx in subcenters.index:
    candidate_cols += [f"dist_{idx}"] #, f"inv_dist_{idx}"]

X_cols = ["dist_km"]+ ["dist_km_sq"]+ candidate_cols
X = sm.add_constant(gdf[X_cols])
y = gdf["ln_density"]

model = sm.OLS(y, X).fit()
print(model.summary())

# --- Step 3: retain subcenters whose distance OR inverse-distance is significant
true_subcenters = []
for idx in subcenters.index:
    p_dist = model.pvalues[f"dist_{idx}"]
    #p_inv = model.pvalues[f"inv_dist_{idx}"]
    if (p_dist < 0.1):# or (p_inv < 0.1):
        true_subcenters.append(idx)

true_subcenters = subcenters.loc[true_subcenters]


# ============================================================
# STEP 5: Plot results
# ============================================================

fig, ax = plt.subplots(figsize=(10, 10))
gdf.plot(ax=ax, column="density", cmap="OrRd", alpha=0.6, legend=True)
plt.scatter(true_subcenters["x"], true_subcenters["y"], c="blue", s=50, label="Subcenters")
gpd.GeoSeries([CBD]).plot(ax=ax, color="blue", marker="*", markersize=100)
plt.legend()
plt.title("Employment Subcenters (McMillen 2001 / 2003 Method)")
plt.axis("off")
plt.show()

cbd_df = pd.DataFrame({
    "cluster": [-1],
    "x": CBD.x,
    "y": CBD.y
})

true_subcenters = pd.concat([true_subcenters, cbd_df], ignore_index=True)



# ============================================================
# STEP 6: Dissolve adjacent subcenter tracts
# ============================================================

#gdf["subcenter"] = gdf.ID.isin(list(true_subcenters.ID))
#sub = gdf[gdf["subcenter"]].copy()
##sub["geometry"] = sub.buffer(0)  # fix geometries
#merged = sub.dissolve().explode(index_parts=True).buffer(0)
#merged = gpd.GeoDataFrame(geometry=merged, crs=gdf.crs).reset_index(drop=True)

# --- Identify tracts belonging to a subcenter (within 1 km of a cluster centroid)
buffer_radius = 750  # meters
gdf["subcenter"] = False

for _, row in true_subcenters.iterrows():
    center_point = Point(row["x"], row["y"])
    within_buffer = gdf.centroid.distance(center_point) <= buffer_radius
    gdf.loc[within_buffer, "subcenter"] = True

# Proceed with dissolve etc.
sub = gdf[gdf["subcenter"]].copy()
sub["geometry"] = sub.buffer(0)
merged = sub.dissolve().explode(index_parts=True).buffer(0)
merged = gpd.GeoDataFrame(geometry=merged, crs=gdf.crs).reset_index(drop=True)



# Jobs per subcenter
def jobs_in_poly(poly):
    return gdf[gdf.centroid.within(poly)]["employment"].sum()

merged["jobs"] = merged.geometry.apply(jobs_in_poly)
merged["buffered_geom"] = merged.geometry.buffer(1000)  # 1 km buffer
merged["jobs_buffered"] = merged["buffered_geom"].apply(jobs_in_poly)

total_jobs = gdf["employment"].sum()
merged["share_total"] = merged["jobs_buffered"] / total_jobs
print(merged[["jobs_buffered", "share_total"]])
print(f"Total share of jobs in subcenters: {merged['share_total'].sum():.2%}")

# ============================================================
# STEP 7: Assign each tract to nearest center (subcenters + CBD)
# ============================================================

centers = merged.copy()
centers["center_id"] = range(len(centers))

def nearest_center(row, centers):
    distances = centers.geometry.distance(row.geometry.centroid)
    return centers.loc[distances.idxmin(), "center_id"]

gdf["center_id"] = gdf.apply(lambda row: nearest_center(row, centers), axis=1)

# ============================================================
# STEP 8: Aggregate jobs by assigned center
# ============================================================

allocation = (
    gdf.groupby("center_id", as_index=False)
       .agg(total_jobs=("employment", "sum"))
       .merge(centers[["center_id", "geometry"]], on="center_id", how="left")
)

allocation["share_total"] = allocation["total_jobs"] / sum(gdf.employment)
print(allocation)


allocation_plot = gpd.GeoDataFrame(allocation, geometry=allocation["geometry"], crs=gdf.crs)
allocation_plot["geometry"] = allocation_plot["geometry"].centroid

# Reset center_id to start at 1
allocation_plot = allocation_plot.reset_index(drop=True)
allocation_plot['center_id'] = allocation_plot.index + 1  # IDs start at 1

# Plot map with bold, black, larger numbers
fig, ax = plt.subplots(figsize=(12, 12))
gdf.plot(ax=ax, color='lightgrey', edgecolor='white', linewidth=0.1)
allocation_plot.plot(ax=ax, markersize=allocation_plot['total_jobs']/200, color='red', label='Cluster Centroids')

# Annotate each subcenter
for idx, row in allocation_plot.iterrows():
    ax.annotate(
        str(row['center_id']),
        xy=(row.geometry.x, row.geometry.y),
        xytext=(3, 3),  # offset
        textcoords="offset points",
        fontsize=16, 
        color='black',
        fontweight='bold'
    )

plt.axis("off")
plt.show()


# ============================================================
# STEP 9: Compare predicted vs actual jobs per municipality
# ============================================================

total_jobs_city = gdf.merge(jobs, on="code_city", how="left")
total_jobs_city = total_jobs_city.loc[:,["code_city", "SPERSONAS"]].drop_duplicates()
total_jobs_city = total_jobs_city.rename(columns={"SPERSONAS": "total_jobs"})

centers_in_city = gpd.sjoin(allocation_plot, gdf[["code_city", "geometry"]], how="left", predicate="within")
jobs_centers_city = centers_in_city.groupby("code_city")["total_jobs"].sum().reset_index()
jobs_centers_city = jobs_centers_city.rename(columns={"total_jobs": "jobs_in_centers"})

comparison = total_jobs_city.merge(jobs_centers_city, on="code_city", how="left")
comparison["jobs_in_centers"].fillna(0, inplace=True)
comparison["share_in_centers"] = comparison["jobs_in_centers"] / comparison["total_jobs"]

print(comparison)

# Plot total jobs vs jobs in centers per city
cities = comparison["code_city"]
total = comparison["total_jobs"]
in_centers = comparison["jobs_in_centers"]

x = np.arange(len(cities))  # the label locations
width = 0.35  # bar width

plt.rcParams.update({'font.size': 14})
fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width/2, total, width, label="Total jobs", color="lightgrey")
rects2 = ax.bar(x + width/2, in_centers, width, label="Jobs in centers", color="royalblue")

# Add labels and formatting
ax.set_ylabel("Number of jobs", fontsize=14)
ax.set_xlabel("City", fontsize=14)
ax.set_title("Total Jobs vs Jobs in Employment Centers by City", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(cities, rotation=45, ha="right", fontsize=14)
ax.legend(fontsize=12)


print(100 * np.nansum(comparison["total_jobs"].loc[comparison.code_city.isin(["08019", "08101", "08194"])]) / np.nansum(comparison["total_jobs"]), "%")
print(100 * np.nansum(comparison["jobs_in_centers"].loc[comparison.code_city.isin(["08019", "08101", "08194"])]) / np.nansum(comparison["total_jobs"]), "%")


allocation_plot["employment_cluster"] = allocation_plot["total_jobs"] * np.nansum(gdf["pop"]) / np.nansum(allocation_plot["total_jobs"])
allocation_plot.loc[:,["geometry", "employment_cluster"]].to_file(path_data + "cluster_employment_catalunya.shp")