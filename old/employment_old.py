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
gdf = import_data(path_data, center, option="SECTION")
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

# ============================================================
# STEP 2: Find CBD (unchanged)
# ============================================================

gdf["centroid"] = gdf.geometry.centroid
gdf["x"] = gdf.centroid.x
gdf["y"] = gdf.centroid.y
gdf["employment"].fillna(0, inplace=True)

x_cbd = np.average(gdf["x"], weights=gdf["employment"])
y_cbd = np.average(gdf["y"], weights=gdf["employment"])
CBD = Point(x_cbd, y_cbd)
print(f"CBD coordinates: ({x_cbd:.3f}, {y_cbd:.3f})")

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
smoothed = lowess(gdf["ln_density"], gdf["dist_km"], frac=0.3, return_sorted=False)
gdf["fitted_ln_density"] = smoothed
gdf["residual"] = gdf["ln_density"] - gdf["fitted_ln_density"]

# --- Step 3.3: Spatial smoothing of residuals (Gaussian kernel)
coords = np.vstack([gdf["x"], gdf["y"]]).T
# Only positive residuals contribute to local employment "excess"
positive_weights = gdf["residual"].clip(lower=0)









# Parameters (tune c between 2 and 4; see recommendations below)
c = 2.5   # multiplier for sqrt(area)
eps = 1e-9

# Prepare arrays
coords = np.vstack([gdf["x"].values, gdf["y"].values]).T   # (n,2)
pos_w = gdf["residual"].clip(lower=0).values               # (n,)
areas = gdf["area"].values                                 # (n,) in m^2

# compute adaptive bandwidth per *target* cell (h_i)
h = c * np.sqrt(areas)            # (n,) in meters

# pairwise distances (n x n). If n large (> ~5000) consider block / KD-tree approach
from scipy.spatial.distance import cdist
D = cdist(coords, coords)         # D[i,j] = dist from target i to source j

# Gaussian kernel: K(d, h_i) = exp(-0.5 * (d / h_i)^2)
# For each target i, compute weights across all j using h_i
H = h[:, None]                    # (n,1)
K = np.exp(-0.5 * (D / (H + eps))**2)    # (n,n)  (broadcasting)

# weighted sum of positive residuals
num = (K * pos_w[None, :]).sum(axis=1)   # (n,)
den = K.sum(axis=1) + eps                # (n,)
gdf["smoothed_resid_adaptive"] = num / den

# Now use this 'smoothed_resid_adaptive' for candidate detection
threshold = np.percentile(gdf["smoothed_resid_adaptive"], 85)   # try 85% or 90%
gdf["candidate"] = gdf["smoothed_resid_adaptive"] >= threshold


#BALLON
#weights = gdf["residual"].clip(lower=0).values  # only positive residuals

# Step 1: Compute location-specific bandwidths (distance to k-th nearest neighbor)
#k = 8 # adjust depending on tract density
#nbrs = NearestNeighbors(n_neighbors=k+1).fit(coords)  # include self
#distances, indices = nbrs.kneighbors(coords)
# Use distance to k-th neighbor as bandwidth for balloon
#h_x = distances[:, -1]
#gdf["bandwidth_balloon"] = h_x

# Step 2: Compute balloon KDE at each centroid
#def balloon_kde(coords, weights, h_x):
#    n = len(coords)
#    dens = np.zeros(n)
#    for i in range(n):
        # compute squared distance from estimation location i to all points
#        diff = coords - coords[i]
#        dist2 = np.sum(diff**2, axis=1)
#        # Gaussian kernel with location-specific bandwidth
#        kern = np.exp(-0.5 * dist2 / (h_x[i]**2))
#        dens[i] = np.sum(weights * kern / (2 * np.pi * h_x[i]**2))
#    return dens

##gdf["smoothed_resid_balloon"] = balloon_kde(coords, weights, h_x)
#gdf.plot("smoothed_resid_balloon")
## Step 3: Identify candidate subcenters (e.g., top 10% of balloon KDE)
#threshold = np.percentile(gdf["smoothed_resid_balloon"], 85)
#gdf["candidate"] = gdf["smoothed_resid_balloon"] >= threshold

#Option2
#kde = KernelDensity(kernel="gaussian", bandwidth=500).fit(coords, sample_weight=positive_weights)
#gdf["smoothed_resid"] = np.exp(kde.score_samples(coords))
#gdf.plot("smoothed_resid")


# --- Step 3.4: Identify candidate subcenters (top 5% smoothed residuals)
#threshold = np.percentile(gdf["smoothed_resid"], 85)
#gdf["candidate"] = gdf["smoothed_resid"] >= threshold
#gdf.plot("candidate")
#####

# --- Step 3.5: Cluster adjacent high-residual tracts (DBSCAN)
candidate_coords = np.vstack([gdf.loc[gdf["candidate"], "x"], gdf.loc[gdf["candidate"], "y"]]).T
clustering = DBSCAN(eps=500, min_samples=1).fit(candidate_coords)
gdf.loc[gdf["candidate"], "cluster"] = clustering.labels_

# --- Step 3.6: Extract subcenter centroids
subcenters = (
    gdf[gdf["candidate"] & (gdf["cluster"] != -1)]
    .groupby("cluster")[["x", "y", "smoothed_resid_adaptive"]]
    .mean()
    .reset_index()
)

# ============================================================
# STEP 4: Validation Regression (McMillen confirmatory step)
# ============================================================

for idx, row in subcenters.iterrows():
    gdf[f"dist_{idx}"] = gdf.centroid.distance(Point(row["x"], row["y"])) / 1000

X = sm.add_constant(gdf[[f"dist_{idx}" for idx in subcenters.index]])
y = gdf["ln_density"]
model = sm.OLS(y, X).fit()
print(model.summary())

# Keep only statistically significant subcenters
significant_idx = [subcenters.index[i] for i, p in enumerate(model.pvalues[1:]) if p < 0.1]
true_subcenters = subcenters.loc[significant_idx]


# ============================================================
# STEP 5: Plot results
# ============================================================

fig, ax = plt.subplots(figsize=(10, 10))
gdf.plot(ax=ax, column="density", cmap="OrRd", alpha=0.6, legend=True)
plt.scatter(true_subcenters["x"], true_subcenters["y"], c="blue", s=50, label="Subcenters")
plt.scatter(x_cbd, y_cbd, c="black", s=80, marker="*", label="CBD")
plt.legend()
plt.title("Employment Subcenters (McMillen 2001 / 2003 Method)")
plt.axis("off")
plt.show()




# ============================================================
# STEP 6: Dissolve adjacent subcenter tracts
# ============================================================

#gdf["subcenter"] = gdf.ID.isin(list(true_subcenters.ID))
#sub = gdf[gdf["subcenter"]].copy()
##sub["geometry"] = sub.buffer(0)  # fix geometries
#merged = sub.dissolve().explode(index_parts=True).buffer(0)
#merged = gpd.GeoDataFrame(geometry=merged, crs=gdf.crs).reset_index(drop=True)

# --- Identify tracts belonging to a subcenter (within 1 km of a cluster centroid)
buffer_radius = 1000  # meters
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
centers["centroid"] = centers.geometry.centroid
centers["center_id"] = range(len(centers))
centers = centers[["center_id", "centroid", "jobs_buffered"]].copy()

CBD_point = Point(x_cbd, y_cbd)
cbd_df = gpd.GeoDataFrame({
    "center_id": [-1],
    "jobs_buffered": [gdf["employment"].sum()],
    "geometry": [CBD_point]
}, crs=gdf.crs)

merged["center_id"] = merged.index
centers = pd.concat([merged, cbd_df], ignore_index=True)

def nearest_center(row, centers):
    distances = centers.centroid.distance(row.geometry.centroid)
    return centers.loc[distances.idxmin(), "center_id"]

gdf["center_id"] = gdf.apply(lambda row: nearest_center(row, centers), axis=1)

# ============================================================
# STEP 8: Aggregate jobs by assigned center
# ============================================================

allocation = (
    gdf.groupby("center_id", as_index=False)
       .agg(total_jobs=("employment", "sum"))
       .merge(centers[["center_id", "jobs_buffered"]], on="center_id", how="left")
)
allocation["share_total"] = allocation["total_jobs"] / total_jobs
print(allocation)

allocation_plot = allocation.merge(centers[["geometry", "center_id"]], on="center_id")
allocation_plot["total"] = allocation_plot["share_total"] * total_jobs

allocation_plot = gpd.GeoDataFrame(allocation_plot, geometry=allocation_plot["geometry"], crs=gdf.crs)
allocation_plot["geometry"] = allocation_plot["geometry"].centroid

# Reset center_id to start at 1
allocation_plot = allocation_plot.reset_index(drop=True)
allocation_plot['center_id'] = allocation_plot.index + 1  # IDs start at 1

# Plot map with bold, black, larger numbers
fig, ax = plt.subplots(figsize=(12, 12))
gdf.plot(ax=ax, color='lightgrey', edgecolor='white', linewidth=0.1)
allocation_plot.plot(ax=ax, markersize=allocation_plot['total']/200, color='red', label='Cluster Centroids')

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
jobs_centers_city = centers_in_city.groupby("code_city")["total"].sum().reset_index()
jobs_centers_city = jobs_centers_city.rename(columns={"total": "jobs_in_centers"})

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




allocation_plot["employment_cluster"] = allocation_plot["total"] * np.nansum(gdf["pop"]) / np.nansum(allocation_plot["total"])
allocation_plot.loc[:,["geometry", "employment_cluster"]].to_file(path_data + "cluster_employment.shp")





gdf.plot(gdf.employment / gdf.area)

