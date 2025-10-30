import geopandas as gpd # type: ignore
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
from libpysal.weights import Queen
from esda import G_Local
from statsmodels.regression.linear_model import OLS

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

land_cover = gpd.read_file(path_data + "cobertes-sol-v1r0-2023.gpkg", layer="cobertes_sol") #, engine="fiona")
layer_styles = gpd.read_file(path_data + "cobertes-sol-v1r0-2023.gpkg", layer="layer_styles") #, engine="fiona")
categories = gpd.read_file(path_data + "cobertes-sol-v1r0-2023.gpkg", layer="cobertes_sol_categories") #, engine="fiona")
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


### Find CBD

gdf["centroid"] = gdf.geometry.centroid
gdf["x"] = gdf.centroid.x
gdf["y"] = gdf.centroid.y
gdf["employment"].loc[np.isnan(gdf["employment"])] = 0
x_cbd = np.average(gdf["x"], weights=gdf["employment"])
y_cbd = np.average(gdf["y"], weights=gdf["employment"])
CBD = Point(x_cbd, y_cbd)
print(f"CBD coordinates: ({x_cbd:.3f}, {y_cbd:.3f})")

gdf["dist_cbd"] = gdf.centroid.distance(CBD)
gdf.plot("dist_cbd")

## McMillen method

# Compute job density (employment / area in km²)
gdf["area_km2"] = gdf.geometry.area / 1e6
gdf["density"] = gdf["employment"] / gdf["area_km2"]

# Take logs for semi-log gradient model
gdf["ln_density"] = np.log(gdf["density"] + 1)
gdf["dist_km"] = gdf["dist_cbd"] / 1000

# distance in km, ln_density computed as before
#smoothed = lowess(gdf["density"], gdf["dist_km"], frac=0.4)  # frac controls smoothing
#gdf["density_fit"] = smoothed[:, 1]

# residuals
#gdf["residual"] = gdf["density"] - gdf["density_fit"]

# Regression: ln(density) = α + β * distance + ε
X = sm.add_constant(gdf["dist_km"].loc[gdf["employment"]>0])
model = sm.OLS(gdf["ln_density"].loc[gdf["employment"]>0], X).fit()
print(model.summary())

gdf["residual"] = model.resid

# Compute residuals
gdf["residual"].loc[gdf["employment"]>0] = model.resid
gdf["residual"].loc[gdf["employment"] == 0] = 0


# Spatial weights (Queen contiguity)
w = Queen.from_dataframe(gdf)
w.transform = "r"

# Local G* statistic (Getis-Ord)
g_local = G_Local(gdf["residual"], w)

# Add to dataframe
gdf["G_star"] = g_local.Gs
gdf["p_value"] = g_local.p_sim

# Identify significant hot spots (e.g. 5% level, positive residuals)
gdf["subcenter"] = (gdf["p_value"] < 0.05) & (gdf["G_star"] > 0)

fig, ax = plt.subplots(figsize=(10, 10))
gdf.plot(ax=ax, column="subcenter", cmap="coolwarm", legend=True)
plt.scatter(x_cbd, y_cbd, color="black", marker="*", s=200, label="CBD")
plt.legend()
plt.title("Identified Employment Subcenters (McMillen, 2001)")
plt.show()

candidate_centers = gdf[gdf["subcenter"]].copy()

for idx, row in candidate_centers.iterrows():
    gdf[f"dist_inv_{idx}"] = 1 / (gdf.geometry.centroid.distance(row.geometry.centroid) + 1e-6)
    gdf[f"dist_{idx}"] = gdf.geometry.centroid.distance(row.geometry.centroid)


X_candidates = gdf[[f"dist_inv_{idx}" for idx in candidate_centers.index] + 
                   [f"dist_{idx}" for idx in candidate_centers.index]]
X_candidates = sm.add_constant(X_candidates)

# 4. Fit OLS
model = OLS(gdf["residual"], X_candidates).fit()
print(model.summary())

# 5. Select significant candidates (p < 0.05)
significant_idx = [candidate_centers.index[i//2]  # each candidate has two columns
                   for i, p in enumerate(model.pvalues[1:]) if p < 0.05]
true_subcenters = candidate_centers.loc[significant_idx]

# Plot the city tracts (optional: color by employment density)
fig, ax = plt.subplots(figsize=(12, 12))
gdf.plot(column="density", cmap="OrRd", legend=True, alpha=0.6, ax=ax)

# Plot true subcenters
true_subcenters.plot(ax=ax, color="blue", markersize=50, label="Subcenters")

# Labels and legend
ax.set_title("Employment Density and True Subcenters", fontsize=14)
ax.legend(fontsize=12)
ax.set_axis_off()
plt.show()




# Dissolve adjacent or close subcenter tracts

gdf["subcenter"] = gdf.ID.isin(list(true_subcenters.ID))

sub = gdf[gdf["subcenter"]].copy()
sub["geometry"] = sub.buffer(0)  # 1.5 km merge buffer (adjust sensitivity)
merged = sub.dissolve().explode(index_parts=True).buffer(0)  # remove buffer
merged = gpd.GeoDataFrame(geometry=merged, crs=gdf.crs).reset_index(drop=True)

# Compute employment within merged subcenters
def jobs_in_poly(poly):
    return gdf[gdf.centroid.within(poly)]["employment"].sum()

merged["jobs"] = merged.geometry.apply(jobs_in_poly)

# 1 km buffer around each subcenter
merged["buffered_geom"] = merged.geometry.buffer(100)

def buffered_jobs(poly):
    return gdf[gdf.centroid.within(poly)]["employment"].sum()

merged["jobs_buffered"] = merged["buffered_geom"].apply(buffered_jobs)

total_jobs = gdf["employment"].sum()
merged["share_total"] = merged["jobs_buffered"] / total_jobs
print(merged[["jobs_buffered", "share_total"]])
print(f"Total share of jobs in subcenters: {merged['share_total'].sum():.2%}")



# Step 1: Define centers (CBD + merged subcenters centroids)
centers = merged.copy()
centers["centroid"] = centers.geometry.centroid
centers["center_id"] = range(len(centers))
centers = centers[["center_id", "centroid", "jobs_buffered"]].copy()

# Add CBD as a center
from shapely.geometry import Point
CBD_point = Point(x_cbd, y_cbd)
cbd_df = gpd.GeoDataFrame({
    "center_id": [-1],
    "jobs_buffered": [gdf["employment"].sum()],
    "geometry": [CBD_point]  # must be named 'geometry'
}, crs=gdf.crs)

merged["center_id"] = merged.index

# Now concatenate with your subcenters
centers = pd.concat([merged, cbd_df], ignore_index=True)

# Step 2: Assign each tract to its nearest center (by centroid distance)
def nearest_center(row, centers):
    distances = centers.centroid.distance(row.geometry.centroid)
    return centers.loc[distances.idxmin(), "center_id"]

gdf["center_id"] = gdf.apply(lambda row: nearest_center(row, centers), axis=1)


# Jobs per assigned center
allocation = (
    gdf.groupby("center_id", as_index=False)
       .agg(total_jobs=("employment", "sum"))
       .merge(centers[["center_id", "jobs_buffered"]], on="center_id", how="left")
)

# Share of total jobs
total_jobs = gdf["employment"].sum()
allocation["share_total"] = allocation["total_jobs"] / total_jobs

print(allocation)
print(f"Sum of shares: {allocation['share_total'].sum():.2%}")  # should be 100%



allocation_plot = allocation.merge(centers.loc[:,["geometry", "center_id"]], on = "center_id")
allocation_plot["total"] = allocation_plot["share_total"] * sum(gdf["employment"])



allocation_plot = gpd.GeoDataFrame(
    allocation_plot,                  # other columns
    geometry=allocation_plot["geometry"],  # assign geometry
    crs=gdf.crs                        # same CRS as original tracts
)

allocation_plot["geometry"] = allocation_plot["geometry"].centroid

fig, ax = plt.subplots(figsize=(12, 12))
gdf.plot(ax=ax, color='lightgrey', edgecolor='white', linewidth=0.1)


# Plot centroids scaled by a variable (e.g. 'jobs')
allocation_plot.plot(
    ax=ax,
    markersize=allocation_plot['total'] / 200,  # adjust scaling as needed
    color='red',
    marker='o',
    label='Cluster Centroids'
)

# --- Add annotation for each center_id ---
for _, row in allocation_plot.iterrows():
    x, y = row.geometry.x, row.geometry.y
    ax.text(
        x + 100, y + 100,               # small offset in map units
        str(row["center_id"]),
        fontsize=10,
        color='black',
        ha='left',
        va='bottom',
        path_effects=[]  # optionally: [patheffects.withStroke(linewidth=2, foreground="white")]
    )

plt.axis('off')
plt.title("Employment subcenters sized by job count")
plt.show()






total_jobs_city = gdf.merge(jobs, on = "code_city", how = "left")
total_jobs_city = total_jobs_city.loc[:,["code_city", "SPERSONAS"]].drop_duplicates()
total_jobs_city = total_jobs_city.rename(columns={"total": "jobs_in_centers"})

# Spatial join: assign each center to the city it is located in
centers_in_city = gpd.sjoin(allocation_plot, gdf[["code_city", "geometry"]], how="left", predicate="within")


# Aggregate jobs per city
jobs_centers_city = centers_in_city.groupby("code_city")["total"].sum().reset_index()
jobs_centers_city = jobs_centers_city.rename(columns={"total": "jobs_in_centers"})

comparison = total_jobs_city.merge(jobs_centers_city, on="code_city", how="left")
#comparison["jobs_in_centers"].fillna(0, inplace=True)  # if some cities have no centers
#comparison["share_in_centers"] = comparison["jobs_in_centers"] / comparison["total_jobs"]

print(comparison)

allocation_plot["employment_cluster"] = allocation_plot["total"] * np.nansum(gdf["pop"]) / np.nansum(allocation_plot["total"])



# Assume `comparison` DataFrame from previous step
cities = comparison["code_city"]
total = comparison["SPERSONAS"]
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
#ax.legend(fontsize=14)

plt.tight_layout()
plt.show()