import geopandas as gpd # type: ignore
import pandas as pd

from import_data import *

path_data = "../data_barcelona/"
#https://catalegs.ide.cat/geonetwork/srv/eng/catalog.search#/search?facet.q=orgName%2FInstitut%20Cartogr%C3%A0fic%20i%20Geol%C3%B2gic%20de%20Catalunya&resultType=details&sortBy=modified&type=dataset&title=cobertes%20sol&from=1&to=20

#layers = fiona.listlayers(path_data + "cobertes-sol-v1r0-2023.gpkg")
land_cover = gpd.read_file(path_data + "cobertes-sol-v1r0-2023.gpkg", layer="cobertes_sol", engine="fiona")
layer_styles = gpd.read_file(path_data + "cobertes-sol-v1r0-2023.gpkg", layer="layer_styles", engine="fiona")
categories = gpd.read_file(path_data + "cobertes-sol-v1r0-2023.gpkg", layer="cobertes_sol_categories", engine="fiona")

gdf = import_data(path_data, center, option = "SECTION")

land_cover = land_cover.to_crs(gdf.crs)

# Spatial intersection
intersection = gpd.overlay(land_cover, gdf, how='intersection')


#Beach
gdf['min_distance_beach'] = gdf.geometry.apply(lambda geom: land_cover.loc[land_cover.nivell_2 == 233,:].distance(geom).min())

# Density of activities
activity = gpd.read_file(path_data + "2018_cohesio_sobreocupacio.gpkg")
activity = activity.to_crs(gdf.crs)
joined = gpd.sjoin(activity[['norm', 'geometry']], gdf[['geometry']], how='inner', predicate='intersects')
mean_dn = joined.groupby('index_right')['norm'].mean()
gdf['mean_activity'] = gdf.index.map(mean_dn)




# Noise and heat and pacified streets: out of Barcelona?
#https://opendata-ajuntament.barcelona.cat/data/en/dataset/rasters-mapa-estrategic-soroll
#noise_data = gpd.read_file(path_data + "2022_raster_total_dia_mapa_estrategic_soroll_bcn.gpkg")
#import rasterio
#from rasterstats import zonal_st

#stats = zonal_stats(
#    vectors=gdf.geometry,
#    raster=path_data + "2022_raster_total_dia_mapa_estrategic_soroll_bcn.gpkg",
#    stats=['mean', 'sum', 'min', 'max'],  # Choose relevant stats
#    geojson_out=True
#)

# Convert results to GeoDataFrame
#import geopandas as gpd
#import json

#gdf_stats = gpd.GeoDataFrame.from_features(stats, crs=gdf.crs)


#with rasterio.open(path_data + "2022_raster_total_dia_mapa_estrategic_soroll_bcn.gpkg") as src:
#    print(src.profile)       # Metadata
#    raster_data = src.read(1)

# Compute area in square meters (adjust if needed)
intersection['area_lc'] = intersection.geometry.area

# Group by tract and land cover, summing areas
grouped = (
    intersection
    .groupby(['ID', 'nivell_2'])['area_lc']
    .sum()
    .reset_index()
)

# Pivot to wide format
pivot = grouped.pivot(index='ID', columns='nivell_2', values='area_lc').fillna(0)

# Optional: reset index to return to a DataFrame
pivot = pivot.reset_index()

land_use = gdf.loc[:,['ID', 'geometry', 'area']].merge(pivot, on = "ID", how = "left")

categories = pd.read_excel("C:/Users/1738037/OneDrive - UAB/1- CLIMGROW Charlotte/1- PSC cities/data_barcelona/categories_land_use.xlsx")

col = list(categories.nivell_2.loc[categories.Urbanisable == 1])
col_alt = list(categories.nivell_2.loc[categories.Urbanisable_alt == 1])
land_use["urb_area"] = np.nansum(land_use[col], axis = 1)
land_use["urb_area_alt"] = np.nansum(land_use[col_alt], axis = 1)
land_use.loc[:,["ID", "urb_area", "urb_area_alt"]].to_excel("C:/Users/1738037/OneDrive - UAB/1- CLIMGROW Charlotte/1- PSC cities/data_barcelona/land_use_urb.xlsx")



list_weird = ['0801506002',
 '0801905001',
 '0802001002',
 '0802001003',
 '0805401006',
 '0806801003',
 '0808903005',
 '0812303003',
 '0816904001',
 '0820501012',
 '0821105005',
 '0824401005',
 '0826601011',
 '0828901001']

land_use_weird = land_use.loc[land_use.ID.isin(list_weird),:]

np.nansum(land_use_weird.iloc[:,3:], axis =0) / sum(np.nansum(land_use_weird.iloc[:,3:], axis =0))