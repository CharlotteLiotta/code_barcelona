import fiona
#https://catalegs.ide.cat/geonetwork/srv/eng/catalog.search#/search?facet.q=orgName%2FInstitut%20Cartogr%C3%A0fic%20i%20Geol%C3%B2gic%20de%20Catalunya&resultType=details&sortBy=modified&type=dataset&title=cobertes%20sol&from=1&to=20
#layers = fiona.listlayers("C:/Users/1738037/OneDrive - UAB/1- CLIMGROW Charlotte/1- PSC cities/data_barcelona/cobertes-sol-v1r0-2023.gpkg")
land_cover = gpd.read_file("C:/Users/1738037/OneDrive - UAB/1- CLIMGROW Charlotte/1- PSC cities/data_barcelona/cobertes-sol-v1r0-2023.gpkg", layer="cobertes_sol", engine="fiona")
layer_styles = gpd.read_file("C:/Users/1738037/OneDrive - UAB/1- CLIMGROW Charlotte/1- PSC cities/data_barcelona/cobertes-sol-v1r0-2023.gpkg", layer="layer_styles", engine="fiona")
categories = gpd.read_file("C:/Users/1738037/OneDrive - UAB/1- CLIMGROW Charlotte/1- PSC cities/data_barcelona/cobertes-sol-v1r0-2023.gpkg", layer="cobertes_sol_categories", engine="fiona")

land_cover = land_cover.to_crs(gdf.crs)

# Spatial intersection
intersection = gpd.overlay(land_cover, gdf, how='intersection')

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
land_use["urb_area"] = np.nansum(land_use[col], axis = 1)
land_use.loc[:,["ID", "urb_area"]].to_excel("C:/Users/1738037/OneDrive - UAB/1- CLIMGROW Charlotte/1- PSC cities/data_barcelona/land_use_urb.xlsx")