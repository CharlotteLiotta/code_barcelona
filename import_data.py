import numpy as np # type: ignore
import pandas as pd
import geopandas as gpd # type: ignore
from shapely import wkt, Point # type: ignore
import requests
from shapely.geometry import shape, LineString, MultiLineString # type: ignore
import os 
import matplotlib.pyplot as plt
import warnings

from import_data import * # type: ignore

def import_data(path_data, center, option):
    """ Import shapefile + population """

    if option == "SECTION":
        #https://www.ine.es/dynt3/inebase/en/index.htm?padre=11676&capsel=11681
        
        #Import population per census tract
        df = pd.read_csv(path_data + '70035.csv', sep = ";", encoding="latin1")
        df = df.loc[:,["Sections", "Total", "Municipalities"]]
        df = df.dropna(subset=["Sections"])
        df["CUSEC"] = df["Sections"].str[:10]

        def fix_decimal(s):
            if isinstance(s, str) and ',' in s:
                integer, decimal = s.split(',', 1)
                if len(decimal) == 2:
                    return f"{integer},{decimal}0"
            return s

        #Merge with shapefile
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            gdf = gpd.read_file(path_data + 'seccionado_2024/SECC_CE_20240101.shp')
        gdf = gdf.loc[gdf.NMUN.isin(["Cornellà de Llobregat", 'Badalona', 'Badia del Vallès', 'Barberà del Vallès', 'Barcelona','Begues','Castellbisbal', 'Castelldefels', 'Cerdanyola del Vallès', 'Cervelló','Corbera de Llobregat','Papiol, El', 'Prat de Llobregat, El', 'Esplugues de Llobregat','Gavà', "Hospitalet de Llobregat, L'",'Palma de Cervelló, La','Molins de Rei', 'Montcada i Reixac', 'Montgat', 'Pallejà', 'Ripollet','Sant Adrià de Besòs', 'Sant Andreu de la Barca','Sant Boi de Llobregat', 'Sant Climent de Llobregat', 'Sant Cugat del Vallès', 'Sant Feliu de Llobregat','Sant Joan Despí','Sant Just Desvern','Sant Vicenç dels Horts', 'Santa Coloma de Cervelló', 'Santa Coloma de Gramenet','Tiana', 'Torrelles de Llobregat', 'Viladecans']),:]
        gdf = gdf.merge(df, on = "CUSEC", how = "left")

        #Compute area, density, distance to city center
        gdf["Total"] = gdf["Total"].apply(fix_decimal)
        gdf["Total"] = pd.to_numeric(gdf["Total"].str.replace(',', ''), errors='coerce')
        gdf.loc[np.isnan(gdf["Total"]), "Total"] = 0
        gdf = gdf.loc[gdf.NMUN.isin(["Cornellà de Llobregat", 'Badalona', 'Badia del Vallès', 'Barberà del Vallès', 'Barcelona','Begues','Castellbisbal', 'Castelldefels', 'Cerdanyola del Vallès', 'Cervelló','Corbera de Llobregat','Papiol, El', 'Prat de Llobregat, El', 'Esplugues de Llobregat','Gavà', "Hospitalet de Llobregat, L'",'Palma de Cervelló, La','Molins de Rei', 'Montcada i Reixac', 'Montgat', 'Pallejà', 'Ripollet','Sant Adrià de Besòs', 'Sant Andreu de la Barca','Sant Boi de Llobregat', 'Sant Climent de Llobregat', 'Sant Cugat del Vallès', 'Sant Feliu de Llobregat','Sant Joan Despí','Sant Just Desvern','Sant Vicenç dels Horts', 'Santa Coloma de Cervelló', 'Santa Coloma de Gramenet','Tiana', 'Torrelles de Llobregat', 'Viladecans']),["CUSEC", "CUMUN", "Shape_Area", "Total", "geometry", "NMUN"]]
        gdf.CUSEC = gdf.CUSEC.astype(str)
        city_center = gdf.loc[gdf.CUSEC == center,:].centroid
        gdf["distance_center"] = gdf.centroid.distance(city_center.iloc[0], align = False) / 1000
        gdf = gdf.loc[:,["CUSEC", "geometry", "Shape_Area", "Total", "distance_center", "NMUN"]]
        
    elif option == "DISTRICT":

        #Import population per district
        df = pd.read_csv(path_data + "2021_densitat.csv")
        df = df.groupby("Nom_Districte").sum("Població")
        
        #Merge with shapefile
        gdf = gpd.read_file(path_data + 'BarcelonaCiutat_Districtes.csv')
        gdf["nom_districte"] = ['Ciutat Vella', 'Eixample', 'Sants-Montjuïc', 'Les Corts', 'Sarrià-Sant Gervasi', 'Gràcia', 'Horta-Guinardó', 'Nou Barris', 'Sant Andreu', 'Sant Martí']
        gdf = gdf.merge(df["Població"], left_on = "nom_districte", right_index = True)
        
        #Compute area, density, distance to city center
        gdf['geometry'] = gdf['geometria_etrs89'].apply(wkt.loads)
        gdf = gpd.GeoDataFrame(geometry="geometry", data=gdf, crs="EPSG:25831")
        gdf["area"] = gdf.area
        city_center = gdf.loc[gdf.nom_districte == "Eixample",:].centroid
        gdf["distance_center"] = gdf.centroid.distance(city_center.iloc[0], align = False) / 1000
        gdf = gdf.loc[:,["nom_districte", "geometry", "area", "Població", "distance_center"]]

    gdf.columns = ["ID", "geometry", "area", "pop", "distance_center", "NMUN"]
    gdf["area"] = gdf["area"] / 1000000
    gdf["density"] = gdf["pop"] / gdf["area"]
    gdf.ID = gdf.ID.astype(str)
    
    return gdf

def import_jobs(gdf, path_data):
    """ Import data on the spatial distribution of employment, previously retrieved from INE API"""

    #Import data on employment
    jobs = pd.read_csv(path_data + 'employment_distrib.csv') #from the INE API
    jobs = jobs.loc[:,["SPERSONAS", "ID_LUGAR_TRAB_N3"]]
    jobs["code_city"] = jobs["ID_LUGAR_TRAB_N3"].str[:5]
    jobs = jobs.loc[:,["SPERSONAS", "code_city"]]

    #Merge with gdf
    gdf["code_city"] = gdf["ID"].str[:5]
    gdf = gdf.merge(jobs, on = "code_city", how = "left")
    size_city = gdf.loc[:,["code_city", "area"]].groupby("code_city").sum("area")
    size_city.columns = ["area_city"]
    gdf = gdf.merge(size_city, on = "code_city", how = "left")
    gdf["density_employment"] = gdf["SPERSONAS"] / gdf["area_city"]
    gdf["employment"] = gdf["SPERSONAS"] * (gdf["area"] / gdf["area_city"])
    gdf = gdf.drop(columns = ["SPERSONAS", "area_city"])

    return gdf

def import_income(gdf, path_data):
    """ Import data on the spatial distribution of incomes"""

    #https://www.ine.es/dynt3/inebase/en/index.htm?padre=12385&capsel=12384
    
    #Import income data
    income = pd.read_csv(path_data + '30824.csv', sep = ";", encoding="latin1")
    income = income.loc[(income.Periodo == 2023) & (income['Average income indicators'] == 'Median income by unit of consumption'),["Sections", "Total"]]
    income = income.dropna(subset=["Sections"])
    income["ID"] = income["Sections"].str[:10]
    income.columns = ['Sections', 'net_income_median', 'ID']
    income.net_income_median = pd.to_numeric(income.net_income_median, errors= "coerce")
    income.net_income_median = income.net_income_median * 1000

    income_mean = pd.read_csv(path_data + '30824.csv', sep = ";", encoding="latin1")
    income_mean = income_mean.loc[(income_mean.Periodo == 2023) & (income_mean['Average income indicators'] == 'Average income by unit of consumption'),["Sections", "Total"]]
    income_mean = income_mean.dropna(subset=["Sections"])
    income_mean["ID"] = income_mean["Sections"].str[:10]
    income_mean.columns = ['Sections', 'net_income_mean', 'ID']
    income_mean.net_income_mean = pd.to_numeric(income_mean.net_income_mean, errors= "coerce")
    income_mean.net_income_mean = income_mean.net_income_mean * 1000


    income_ineq = pd.read_csv(path_data + '30901.csv', sep = ";", encoding="latin1")
    income_low = income_ineq.loc[(income_ineq.Periodo == 2023) & (income_ineq['Distribución de la renta por unidad de consumo'] == 'Población con ingresos por unidad de consumo por debajo 60% de la mediana'),["Secciones", "Total"]]
    income_low = income_low.dropna(subset=["Secciones"])
    income_low["ID"] = income_low["Secciones"].str[:10]
    income_low.columns = ['Secciones', 'share_low_income', 'ID']
    income_low['share_low_income'] = income_low['share_low_income'].str.replace(',', '.', regex=False)
    income_low.share_low_income = pd.to_numeric(income_low.share_low_income, errors= "coerce")

    income_high = income_ineq.loc[(income_ineq.Periodo == 2023) & (income_ineq['Distribución de la renta por unidad de consumo'] == 'Población con ingresos por unidad de consumo por encima 140% de la mediana'),["Secciones", "Total"]]
    income_high = income_high.dropna(subset=["Secciones"])
    income_high["ID"] = income_high["Secciones"].str[:10]
    income_high.columns = ['Secciones', 'share_high_income', 'ID']
    income_high['share_high_income'] = income_high['share_high_income'].str.replace(',', '.', regex=False)
    income_high.share_high_income = pd.to_numeric(income_high.share_high_income, errors= "coerce")




    #Merge with gdf
    gdf = gdf.merge(income.loc[:,['net_income_median', 'ID']], on = "ID", how = "left")
    gdf = gdf.merge(income_mean.loc[:,['net_income_mean', 'ID']], on = "ID", how = "left")
    gdf = gdf.merge(income_low.loc[:,['share_low_income', 'ID']], on = "ID", how = "left")
    gdf = gdf.merge(income_high.loc[:,['share_high_income', 'ID']], on = "ID", how = "left")

    gdf["net_income_median"] = (gdf["net_income_median"]) / 12 #/ gdf["active_per_hh"]
    gdf["net_income_mean"] = (gdf["net_income_mean"]) / 12 #/ gdf["active_per_hh"]
    #Compute average income
    #Y = (np.nansum(gdf.net_income * gdf["pop"]) / np.nansum(gdf["pop"]))
    def weighted_median(values, weights):
        mask = ~pd.isna(values) & ~pd.isna(weights)
        values = np.asarray(values[mask])
        weights = np.asarray(weights[mask])

        # Sort by value
        sorted_idx = np.argsort(values)
        values_sorted = values[sorted_idx]
        weights_sorted = weights[sorted_idx]

        # Cumulative weights
        cum_weights = np.cumsum(weights_sorted)
        cutoff = 0.5 * np.sum(weights_sorted)

        # Weighted median
        return values_sorted[np.searchsorted(cum_weights, cutoff)]
    Y_median = weighted_median(gdf["net_income_median"], gdf["pop"])

    gdf["pop_LOW"] = gdf["pop"] * gdf["share_low_income"] / 100
    gdf["pop_HIGH"] = gdf["pop"] * gdf["share_high_income"] / 100
    gdf["pop_MED"] = gdf["pop"] - gdf["pop_HIGH"] - gdf["pop_LOW"]
    return Y_median, gdf

def import_land_use(gdf, path_data):
    """ Import data on land use - pretreated in import_land_use """

    land_use = pd.read_excel(path_data + "land_use_urb.xlsx")
    land_use.ID = land_use.ID.astype(str).str.zfill(10)
    gdf = gdf.merge(land_use, on = "ID", how = "left")
    return gdf

def import_rent_and_size(gdf, path_data):
    """ Import rent per m2 and dwelling size data """

    #https://habitatge.gencat.cat/ca/dades/indicadors_estadistiques/estadistiques_de_construccio_i_mercat_immobiliari/mercat_de_lloguer/lloguers-municipis-amb/

    ### RENTS

    #Section level - AMB
    rent = pd.read_excel(path_data + "AMB_lloguer_m2.xlsx", header = 5)
    rent = rent.loc[:,["Codi_àmbit", "IV"]]
    gdf["Codi_àmbit"] = pd.to_numeric(gdf["ID"].str[:7])
    rent["IV"] = pd.to_numeric(rent["IV"], errors = "coerce")
    rent.columns = ['Codi_àmbit', 'rent_AMB_section']
    gdf = gdf.merge(rent, on = "Codi_àmbit", how = "left")

    #City level - AMB
    rent = pd.read_excel(path_data + "AMB_lloguer_m2.xlsx", header = 5)
    rent = rent.loc[np.isnan(rent.Codi_àmbit),["Codi_INE", "IV"]]
    gdf["Codi_INE"] = pd.to_numeric(gdf["code_city"])
    rent["IV"] = pd.to_numeric(rent["IV"], errors = "coerce")
    rent = rent.iloc[0:29]
    rent.columns = ['Codi_INE', 'rent_AMB_city']
    rent.Codi_INE = rent.Codi_INE.astype(int)
    gdf = gdf.merge(rent, on = "Codi_INE", how = "left")

    #Barri level - Barcelona
    rent = pd.read_excel(path_data + "trimestral_bcn_lloguer_m2.xlsx", header = 20, sheet_name = "2023")
    rent = rent.iloc[:,[0,5]]
    rent.columns = ["code_barri", "rent_barcelona_barri"]
    admin = pd.read_excel(path_data + "BarcelonaCiutat_SeccionsCensals.xlsx")
    admin = admin.loc[:,["codi_districte", "codi_barri", "codi_seccio_censal"]]
    admin["ID"] = "08019" + admin["codi_districte"].astype(str).str.zfill(2) + admin["codi_seccio_censal"].astype(str).str.zfill(3)
    rent = rent.merge(admin, left_on = "code_barri", right_on = "codi_barri", how = "left")
    gdf = gdf.merge(rent.loc[:,["rent_barcelona_barri", "ID"]], on = "ID", how = "left")

    #District level - Barcelona
    rent = pd.read_excel(path_data + "trimestral_bcn_lloguer_m2.xlsx", header = 8, sheet_name = "2023")
    rent = rent.iloc[0:10,[0,5]]
    rent.columns = ["codi_districte", "rent_barcelona_district"]
    rent = rent.merge(admin, on = "codi_districte", how = "left")
    gdf = gdf.merge(rent.loc[:,["rent_barcelona_district", "ID"]], on = "ID", how = "left")

    #Consolidate
    gdf["rent_m2"] = gdf['rent_AMB_section']
    gdf.loc[np.isnan(gdf.rent_m2), "rent_m2"] = gdf.loc[np.isnan(gdf.rent_m2), "rent_barcelona_barri"]
    #gdf.loc[gdf.rent_m2 < 0.5, "rent_m2"] = np.nan
    gdf.loc[np.isnan(gdf.rent_m2), "rent_m2"] = gdf.loc[np.isnan(gdf.rent_m2), "rent_AMB_city"]
    gdf.loc[np.isnan(gdf.rent_m2), "rent_m2"] = gdf.loc[np.isnan(gdf.rent_m2), "rent_barcelona_district"]

    ### DWELLING SIZE

    #Section level - AMB
    size = pd.read_excel(path_data + "AMB_Superficie.xlsx", header = 5)
    size = size.loc[:,["Codi_àmbit", "IV"]]
    size["IV"] = pd.to_numeric(size["IV"], errors = "coerce")
    size.columns = ['Codi_àmbit', 'size_AMB_section']
    gdf = gdf.merge(size, on = "Codi_àmbit", how = "left")

    #City level - AMB
    size = pd.read_excel(path_data + "AMB_Superficie.xlsx", header = 5)
    size = size.loc[np.isnan(size.Codi_àmbit),["Codi_INE", "IV"]]
    size["IV"] = pd.to_numeric(size["IV"], errors = "coerce")
    size = size.iloc[0:29]
    size.columns = ['Codi_INE', 'size_AMB_city']
    size.Codi_INE = size.Codi_INE.astype(int)
    gdf = gdf.merge(size, on = "Codi_INE", how = "left")

    #Barri level - Barcelona
    size = pd.read_excel(path_data + "trimestral_bcn_sup.xlsx", header = 20, sheet_name = "2023")
    size = size.iloc[:,[0,5]]
    size.columns = ["code_barri", "size_barcelona_barri"]
    size = size.merge(admin, left_on = "code_barri", right_on = "codi_barri", how = "left")
    gdf = gdf.merge(size.loc[:,["size_barcelona_barri", "ID"]], on = "ID", how = "left")

    #District level - Barcelona
    size = pd.read_excel(path_data + "trimestral_bcn_sup.xlsx", header = 8, sheet_name = "2023")
    size = size.iloc[0:10,[0,5]]
    size.columns = ["codi_districte", "size_barcelona_district"]
    size = size.merge(admin, on = "codi_districte", how = "left")
    gdf = gdf.merge(size.loc[:,["size_barcelona_district", "ID"]], on = "ID", how = "left")

    #Consolidate
    gdf["size_AMB"] = gdf['size_AMB_section']
    gdf.loc[np.isnan(gdf["size_AMB"]), "size_AMB"] = gdf.loc[np.isnan(gdf["size_AMB"]), "size_barcelona_barri"]
    gdf.loc[gdf["size_AMB"] < 0.5, "size_AMB"] = np.nan
    gdf.loc[np.isnan(gdf["size_AMB"]), "size_AMB"] = gdf.loc[np.isnan(gdf["size_AMB"]), "size_AMB_city"]
    gdf.loc[np.isnan(gdf["size_AMB"]), "size_AMB"] = gdf.loc[np.isnan(gdf["size_AMB"]), "size_barcelona_district"]

    gdf["size_AMB"] = gdf["size_AMB"]/gdf["active_per_hh"]
    
    gdf = gdf.drop(columns = ['Codi_àmbit', 'rent_AMB_section',
       'Codi_INE', 'rent_AMB_city', 'rent_barcelona_barri',
       'rent_barcelona_district', 'size_AMB_section',
       'size_AMB_city', 'size_barcelona_barri', 'size_barcelona_district'])
    
    return gdf

def import_ppl_per_hh(gdf, path_data):
    """ Use data retrieved with the INE API """

    ppl_per_hh = pd.read_csv(path_data + 'ppl_per_hh.csv')
    ppl_per_hh["ID_7D"] = ppl_per_hh["ID_RESIDENCIA_N4"].str[9:]
    ppl_per_hh.loc[ppl_per_hh["ID_ACTI_HOG_1"] == '3 or more', "ID_ACTI_HOG_1"] = 3
    ppl_per_hh.loc[ppl_per_hh["ID_ACTI_HOG_2"] == '3 or more', "ID_ACTI_HOG_2"] = 3
    ppl_per_hh["ID_ACTI_HOG_1"] = pd.to_numeric(ppl_per_hh["ID_ACTI_HOG_1"])
    ppl_per_hh["ID_ACTI_HOG_2"] = pd.to_numeric(ppl_per_hh["ID_ACTI_HOG_2"])
    ppl_per_hh["active_per_hh"] = ppl_per_hh["ID_ACTI_HOG_1"] + ppl_per_hh["ID_ACTI_HOG_2"]
    
    ppl_per_hh["active_per_hh"] = ppl_per_hh["active_per_hh"] * ppl_per_hh["SHOGARES"]
    
    ppl_per_hh = ppl_per_hh.loc[:,["active_per_hh", "ID_7D", "SHOGARES"]].groupby("ID_7D").sum()
    ppl_per_hh["active_per_hh"] = ppl_per_hh["active_per_hh"] / ppl_per_hh["SHOGARES"]
    ppl_per_hh = ppl_per_hh.loc[:,["active_per_hh"]]
    
    gdf["ID_7D"] = gdf["ID"].str[:7]
    gdf = gdf.merge(ppl_per_hh, on = "ID_7D", how = "left")

    area_housing = pd.read_csv(path_data + 'area_housing.csv')
    area_housing["ID_7D"] = area_housing["ID_RESIDENCIA_N4"].str[9:]
    area_housing.loc[area_housing["ID_SUP_VIV"] == '106-120 m2', "ID_SUP_VIV"] = 113
    area_housing.loc[area_housing["ID_SUP_VIV"] == '121-150 m2', "ID_SUP_VIV"] = 135.5
    area_housing.loc[area_housing["ID_SUP_VIV"] == '151-180 m2', "ID_SUP_VIV"] = 165.5
    area_housing.loc[area_housing["ID_SUP_VIV"] == '30-45 m2', "ID_SUP_VIV"] = 37.5
    area_housing.loc[area_housing["ID_SUP_VIV"] == '46-60 m2', "ID_SUP_VIV"] = 53
    area_housing.loc[area_housing["ID_SUP_VIV"] == '61-75 m2', "ID_SUP_VIV"] = 68
    area_housing.loc[area_housing["ID_SUP_VIV"] == '76-90 m2', "ID_SUP_VIV"] = 83
    area_housing.loc[area_housing["ID_SUP_VIV"] == '91-105 m2', "ID_SUP_VIV"] = 98
    area_housing.loc[area_housing["ID_SUP_VIV"] == 'Less than 30 m2', "ID_SUP_VIV"] = 15
    area_housing.loc[area_housing["ID_SUP_VIV"] == 'More than 180 m2', "ID_SUP_VIV"] = 200

    area_housing["ID_SUP_VIV"] = pd.to_numeric(area_housing["ID_SUP_VIV"], errors = "coerce")
    area_housing["ID_SUP_VIV"] = area_housing["ID_SUP_VIV"] * area_housing["SHOGARES"]
    
    area_housing = area_housing.loc[:,["SHOGARES", "ID_SUP_VIV", "ID_7D"]].groupby("ID_7D").sum()
    area_housing["ID_SUP_VIV"] = area_housing["ID_SUP_VIV"] / area_housing["SHOGARES"]
    
    area_housing = area_housing.merge(ppl_per_hh, on = "ID_7D", how = "left")
    area_housing["size_census"] = area_housing["ID_SUP_VIV"] / area_housing["active_per_hh"]
    gdf["ID_7D"] = gdf["ID"].str[:7]
    gdf = gdf.merge(area_housing.loc[:,["size_census"]], left_on = "ID_7D", right_index = True, how = "left")

    return gdf

def load_transport_times(gdf, path_data, center):
    """ Load transport times previously retrieved with import_transport_time """

    def load_transport_data(mode):
        i = 100
        travel_time_matrix = np.load(path_data + "/travel_time_matrix_" + mode + "_" + str(i) + ".npy", allow_pickle= True) #"tt_" + center + 
    
        while i < len(gdf) - 100:
            i = i + 100
            temp = np.load(path_data + "/travel_time_matrix_" + mode + "_" + str(i) + ".npy", allow_pickle= True) #"tt_" + center + 
            travel_time_matrix = np.concatenate((travel_time_matrix, temp), axis=0)
        
        temp = np.load(path_data + "/travel_time_matrix_" + mode +  "_" + str(len(gdf)) + ".npy", allow_pickle= True) #"tt_" + center + 
        travel_time_matrix = np.concatenate((travel_time_matrix, temp), axis=0)
    
        travel_time_matrix = pd.DataFrame(travel_time_matrix, columns = ['from_id', 'to_id', 'travel_time'])
        travel_time_matrix['travel_time'] = pd.to_numeric(travel_time_matrix['travel_time'], errors='coerce')
        return travel_time_matrix

    return load_transport_data("car"), load_transport_data("transit")

def load_transport_times_poly(gdf, path_data, center, option = ""):
    """ Load transport times previously retrieved with import_transport_time """

    def load_transport_data(mode):
        i = 100
        travel_time_matrix = np.load(path_data + "/travel_time_matrix_poly_" + mode + "_" + str(i) + option + ".npy", allow_pickle= True) #"tt_" + center + 
    
        while i < len(gdf) - 100:
            i = i + 100
            temp = np.load(path_data + "/travel_time_matrix_poly_" + mode + "_" + str(i) + option + ".npy", allow_pickle= True) #"tt_" + center + 
            travel_time_matrix = np.concatenate((travel_time_matrix, temp), axis=0)
        
        temp = np.load(path_data + "/travel_time_matrix_poly_" + mode +  "_" + str(len(gdf)) + option + ".npy", allow_pickle= True) #"tt_" + center + 
        travel_time_matrix = np.concatenate((travel_time_matrix, temp), axis=0)
    
        travel_time_matrix = pd.DataFrame(travel_time_matrix, columns = ['from_id', 'to_id', 'travel_time'])
        travel_time_matrix['travel_time'] = pd.to_numeric(travel_time_matrix['travel_time'], errors='coerce')
        return travel_time_matrix

    return load_transport_data("car"), load_transport_data("transit")


def load_transport_distance(gdf, path_data, center):
    """ Load transport times previously retrieved with import_transport_time """

    i = 100
    travel_distance_matrix = np.load(path_data + "detailed_itin_car_" + str(i) + ".npy", allow_pickle= True)
    
    while i < len(gdf) - 100:
        i = i + 100
        temp = np.load(path_data + "detailed_itin_car_" + str(i) + ".npy", allow_pickle= True)
        travel_distance_matrix = np.concatenate((travel_distance_matrix, temp), axis=0)
        
    temp = np.load(path_data + "detailed_itin_car_" + str(len(gdf)) + ".npy", allow_pickle= True)
    travel_distance_matrix = np.concatenate((travel_distance_matrix, temp), axis=0)
    
    travel_distance_matrix = pd.DataFrame(travel_distance_matrix)
    travel_distance_matrix = travel_distance_matrix.iloc[:,[0,1,6]]
    travel_distance_matrix.columns = ["from_id", "to_id", "distance_car"]
    travel_distance_matrix['distance_car'] = pd.to_numeric(travel_distance_matrix['distance_car'], errors='coerce')
    return travel_distance_matrix


def merge_transport(gdf, travel_time_matrix, center, name):
    """ Merge transport times to a point with gdf """

    gdf = gdf.merge(travel_time_matrix.loc[travel_time_matrix.to_id == center,:], left_on = "ID", right_on = "from_id", how = "left")
    gdf = gdf.drop(columns = ['from_id', 'to_id'])
    gdf = gdf.rename(columns={"travel_time": name})
    return gdf

def add_transport(gdf, travel_time_matrix_car, travel_time_matrix_transit, center):
    """ Merge transport times to a point with gdf """

    gdf = merge_transport(gdf, travel_time_matrix_transit, center, "travel_time_transit")
    gdf = merge_transport(gdf, travel_time_matrix_car, center, "travel_time_car")
    return gdf

def import_cost_transit(gdf):
    """ Cost of the monthly transport pass - per TMB zone """

    gdf["monthly_cost_transit"] = np.nan
    gdf.loc[gdf.code_city.isin(['08015', '08019', '08056', '08077','08101', '08089', '08125', '08126','08169', '08194', '08200', '08211', '08217', '08221', '08245', '08282', '08301', "08073"]), "monthly_cost_transit"] = 22
    gdf.loc[gdf.code_city.isin(['08020', '08054', '08068', '08072', '08123','08157', '08158', '08180', '08196', '08204','08205', '08244', '08252','08263', '08289', '08904', '08905', '08266']), "monthly_cost_transit"] = 29.65
    return gdf

def import_trans_mode(path_data):
    """ Import data on transport mode per census section - for the calibration """

    #https://www.ine.es/dynt3/inebase/en/index.htm?padre=8981&capsel=8982
    
    trans_mode = pd.read_csv(path_data + '55377.csv', sep = ";")
    trans_mode = trans_mode.loc[trans_mode.Municipalities.isin(['Badalona', 'Barcelona', 'Castelldefels', 'Cerdanyola del Vallès', 'Cornellà de Llobregat', 'Granollers', "Hospitalet de Llobregat, L'", 'Manresa', 'Mollet del Vallès', 'Prat de Llobregat, El', 'Sabadell', 'Sant Boi de Llobregat', 'Sant Cugat del Vallès', 'Santa Coloma de Gramenet', 'Terrassa', 'Viladecans', 'Vilanova i la Geltrú']),:] #check manual jusqu'a fuengirola
    trans_mode = trans_mode.loc[trans_mode.Age == "Total",:]
    trans_mode = trans_mode.loc[trans_mode.Sex == "Both sexes",:]
    trans_mode = trans_mode.loc[:,["Municipalities", "Means of transport", "Total"]]
    trans_mode = trans_mode.pivot(index='Municipalities', columns='Means of transport', values='Total')
    trans_mode['Total'] = pd.to_numeric(trans_mode['Total'].str.replace(",", ""), errors="coerce")
    for col in ['Public', 'Particular', 'Walking', 'Company or other media']:
        trans_mode[col] = pd.to_numeric(trans_mode[col].str.replace(",", ""), errors="coerce")
        trans_mode[col] = trans_mode[col] / trans_mode['Total']

    mapping = {
        'Badalona': '08015',
        'Barcelona': '08019',
        'Castelldefels': '08056',
        'Cerdanyola del Vallès': '08266',
        'Cornellà de Llobregat': '08073',
        'Granollers': '08096',
        "Hospitalet de Llobregat, L'": '08101',
        'Manresa': '08113',
        'Mollet del Vallès': '08124',
        'Prat de Llobregat, El': '08169',
        'Sabadell': '08187',
        'Sant Boi de Llobregat': '08200',
        'Sant Cugat del Vallès': '08205',
        'Santa Coloma de Gramenet': '08245',
        'Terrassa': '08279',
        'Viladecans': '08301',
        'Vilanova i la Geltrú': '08307'
        # ... add all 17 mappings here
        }

    trans_mode['code_city'] = trans_mode.index.map(mapping)
    trans_mode["share_car"] = trans_mode["Particular"] + trans_mode["Company or other media"]
    return trans_mode

#add emf -> meilleures données transport (modes et durées, mais pas emploi)
#emf = pd.read_csv(path_data + "emef/Microdades Ús públic_EMEF2023_Desplaçaments (2).csv", sep = ";")
#emf = emf.loc[(emf.TIPOL == 1) & (emf.V03A == 3),:]
#emf = emf.loc[:,["ID", "DISTANCIA_ORTO_REC_R1", "COM_O2", "COM_D2", "V03G_R3"]]
#emf = emf.groupby("ID").first()
#emf["indic"] = 1
#emf.loc[:,["indic", "COM_D2"]].groupby("COM_D2").sum()

def import_beach(gdf, path_data):
    """ Import distance to the beach from land cover data """

    land_cover = gpd.read_file(path_data + "cobertes-sol-v1r0-2023.gpkg", layer="cobertes_sol", engine="fiona")
    land_cover = land_cover.to_crs(gdf.crs)
    gdf['min_distance_beach'] = gdf.centroid.apply(
        lambda center: land_cover.loc[land_cover.nivell_2 == 233, :].distance(center).min()
        )
    gdf["beach_500m"] = (gdf['min_distance_beach'] < 500) * 1
    gdf["beach_500m_1km"] = ((gdf['min_distance_beach'] < 1000) & (gdf['min_distance_beach'] > 500)) * 1
    gdf["beach_1km_2km"] = ((gdf['min_distance_beach'] < 2000) & (gdf['min_distance_beach'] > 1000)) * 1
    return gdf.loc[:,['ID', 'min_distance_beach', "beach_500m", "beach_500m_1km", "beach_1km_2km"]]

def import_parcs(gdf, path_data):
    """ Import distance to parcs from both Barcelona and AMB data """

    gdf["barcelona"] = (gdf.code_city == "08019")
    parcs = gpd.read_file(path_data + "pev_parcs_od.gpkg")
    parcs = parcs.to_crs(gdf.crs)

    parcs2 = gpd.read_file(path_data + "equipaments_pro.kml")
    parcs2 = parcs2.to_crs(gdf.crs)

    parcs_combined = pd.concat([parcs["geometry"], parcs2["geometry"]])
    parcs_combined = gpd.GeoDataFrame(parcs_combined, geometry="geometry", crs=parcs.crs)

    parcs_combined = gpd.GeoDataFrame(
        geometry=pd.concat([parcs.loc[parcs.area_ha > 2, "geometry"], parcs2["geometry"]]),
        crs=parcs.crs
    )

    gdf['min_distance_parc_combined'] = gdf.centroid.apply(lambda geom: parcs_combined.distance(geom).min())

    gdf["parc_500m"] = (gdf['min_distance_parc_combined'] < 500) * 1
    gdf["parc_500m_1km"] = ((gdf['min_distance_parc_combined'] < 1000) & (gdf['min_distance_parc_combined'] > 500)) * 1
    gdf["parc_1km_2km"] = ((gdf['min_distance_parc_combined'] < 2000) & (gdf['min_distance_parc_combined'] > 1000)) * 1

    gdf["parc_500m_b"] = gdf["parc_500m"] * gdf["barcelona"]
    gdf["parc_500m_1km_b"] = gdf["parc_500m_1km"] * gdf["barcelona"]
    gdf["parc_1km_2km_b"] = gdf["parc_1km_2km"] * gdf["barcelona"]

    return gdf.loc[:,["ID", 'min_distance_parc_combined', "parc_500m", "parc_500m_1km", "parc_1km_2km", "parc_500m_b", "parc_500m_1km_b", "parc_1km_2km_b"]]

def import_stations(gdf, option = "sants_only"):
    """ Import the locations of the main train stations of Barcelona """
    
    stations = [
    {"name": "Barcelona Sants", "lon": 2.1399, "lat": 41.3809},
    {"name": "Estació de França", "lon": 2.1875, "lat": 41.3836},
    {"name": "Passeig de Gràcia", "lon": 2.1665, "lat": 41.3916},
    {"name": "Plaça de Catalunya", "lon": 2.1701, "lat": 41.3870},
    {"name": "El Clot-Aragó", "lon": 2.1924, "lat": 41.4112},
    {"name": "Arc de Triomf", "lon": 2.1802, "lat": 41.3912},
    {"name": "Sant Andreu", "lon": 2.1897, "lat": 41.4351}
    ]

    # Create GeoDataFrame in EPSG:4326
    stations = gpd.GeoDataFrame(
        stations,
        geometry=[Point(s["lon"], s["lat"]) for s in stations],
        crs="EPSG:4326"
    )

    stations = stations.to_crs(gdf.crs)

    if option == "sants_only":
        station_geom = stations.geometry.iloc[0]
    else:
        station_geom = stations.geometry

    gdf['min_distance_stations'] = gdf.centroid.distance(station_geom)

    gdf["station_500m"] = (gdf['min_distance_stations'] < 500) * 1
    gdf["station_500m_1km"] = ((gdf['min_distance_stations'] < 1000) & (gdf['min_distance_stations'] > 500)) * 1
    gdf["station_1km_2km"] = ((gdf['min_distance_stations'] < 2000) & (gdf['min_distance_stations'] > 1000)) * 1

    return gdf.loc[:,["ID", 'min_distance_stations', "station_500m", "station_500m_1km", "station_1km_2km"]]

def import_airport(gdf):
    """ Import the location of the airport """

    lon, lat = 2.0783, 41.2969
    airport_wgs84 = gpd.GeoDataFrame(
        {'name': ['Barcelona Airport']},
        geometry=[Point(lon, lat)],
        crs='EPSG:4326'  # WGS84
    )
    airport_25830 = airport_wgs84.to_crs(epsg=25830)
    gdf['min_distance_airport'] = gdf.centroid.apply(lambda geom: airport_25830.distance(geom).min())
    gdf["airport_500m"] = (gdf['min_distance_airport'] < 500) * 1
    
    return gdf.loc[:,["ID", "min_distance_airport", "airport_500m"]]

def import_touristic_areas(gdf, path_data):
    """ Import the locations of touristic areas
    
    https://opendata-ajuntament.barcelona.cat/data/en/dataset/habitatges-us-turistic
    There are other data to check as well.
    """

    tourism = gpd.read_file(path_data + "2019_turisme_allotjament.gpkg")
    tourism = tourism.to_crs(gdf.crs)
    joined = gpd.sjoin(tourism[['DN', 'geometry']], gdf[['geometry']], how='inner', predicate='intersects')
    mean_dn = joined.groupby('index_right')['DN'].mean()
    gdf['index_tourism'] = gdf.index.map(mean_dn)
    gdf.loc[np.isnan(gdf.index_tourism), "index_tourism"] = 0

    gdf["high_tourism"] = (gdf['index_tourism'] > 50) * 1
    gdf["medium_tourism"] = ((gdf['index_tourism'] < 50) & (gdf['index_tourism'] > 30)) * 1

    return gdf.loc[:,["ID", "high_tourism", "medium_tourism"]]

def import_activity(gdf, path_data):
    """ Import data about activity from Barcelona """

    activity = gpd.read_file(path_data + "2018_cohesio_sobreocupacio.gpkg")
    activity = activity.to_crs(gdf.crs)
    joined = gpd.sjoin(activity[['norm', 'geometry']], gdf[['geometry']], how='inner', predicate='intersects')
    mean_dn = joined.groupby('index_right')['norm'].mean()
    gdf['mean_activity'] = gdf.index.map(mean_dn)
    gdf.loc[np.isnan(gdf.mean_activity), "mean_activity"] = 0

    return gdf.loc[:,["ID", "mean_activity"]]

def import_pedestrian_streets(gdf, path_data):
    """ Import data on the density of pedestrian streets, both from OSM and Barcelona Open Data Platform"""
    
    amb_polygon = gdf.to_crs(epsg=4326).unary_union
    minx, miny, maxx, maxy = amb_polygon.bounds
    bbox = f"{miny},{minx},{maxy},{maxx}"

    # Overpass QL query to get ways tagged as pedestrian or footway within bbox
    overpass_url = "http://overpass-api.de/api/interpreter"
    query = f"""
    [out:json][timeout:25];
    (
    way["highway"="pedestrian"]({bbox});
    );
    out geom;
    """

    response = requests.post(overpass_url, data={'data': query})
    data = response.json()

    # Extract ways and convert to shapely LineStrings
    lines = []
    for element in data['elements']:
        if element['type'] == 'way' and 'geometry' in element:
            coords = [(pt['lon'], pt['lat']) for pt in element['geometry']]
            if len(coords) > 1:
                lines.append(LineString(coords))

    # Create GeoDataFrame
    gdf_pedestrian = gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326")
    pedestrian_data = gpd.read_file(path_data + "Carrers_Amb_Prioritat_Vianants/Carrers_Prioritat_Vianants.shp")
    gdf = compute_pedestrian_density(gdf, gdf_pedestrian, "pedestrian_density")
    gdf = compute_pedestrian_density(gdf, pedestrian_data, "pedestrian_data_density")
    
    return gdf.loc[:, ["ID", "pedestrian_density", "pedestrian_data_density"]]

def compute_pedestrian_density(gdf, gdf_pedestrian, name):
    gdf_pedestrian = gdf_pedestrian.to_crs(gdf.crs)
    gdf_pedestrian["length_m"] = gdf_pedestrian.geometry.length
    joined = gpd.sjoin(gdf_pedestrian, gdf, predicate="intersects")
    agg_length = joined.groupby("index_right")["length_m"].sum()
    gdf[name] = gdf.index.map(agg_length).fillna(0)
    gdf[name] = gdf[name]  / gdf["area"] 
    return gdf

def import_rodalies(gdf, path_data):
    """ Import location of rodalies stations """

    rodalies = pd.read_excel(path_data + "listado-estaciones-rodalies-barcelona.xlsx")
    rodalies = gpd.GeoDataFrame(
        rodalies, 
        geometry=gpd.points_from_xy(rodalies.LONGITUD, rodalies.LATITUD),
        crs="EPSG:4326"  # WGS 84
        )

    rodalies = rodalies.to_crs(gdf.crs)
    gdf['min_distance_rodalies'] = gdf.centroid.apply(lambda geom: rodalies.distance(geom).min())
    gdf["rodalies_500m"] = (gdf['min_distance_rodalies'] < 500) * 1

    return gdf.loc[:,["ID", "min_distance_rodalies", "rodalies_500m"]]

def import_fgc(gdf, path_data):
    """ Import locations of fgc stations """
    
    fgc = pd.read_excel(path_data + "gtfs_stops.xlsx")
    fgc[['lat', 'lon']] = fgc['stop_coordinates'].str.split(',', expand=True).astype(float)
    fgc = gpd.GeoDataFrame(
        fgc, 
        geometry=gpd.points_from_xy(fgc.lon, fgc.lat),
        crs="EPSG:4326"  # WGS 84
    )

    fgc = fgc.to_crs(gdf.crs)
    gdf['min_distance_fgc'] = gdf.centroid.apply(lambda geom: fgc.distance(geom).min())
    gdf["fgc_500m"] = (gdf['min_distance_fgc'] < 500) * 1

    return gdf.loc[:,["ID", "min_distance_fgc", "fgc_500m"]]

def import_slope(gdf, path_data):
    """ Import slope/elevation data from IGC """

    root_dir = path_data + "mp20p5m_ETRS89zt1751632652072"
    slope_data = []

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".zip"):
                zip_path = os.path.join(dirpath, filename)
                try:
                    slope_here = gpd.read_file(f"zip://{zip_path}")
                    slope_data.append(slope_here)
                except Exception as e:
                    print(f"Failed to read {zip_path}: {e}")

    # Concatenate all GeoDataFrames into one (optional)
    if slope_data:
        full_gdf = gpd.GeoDataFrame(pd.concat(slope_data, ignore_index=True), crs=slope_data[0].crs)

    # Ensure both GeoDataFrames use the same projected CRS (for accurate area computation)
    full_gdf = full_gdf.to_crs(gdf.crs)

    # Spatial join: restrict full_gdf polygons to only those that intersect each gdf polygon
    results = []

    for idx, row in gdf.iterrows():
        target_geom = row.geometry
        intersections = full_gdf[full_gdf.intersects(target_geom)].copy()
        if intersections.empty:
            results.append(0.0)
            continue

        intersections['intersection'] = intersections.geometry.intersection(target_geom)
        covered_area = intersections['intersection'].area.sum()
        total_area = target_geom.area
        share = covered_area / total_area
        results.append(share)

    gdf['slope_20'] = results

    return gdf.loc[:,["ID", "slope_20"]]

def import_amenities(gdf, path_data, option_load, option_save):

    if option_load == 1:

        data_amenity = import_beach(gdf, path_data)
        data_amenity = data_amenity.merge(import_parcs(gdf, path_data), on = "ID", how = "left")
        data_amenity = data_amenity.merge(import_stations(gdf, option = "sants_only"), on = "ID", how = "left")
        data_amenity = data_amenity.merge(import_airport(gdf), on = "ID", how = "left")
        data_amenity = data_amenity.merge(import_touristic_areas(gdf, path_data), on = "ID", how = "left")
        data_amenity = data_amenity.merge(import_activity(gdf, path_data), on = "ID", how = "left")
        data_amenity = data_amenity.merge(import_pedestrian_streets(gdf, path_data), on = "ID", how = "left")
        data_amenity = data_amenity.merge(import_rodalies(gdf, path_data), on = "ID", how = "left")
        data_amenity = data_amenity.merge(import_fgc(gdf, path_data), on = "ID", how = "left")
        data_amenity = data_amenity.merge(import_slope(gdf, path_data), on = "ID", how = "left")

        if option_save == 1:

            data_amenity.to_excel(path_data + "data_amenity.xlsx")

    elif option_load == 0:

        data_amenity = pd.read_excel(path_data + "data_amenity.xlsx", index_col = 0)
        data_amenity['ID'] = '0' + data_amenity['ID'].astype(str)

    gdf["ID"] = gdf["ID"].astype(str)
    return gdf.merge(data_amenity, on = "ID", how = "left")

def import_rent_idealista(gdf, path_data):
    rent_idea = gpd.read_file(path_data + "barcelona_rent_idealista.gpkg", layer="points")
    rent_idea.plot("UNITPRICE", s = 1, legend = True)
    rent_idea = rent_idea.to_crs(gdf.crs)
    joined = gpd.sjoin(rent_idea, gdf, how="inner", predicate="within")
    agg = joined.groupby("ID").agg(
        count_dea_data=("UNITPRICE", "count"),
        mean_rent_per_sqm=("UNITPRICE", "mean"),
        median_rent_per_sqm=("UNITPRICE", "median")
        ).reset_index()
    
    gdf = gdf.merge(agg, on="ID", how = "left")
    return gdf

def import_tax_zone(gdf, employment_centers):
    zone_tax = gdf.loc[gdf.ID.str[:5].isin(["08019", "08101", "08194"]),:]
    #fig, ax = plt.subplots(figsize=(8, 8))
    #gdf.plot(ax = ax, color = "lightgrey")
    #zone_tax.plot(ax = ax)
    #plt.show()
    zone_union = zone_tax.union_all()
    clusters_in_zone = employment_centers[employment_centers.within(zone_union)]["cluster"].unique().tolist()
    house_in_zone = gdf[gdf.centroid.within(zone_union)]["ID"].unique().tolist()
    return clusters_in_zone, house_in_zone, zone_union