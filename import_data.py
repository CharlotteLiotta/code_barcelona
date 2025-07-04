import numpy as np # type: ignore
import pandas as pd
import geopandas as gpd # type: ignore
from shapely import wkt # type: ignore

from import_data import * # type: ignore

def import_data(path_data, center, option):
    """ Import shapefile + population """

    if option == "SECTION":
        #https://www.ine.es/dynt3/inebase/en/index.htm?padre=11676&capsel=11681
        
        #Import population per census tract
        df = pd.read_csv(path_data + '70035.csv', sep = ";", encoding="latin1")
        df = df.loc[:,["Sections", "Total"]]
        df = df.dropna(subset=["Sections"])
        df["CUSEC"] = df["Sections"].str[:10]

        def fix_decimal(s):
            if isinstance(s, str) and ',' in s:
                integer, decimal = s.split(',', 1)
                if len(decimal) == 2:
                    return f"{integer},{decimal}0"
            return s

        #Merge with shapefile
        gdf = gpd.read_file(path_data + 'seccionado_2024/SECC_CE_20240101.shp')
        gdf = gdf.merge(df, on = "CUSEC")

        #Compute area, density, distance to city center
        gdf["Total"] = gdf["Total"].apply(fix_decimal)
        gdf["Total"] = pd.to_numeric(gdf["Total"].str.replace(',', ''), errors='coerce')
        gdf = gdf.loc[gdf.NMUN.isin(['Badalona', 'Badia del Vallès', 'Barberà del Vallès', 'Barcelona','Begues','Castellbisbal', 'Castelldefels', 'Cerdanyola del Vallès', 'Cervelló','Corbera de Llobregat','Papiol, El', 'Prat de Llobregat, El', 'Esplugues de Llobregat','Gavà', "Hospitalet de Llobregat, L'",'Palma de Cervelló, La','Molins de Rei', 'Montcada i Reixac', 'Montgat', 'Pallejà', 'Ripollet','Sant Adrià de Besòs', 'Sant Andreu de la Barca','Sant Boi de Llobregat', 'Sant Climent de Llobregat', 'Sant Cugat del Vallès', 'Sant Feliu de Llobregat','Sant Joan Despí','Sant Just Desvern','Sant Vicenç dels Horts', 'Santa Coloma de Cervelló', 'Santa Coloma de Gramenet','Tiana', 'Torrelles de Llobregat', 'Viladecans']),["CUSEC", "CUMUN", "Shape_Area", "Total", "geometry", "NMUN"]]
        gdf.CUSEC = gdf.CUSEC.astype(str)
        city_center = gdf.loc[gdf.CUSEC == center,:].centroid
        gdf["distance_center"] = gdf.centroid.distance(city_center.iloc[0], align = False) / 1000
        gdf = gdf.loc[:,["CUSEC", "geometry", "Shape_Area", "Total", "distance_center"]]
        
    elif option == "DISTRICT":

        #Import population per district
        df = pd.read_csv(path_data + "2021_densitat.csv")
        df = df.groupby("Nom_Districte").sum("Població")
        
        #Merge with shapefile
        gdf = gpd.read_file(path_data + 'BarcelonaCiutat_Districtes.csv')
        gdf = gdf.merge(df["Població"], left_on = "nom_districte", right_index = True)
        
        #Compute area, density, distance to city center
        gdf['geometry'] = gdf['geometria_etrs89'].apply(wkt.loads)
        gdf = gpd.GeoDataFrame(geometry="geometry", data=gdf, crs="EPSG:25831")
        gdf["area"] = gdf.area
        city_center = gdf.loc[gdf.nom_districte == "Eixample",:].centroid
        gdf["distance_center"] = gdf.centroid.distance(city_center.iloc[0], align = False) / 1000
        gdf = gdf.loc[:,["Codi_Districte", "geometry", "area", "Població", "distance_center"]]

    gdf.columns = ["ID", "geometry", "area", "pop", "distance_center"]
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
    income = pd.read_csv(path_data + '30896.csv', sep = ";", encoding="latin1")
    income = income.loc[(income.Periodo == 2022) & (income['Mean and median income indicators'] == 'Average household net income'),["Sections", "Total"]]
    income = income.dropna(subset=["Sections"])
    income["ID"] = income["Sections"].str[:10]
    income.columns = ['Sections', 'net_income', 'ID']
    income.net_income = pd.to_numeric(income.net_income, errors= "coerce")
    income.net_income = income.net_income * 1000

    #Merge with gdf
    gdf = gdf.merge(income.loc[:,['net_income', 'ID']], on = "ID", how = "left")
    
    gdf["net_income"] = gdf["net_income"] / gdf["active_per_hh"]
    #Compute average income
    Y = (np.nansum(gdf.net_income * gdf["pop"]) / np.nansum(gdf["pop"]))
    
    return Y / 12, gdf

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
    gdf["size"] = gdf['size_AMB_section']
    gdf.loc[np.isnan(gdf["size"]), "size"] = gdf.loc[np.isnan(gdf["size"]), "size_barcelona_barri"]
    gdf.loc[gdf["size"] < 0.5, "size"] = np.nan
    gdf.loc[np.isnan(gdf["size"]), "size"] = gdf.loc[np.isnan(gdf["size"]), "size_AMB_city"]
    gdf.loc[np.isnan(gdf["size"]), "size"] = gdf.loc[np.isnan(gdf["size"]), "size_barcelona_district"]

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
    area_housing["size"] = area_housing["ID_SUP_VIV"] / area_housing["active_per_hh"]
    gdf["ID_7D"] = gdf["ID"].str[:7]
    gdf = gdf.drop(columns = ["size"])
    gdf = gdf.merge(area_housing.loc[:,["size"]], left_on = "ID_7D", right_index = True, how = "left")

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
    gdf.loc[gdf.code_city.isin(['08015', '08019', '08056', '08077','08101', '08089', '08125', '08126','08169', '08194', '08200', '08211', '08217', '08221', '08245', '08282', '08301', ]), "monthly_cost_transit"] = 22
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