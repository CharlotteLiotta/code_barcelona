import pyreadstat # type: ignore
import statsmodels.api as sm # type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore
import pandas as pd
import matplotlib.pyplot as plt

from import_data import *

def import_opinion_parameters_old(path_data):

    df, meta = pyreadstat.read_sav(path_data + 'oltra_barcelona/data.sav')

    # Regression as in Konc et al.

    #Acceptability: Q17 (acceptability LEZ), Q16 (evaluation LEZ), Q19_4 (evaluation toll)
    #Ideology: P8 (political ideology) or Q3_1 (environmental behavior) or Q2_4 (pollution as a problem)
    #Effectiveness: Q10_1 (pollution) or Q10_2 (traffic) or Q12_2, Q12_3, Q12_4, Q12_5
    #Well-being: Q11_1 (quality of life), Q11_2 (freedom)

    df_reg = df.loc[:,["Q17", "Q10_1", "Q11_2", "Q13", "P8", "Q3_1", "P1", "qtAge", 'qtEducation', "P9", "Q14"]]
    df_reg.columns = ["Acceptability", "Effectiveness", "Quality of life", "Justice", "Ideology", "Pro_envt", "Gender", "Age", "Education", "Children", "Democratic"]
    df_reg = df_reg.loc[df_reg.Ideology != 6,:]

    #df_reg = df_reg.loc[df_reg["Quality of life"] != 3,:]
    #df_reg["Quality of life"] = (df_reg["Quality of life"] == 1) * 1


    #df_reg["Effectiveness_I"] = df_reg["Effectiveness"] * df_reg["Ideology"] 
    #df_reg["Quality of life_I"] = df_reg["Quality of life"] * df_reg["Ideology"] 
    #df_reg["Justice_I"] = df_reg["Justice"] * df_reg["Ideology"] 
    df_reg["Man"] = (df_reg["Gender"] == 1) * 1
    df_reg["age_l40"] = (df_reg["Age"] == 1) * 1
    #df_reg["age_h65"] = (df_reg["Age"] == 3) * 1
    #df_reg["University"] = (df_reg["Education"] == 2) * 1
    #df_reg["Children"] = (df_reg["Children"] == 1) * 1
    df_reg["Quality of life"] = 6 - df_reg["Quality of life"]
    
    #X_raw = df_reg[["Effectiveness", "Quality of life", "Justice", "age_l40", "Democratic", "Pro_envt"]]
    X_raw = df_reg[["Effectiveness", "Quality of life", "Justice"]]
    y_raw = df_reg["Acceptability"].values.reshape(-1, 1)

    # Min-max normalization for X
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X_raw)
    X_scaled = pd.DataFrame(X_scaled, columns=X_raw.columns)

    # Min-max normalization for y
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y_raw).flatten()

    # Add intercept
    X_scaled = sm.add_constant(X_scaled)

    model_statsmodel = sm.OLS(y_scaled, X_scaled).fit()
    print(model_statsmodel.summary())

    return np.array(model_statsmodel.params)

def import_opinion_parameters(path_data):

    df_reg, meta = pyreadstat.read_sav(path_data + '040023 En moviment pel clima 2025_V01 - còpia.sav')
    
    df_reg = gpd.GeoDataFrame(df_reg, geometry = gpd.points_from_xy(df_reg.GEO_X, df_reg.GEO_Y), crs="EPSG:4326")
    gdf = gpd.read_file(path_data + "income_loss.geojson")
    df_reg = df_reg.to_crs(gdf.crs)
    df_reg = gpd.sjoin(df_reg, gdf, predicate="within")

    df_reg.rename(columns={'P22_4': 'acceptability'}, inplace=True)
    df_reg = df_reg.loc[~np.isnan(df_reg.acceptability) & (df_reg.acceptability < 97)]

    df_reg.rename(columns={'P21_3': 'climate_change'}, inplace=True)
    df_reg = df_reg.loc[~np.isnan(df_reg.climate_change) & (df_reg.climate_change < 80)]
    
    #df_reg.rename(columns={'P11': 'declared_impacts'}, inplace=True)
    #df_reg = df_reg.loc[(df_reg.declared_impacts != 3) & (df_reg.declared_impacts != 4),:]
    #df_reg.declared_impacts = (df_reg.declared_impacts == 1) * 1

    df_reg["log_absolute_loss"] = - np.log(-df_reg["income_loss_absolute"])

    df_reg.rename(columns={'P21_2': 'quality_of_life'}, inplace=True)
    df_reg = df_reg.loc[~np.isnan(df_reg.quality_of_life) & (df_reg.quality_of_life < 80)]

    df_reg.rename(columns={'P21_1': 'congestion'}, inplace=True)
    df_reg = df_reg.loc[~np.isnan(df_reg.congestion) & (df_reg.congestion < 80)]

    X_raw = df_reg[["climate_change", "log_absolute_loss", "quality_of_life", 'congestion']]
    y_raw = df_reg["acceptability"].values.reshape(-1, 1)

    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X_raw)
    X_scaled = pd.DataFrame(X_scaled, columns=X_raw.columns)

    #X_scaled["Traffic_I"] = X_scaled["Traffic"] * X_scaled["Ideology"] 
    #X_scaled["Quality_life_I"] = X_scaled["Quality_life"] * X_scaled["Ideology"] 
    #X_scaled["Climate_change_I"] = X_scaled["Climate_change"] * X_scaled["Ideology"] 

    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y_raw).flatten()

    X_scaled = sm.add_constant(X_scaled)

    model_statsmodel = sm.WLS(y_scaled, X_scaled, weights=df_reg['PESAIX']).fit()
    print(model_statsmodel.summary())

    return np.array(model_statsmodel.params)

def compute_political_opinion(score_welfare, score_quality_of_life, score_emissions, score_congestion, BETA_OPINION):
    """ Compute public support based on the regression on the survey data """
    
    return BETA_OPINION[0] + BETA_OPINION[1] * score_emissions + BETA_OPINION[2] * score_welfare + BETA_OPINION[3] * score_quality_of_life + BETA_OPINION[4] * score_congestion
