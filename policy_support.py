import pyreadstat # type: ignore
import statsmodels.api as sm # type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore
import pandas as pd
import matplotlib.pyplot as plt

from import_data import *

def import_opinion_parameters(path_data):

    
    def weighted_median(values, weights):
        # Sort values and weights by values
        sorted_idx = np.argsort(values)
        values, weights = values[sorted_idx], weights[sorted_idx]
    
        # Compute cumulative weights
        cum_weights = np.cumsum(weights) / np.sum(weights)
    
        # Find first value where cum_weight >= 0.5
        return values[cum_weights >= 0.5][0]

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
    #X_raw = df_reg[["climate_change", "log_absolute_loss", 'congestion']]
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

    #X_scaled["PESAIX"] = df_reg["PESAIX"]

    return np.array(model_statsmodel.params), weighted_median(y_scaled, df_reg["PESAIX"].values) #np.nanmedian(y_scaled)

def import_price_parameters(path_data):

    def weighted_median(values, weights):
        # Sort values and weights by values
        sorted_idx = np.argsort(values)
        values, weights = values[sorted_idx], weights[sorted_idx]
    
        # Compute cumulative weights
        cum_weights = np.cumsum(weights) / np.sum(weights)
    
        # Find first value where cum_weight >= 0.5
        return values[cum_weights >= 0.5][0]

    df_reg, meta = pyreadstat.read_sav(path_data + '040023 En moviment pel clima 2025_V01 - còpia.sav')
    
    df_reg = gpd.GeoDataFrame(df_reg, geometry = gpd.points_from_xy(df_reg.GEO_X, df_reg.GEO_Y), crs="EPSG:4326")
    gdf = gpd.read_file(path_data + "income_loss.geojson")
    df_reg = df_reg.to_crs(gdf.crs)
    df_reg = gpd.sjoin(df_reg, gdf, predicate="within")

    #df_reg.rename(columns={'P18': 'acceptable_price'}, inplace=True)
    df_reg.rename(columns={'P17': 'acceptable_price'}, inplace=True)
    
    df_reg = df_reg.loc[~np.isnan(df_reg.acceptable_price) & (df_reg.acceptable_price < 15)]

    #df_reg.acceptable_price = df_reg.acceptable_price / 2

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
    #X_raw = df_reg[["climate_change", "log_absolute_loss", 'congestion']]
    y_raw = (df_reg["acceptable_price"].values.reshape(-1, 1))

    
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X_raw)
    X_scaled = pd.DataFrame(X_scaled, columns=X_raw.columns)

    #X_scaled["Traffic_I"] = X_scaled["Traffic"] * X_scaled["Ideology"] 
    #X_scaled["Quality_life_I"] = X_scaled["Quality_life"] * X_scaled["Ideology"] 
    #X_scaled["Climate_change_I"] = X_scaled["Climate_change"] * X_scaled["Ideology"] 

    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y_raw).flatten()

    X_scaled = sm.add_constant(X_scaled)

    model_statsmodel = sm.WLS(y_raw, X_scaled, weights=df_reg['PESAIX']).fit()
    print(model_statsmodel.summary())

    print(np.nanmedian(X_scaled, 0))

    return np.array(model_statsmodel.params), weighted_median(df_reg["acceptable_price"].values, df_reg["PESAIX"].values) #np.nanmedian(y_scaled) #np.nanmedian(y_raw)

def compute_political_opinion(score_welfare, score_quality_of_life, score_emissions, score_congestion, BETA_OPINION):
    """ Compute public support based on the regression on the survey data """
    
    return BETA_OPINION[0] + BETA_OPINION[1] * score_emissions + BETA_OPINION[2] * score_welfare + BETA_OPINION[3] * score_quality_of_life + BETA_OPINION[4] * score_congestion

def compute_price(score_welfare, score_quality_of_life, score_emissions, score_congestion, BETA_OPINION):
    """ Compute public support based on the regression on the survey data """
    
    return BETA_OPINION[0] + BETA_OPINION[1] * score_emissions + BETA_OPINION[2] * score_welfare + BETA_OPINION[3] * score_quality_of_life + BETA_OPINION[4] * score_congestion
