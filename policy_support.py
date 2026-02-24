import pyreadstat # type: ignore
import statsmodels.api as sm # type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from import_data import *
from plotting_tools import *

def compute_opinion2(score_welfare, score_quality_of_life, score_emissions, BETA_OPINION, scenario, lvl, score_knowledge):
    """ Compute public support based on the regression on the survey data """
    
    dummy_low = (lvl == "LOW")
    dummy_high = (lvl == "HIGH")

    if scenario == "increasing_knowledge":
        opinion = np.nan #BETA_OPINION[0] + BETA_OPINION[1] * score_emissions + BETA_OPINION[2] * score_quality_of_life + BETA_OPINION[3] * score_welfare + BETA_OPINION[4] * score_knowledge
    else:
        opinion = BETA_OPINION[0] + BETA_OPINION[1] * score_emissions + BETA_OPINION[2] * score_quality_of_life + BETA_OPINION[3] * score_welfare + BETA_OPINION[4] * dummy_low + BETA_OPINION[5] * dummy_high
        
    return opinion

def import_opinion_parameters2(path_data, scenario, expected_welfare_loss):
    
    def weighted_median(values, weights):
        # Sort values and weights by values
        sorted_idx = np.argsort(values)
        values, weights = values[sorted_idx], weights[sorted_idx]
    
        # Compute cumulative weights
        cum_weights = np.cumsum(weights) / np.sum(weights)
    
        # Find first value where cum_weight >= 0.5
        return values[cum_weights >= 0.5][0]

    #import survey
    df_reg, meta = pyreadstat.read_sav(path_data + '040023 En moviment pel clima 2025_V01 - còpia.sav')

    #add income variable (Joan)
    income_var = pd.read_excel(path_data + "MOBCLIMA_amb_renda_codipaisINE.xlsx")
    df_reg = df_reg.merge(income_var.loc[:,["NUME", "categoria_quintils", "ingressos_estimats"]], on = "NUME")

    #add welfare losses
    df_reg = gpd.GeoDataFrame(df_reg, geometry = gpd.points_from_xy(df_reg.GEO_X, df_reg.GEO_Y), crs="EPSG:4326")
    df_reg = df_reg.to_crs(expected_welfare_loss.crs)
    df_reg = gpd.sjoin(df_reg, expected_welfare_loss, how = "left", predicate="within")

    #prepare variables

    #dependent var 1: acceptability
    df_reg.rename(columns={'P22_4': 'acceptability'}, inplace=True)
    
    #dependent var 2: acceptable price
    df_reg.rename(columns={'P18': 'acceptable_price'}, inplace=True)
    df_reg.acceptable_price = df_reg.acceptable_price / 2

    
    print(spearmanr(df_reg["acceptability"], df_reg["acceptable_price"], nan_policy="omit"))

    # 4. Conditional means with 95% CI
    df_reg_here = df_reg.loc[~np.isnan(df_reg.acceptability) & (df_reg.acceptability < 97)]
    df_reg_here = df_reg_here.loc[~np.isnan(df_reg_here.acceptable_price) & (df_reg_here.acceptable_price < 20)]
    summary = (
        df_reg_here.groupby("acceptability")["acceptable_price"]
        .agg(["mean", "std", "count"])
    )
    summary["se"] = summary["std"] / (summary["count"] ** 0.5)

    plt.figure()
    plt.errorbar(
        summary.index,
        summary["mean"],
        yerr=1.96 * summary["se"],
        fmt="o"
    )
    plt.xlabel("Acceptability (0–10)")
    plt.ylabel("Mean acceptable price")
    plt.show()

    #explanatory vars

    df_reg.rename(columns={'P21_3': 'climate_change'}, inplace=True)

    df_reg.rename(columns={'P21_2': 'quality_of_life'}, inplace=True)
    
    df_reg["LOW"] = (df_reg.ingressos_estimats  < 11550) * 1
    df_reg["HIGH"] = (df_reg.ingressos_estimats > 26950) * 1

    #df_reg["score_welfare"] = df_reg["score_welfare_low"] * df_reg["LOW"] + df_reg["score_welfare_high"] * df_reg["HIGH"]+ df_reg["score_welfare_med"] * (1 - df_reg["HIGH"] - df_reg["LOW"])
    df_reg["score_welfare"] = df_reg["score_welfare_med"]
    for i in range(len(df_reg["score_welfare"])):
        if df_reg["LOW"].iloc[i] == 1:
            df_reg["score_welfare"].iloc[i] = df_reg["score_welfare_low"].iloc[i]
        elif df_reg["HIGH"].iloc[i] == 1:
            df_reg["score_welfare"].iloc[i] = df_reg["score_welfare_high"].iloc[i]
    #df_reg = df_reg.loc[~np.isnan(df_reg.score_welfare)]
    sum(np.isnan(df_reg.score_welfare))

    df_reg["vehicle_ownership_license"] = 1 * (((df_reg.P34A > 0) &(df_reg.P33A  == 1))| ((df_reg.P34B > 0) &(df_reg.P33B  == 1)) |((df_reg.P34C > 0) &(df_reg.P33A  == 1))) 
    df_reg["score_welfare_vehicle_ownership_license"] = df_reg["score_welfare"]
    df_reg.loc[df_reg["vehicle_ownership_license"] == 0, "score_welfare_vehicle_ownership_license"] = 0.5

    #df_reg = df_reg.loc[~np.isnan(df_reg.acceptability) & (df_reg.acceptability < 97)]
    #df_reg = df_reg.loc[~np.isnan(df_reg.acceptable_price) & (df_reg.acceptable_price < 20)]
    #df_reg = df_reg.loc[~np.isnan(df_reg.climate_change) & (df_reg.climate_change < 80)]
    #df_reg = df_reg.loc[~np.isnan(df_reg.quality_of_life) & (df_reg.quality_of_life < 80)]
    #df_reg = df_reg.loc[~np.isnan(df_reg.score_welfare)]
    
    #Export plots
    plot_mobility_loss("score_welfare_vehicle_ownership_license","acceptable_price", df_reg.loc[~np.isnan(df_reg.acceptable_price) & (df_reg.acceptable_price < 20) & ~np.isnan(df_reg.score_welfare)])
     
    def plot_hist_survey(df_reg, var, xlabel):

        plt.figure(figsize=(8, 5))
        bins = np.arange(-0.5, 11.5, 1)  
        plt.hist(
            df_reg[var],
            weights=(df_reg["PESAIX"] / df_reg["PESAIX"].sum()) * 100,
            bins=bins,
            color='#1f77b4',
            alpha=0.85,
            edgecolor='black'  # clearer bar separation
        )

        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel('Respondents (%)', fontsize=14)

        plt.xticks(range(0, 11), fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.show()

    plot_hist_survey(df_reg.loc[~np.isnan(df_reg.acceptability) & (df_reg.acceptability < 97)], 'acceptability', 'Acceptability (0-10 scale)')
    plot_hist_survey(df_reg.loc[~np.isnan(df_reg.acceptable_price) & (df_reg.acceptable_price < 20)], 'acceptable_price', 'Acceptable price per trip')
    plot_hist_survey(df_reg.loc[~np.isnan(df_reg.climate_change) & (df_reg.climate_change < 80)], 'climate_change', 'Perceived benefits on GHG emissions')
    plot_hist_survey(df_reg.loc[~np.isnan(df_reg.quality_of_life) & (df_reg.quality_of_life < 80)], 'quality_of_life', 'Perceived benefits on air and noise pollution and safety')
    plot_hist_survey(df_reg.loc[~np.isnan(df_reg.P20_1) & (df_reg.P20_1 < 97)], 'P20_1', '')
    plot_hist_survey(df_reg.loc[~np.isnan(df_reg.P20_4) & (df_reg.P20_4 < 97)], 'P20_4', '')
    plot_hist_survey(df_reg.loc[~np.isnan(df_reg.P20_7) & (df_reg.P20_7 < 97)], 'P20_7', '')

    #initial values
    INITIAL_OPINION = weighted_median(df_reg["acceptability"].loc[~np.isnan(df_reg.acceptability) & (df_reg.acceptability < 97)].values, df_reg["PESAIX"].values)
    INITIAL_PRICE = weighted_median(df_reg["acceptable_price"].loc[~np.isnan(df_reg.acceptable_price) & (df_reg.acceptable_price < 20)].values, df_reg["PESAIX"].values)
    INITIAL_CC = weighted_median(df_reg["climate_change"].loc[~np.isnan(df_reg.climate_change) & (df_reg.climate_change < 80)].values, df_reg["PESAIX"].values)
    INITIAL_WELFARE = weighted_median(df_reg["score_welfare_vehicle_ownership_license"].loc[~np.isnan(df_reg.score_welfare)].values, df_reg["PESAIX"].values) #_vehicle_ownership_license
    INITIAL_QOL = weighted_median(df_reg["quality_of_life"].loc[~np.isnan(df_reg.quality_of_life) & (df_reg.quality_of_life < 80)].values, df_reg["PESAIX"].values)

    if scenario == "increasing_knowledge":
        INITIAL_KNOWLEDGE = weighted_median(df_reg["knowledge"].values, df_reg["PESAIX"].values) #np.nanmedian(y_scaled) #np.nanmedian(y_raw)
    else:
        INITIAL_KNOWLEDGE = 0

    #regression model
    df_reg_here = df_reg.loc[~np.isnan(df_reg.acceptability) & (df_reg.acceptability < 97) & ~np.isnan(df_reg.acceptable_price) & (df_reg.acceptable_price < 20) & ~np.isnan(df_reg.climate_change) & (df_reg.climate_change < 80) & ~np.isnan(df_reg.score_welfare) &~np.isnan(df_reg.quality_of_life) & (df_reg.quality_of_life < 80),:]
    df_reg_here[["climate_change", "quality_of_life", "acceptability"]] = df_reg_here[["climate_change", "quality_of_life", "acceptability"]] / 10
    
    X = df_reg_here[["climate_change", "quality_of_life", "score_welfare_vehicle_ownership_license", "LOW", "HIGH"]] #score_welfare_vehicle_ownership_license
    X = sm.add_constant(X)
    y_acceptability = df_reg_here["acceptability"].values.reshape(-1, 1).flatten()
    y_price = (df_reg_here["acceptable_price"].values.reshape(-1, 1)).flatten()

    model_acceptability = sm.WLS(y_acceptability, X, weights=df_reg_here['PESAIX']).fit()
    print(model_acceptability.summary())

    model_price = sm.WLS(y_price, X, weights=df_reg_here['PESAIX']).fit()
    print(model_price.summary())

    ## EXPLORATION
    #df_reg["score_welfare_vehicle_ownership_license"] = df_reg["score_welfare"]
    #df_reg.loc[df_reg["vehicle_ownership_license"] == 0, "score_welfare_vehicle_ownership_license"] = 0.5

    df_reg.rename(columns={'P02': 'age'}, inplace=True)
    df_reg["age"] = 2025 - df_reg["age"]

    df_reg.rename(columns={'P01': 'man'}, inplace=True)
    df_reg["man"] = (df_reg["man"] == 1) * 1

    df_reg.rename(columns={'P30': 'education'}, inplace=True)
    
    df_reg["children"] = 1 * ((df_reg.P03B_1 < 18) & (df_reg.P03C_1 == 2)) + 1 * ((df_reg.P03B_2 < 18) & (df_reg.P03C_2 == 2)) + 1 * ((df_reg.P03B_3 < 18) & (df_reg.P03C_3 == 2)) + 1 * ((df_reg.P03B_4 < 18) & (df_reg.P03C_4 == 2)) + 1 * ((df_reg.P03B_5 < 18) & (df_reg.P03C_5 == 2)) + 1 * ((df_reg.P03B_6 < 18) & (df_reg.P03C_6 == 2)) + 1 * ((df_reg.P03B_7 < 18) & (df_reg.P03C_7 == 2)) + 1 * ((df_reg.P03B_8 < 18) & (df_reg.P03C_8 == 2)) + 1 * ((df_reg.P03B_9 < 18) & (df_reg.P03C_9 == 2)) + 1 * ((df_reg.P03B_10 < 18) & (df_reg.P03C_10 == 2))
    df_reg["children2"] = 1 * ((df_reg.P03B_1 < 18)) + 1 * ((df_reg.P03B_2 < 18)) + 1 * ((df_reg.P03B_3 < 18)) + 1 * ((df_reg.P03B_4 < 18)) + 1 * ((df_reg.P03B_5 < 18)) + 1 * ((df_reg.P03B_6 < 18)) + 1 * ((df_reg.P03B_7 < 18)) + 1 * ((df_reg.P03B_8 < 18)) + 1 * ((df_reg.P03B_9 < 18)) + 1 * ((df_reg.P03B_10 < 18))
    df_reg["children3"] = (df_reg["children2"] > 0) * 1

    df_reg.rename(columns={'P35': 'political_ideology'}, inplace=True)
    
    df_reg.rename(columns={'P23': 'institutional_trust'}, inplace=True)
    
    df_reg.rename(columns={'P27_1': 'ecoanxiety'}, inplace=True)

    df_reg.rename(columns={'P26_5': 'ecological_paradigm'}, inplace=True)
    
    df_reg.rename(columns={'P24': 'knowledge'}, inplace=True)
    
    df_reg_here = df_reg.loc[~np.isnan(df_reg.acceptability) & (df_reg.acceptability < 97) & ~np.isnan(df_reg.acceptable_price) & (df_reg.acceptable_price < 20) & ~np.isnan(df_reg.climate_change) & (df_reg.climate_change < 80) & ~np.isnan(df_reg.score_welfare) &~np.isnan(df_reg.quality_of_life) & (df_reg.quality_of_life < 80),:]
    
    df_reg_here = df_reg_here.loc[~np.isnan(df_reg_here.knowledge) & (df_reg_here.knowledge < 80)]
    df_reg_here = df_reg_here.loc[~np.isnan(df_reg_here.ecological_paradigm) & (df_reg_here.ecological_paradigm < 80)]
    df_reg_here = df_reg_here.loc[~np.isnan(df_reg_here.ecoanxiety) & (df_reg_here.ecoanxiety < 80)]
    df_reg_here = df_reg_here.loc[~np.isnan(df_reg_here.institutional_trust) & (df_reg_here.institutional_trust < 80)]
    df_reg_here = df_reg_here.loc[~np.isnan(df_reg_here.political_ideology) & (df_reg_here.political_ideology < 80)]
    df_reg_here = df_reg_here.loc[~np.isnan(df_reg_here.education) & (df_reg_here.education < 6)]


    X = df_reg_here[["climate_change", "quality_of_life", "score_welfare_vehicle_ownership_license", "LOW", "HIGH", "age", "man", "education", "children2", 'political_ideology', 'ecoanxiety', "knowledge"]] #score_welfare_vehicle_ownership_license
    #X = df_reg[["climate_change", "quality_of_life", "score_welfare", "LOW", "HIGH"]]
    X = sm.add_constant(X)
    y_acceptability = df_reg_here["acceptability"].values.reshape(-1, 1).flatten()
    y_price = (df_reg_here["acceptable_price"].values.reshape(-1, 1)).flatten()

    model_acceptability_full = sm.WLS(y_acceptability, X, weights=df_reg_here['PESAIX']).fit()
    print(model_acceptability_full.summary())

    model_price_full = sm.WLS(y_price, X, weights=df_reg_here['PESAIX']).fit()
    print(model_price_full.summary())

    #Export results
    BETA_OPINION = np.array(model_acceptability.params)
    BETA_PRICE = np.array(model_price.params)
    

    return BETA_OPINION, BETA_PRICE, INITIAL_OPINION, INITIAL_PRICE, INITIAL_CC, INITIAL_WELFARE, INITIAL_QOL, INITIAL_KNOWLEDGE

def import_opinion_parameters(path_data, scenario):
    def weighted_mean(values, weights):
        return np.sum(values * weights) / np.sum(weights)
    
    def weighted_median(values, weights):
        # Sort values and weights by values
        sorted_idx = np.argsort(values)
        values, weights = values[sorted_idx], weights[sorted_idx]
    
        # Compute cumulative weights
        cum_weights = np.cumsum(weights) / np.sum(weights)
    
        # Find first value where cum_weight >= 0.5
        return values[cum_weights >= 0.5][0]

    df_reg, meta = pyreadstat.read_sav(path_data + '040023 En moviment pel clima 2025_V01 - còpia.sav')
    
    #add income variable (Joan)
    income_var = pd.read_excel(path_data + "MOBCLIMA_amb_renda_codipaisINE.xlsx")
    df_reg = df_reg.merge(income_var.loc[:,["NUME", "categoria_quintils", "ingressos_estimats"]], on = "NUME")

    df_reg = gpd.GeoDataFrame(df_reg, geometry = gpd.points_from_xy(df_reg.GEO_X, df_reg.GEO_Y), crs="EPSG:4326")
    gdf = gpd.read_file(path_data + "income_loss.geojson")
    df_reg = df_reg.to_crs(gdf.crs)
    df_reg = gpd.sjoin(df_reg, gdf, predicate="within")

    #if scenario == "discount_residents":
    #    df_reg.rename(columns={'P20_4': 'acceptability'}, inplace=True)
    #else:
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

    df_reg["LOW"] = (df_reg.ingressos_estimats  < 11550) * 1
    df_reg["HIGH"] = (df_reg.ingressos_estimats > 26950) * 1

    X_raw = df_reg[["climate_change", "log_absolute_loss", "quality_of_life", "LOW", "HIGH"]]
        
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

def import_price_parameters(path_data, scenario):

    def weighted_mean(values, weights):
        return np.sum(values * weights) / np.sum(weights)
    

    def weighted_median(values, weights):
        # Sort values and weights by values
        sorted_idx = np.argsort(values)
        values, weights = values[sorted_idx], weights[sorted_idx]
    
        # Compute cumulative weights
        cum_weights = np.cumsum(weights) / np.sum(weights)
    
        # Find first value where cum_weight >= 0.5
        return values[cum_weights >= 0.5][0]

    df_reg, meta = pyreadstat.read_sav(path_data + '040023 En moviment pel clima 2025_V01 - còpia.sav')
    
    #add income variable (Joan)
    income_var = pd.read_excel(path_data + "MOBCLIMA_amb_renda_codipaisINE.xlsx")
    df_reg = df_reg.merge(income_var.loc[:,["NUME", "categoria_quintils", "ingressos_estimats"]], on = "NUME")

    df_reg = gpd.GeoDataFrame(df_reg, geometry = gpd.points_from_xy(df_reg.GEO_X, df_reg.GEO_Y), crs="EPSG:4326")
    gdf = gpd.read_file(path_data + "income_loss.geojson")
    df_reg = df_reg.to_crs(gdf.crs)
    df_reg = gpd.sjoin(df_reg, gdf, predicate="within")

    if scenario == "tax_question_P17":
        df_reg.rename(columns={'P17': 'acceptable_price'}, inplace=True)
    else:
        df_reg.rename(columns={'P18': 'acceptable_price'}, inplace=True)
    
    df_reg = df_reg.loc[~np.isnan(df_reg.acceptable_price) & (df_reg.acceptable_price < 15)]

    df_reg.acceptable_price = df_reg.acceptable_price / 2

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

    df_reg.rename(columns={'P24': 'knowledge'}, inplace=True)
    df_reg = df_reg.loc[~np.isnan(df_reg.knowledge) & (df_reg.knowledge < 80)]

    df_reg["LOW"] = (df_reg.ingressos_estimats  < 11550) * 1
    df_reg["HIGH"] = (df_reg.ingressos_estimats > 26950) * 1
    
    if scenario == "increasing_knowledge":
        X_raw = df_reg[["climate_change", "log_absolute_loss", "quality_of_life", "LOW", "HIGH", "knowledge"]]
    else:
        X_raw = df_reg[["climate_change", "log_absolute_loss", "quality_of_life", "LOW", "HIGH"]]
    
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

    if scenario == "increasing_knowledge":
        median_knowledge = weighted_median(X_scaled["knowledge"].values, df_reg["PESAIX"].values) #np.nanmedian(y_scaled) #np.nanmedian(y_raw)
    else:
        median_knowledge = 0

    return np.array(model_statsmodel.params), weighted_median(df_reg["acceptable_price"].values, df_reg["PESAIX"].values), weighted_median(X_scaled["climate_change"].values, df_reg["PESAIX"].values), weighted_median(X_scaled["log_absolute_loss"].values, df_reg["PESAIX"].values), weighted_median(X_scaled["quality_of_life"].values, df_reg["PESAIX"].values), median_knowledge

def compute_political_opinion(score_welfare, score_quality_of_life, score_emissions, score_congestion, BETA_OPINION, option_congestion_in_support):
    """ Compute public support based on the regression on the survey data """
    
    if option_congestion_in_support == True:
        outcome = BETA_OPINION[0] + BETA_OPINION[1] * score_emissions + BETA_OPINION[2] * score_welfare + BETA_OPINION[3] * score_quality_of_life + BETA_OPINION[4] * score_congestion
    elif option_congestion_in_support == False:
        outcome = BETA_OPINION[0] + BETA_OPINION[1] * score_emissions + BETA_OPINION[2] * score_welfare + BETA_OPINION[3] * score_quality_of_life

    return outcome
    

def compute_price(score_welfare, score_quality_of_life, score_emissions, score_congestion, BETA_OPINION, option_congestion_in_support, scenario, score_knowledge, lvl):
    """ Compute public support based on the regression on the survey data """
    
    dummy_low = (lvl == "LOW")
    dummy_high = (lvl == "HIGH")

    if scenario == "increasing_knowledge":
        outcome = np.nan #BETA_OPINION[0] + BETA_OPINION[1] * score_emissions + BETA_OPINION[2] * score_welfare + BETA_OPINION[3] * score_quality_of_life + BETA_OPINION[4] * score_knowledge
    elif option_congestion_in_support == False:
        outcome = BETA_OPINION[0] + BETA_OPINION[1] * score_emissions + BETA_OPINION[2] * score_welfare + BETA_OPINION[3] * score_quality_of_life + BETA_OPINION[4] * dummy_low + BETA_OPINION[5] * dummy_high

    return outcome