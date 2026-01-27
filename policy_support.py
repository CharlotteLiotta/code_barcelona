import pyreadstat # type: ignore
import statsmodels.api as sm # type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from import_data import *

def import_opinion_parameters(path_data, scenario, expected_welfare_loss):
    
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
    df_reg = gpd.sjoin(df_reg, expected_welfare_loss, predicate="within")

    #prepare variables

    #dependent var 1: acceptability
    df_reg.rename(columns={'P22_4': 'acceptability'}, inplace=True)
    df_reg = df_reg.loc[~np.isnan(df_reg.acceptability) & (df_reg.acceptability < 97)]

    #dependent var 2: acceptable price
    df_reg.rename(columns={'P18': 'acceptable_price'}, inplace=True)
    
    df_reg = df_reg.loc[~np.isnan(df_reg.acceptable_price) & (df_reg.acceptable_price < 20)]
    df_reg.acceptable_price = df_reg.acceptable_price / 2

    print(spearmanr(df_reg["acceptability"], df_reg["acceptable_price"], nan_policy="omit"))

    # 4. Conditional means with 95% CI
    summary = (
        df_reg.groupby("acceptability")["acceptable_price"]
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
    df_reg = df_reg.loc[~np.isnan(df_reg.climate_change) & (df_reg.climate_change < 80)]

    df_reg.rename(columns={'P21_2': 'quality_of_life'}, inplace=True)
    df_reg = df_reg.loc[~np.isnan(df_reg.quality_of_life) & (df_reg.quality_of_life < 80)]
    
    df_reg["LOW"] = (df_reg.ingressos_estimats  < 11550) * 1
    df_reg["HIGH"] = (df_reg.ingressos_estimats > 26950) * 1

    df_reg["score_welfare"] = df_reg["score_welfare_low"] * df_reg["LOW"] + df_reg["score_welfare_high"] * df_reg["HIGH"]+ df_reg["score_welfare_med"] * (1 - df_reg["HIGH"] - df_reg["LOW"])
    df_reg["vehicle_ownership_license"] = 1 * (((df_reg.P34A > 0) &(df_reg.P33A  == 1))| ((df_reg.P34B > 0) &(df_reg.P33B  == 1)) |((df_reg.P34C > 0) &(df_reg.P33A  == 1))) 
    df_reg["score_welfare_vehicle_ownership_license"] = df_reg["score_welfare"]
    df_reg.loc[df_reg["vehicle_ownership_license"] == 0, "score_welfare_vehicle_ownership_license"] = 0.5

    #Export plots
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

    plot_hist_survey(df_reg, 'acceptability', 'Acceptability (0-10 scale)')
    plot_hist_survey(df_reg, 'acceptable_price', 'Acceptable price per trip')
    plot_hist_survey(df_reg, 'climate_change', 'Perceived benefits on GHG emissions')
    plot_hist_survey(df_reg, 'quality_of_life', 'Perceived benefits on air and noise pollution and safety')
    plot_hist_survey(df_reg.loc[~np.isnan(df_reg.P20_1) & (df_reg.P20_1 < 97)], 'P20_1', '')
    plot_hist_survey(df_reg.loc[~np.isnan(df_reg.P20_4) & (df_reg.P20_4 < 97)], 'P20_4', '')
    plot_hist_survey(df_reg.loc[~np.isnan(df_reg.P20_7) & (df_reg.P20_7 < 97)], 'P20_7', '')

    #regression model

    df_reg[["climate_change", "quality_of_life", "acceptability"]] = df_reg[["climate_change", "quality_of_life", "acceptability"]] / 10

    X = df_reg[["climate_change", "quality_of_life", "score_welfare_vehicle_ownership_license", "LOW", "HIGH"]]
    X = sm.add_constant(X)
    y_acceptability = df_reg["acceptability"].values.reshape(-1, 1).flatten()
    y_price = (df_reg["acceptable_price"].values.reshape(-1, 1)).flatten()

    model_acceptability = sm.WLS(y_acceptability, X, weights=df_reg['PESAIX']).fit()
    print(model_acceptability.summary())

    model_price = sm.WLS(y_price, X, weights=df_reg['PESAIX']).fit()
    print(model_price.summary())

    #Export results
    BETA_OPINION = np.array(model_acceptability.params)
    BETA_PRICE = np.array(model_price.params)
    INITIAL_OPINION = weighted_median(y_acceptability, df_reg["PESAIX"].values)
    INITIAL_PRICE = weighted_median(y_price, df_reg["PESAIX"].values)
    INITIAL_CC = weighted_median(X["climate_change"].values, df_reg["PESAIX"].values)
    INITIAL_WELFARE = weighted_median(X["score_welfare_vehicle_ownership_license"].values, df_reg["PESAIX"].values)
    INITIAL_QOL = weighted_median(X["quality_of_life"].values, df_reg["PESAIX"].values)

    if scenario == "increasing_knowledge":
        INITIAL_KNOWLEDGE = weighted_median(X["knowledge"].values, df_reg["PESAIX"].values) #np.nanmedian(y_scaled) #np.nanmedian(y_raw)
    else:
        INITIAL_KNOWLEDGE = 0

    return BETA_OPINION, BETA_PRICE, INITIAL_OPINION, INITIAL_PRICE, INITIAL_CC, INITIAL_WELFARE, INITIAL_QOL, INITIAL_KNOWLEDGE


def compute_opinion(score_welfare, score_quality_of_life, score_emissions, BETA_OPINION, scenario, lvl, score_knowledge):
    """ Compute public support based on the regression on the survey data """
    
    dummy_low = (lvl == "LOW")
    dummy_high = (lvl == "HIGH")

    if scenario == "increasing_knowledge":
        opinion = np.nan #BETA_OPINION[0] + BETA_OPINION[1] * score_emissions + BETA_OPINION[2] * score_quality_of_life + BETA_OPINION[3] * score_welfare + BETA_OPINION[4] * score_knowledge
    else:
        opinion = BETA_OPINION[0] + BETA_OPINION[1] * score_emissions + BETA_OPINION[2] * score_quality_of_life + BETA_OPINION[3] * score_welfare + BETA_OPINION[4] * dummy_low + BETA_OPINION[5] * dummy_high
        
    return opinion