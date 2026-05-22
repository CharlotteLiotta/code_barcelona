import pyreadstat # type: ignore
import statsmodels.api as sm # type: ignore
import pandas as pd
from scipy.stats import spearmanr
from stargazer.stargazer import Stargazer
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from scipy.stats import norm
from statsmodels.base.model import GenericLikelihoodModel

from import_data import *
from plotting_tools import *
from policy_support import *


def print_me_tobit(results, X_tobit):

    params = results.params
    se = results.bse
    pvalues = results.pvalues

    beta = params[:-1]
    beta_se = se[:-1]
    beta_p = pvalues[:-1]

    sigma = np.exp(params[-1])

    # Mean marginal effects
    x_mean = np.mean(X_tobit, axis=0)
    xb = np.dot(x_mean, beta)

    phi_term = norm.cdf(xb / sigma)

    marginal_effects = phi_term * beta

    me_table = pd.DataFrame({
        'Variable': X_tobit.columns,
        'Coef': beta,
        'StdErr': beta_se,
        'p_value': beta_p,
        'MarginalEffect': marginal_effects
    })

    print(me_table)


def fit_weighted_tobit(y, X, start_params, weights=None, left=0):

    model = GenericLikelihoodModel(
        endog=y,
        exog=X,
        loglikeobs=lambda params: weighted_tobit_loglike(
            params,
            y=y,
            X=X,
            weights=weights,
            left=left
        )
    )

    results = model.fit(
        start_params=start_params,
        method='bfgs',
        maxiter=5000,
        disp=True
    )

    return results

def weighted_tobit_loglike(params, y, X, weights=None, left=0):
    beta = params[:-1]
    sigma = np.exp(params[-1])

    xb = X @ beta

    cens = (y <= left)
    uncens = (y > left)

    ll = np.zeros(len(y))

    # Censored observations
    ll[cens]   = norm.logcdf((left - xb[cens]) / sigma)
    ll[uncens] = -np.log(sigma) + norm.logpdf((y[uncens] - xb[uncens]) / sigma)

    if weights is not None:
        ll = ll * weights

    return ll  # array, not sum

def compute_opinion(score_welfare, score_quality_of_life, score_emissions, BETA_OPINION, scenario, lvl, score_knowledge, tobit = False):
    """ Compute public support based on the regression on the survey data """
    
    dummy_low = (lvl == "LOW")
    dummy_high = (lvl == "HIGH")

    if tobit == True:
        sigma = np.exp(BETA_OPINION[-1])
        BETA_OPINION = BETA_OPINION[:-1]
        

    if scenario == "increasing_knowledge":
        opinion = BETA_OPINION[0] + BETA_OPINION[1] * score_emissions + BETA_OPINION[2] * score_quality_of_life + BETA_OPINION[3] * score_welfare + BETA_OPINION[4] * dummy_low + BETA_OPINION[5] * dummy_high + BETA_OPINION[6] * score_knowledge
    else:
        opinion = BETA_OPINION[0] + BETA_OPINION[1] * score_emissions + BETA_OPINION[2] * score_quality_of_life + BETA_OPINION[3] * score_welfare + BETA_OPINION[4] * dummy_low + BETA_OPINION[5] * dummy_high
        
    if tobit == True:
        z = opinion / sigma
        opinion = norm.cdf(z) * opinion + sigma * norm.pdf(z)
    else:
        opinion = 1 / (1 + np.exp(-opinion))

    return opinion

def weighted_median(values, weights):
    # Sort values and weights by values
    sorted_idx = np.argsort(values)
    values, weights = values[sorted_idx], weights[sorted_idx]
    
    # Compute cumulative weights
    cum_weights = np.cumsum(weights) / np.sum(weights)
    
    # Find first value where cum_weight >= 0.5
    return values[cum_weights >= 0.5][0]

def import_opinion_parameters(path_data, scenario, expected_welfare_loss, weighted_median, Y_median, wage_factors):
    
    #import survey
    df_reg, meta = pyreadstat.read_sav(path_data + '040023 En moviment pel clima 2025_V01 - còpia.sav')

    #add income variable (Joan)
    income_var = pd.read_excel(path_data + "MOBCLIMA_amb_renda_codipaisINE.xlsx")
    df_reg = df_reg.merge(income_var.loc[:,["NUME", "categoria_quintils", "ingressos_estimats"]], on = "NUME")

    #add welfare losses
    df_reg = gpd.GeoDataFrame(df_reg, geometry = gpd.points_from_xy(df_reg.GEO_X, df_reg.GEO_Y), crs="EPSG:4326")
    df_reg = df_reg.to_crs(expected_welfare_loss.crs)
    df_reg = gpd.sjoin(df_reg, expected_welfare_loss, how = "left", predicate="within")

    #prepare variables: dependent var

    df_reg.rename(columns={'P22_4': 'acceptability'}, inplace=True)
    df_reg.rename(columns={'P18': 'acceptable_price'}, inplace=True)
    df_reg.acceptable_price = df_reg.acceptable_price / 2
    print(spearmanr(df_reg["acceptability"], df_reg["acceptable_price"], nan_policy="omit"))
    plot_acceptability_price(df_reg)

    #prepare variables:explanatory vars

    df_reg.rename(columns={'P21_3': 'climate_change'}, inplace=True)
    df_reg.rename(columns={'P21_2': 'quality_of_life'}, inplace=True)
    df_reg["LOW"] = (df_reg.ingressos_estimats  < (12 * Y_median * wage_factors["LOW"])) * 1
    df_reg["HIGH"] = (df_reg.ingressos_estimats > (12 * Y_median * wage_factors["HIGH"])) * 1
    
    df_reg["score_welfare"] = df_reg["score_welfare_med"]
    df_reg["score_welfare"][np.isnan(df_reg["score_welfare"])] = df_reg["score_welfare_low"]
    df_reg["score_welfare"][np.isnan(df_reg["score_welfare"])] = df_reg["score_welfare_high"]

    for i in range(len(df_reg["score_welfare"])):
        if df_reg["LOW"].iloc[i] == 1:
            df_reg.loc[i, "score_welfare"] = df_reg.loc[i, "score_welfare_low"]
        elif df_reg["HIGH"].iloc[i] == 1:
            df_reg.loc[i, "score_welfare"] = df_reg.loc[i, "score_welfare_high"]

    df_reg["vehicle_ownership_license"] = 1 * (((df_reg.P34A > 0) &(df_reg.P33A  == 1))| ((df_reg.P34B > 0) &(df_reg.P33B  == 1)) |((df_reg.P34C > 0) &(df_reg.P33A  == 1))) 
    #df_reg["score_welfare"] = df_reg["score_welfare"] - 0.5
    #df_reg["score_welfare_vehicle_ownership_license"] = df_reg["score_welfare"] * df_reg["vehicle_ownership_license"]
    df_reg["score_welfare_vehicle_ownership_license"] = df_reg["score_welfare"]
    df_reg.loc[df_reg["vehicle_ownership_license"] == 0, "score_welfare_vehicle_ownership_license"] = 0.5

    #Export plots
    plot_mobility_loss("score_welfare_vehicle_ownership_license","acceptable_price", df_reg.loc[~np.isnan(df_reg.acceptable_price) & (df_reg.acceptable_price < 20) & ~np.isnan(df_reg.score_welfare)])
    plot_hist_survey(df_reg.loc[~np.isnan(df_reg.acceptability) & (df_reg.acceptability < 97)], 'acceptability', 'Acceptability (0-10 scale)')
    plot_hist_survey(df_reg.loc[~np.isnan(df_reg.acceptable_price) & (df_reg.acceptable_price < 20)], 'acceptable_price', 'Acceptable price per day')
    plot_hist_survey(df_reg.loc[~np.isnan(df_reg.climate_change) & (df_reg.climate_change < 80)], 'climate_change', 'Perceived benefits on GHG emissions')
    plot_hist_survey(df_reg.loc[~np.isnan(df_reg.quality_of_life) & (df_reg.quality_of_life < 80)], 'quality_of_life', 'Perceived benefits on air and noise pollution and safety')
    plot_hist_survey(df_reg.loc[~np.isnan(df_reg.P20_1) & (df_reg.P20_1 < 97)], 'P20_1', '')
    plot_hist_survey(df_reg.loc[~np.isnan(df_reg.P20_4) & (df_reg.P20_4 < 97)], 'P20_4', '')
    plot_hist_survey(df_reg.loc[~np.isnan(df_reg.P20_7) & (df_reg.P20_7 < 97)], 'P20_7', '')

    #initial values
    INITIAL_OPINION = weighted_median(df_reg["acceptability"].loc[~np.isnan(df_reg.acceptability) & (df_reg.acceptability < 97)].values, df_reg["PESAIX"].loc[~np.isnan(df_reg.acceptability) & (df_reg.acceptability < 97)].values)
    INITIAL_PRICE = weighted_median(df_reg["acceptable_price"].loc[~np.isnan(df_reg.acceptable_price) & (df_reg.acceptable_price < 20)].values, df_reg["PESAIX"].loc[~np.isnan(df_reg.acceptable_price) & (df_reg.acceptable_price < 20)].values)
    INITIAL_CC = weighted_median(df_reg["climate_change"].loc[~np.isnan(df_reg.climate_change) & (df_reg.climate_change < 80)].values, df_reg["PESAIX"].loc[~np.isnan(df_reg.climate_change) & (df_reg.climate_change < 80)].values)
    INITIAL_WELFARE = weighted_median(df_reg["score_welfare_vehicle_ownership_license"].loc[~np.isnan(df_reg.score_welfare)].values, df_reg["PESAIX"].loc[~np.isnan(df_reg.score_welfare)].values) #_vehicle_ownership_license
    INITIAL_QOL = weighted_median(df_reg["quality_of_life"].loc[~np.isnan(df_reg.quality_of_life) & (df_reg.quality_of_life < 80)].values, df_reg["PESAIX"].loc[~np.isnan(df_reg.quality_of_life) & (df_reg.quality_of_life < 80)].values)

    df_reg.rename(columns={'P24': 'knowledge'}, inplace=True)
    df_reg.knowledge = df_reg.knowledge/10

    if scenario == "increasing_knowledge":
        INITIAL_KNOWLEDGE = weighted_median(df_reg["knowledge"].loc[~np.isnan(df_reg.knowledge) & (df_reg.knowledge < 8)].values, df_reg["PESAIX"].loc[~np.isnan(df_reg.knowledge) & (df_reg.knowledge < 80)].values) #np.nanmedian(y_scaled) #np.nanmedian(y_raw)
    else:
        INITIAL_KNOWLEDGE = 0


    #regression model 1
    df_reg_here = df_reg.loc[~np.isnan(df_reg.acceptability) & (df_reg.acceptability < 97) & ~np.isnan(df_reg.acceptable_price) & (df_reg.acceptable_price < 20) & ~np.isnan(df_reg.climate_change) & (df_reg.climate_change < 80) & ~np.isnan(df_reg.score_welfare) &~np.isnan(df_reg.quality_of_life) & (df_reg.quality_of_life < 80),:]
    df_reg_here.loc[:,["climate_change", "quality_of_life", "acceptability"]] = df_reg_here.loc[:,["climate_change", "quality_of_life", "acceptability"]] / 10
    
    

    if scenario == "increasing_knowledge":
        df_reg_here = df_reg_here.loc[~np.isnan(df_reg_here.knowledge) & (df_reg_here.knowledge < 8)]
        X = df_reg_here[["climate_change", "quality_of_life", "score_welfare_vehicle_ownership_license", "LOW", "HIGH", "knowledge"]] #score_welfare_vehicle_ownership_license
    else:
        X = df_reg_here[["climate_change", "quality_of_life", "score_welfare_vehicle_ownership_license", "LOW", "HIGH"]] #score_welfare_vehicle_ownership_license
    X = sm.add_constant(X)
    y_acceptability = df_reg_here["acceptability"].values.reshape(-1, 1).flatten()
    y_price = (df_reg_here["acceptable_price"].values.reshape(-1, 1)).flatten()

    #export_df = X.copy()  # includes const + explanatory variables
    #export_df["y_price"] = y_price
    #export_df["y_acceptability"] = y_acceptability
    #export_df["weights"] = df_reg_here["PESAIX"].values

    # Export to CSV
    #export_df.to_csv("regression_data.csv", index=False) 

    #OLS
    model_acceptability = sm.WLS(y_acceptability, X, weights=df_reg_here['PESAIX']).fit() #cov_type='HC3'
    print(model_acceptability.summary())

    #PROBIT
    model_acceptability = GLM(
        y_acceptability,
        X,
        family=families.Binomial(link=families.links.Probit()), #families.Binomial(link=families.links.Logit()),
        var_weights=df_reg_here['PESAIX']   # swap freq_weights for var_weights if your weights are sampling/probability weights
        ).fit() #cov_type='HC3'
    
    print(model_acceptability.summary())

    #PROBIT - MARGINAL EFFECTS
    marginal_effects = model_acceptability.get_margeff()
    print(marginal_effects.summary())

    #LOGIT
    model_acceptability = GLM(
        y_acceptability,
        X,
        family=families.Binomial(link=families.links.Logit()), #families.Binomial(link=families.links.Logit()),
        var_weights=df_reg_here['PESAIX'] * (len(df_reg_here['PESAIX'] ) / sum(df_reg_here['PESAIX'] ))   # swap freq_weights for var_weights if your weights are sampling/probability weights
        #var_weights = df_reg_here['PESAIX'],
        ).fit() #cov_type='HC3'
    
    print(model_acceptability.summary())

    #LOGIT - MARGINAL EFFECTS
    marginal_effects = model_acceptability.get_margeff()
    print(marginal_effects.summary())

    #MULTINOMIAL LOGIT
    from statsmodels.discrete.discrete_model import MNLogit

    # MNLogit expects integer-encoded categories (0, 1, 2, ...)
    #Make sure y_acceptability contains integer category codes
    model_acceptability = MNLogit(
        (10 * y_acceptability).astype(int),
        X
        ).fit(
                method='lbfgs',  # or 'bfgs', 'lbfgs'
                maxiter=100,
                disp=True
            )

    print(model_acceptability.summary())

    # MULTINOMIAL LOGIT - MARGINAL EFFECTS
    marginal_effects = model_acceptability.get_margeff()
    print(marginal_effects.summary())

    #OLS
    model_price = sm.WLS(y_price, X, weights=df_reg_here['PESAIX']).fit() #
    print(model_price.summary())

    #TOBIT
    start_beta = model_price.params
    start_sigma = np.log(np.std(model_price.resid))
    start_params = np.append(start_beta, start_sigma)

    results_tobit = fit_weighted_tobit(
        y=y_price,
        X=X,
        start_params=start_params,
        weights=df_reg_here['PESAIX'] * (len(df_reg_here['PESAIX'] ) / sum(df_reg_here['PESAIX'] )),
        left=0
    )

    print(results_tobit.summary())

    # Extract parameters
    print_me_tobit(results_tobit, X)

    #regression model 2
    df_reg.rename(columns={'P02': 'age'}, inplace=True)
    df_reg["age"] = 2025 - df_reg["age"]

    df_reg.rename(columns={'P01': 'man'}, inplace=True)
    df_reg["man"] = (df_reg["man"] == 1) * 1

    df_reg.rename(columns={'P30': 'education'}, inplace=True)
    
    df_reg["children"] = 1 * ((df_reg.P03B_1 < 18) & (df_reg.P03C_1 == 2)) + 1 * ((df_reg.P03B_2 < 18) & (df_reg.P03C_2 == 2)) + 1 * ((df_reg.P03B_3 < 18) & (df_reg.P03C_3 == 2)) + 1 * ((df_reg.P03B_4 < 18) & (df_reg.P03C_4 == 2)) + 1 * ((df_reg.P03B_5 < 18) & (df_reg.P03C_5 == 2)) + 1 * ((df_reg.P03B_6 < 18) & (df_reg.P03C_6 == 2)) + 1 * ((df_reg.P03B_7 < 18) & (df_reg.P03C_7 == 2)) + 1 * ((df_reg.P03B_8 < 18) & (df_reg.P03C_8 == 2)) + 1 * ((df_reg.P03B_9 < 18) & (df_reg.P03C_9 == 2)) + 1 * ((df_reg.P03B_10 < 18) & (df_reg.P03C_10 == 2))
    df_reg["children2"] = 1 * ((df_reg.P03B_1 < 18)) + 1 * ((df_reg.P03B_2 < 18)) + 1 * ((df_reg.P03B_3 < 18)) + 1 * ((df_reg.P03B_4 < 18)) + 1 * ((df_reg.P03B_5 < 18)) + 1 * ((df_reg.P03B_6 < 18)) + 1 * ((df_reg.P03B_7 < 18)) + 1 * ((df_reg.P03B_8 < 18)) + 1 * ((df_reg.P03B_9 < 18)) + 1 * ((df_reg.P03B_10 < 18))
    df_reg["children3"] = (df_reg["children"] > 0) * 1

    df_reg.rename(columns={'P35': 'political_ideology'}, inplace=True)
    
    df_reg.rename(columns={'P23': 'institutional_trust'}, inplace=True)
    
    df_reg.rename(columns={'P27_1': 'ecoanxiety'}, inplace=True)

    df_reg.rename(columns={'P26_5': 'ecological_paradigm'}, inplace=True)
    
    
    df_reg_here = df_reg.loc[~np.isnan(df_reg.acceptability) & (df_reg.acceptability < 97) & ~np.isnan(df_reg.acceptable_price) & (df_reg.acceptable_price < 20) & ~np.isnan(df_reg.climate_change) & (df_reg.climate_change < 80) & ~np.isnan(df_reg.score_welfare) &~np.isnan(df_reg.quality_of_life) & (df_reg.quality_of_life < 80),:]
    
    df_reg_here = df_reg_here.loc[~np.isnan(df_reg_here.knowledge) & (df_reg_here.knowledge < 8)]
    df_reg_here = df_reg_here.loc[~np.isnan(df_reg_here.ecological_paradigm) & (df_reg_here.ecological_paradigm < 80)]
    df_reg_here = df_reg_here.loc[~np.isnan(df_reg_here.ecoanxiety) & (df_reg_here.ecoanxiety < 80)]
    df_reg_here = df_reg_here.loc[~np.isnan(df_reg_here.institutional_trust) & (df_reg_here.institutional_trust < 80)]
    df_reg_here = df_reg_here.loc[~np.isnan(df_reg_here.political_ideology) & (df_reg_here.political_ideology < 80)]
    df_reg_here = df_reg_here.loc[~np.isnan(df_reg_here.education) & (df_reg_here.education < 5)]


    X = df_reg_here[["climate_change", "quality_of_life", "score_welfare_vehicle_ownership_license", "LOW", "HIGH", "age", "man", "education", "political_ideology", 'ecoanxiety', 'ecological_paradigm', "knowledge", "children2"]]   #, "age"           #score_welfare_vehicle_ownership_license
    
    X = sm.add_constant(X)
    y_acceptability = df_reg_here["acceptability"].values.reshape(-1, 1).flatten() / 10
    y_price = (df_reg_here["acceptable_price"].values.reshape(-1, 1)).flatten()

    model_acceptability_full = sm.WLS(y_acceptability, X, weights=df_reg_here['PESAIX']).fit() #, weights=df_reg_here['PESAIX']
    print(model_acceptability_full.summary())

    model_acceptability_full = GLM(
        y_acceptability,
        X,
        family=families.Binomial(link=families.links.Logit()), #families.Binomial(link=families.links.Probit()),
        var_weights=df_reg_here['PESAIX']   # swap freq_weights for var_weights if your weights are sampling/probability weights
        ).fit()
    print(model_acceptability_full.summary())
    marginal_effects = model_acceptability_full.get_margeff()
    print(marginal_effects.summary())

    model_price_full = sm.WLS(y_price, X, weights=df_reg_here['PESAIX']).fit()
    print(model_price_full.summary())

    #TOBIT
    start_beta = model_price_full.params
    start_sigma = np.log(np.std(model_price_full.resid))
    start_params = np.append(start_beta, start_sigma)

    results_tobit_full = fit_weighted_tobit(
        y=y_price,
        X=X,
        start_params=start_params,
        weights=df_reg_here['PESAIX'].values,
        left=0
    )

    print(results_tobit_full.summary())

    # Extract parameters
    print_me_tobit(results_tobit_full, X)

    BETA_OPINION = np.array(model_acceptability.params)
    BETA_PRICE = np.array(results_tobit.params)
    
    #stargazer = Stargazer([model_price, model_acceptability, model_price_full, model_acceptability_full])
    #print(stargazer.render_latex())
    
    return BETA_OPINION, BETA_PRICE, INITIAL_OPINION, INITIAL_PRICE, INITIAL_CC, INITIAL_WELFARE, INITIAL_QOL, INITIAL_KNOWLEDGE
