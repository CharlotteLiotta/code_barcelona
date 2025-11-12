import pyreadstat
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols

from import_data import *

# Import the data
path_data = "../data_barcelona/"

df, meta = pyreadstat.read_sav(path_data + 'oltra_barcelona/data.sav')

label_table = pd.DataFrame({
    "column_name": meta.column_names,
    "column_label": meta.column_labels
})

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


df_reg["Effectiveness_I"] = df_reg["Effectiveness"] * df_reg["Ideology"] 
df_reg["Quality of life_I"] = df_reg["Quality of life"] * df_reg["Ideology"] 
df_reg["Justice_I"] = df_reg["Justice"] * df_reg["Ideology"] 
df_reg["Man"] = (df_reg["Gender"] == 1) * 1
df_reg["age_l40"] = (df_reg["Age"] == 1) * 1
df_reg["age_h65"] = (df_reg["Age"] == 3) * 1
df_reg["University"] = (df_reg["Education"] == 2) * 1
df_reg["Children"] = (df_reg["Children"] == 1) * 1

y = df_reg["Acceptability"] #Acceptability
X = df_reg.loc[:,["Effectiveness", "Quality of life", "Justice", "age_l40", "Democratic", "Pro_envt"]] 
X = sm.add_constant(X)  # Adds intercept
model_statsmodel = sm.OLS(y, X).fit()
print(model_statsmodel.summary())

# Spatial in acceptability

path_data = "../data_barcelona/"
center = "0801901025"
gdf = import_data(path_data, center, option = "DISTRICT")
df["P4"] = df["P4"]-1

acceptability_district = df.loc[:,["P4", "Q17"]].groupby("P4").mean()
acceptability_district = gdf.merge(acceptability_district, left_index=True, right_index=True)
acceptability_district.plot("Q17", legend = True)

model = ols('Q17 ~ C(P4)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

model = ols('Q17 ~ distance_center', data=acceptability_district).fit()
print(model.summary())

# Spatial variations in well being, inequality, effectiveness)

#justice
acceptability_district = df.loc[:,["P4", "Q13"]].groupby("P4").mean()
acceptability_district = gdf.merge(acceptability_district, left_index=True, right_index=True)
acceptability_district.plot("Q13", legend = True)

model = ols('Q13 ~ C(P4)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

model = ols('Q13 ~ distance_center', data=acceptability_district).fit()
print(model.summary())

#effectiveness
acceptability_district = df.loc[:,["P4", "Q10_1"]].groupby("P4").mean()
acceptability_district = gdf.merge(acceptability_district, left_index=True, right_index=True)
acceptability_district.plot("Q10_1", legend = True)

model = ols('Q10_1 ~ C(P4)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

model = ols('Q10_1 ~ distance_center', data=acceptability_district).fit()
print(model.summary())

# And compare experienced and perceived well-being

#MORE "PERCEIVED": Q11_1 and Q11_2(mejora mi calidad de vida, reduce mi libertad)

acceptability_district = df.loc[:,["P4", "Q11_2"]].groupby("P4").mean()
acceptability_district = gdf.merge(acceptability_district, left_index=True, right_index=True)
acceptability_district.plot("Q11_2", legend = True)

model = ols('Q11_2 ~ C(P4)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

model = ols('Q11_2 ~ distance_center', data=acceptability_district).fit()
print(model.summary())

#IN BETWEEN: Q5 (te has visto direcatmente afectado)

acceptability_district = df.loc[df.Q5 != 3,["P4", "Q5"]].groupby("P4").mean()
acceptability_district = gdf.merge(acceptability_district, left_index=True, right_index=True)
acceptability_district.plot("Q5", legend = True)

model = ols('Q5 ~ C(P4)', data=df.loc[df.Q5 != 3,:]).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

model = ols('Q5 ~ distance_center', data=acceptability_district).fit()
print(model.summary())

#MORE "EXPERIENCED": full model

#Correlations
df_here = df.loc[df.Q5 != 3,:]
df_here.Q5 = (df_here.Q5 == 1) * 1
df_here.Q11_1 = 6 - df_here.Q11_1

from scipy.stats import pearsonr

vars = ['Q11_1', 'Q11_2', 'Q5']
corr_matrix = df_here[vars].corr()

pval_matrix = pd.DataFrame(index=vars, columns=vars)
for i in vars:
    for j in vars:
        _, pval = pearsonr(df_here[i], df_here[j])
        pval_matrix.loc[i, j] = pval

print("Correlation matrix:\n", corr_matrix)
print("P-value matrix:\n", pval_matrix)

#Comparaison

df_reg = df.loc[:,["Q17", "Q10_1", "Q11_1", "Q13", "P8", "Q3_1", "P1", "qtAge", 'qtEducation', "P9", "Q14"]]
df_reg.columns = ["Acceptability", "Effectiveness", "Quality of life", "Justice", "Ideology", "Pro_envt", "Gender", "Age", "Education", "Children", "Democratic"]
df_reg = df_reg.loc[df_reg.Ideology != 6,:]

df_reg["Effectiveness_I"] = df_reg["Effectiveness"] * df_reg["Ideology"] 
df_reg["Quality of life_I"] = df_reg["Quality of life"] * df_reg["Ideology"] 
df_reg["Justice_I"] = df_reg["Justice"] * df_reg["Ideology"] 
df_reg["Man"] = (df_reg["Gender"] == 1) * 1
df_reg["age_l40"] = (df_reg["Age"] == 1) * 1
df_reg["age_h65"] = (df_reg["Age"] == 3) * 1
df_reg["University"] = (df_reg["Education"] == 2) * 1
df_reg["Children"] = (df_reg["Children"] == 1) * 1

y = df_reg["Acceptability"] #Acceptability
X = df_reg.loc[:,["Effectiveness", "Quality of life", "Justice", "age_l40", "Democratic", "Pro_envt"]] 
X = sm.add_constant(X)  # Adds intercept
model_statsmodel = sm.OLS(y, X).fit()
print(model_statsmodel.summary())

df_reg = df.loc[:,["Q17", "Q10_1", "Q11_2", "Q13", "P8", "Q3_1", "P1", "qtAge", 'qtEducation', "P9", "Q14"]]
df_reg.columns = ["Acceptability", "Effectiveness", "Quality of life", "Justice", "Ideology", "Pro_envt", "Gender", "Age", "Education", "Children", "Democratic"]
df_reg = df_reg.loc[df_reg.Ideology != 6,:]

df_reg["Effectiveness_I"] = df_reg["Effectiveness"] * df_reg["Ideology"] 
df_reg["Quality of life_I"] = df_reg["Quality of life"] * df_reg["Ideology"] 
df_reg["Justice_I"] = df_reg["Justice"] * df_reg["Ideology"] 
df_reg["Man"] = (df_reg["Gender"] == 1) * 1
df_reg["age_l40"] = (df_reg["Age"] == 1) * 1
df_reg["age_h65"] = (df_reg["Age"] == 3) * 1
df_reg["University"] = (df_reg["Education"] == 2) * 1
df_reg["Children"] = (df_reg["Children"] == 1) * 1

y = df_reg["Acceptability"] #Acceptability
X = df_reg.loc[:,["Effectiveness", "Quality of life", "Justice", "age_l40", "Democratic", "Pro_envt"]] 
X = sm.add_constant(X)  # Adds intercept
model_statsmodel = sm.OLS(y, X).fit()
print(model_statsmodel.summary())

df_reg = df.loc[:,["Q17", "Q10_1", "Q5", "Q13", "P8", "Q3_1", "P1", "qtAge", 'qtEducation', "P9", "Q14"]]
df_reg.columns = ["Acceptability", "Effectiveness", "Quality of life", "Justice", "Ideology", "Pro_envt", "Gender", "Age", "Education", "Children", "Democratic"]
df_reg = df_reg.loc[df_reg.Ideology != 6,:]

df_reg["Effectiveness_I"] = df_reg["Effectiveness"] * df_reg["Ideology"] 
df_reg["Quality of life_I"] = df_reg["Quality of life"] * df_reg["Ideology"] 
df_reg["Justice_I"] = df_reg["Justice"] * df_reg["Ideology"] 
df_reg["Man"] = (df_reg["Gender"] == 1) * 1
df_reg["age_l40"] = (df_reg["Age"] == 1) * 1
df_reg["age_h65"] = (df_reg["Age"] == 3) * 1
df_reg["University"] = (df_reg["Education"] == 2) * 1
df_reg["Children"] = (df_reg["Children"] == 1) * 1

y = df_reg["Acceptability"] #Acceptability
X = df_reg.loc[:,["Effectiveness", "Quality of life", "Justice", "age_l40", "Democratic", "Pro_envt"]] 
X = sm.add_constant(X)  # Adds intercept
model_statsmodel = sm.OLS(y, X).fit()
print(model_statsmodel.summary())