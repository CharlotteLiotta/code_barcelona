import pyreadstat # type: ignore
import statsmodels.api as sm # type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd # type: ignore
import numpy as np 
from copy import deepcopy
from scipy.sparse import csr_matrix # type: ignore
import jpype # type: ignore
import os
#os.environ["R5_JAR"] = "C:/Users/1738037/AppData/Local/miniforge3/envs/r5py/Lib/site-packages/r5py/data/r5-v6.8-all.jar"
#jpype.startJVM(classpath=[os.environ["R5_JAR"]])
import datetime
import warnings
import pickle
from stargazer.stargazer import Stargazer

from functions import *
from import_data import * # type: ignore
from calibration import * # type: ignore
from model import * # type: ignore
from plotting_tools import * # type: ignore
from import_transport import *
from policy_support import *

path_data = "../../data_barcelona/"

#gdf = gpd.read_file(path_data + "experienced_impact.geojson")

# LOAD DATA
df, meta = pyreadstat.read_sav(path_data + '040023 En moviment pel clima 2025_V01 - còpia.sav')
print(df.head())       # preview data
print(meta.column_names)  # variable names
print(meta.column_labels)
tab_label = pd.DataFrame([meta.column_names, meta.column_labels]).T

# MERGE WITH GDF TO PLOT SPATIAL VARIATIONS IN ACCEPTABILITY
df = gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df.GEO_X, df.GEO_Y), crs="EPSG:4326")
gdf = gpd.read_file(path_data + "income_loss.geojson")
df = df.to_crs(gdf.crs)
df.loc[df.P22_4 > 96, "P22_4"] = np.nan
df_plot = gpd.sjoin(df, gdf, predicate="within")
averages = df_plot.groupby("index_right")["P22_4"].agg(lambda x: np.nanmean(x))
gdf["avg_acceptability"] = gdf.index.map(averages)
plot_with_missing(gdf, gdf["avg_acceptability"])

# CREATE DATASET FOR THE REGRESSIONS
df_reg = gpd.sjoin(df, gdf, predicate="within")

# 0- ACCEPTABILITY
df_reg.rename(columns={'P15': 'acceptability_new'}, inplace=True)
#df_reg = df_reg.loc[~np.isnan(df_reg.acceptability) & (df_reg.acceptability < 97)]
#plt.hist(df_reg.acceptability_new)


# Count frequency of each response
#counts = df_reg['acceptability_new'].value_counts().sort_index()
counts = df_reg.groupby("acceptability_new")["PESAIX"].sum()

# Labels and sizes
labels = ["In favor", "Against", "Depends on \n implementation", "      Don't know"]
sizes = counts.values
sizes[3] = sizes[4] + sizes[3]
sizes = sizes[0:4]

# Colors (optional: use a colormap for consistent shading)
cmap = plt.get_cmap('Greys')
colors = list(cmap(np.linspace(0.45, 0.85, 4)))
# Swap in a bright blue for the Lacrosse color.
colors[0] = 'dodgerblue'

fig, ax = plt.subplots(figsize=(6, 6))

# Pie chart
wedges, texts, autotexts = ax.pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%',
    startangle=90,
    pctdistance=0.8,    # push percentages inward
    labeldistance=1.1,  # push labels outward,
    counterclock=False,
    colors=colors,
    wedgeprops={'edgecolor': 'white', 'linewidth': 0.8}
)

# Style labels and percentages
plt.setp(texts, fontsize=14)
plt.setp(autotexts, fontsize=14, color='white', weight='bold')
texts[3].set_position((0.2, 1.2))  # (x, y) in axes coordinates
# Ensure it's a circle
ax.axis('equal')


plt.tight_layout()
plt.show()



df_reg.rename(columns={'P22_4': 'acceptability'}, inplace=True)
df_reg = df_reg.loc[~np.isnan(df_reg.acceptability) & (df_reg.acceptability < 97)]

plt.figure(figsize=(8, 5))

# Define bins for discrete values 0–10 (11 bins)
bins = np.arange(-0.5, 11.5, 1)  

plt.hist(
    df_reg['acceptability'],
    weights=(df_reg["PESAIX"] / df_reg["PESAIX"].sum()) * 100,
    bins=bins,
    color='#1f77b4',
    alpha=0.85,
    edgecolor='black'  # clearer bar separation
)

# Axis labels and style
plt.xlabel('Acceptability (0-10 scale)', fontsize=14)
plt.ylabel('Respondents (%)', fontsize=14)

# Ensure ticks line up with integer values
plt.xticks(range(0, 11), fontsize=12)
plt.yticks(fontsize=12)

# Gridlines only on y-axis for readability
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Tight layout for clean spacing
plt.tight_layout()

plt.show()




df_reg = df_reg.loc[~np.isnan(df_reg.P20_7) & (df_reg.P20_7 < 97)]

plt.figure(figsize=(8, 5))

# Define bins for discrete values 0–10 (11 bins)
bins = np.arange(-0.5, 11.5, 1)  

plt.hist(
    df_reg['P20_7'],
    weights=(df_reg["PESAIX"] / df_reg["PESAIX"].sum()) * 100,
    bins=bins,
    color='#1f77b4',
    alpha=0.85,
    edgecolor='black'  # clearer bar separation
)

# Axis labels and style
#plt.xlabel('Acceptability (0-10 scale)', fontsize=14)
plt.ylabel('Respondents (%)', fontsize=14)

# Ensure ticks line up with integer values
plt.xticks(range(0, 11), fontsize=12)
plt.yticks(fontsize=12)

# Gridlines only on y-axis for readability
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Tight layout for clean spacing
plt.tight_layout()

plt.show()





df_reg.rename(columns={'P17': 'acceptable_price'}, inplace=True)
df_reg = df_reg.loc[~np.isnan(df_reg.acceptable_price) & (df_reg.acceptable_price < 10.1)]

plt.figure(figsize=(8,5))
plt.hist(
    df_reg['acceptable_price'],
    weights=(df_reg["PESAIX"] / df_reg["PESAIX"].sum()) * 100,
    color='#1f77b4',
    alpha=0.85,
    edgecolor='black'  # clearer bar separation
)
plt.xlabel('Acceptable price', fontsize=14)
plt.ylabel('Respondents (%)', fontsize=14)
# Ensure ticks line up with integer values
plt.xticks(range(0, 11), fontsize=12)
plt.yticks(fontsize=12)

# Gridlines only on y-axis for readability
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Tight layout for clean spacing
plt.tight_layout()


# 1- SOCIODEMOGRAPHIC VARIABLES

#age
df_reg.rename(columns={'P02': 'age'}, inplace=True)
df_reg["age"] = 2025 - df_reg["age"]
plt.hist(df_reg.age)

age_bins = [18, 30, 40, 50, 60, 70, 80, 90]
df_reg['age_group'] = pd.cut(df_reg['age'], bins=age_bins, right=False)
data_to_plot = [df_reg.loc[df_reg['age_group'] == grp, 'acceptability'] for grp in df_reg['age_group'].cat.categories]
plt.figure(figsize=(8,5))
plt.boxplot(data_to_plot, labels=[str(cat) for cat in df_reg['age_group'].cat.categories])
plt.xlabel('Age Group')
plt.ylabel('Acceptability (0-10)')
plt.title('Policy Acceptability by Age Group')
plt.show()

#gender
df_reg.rename(columns={'P01': 'man'}, inplace=True)
df_reg["man"] = (df_reg["man"] == 1) * 1
plt.hist(df_reg.man)
data_to_plot = [
    df_reg.loc[df_reg['man'] == 0, 'acceptability'],
    df_reg.loc[df_reg['man'] == 1, 'acceptability']]
plt.figure(figsize=(6,5))
plt.boxplot(data_to_plot, labels=['Women', 'Men'])
plt.xlabel('Gender')
plt.ylabel('Acceptability (0-10)')
plt.title('Policy Acceptability by Gender')
plt.show()

#education
df_reg.rename(columns={'P30': 'education'}, inplace=True)
df_reg = df_reg.loc[~np.isnan(df_reg.education) & (df_reg.education < 6)]
plt.hist(df_reg.education)
data_to_plot = [
    df_reg.loc[df_reg['education'] == 1, 'acceptability'],
    df_reg.loc[df_reg['education'] == 2, 'acceptability'],
    df_reg.loc[df_reg['education'] == 3, 'acceptability'],
    df_reg.loc[df_reg['education'] == 4, 'acceptability']]
plt.figure(figsize=(8,5))
plt.boxplot(data_to_plot, labels=['1', '2', '3', '4'])
plt.xlabel('Education Level')
plt.ylabel('Acceptability (0-10)')
plt.title('Policy Acceptability by Education Level')
plt.show()

#dependent children
df_reg["children"] = 1 * ((df_reg.P03B_1 < 18) & (df_reg.P03C_1 == 2)) + 1 * ((df_reg.P03B_2 < 18) & (df_reg.P03C_2 == 2)) + 1 * ((df_reg.P03B_3 < 18) & (df_reg.P03C_3 == 2)) + 1 * ((df_reg.P03B_4 < 18) & (df_reg.P03C_4 == 2)) + 1 * ((df_reg.P03B_5 < 18) & (df_reg.P03C_5 == 2)) + 1 * ((df_reg.P03B_6 < 18) & (df_reg.P03C_6 == 2)) + 1 * ((df_reg.P03B_7 < 18) & (df_reg.P03C_7 == 2)) + 1 * ((df_reg.P03B_8 < 18) & (df_reg.P03C_8 == 2)) + 1 * ((df_reg.P03B_9 < 18) & (df_reg.P03C_9 == 2)) + 1 * ((df_reg.P03B_10 < 18) & (df_reg.P03C_10 == 2))
df_reg["children2"] = 1 * ((df_reg.P03B_1 < 18)) + 1 * ((df_reg.P03B_2 < 18)) + 1 * ((df_reg.P03B_3 < 18)) + 1 * ((df_reg.P03B_4 < 18)) + 1 * ((df_reg.P03B_5 < 18)) + 1 * ((df_reg.P03B_6 < 18)) + 1 * ((df_reg.P03B_7 < 18)) + 1 * ((df_reg.P03B_8 < 18)) + 1 * ((df_reg.P03B_9 < 18)) + 1 * ((df_reg.P03B_10 < 18))
plt.hist(df_reg["children"], bins = np.arange(df_reg["children"].min(), df_reg["children"].max() + 2) - 0.5)
plt.hist(df_reg["children2"], bins = np.arange(df_reg["children2"].min(), df_reg["children2"].max() + 2) - 0.5)
df_reg["children3"] = df_reg["children2"] > 0

# 2- PSYCHOLOGY/VALUES

#political ideology
df_reg.rename(columns={'P35': 'political_ideology'}, inplace=True)
df_reg = df_reg.loc[~np.isnan(df_reg.political_ideology) & (df_reg.political_ideology < 80)]
plt.hist(df_reg.political_ideology, bins = 11)
bins =  [-0.1, 1.1, 3.1, 5.1, 7.1, 10.1]
df_reg['ideology_group'] = pd.cut(df_reg['political_ideology'], bins=bins, right=False)
data_to_plot = [df_reg.loc[df_reg['ideology_group'] == grp, 'acceptability'] for grp in df_reg['ideology_group'].cat.categories]
plt.figure(figsize=(10,5))
plt.boxplot(data_to_plot, labels=[str(cat) for cat in df_reg['ideology_group'].cat.categories])
plt.xlabel('Political Ideology')
plt.ylabel('Acceptability (0-10)')
plt.title('Policy Acceptability by Political Ideology')
plt.show()

#institutional trust
df_reg.rename(columns={'P23': 'institutional_trust'}, inplace=True)
df_reg = df_reg.loc[~np.isnan(df_reg.institutional_trust) & (df_reg.institutional_trust < 80)]
plt.hist(df_reg.institutional_trust, bins = 11)
bins =  [-0.1, 0.1, 2.1, 4.1, 6.1, 8.1, 10.1]
df_reg['institutional_trust_group'] = pd.cut(df_reg['institutional_trust'], bins=bins, right=False)
data_to_plot = [df_reg.loc[df_reg['institutional_trust_group'] == grp, 'acceptability'] for grp in df_reg['institutional_trust_group'].cat.categories]
plt.figure(figsize=(10,5))
plt.boxplot(data_to_plot, labels=[str(cat) for cat in df_reg['institutional_trust_group'].cat.categories])
plt.xlabel('Institutional Trust')
plt.ylabel('Acceptability (0-10)')
plt.title('Policy Acceptability by Institutional Trust')
plt.show()

#peer effect
df_reg.rename(columns={'P28_1': 'peer_effect'}, inplace=True)
df_reg = df_reg.loc[~np.isnan(df_reg.peer_effect) & (df_reg.peer_effect < 80)]
plt.hist(df_reg.peer_effect, bins = 11)
bins =  [-0.1, 0.1, 2.1, 4.1, 6.1, 8.1, 10.1]
df_reg['peer_effect_group'] = pd.cut(df_reg['peer_effect'], bins=bins, right=False)
data_to_plot = [df_reg.loc[df_reg['peer_effect_group'] == grp, 'acceptability'] for grp in df_reg['peer_effect_group'].cat.categories]
plt.figure(figsize=(10,5))
plt.boxplot(data_to_plot, labels=[str(cat) for cat in df_reg['peer_effect_group'].cat.categories])
plt.xlabel('Peer Effect')
plt.ylabel('Acceptability (0-10)')
plt.title('Policy Acceptability by Peer Effect')
plt.show()

#Eco-anxiety
df_reg.rename(columns={'P27_1': 'ecoanxiety'}, inplace=True)
df_reg = df_reg.loc[~np.isnan(df_reg.ecoanxiety) & (df_reg.ecoanxiety < 80)]
plt.hist(df_reg.ecoanxiety, bins = 11)
bins =  [-0.1, 0.1, 2.1, 4.1, 6.1, 8.1, 10.1]
df_reg['ecoanxiety_group'] = pd.cut(df_reg['ecoanxiety'], bins=bins, right=False)
data_to_plot = [df_reg.loc[df_reg['ecoanxiety_group'] == grp, 'acceptability'] for grp in df_reg['ecoanxiety_group'].cat.categories]
plt.figure(figsize=(10,5))
plt.boxplot(data_to_plot, labels=[str(cat) for cat in df_reg['ecoanxiety_group'].cat.categories])
plt.xlabel('Ecoanxiety')
plt.ylabel('Acceptability (0-10)')
plt.title('Policy Acceptability by Ecoanxiety')
plt.show()

#Ecological paradigm
df_reg.rename(columns={'P26_5': 'ecological_paradigm'}, inplace=True)
df_reg = df_reg.loc[~np.isnan(df_reg.ecological_paradigm) & (df_reg.ecological_paradigm < 80)]
plt.hist(df_reg.ecological_paradigm, bins = 11)
bins =  [-0.1, 0.1, 2.1, 4.1, 6.1, 8.1, 10.1]
df_reg['ecological_paradigm_group'] = pd.cut(df_reg['ecological_paradigm'], bins=bins, right=False)
data_to_plot = [df_reg.loc[df_reg['ecological_paradigm_group'] == grp, 'acceptability'] for grp in df_reg['ecological_paradigm_group'].cat.categories]
plt.figure(figsize=(10,5))
plt.boxplot(data_to_plot, labels=[str(cat) for cat in df_reg['ecological_paradigm_group'].cat.categories])
plt.xlabel('Ecological Paradigm')
plt.ylabel('Acceptability (0-10)')
plt.title('Policy Acceptability by Ecological Paradigm')
plt.show()

#Knowledge
df_reg.rename(columns={'P24': 'knowledge'}, inplace=True)
df_reg = df_reg.loc[~np.isnan(df_reg.knowledge) & (df_reg.knowledge < 80)]
plt.hist(df_reg.knowledge, bins = 11)
bins =  [-0.1, 0.1, 2.1, 4.1, 6.1, 8.1, 10.1]
df_reg['knowledge_group'] = pd.cut(df_reg['knowledge'], bins=bins, right=False)
data_to_plot = [df_reg.loc[df_reg['knowledge_group'] == grp, 'acceptability'] for grp in df_reg['knowledge_group'].cat.categories]
plt.figure(figsize=(10,5))
plt.boxplot(data_to_plot, labels=[str(cat) for cat in df_reg['knowledge_group'].cat.categories])
plt.xlabel('Knowledge')
plt.ylabel('Acceptability (0-10)')
plt.title('Policy Acceptability by Knowledge')
plt.show()

# 3- PERCEIVED BENEFITS

#On climate change
df_reg.rename(columns={'P21_3': 'climate_change'}, inplace=True)
df_reg = df_reg.loc[~np.isnan(df_reg.climate_change) & (df_reg.climate_change < 80)]
plt.hist(df_reg.climate_change, bins = 11)

plt.figure(figsize=(8,5))
plt.hist(
    df_reg['climate_change'],
    weights=(df_reg["PESAIX"] / df_reg["PESAIX"].sum()) * 100,
    color='#1f77b4',
    alpha=0.85,
    edgecolor='black'  # clearer bar separation
)
plt.xlabel('Perceived benefits on GHG emissions', fontsize=14)
plt.ylabel('Respondents (%)', fontsize=14)
# Ensure ticks line up with integer values
plt.xticks(range(0, 11), fontsize=12)
plt.yticks(fontsize=12)

# Gridlines only on y-axis for readability
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Tight layout for clean spacing
plt.tight_layout()





bins =  [-0.1, 0.1, 2.1, 4.1, 6.1, 8.1, 10.1]
df_reg['climate_change_group'] = pd.cut(df_reg['climate_change'], bins=bins, right=False)
data_to_plot = [df_reg.loc[df_reg['climate_change_group'] == grp, 'acceptability'] for grp in df_reg['climate_change_group'].cat.categories]
plt.figure(figsize=(10,5))
plt.boxplot(data_to_plot, labels=[str(cat) for cat in df_reg['climate_change_group'].cat.categories])
plt.xlabel('Perceived benefits on climate change')
plt.ylabel('Acceptability (0-10)')
plt.title('Policy Acceptability by Perceived benefits on climate change')
plt.show()

#On traffic
df_reg.rename(columns={'P21_1': 'congestion'}, inplace=True)
df_reg = df_reg.loc[~np.isnan(df_reg.congestion) & (df_reg.congestion < 80)]
plt.hist(df_reg.congestion, bins = 11)

plt.figure(figsize=(8,5))
plt.hist(
    df_reg['congestion'],
    weights=(df_reg["PESAIX"] / df_reg["PESAIX"].sum()) * 100,
    color='#1f77b4',
    alpha=0.85,
    edgecolor='black'  # clearer bar separation
)
plt.xlabel('Perceived benefits on congestion', fontsize=14)
plt.ylabel('Respondents (%)', fontsize=14)
# Ensure ticks line up with integer values
plt.xticks(range(0, 11), fontsize=12)
plt.yticks(fontsize=12)

# Gridlines only on y-axis for readability
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Tight layout for clean spacing
plt.tight_layout()







bins =  [-0.1, 0.1, 2.1, 4.1, 6.1, 8.1, 10.1]
df_reg['congestion_group'] = pd.cut(df_reg['congestion'], bins=bins, right=False)
data_to_plot = [df_reg.loc[df_reg['congestion_group'] == grp, 'acceptability'] for grp in df_reg['congestion_group'].cat.categories]
plt.figure(figsize=(10,5))
plt.boxplot(data_to_plot, labels=[str(cat) for cat in df_reg['congestion_group'].cat.categories])
plt.xlabel('Perceived benefits on congestion')
plt.ylabel('Acceptability (0-10)')
plt.title('Policy Acceptability by Perceived benefits on congestion')
plt.show()

#On quality of life
df_reg.rename(columns={'P21_2': 'quality_of_life'}, inplace=True)
df_reg = df_reg.loc[~np.isnan(df_reg.quality_of_life) & (df_reg.quality_of_life < 80)]
plt.hist(df_reg.quality_of_life, bins = 11)

plt.figure(figsize=(8,5))
plt.hist(
    df_reg['quality_of_life'],
    weights=(df_reg["PESAIX"] / df_reg["PESAIX"].sum()) * 100,
    color='#1f77b4',
    alpha=0.85,
    edgecolor='black'  # clearer bar separation
)
plt.xlabel('Perceived benefits on air and noise pollution and safety', fontsize=14)
plt.ylabel('Respondents (%)', fontsize=14)
# Ensure ticks line up with integer values
plt.xticks(range(0, 11), fontsize=12)
plt.yticks(fontsize=12)

# Gridlines only on y-axis for readability
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Tight layout for clean spacing
plt.tight_layout()





bins =  [-0.1, 0.1, 2.1, 4.1, 6.1, 8.1, 10.1]
df_reg['quality_of_life_group'] = pd.cut(df_reg['quality_of_life'], bins=bins, right=False)
data_to_plot = [df_reg.loc[df_reg['quality_of_life_group'] == grp, 'acceptability'] for grp in df_reg['quality_of_life_group'].cat.categories]
plt.figure(figsize=(10,5))
plt.boxplot(data_to_plot, labels=[str(cat) for cat in df_reg['quality_of_life_group'].cat.categories])
plt.xlabel('Perceived benefits on quality of life')
plt.ylabel('Acceptability (0-10)')
plt.title('Policy Acceptability by Perceived benefits on quality of life')
plt.show()

# 4- PERCEIVED IMPACTS

#OPTION1 - DIRECT SURVEY QUESTION

#declared impacts
df_reg.rename(columns={'P11': 'declared_impacts'}, inplace=True)
df_reg = df_reg.loc[(df_reg.declared_impacts != 3) & (df_reg.declared_impacts != 4),:]
df_reg.declared_impacts = (df_reg.declared_impacts == 1) * 1
plt.hist(df_reg.declared_impacts)
data_to_plot = [
    df_reg.loc[df_reg['declared_impacts'] == 0, 'acceptability'],
    df_reg.loc[df_reg['declared_impacts'] == 1, 'acceptability']]
plt.figure(figsize=(6,5))
plt.boxplot(data_to_plot, labels=['No impact', 'Impact'])
plt.xlabel('Declared Impacts')
plt.ylabel('Acceptability (0-10)')
plt.title('Policy Acceptability by Declared Impacts')
plt.show()

#declared impacts frequency
df_reg.rename(columns={'P12': 'declared_impacts_frequency'}, inplace=True)
df_reg = df_reg.loc[(df_reg.declared_impacts_frequency != 8) & (df_reg.declared_impacts_frequency != 9),:]
df_reg.declared_impacts_frequency.loc[df_reg.declared_impacts == 0] = 7
plt.hist(df_reg.declared_impacts_frequency, bins = 7)
#bins =  [0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1]
bins =  [0.1, 2.1, 5.1, 7.1]
df_reg['declared_impacts_frequency_group'] = pd.cut(df_reg['declared_impacts_frequency'], bins=bins, right=False)
data_to_plot = [df_reg.loc[df_reg['declared_impacts_frequency_group'] == grp, 'acceptability'] for grp in df_reg['declared_impacts_frequency_group'].cat.categories]
plt.figure(figsize=(10,5))
plt.boxplot(data_to_plot, labels=[str(cat) for cat in df_reg['declared_impacts_frequency_group'].cat.categories])
plt.xlabel('Declared impacts frequency')
plt.ylabel('Acceptability (0-10)')
plt.title('Policy Acceptability by Declared impacts frequency')
plt.show()

#how would the toll affect your mobility
df_reg.rename(columns={'P19': 'impact_mobility'}, inplace=True)
df_reg = df_reg.loc[(df_reg.impact_mobility <90),:]

labels = {
    1: "1- Would not change",
    2: "2- Move outside Barcelona",
    3: "3- Go to Barcelona less",
    4: "4- Use private vehicle more",
    5: "5- Use public transport/cycle/walk more",
    6: "6- Stop using private vehicle",
    7: "7- Depends on toll price",
    8: "8- Other"
}

df_reg['impact_mobility'].map(labels).value_counts().sort_index().plot(kind='bar', figsize=(10,6))
plt.ylabel("Count")
plt.xlabel("Mobility response")
plt.title("Distribution of Mobility Responses")
plt.xticks(rotation=45, ha='right')
plt.show()

groups = [df_reg.loc[df_reg['impact_mobility']==k, 'acceptability'].dropna() for k in sorted(labels.keys())]
plt.figure(figsize=(12,6))
plt.boxplot(groups, labels=[labels[k] for k in sorted(labels.keys())], patch_artist=True)
plt.xticks(rotation=45, ha='right')
plt.xlabel("Mobility response")
plt.ylabel("Acceptability")
plt.title("Distribution of acceptability by Mobility Response")
plt.show()

df_reg["no_change"] = (df_reg.impact_mobility == 1) * 1
df_reg["use_car_more"] = (df_reg.impact_mobility == 4) * 1
df_reg["use_car_less"] = (df_reg.impact_mobility.isin([2, 3, 5, 6])) * 1


df_reg["P19_2"] = (df_reg.impact_mobility == 2) * 1
df_reg["P19_3"] = (df_reg.impact_mobility == 3) * 1
df_reg["P19_4"] = (df_reg.impact_mobility == 4) * 1
df_reg["P19_5"] = (df_reg.impact_mobility == 5) * 1
df_reg["P19_6"] = (df_reg.impact_mobility == 6) * 1
df_reg["P19_7"] = (df_reg.impact_mobility == 7) * 1
df_reg["P19_8"] = (df_reg.impact_mobility == 8) * 1

pd.crosstab(df_reg['declared_impacts'], df_reg['impact_mobility'])

#OPTION2 - CAR OWERNSHIP + LIVE OR WORK IN THE AFFECTED AREA

#vehicle ownership
df_reg["vehicle_ownership"] = 1 * ((df_reg.P34A > 0) | (df_reg.P34B > 0) |(df_reg.P34C > 0)) 
plt.hist(df_reg["vehicle_ownership"])
data_to_plot = [
    df_reg.loc[df_reg['vehicle_ownership'] == 0, 'acceptability'],
    df_reg.loc[df_reg['vehicle_ownership'] == 1, 'acceptability']]
plt.figure(figsize=(6,5))
plt.boxplot(data_to_plot, labels=['No vehicle', 'Vehicle'])
plt.xlabel('Vehicle ownership')
plt.ylabel('Acceptability (0-10)')
plt.title('Policy Acceptability by Vehicle ownership')
plt.show()

#vehicle ownership and license
df_reg["vehicle_ownership_license"] = 1 * (((df_reg.P34A > 0) &(df_reg.P33A  == 1))| ((df_reg.P34B > 0) &(df_reg.P33B  == 1)) |((df_reg.P34C > 0) &(df_reg.P33A  == 1))) 
plt.hist(df_reg["vehicle_ownership_license"])
data_to_plot = [
    df_reg.loc[df_reg['vehicle_ownership_license'] == 0, 'acceptability'],
    df_reg.loc[df_reg['vehicle_ownership_license'] == 1, 'acceptability']]
plt.figure(figsize=(6,5))
plt.boxplot(data_to_plot, labels=['No vehicle', 'Vehicle'])
plt.xlabel('Vehicle ownership and license')
plt.ylabel('Acceptability (0-10)')
plt.title('Policy Acceptability by Vehicle ownership')
plt.show()

#live in affected area
df_reg["live_in_tax_area"]= 1 * (df_reg.ID.str[:5].isin(["08019", "08101", "08194"]))
plt.hist(df_reg["live_in_tax_area"])
data_to_plot = [
    df_reg.loc[df_reg['live_in_tax_area'] == 0, 'acceptability'],
    df_reg.loc[df_reg['live_in_tax_area'] == 1, 'acceptability']]
plt.figure(figsize=(6,5))
plt.boxplot(data_to_plot, labels=['Live out of tax area', 'Live in tax area'])
plt.xlabel('Residence')
plt.ylabel('Acceptability (0-10)')
plt.title('Policy Acceptability by Residence')
plt.show()

#work in affected area
df_reg["work_in_tax_area"]= 1 * (df_reg.P32_MUNICOD.isin([8019, 8101, 8194]))

gdf_job = gpd.GeoDataFrame(df_reg, geometry = gpd.points_from_xy(df_reg.P32_GEO_X, df_reg.P32_GEO_Y), crs="EPSG:4326")
#gdf_job = gdf_job[gdf_job.geometry.notnull() & gdf_job.is_valid & ~gdf_job.is_empty]
gdf_job.to_crs(gdf.crs).plot()
fig, ax = plt.subplots(figsize=(8,6))
xmin, ymin, xmax, ymax = df.total_bounds
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
# Plot the second GeoDataFrame
gdf_job.to_crs(gdf.crs).plot(df_reg["work_in_tax_area"], ax=ax, marker='x', legend = True)
plt.legend()
plt.show()

#OPTION3 - EXPERIENCED IMPACTS

plt.hist(df_reg.income_loss_absolute)

plt.hist(df_reg.income_loss_relative)

### Regression

#v0 - Without impacts

X_raw = df_reg[["climate_change", "congestion", "quality_of_life", "political_ideology", "institutional_trust", "knowledge", "peer_effect", "ecoanxiety", "ecological_paradigm", "age", "man", "education", "children3"]]
y_raw = df_reg["acceptability"].values.reshape(-1, 1)

scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X_raw)
X_scaled = pd.DataFrame(X_scaled, columns=X_raw.columns)
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y_raw).flatten()
X_scaled = sm.add_constant(X_scaled)

model_statsmodel = sm.WLS(y_scaled, X_scaled, weights=df_reg['PESAIX']).fit()
print(model_statsmodel.summary())

model_statsmodel = sm.OLS(y_scaled, X_scaled).fit()
print(model_statsmodel.summary())

#expected effect:
# PSYCHOLOGY: political ideology -, institutional trust +, knowledge +, peer effect +, ecoanxiety +, ecological paradigm +
# SOCIODEMO: age -, man -, education +, children -
# PERCEIVED BENEFITS: climate change +, congestion +, quality of life +

#v1 - With perceived impacts

X_raw = df_reg[["declared_impacts", "climate_change", "congestion", "quality_of_life", "political_ideology", "institutional_trust", "knowledge", "ecoanxiety", "ecological_paradigm", "age", "man", "education", "children"]]
#X_raw = df_reg[["declared_impacts"]]
#X_raw = df_reg[["declared_impacts", "climate_change", "congestion", "quality_of_life"]]
#X_raw = df_reg[["declared_impacts", "climate_change", "congestion", "quality_of_life", "institutional_trust", "ecoanxiety"]]
#X_raw = df_reg[["declared_impacts", "climate_change", "congestion", "quality_of_life", "institutional_trust", "ecoanxiety", "man", "education"]]
y_raw = df_reg["acceptability"].values.reshape(-1, 1)

#corr_df = pd.DataFrame(np.corrcoef(X_raw.T), index = X_raw.columns, columns = X_raw.columns)

scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X_raw)
X_scaled = pd.DataFrame(X_scaled, columns=X_raw.columns)
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y_raw).flatten()
X_scaled = sm.add_constant(X_scaled)

model_statsmodel = sm.WLS(y_scaled, X_scaled, weights=df_reg['PESAIX']).fit()
print(model_statsmodel.summary())

model_statsmodel = sm.OLS(y_scaled, X_scaled).fit()
print(model_statsmodel.summary())

# IMPACTS: declared impact -, declared impact frequency +, no change/use car more/use car less
# PSYCHOLOGY: political ideology -, institutional trust +, knowledge +, peer effect +, ecoanxiety +, ecological paradigm +
# SOCIODEMO: age -, man -, education +, children -
# PERCEIVED BENEFITS: climate change +, congestion +, quality of life +

#endogeneity: ppl declaring they would use car less are also those that are willing to change things

#v2 - Car ownership + live or work in the affected area

### car ownership variable to improve

df_reg["live_or_work_in_tax_area"] = (df_reg["live_in_tax_area"] + df_reg["work_in_tax_area"]) > 0
df_reg["live_or_work_in_tax_area_car"] = (df_reg["live_or_work_in_tax_area"] * df_reg["vehicle_ownership"])
df_reg["live_in_tax_area_qol"] = df_reg["live_in_tax_area"] * df_reg["quality_of_life"]
df_reg["work_in_tax_area_qol"] = df_reg["work_in_tax_area"] * df_reg["quality_of_life"]
df_reg["live_or_work_in_tax_area_congestion"] = df_reg["live_or_work_in_tax_area"] * df_reg["congestion"]
df_reg["live_or_work_in_tax_area_qol"] = df_reg["live_or_work_in_tax_area"] * df_reg["quality_of_life"]
df_reg["work_in_tax_area_congestion"] = df_reg["work_in_tax_area"] * df_reg["congestion"]


X_raw = df_reg[["live_or_work_in_tax_area", "vehicle_ownership_license", "climate_change", "live_or_work_in_tax_area_qol", "live_or_work_in_tax_area_congestion"]]
X_raw = df_reg[["live_or_work_in_tax_area", "vehicle_ownership_license", "climate_change", "live_or_work_in_tax_area_qol", "live_or_work_in_tax_area_congestion", "political_ideology", "institutional_trust", "knowledge", "peer_effect", "ecoanxiety", "ecological_paradigm", "age", "man", "education", "children"]]
X_raw = df_reg[["live_or_work_in_tax_area", "vehicle_ownership_license", "climate_change", "live_or_work_in_tax_area_qol", "live_or_work_in_tax_area_congestion", "institutional_trust", "knowledge", "ecoanxiety", "education"]]

#X_raw = df_reg[["vehicle_ownership", "live_in_tax_area", "climate_change", "live_in_tax_area_qol", "live_in_tax_area_congestion", "institutional_trust", "knowledge", "ecoanxiety", "education"]]
y_raw = df_reg["acceptability"].values.reshape(-1, 1)

scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X_raw)
X_scaled = pd.DataFrame(X_scaled, columns=X_raw.columns)
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y_raw).flatten()
X_scaled = sm.add_constant(X_scaled)

model_statsmodel = sm.WLS(y_scaled, X_scaled, weights=df_reg['PESAIX']).fit()
print(model_statsmodel.summary())

model_statsmodel = sm.OLS(y_scaled, X_scaled).fit()
print(model_statsmodel.summary())

# IMPACTS: car ownership -, live in tax area -
# PSYCHOLOGY: political ideology -, institutional trust +, knowledge +, peer effect +, ecoanxiety +, ecological paradigm +
# SOCIODEMO: age -, man -, education +, children -
# PERCEIVED BENEFITS: climate change +, congestion +, quality of life +

#v3 - Full model

plt.hist(df_reg.P18[df_reg.P18 < 20])
np.nansum(df_reg.P18 < 1.001) / np.nansum(df_reg.P18 < 90)

df_reg["experienced_impacts_declared_impacts"] = - df_reg["declared_impacts"] * df_reg["experienced_impact"]
df_reg["experienced_impacts_declared_impacts2"] = df_reg["experienced_impacts_declared_impacts"]  * df_reg["experienced_impacts_declared_impacts"] 
df_reg["experienced_impacts_declared_impacts3"] = df_reg["experienced_impact"] < - 0.015

X_raw = df_reg[["experienced_impacts_declared_impacts", "experienced_impacts_declared_impacts2", "climate_change", "congestion", "quality_of_life", "political_ideology", "institutional_trust", "knowledge", "peer_effect", "ecoanxiety", "ecological_paradigm", "age", "man", "education", "children"]]
X_raw = df_reg[["experienced_impacts_declared_impacts", "climate_change", "congestion", "quality_of_life"]]
X_raw = df_reg[["experienced_impact", "climate_change", "congestion", "quality_of_life", "institutional_trust"]]

#"live_or_work_in_tax_area", "vehicle_ownership_license"
#"declared_impacts"
df_reg["log_absolute_loss"] = -np.log(-df_reg["income_loss_absolute"])
df_reg["log_relative_loss"] = np.log(-df_reg["income_loss_relative"])

df_reg["log_absolute_loss_declared_impact"] = df_reg["log_absolute_loss"] * (df_reg["declared_impacts"])
df_reg["relative_loss_declared_impact"] = df_reg["income_loss_relative"] * df_reg["live_or_work_in_tax_area"]* df_reg["vehicle_ownership_license"]
df_reg["log_relative_loss_declared_impact"] = df_reg["log_relative_loss"] * (7 - df_reg["declared_impacts_frequency"])

df_reg["log_absolute_loss_eco"] = - df_reg["log_absolute_loss"] * df_reg["political_ideology"]
df_reg["climate_change_eco"] = - df_reg["climate_change"] * df_reg["ecological_paradigm"]
df_reg["congestion_eco"] = - df_reg["congestion"] * df_reg["political_ideology"]
df_reg["quality_of_life_eco"] = - df_reg["quality_of_life"] * df_reg["climate_change_eco"]

X_raw = df_reg[["log_absolute_loss", "climate_change", "congestion", "quality_of_life", "political_ideology", "institutional_trust", "knowledge", "ecoanxiety", "ecological_paradigm", "age", "man", "education", "children"]]

X_raw = df_reg[["log_absolute_loss_declared_impact", "climate_change", "congestion", "quality_of_life"]]


X_raw = df_reg[["log_absolute_loss", "climate_change", "congestion", "quality_of_life"]]

X_raw = df_reg[["log_absolute_loss", "climate_change", "congestion", "quality_of_life", "knowledge", "climate_change_eco", "quality_of_life_eco"]]

y_raw = df_reg["acceptable_price"].values.reshape(-1, 1)

#corr_df = pd.DataFrame(np.corrcoef(X_raw.T), index = X_raw.columns, columns = X_raw.columns)

scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X_raw)
X_scaled = pd.DataFrame(X_scaled, columns=X_raw.columns)
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y_raw).flatten()
X_scaled = sm.add_constant(X_scaled)

X_raw = sm.add_constant(X_raw)

model_statsmodel = sm.WLS(y_raw, X_scaled, weights=df_reg['PESAIX']).fit()
#model_statsmodel = sm.WLS(y_raw, X_raw, weights=df_reg['PESAIX']).fit()

print(model_statsmodel.summary())

model_statsmodel = sm.OLS(y_scaled, X_scaled).fit()
print(model_statsmodel.summary())

#other

X_raw = df_reg[["declared_impacts", "congestion", "quality_of_life", "institutional_trust"]]

y_raw = df_reg["acceptability"].values.reshape(-1, 1)

#corr_df = pd.DataFrame(np.corrcoef(X_raw.T), index = X_raw.columns, columns = X_raw.columns)

scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X_raw)
X_scaled = pd.DataFrame(X_scaled, columns=X_raw.columns)
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y_raw).flatten()
X_scaled = sm.add_constant(X_scaled)

model_statsmodel = sm.WLS(y_scaled, X_scaled, weights=df_reg['PESAIX']).fit()
print(model_statsmodel.summary())


df_reg["log_absolute_loss_eco"] = - df_reg["log_absolute_loss"] * df_reg["ecological_paradigm"]
df_reg["climate_change_eco"] = - df_reg["climate_change"] * df_reg["ecological_paradigm"]
df_reg["congestion_eco"] = - df_reg["congestion"] * df_reg["ecological_paradigm"]
df_reg["quality_of_life_eco"] = - df_reg["quality_of_life"] * df_reg["ecological_paradigm"]


df_reg2 = df_reg #.loc[(df_reg.acceptability > 0),:]
X_raw = (df_reg2[["log_absolute_loss", "congestion", "quality_of_life", "climate_change"]])

y_raw = ((df_reg2["acceptable_price"].values.reshape(-1, 1)))

#corr_df = pd.DataFrame(np.corrcoef(X_raw.T), index = X_raw.columns, columns = X_raw.columns)

scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X_raw)
X_scaled = pd.DataFrame(X_scaled, columns=X_raw.columns)
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y_raw).flatten()
X_scaled = sm.add_constant(X_scaled)

model_statsmodel = sm.WLS(y_raw, X_scaled, weights=df_reg2['PESAIX']).fit()
print(model_statsmodel.summary())



















### Descriptive statistics



x = df_reg.declared_impacts_frequency
y = df_reg.acceptable_price
w = df_reg.PESAIX   # survey weights

# Define custom bins and labels
bins = [0.5, 3.5, 6.5, 7.5]  
labels = ["> once a week", "< once a week", "Never"]
df_reg['bin'] = pd.cut(x, bins=bins, labels=labels)

# Weighted mean function
def weighted_mean(series, weights):
    return np.average(series, weights=weights)

# Group by bin and calculate weighted stats
grouped = (
    df_reg.groupby('bin')
    .apply(lambda g: pd.Series({
        "mean": weighted_mean(g[y.name], g[w.name]),
        "share": g[w.name].sum() / w.sum() * 100  # percentage of population
    }))
)

# Optional: drop bins with negligible share
grouped = grouped[grouped['share'] >= 1]

# --- Plot ---
fig, ax1 = plt.subplots()

# Line: weighted mean acceptable price
ax1.plot(grouped.index, grouped['mean'], marker='o', color='blue')
ax1.set_xlabel('Frequency of private vehicle use in the zone', fontsize=14)
ax1.set_ylabel('Acceptable toll (€)', fontsize=14, color='blue')
ax1.tick_params(axis='both', labelsize=12)
ax1.tick_params(axis='y', labelcolor='blue')

# Bars: share of respondents
ax2 = ax1.twinx()
ax2.bar(grouped.index, grouped['share'], alpha=0.3, color='gray')
ax2.tick_params(axis='y', labelsize=12, labelcolor='black')
ax2.set_ylabel('Share of respondents (%)', color='black', fontsize=14)

plt.show()



x = df_reg.declared_impacts_frequency
y = df_reg.acceptability
w = df_reg.PESAIX   # survey weights

# Define custom bins and labels
bins = [0.5, 3.5, 6.5, 7.5]  
labels = ["> once a week", "< once a week", "Never"]
df_reg['bin'] = pd.cut(x, bins=bins, labels=labels)

# Weighted mean function
def weighted_mean(series, weights):
    return np.average(series, weights=weights)

# Group by bin and calculate weighted stats
grouped = (
    df_reg.groupby('bin')
    .apply(lambda g: pd.Series({
        "mean": weighted_mean(g[y.name], g[w.name]),
        "share": g[w.name].sum() / w.sum() * 100  # percentage of population
    }))
)

# Optional: drop bins with negligible share
grouped = grouped[grouped['share'] >= 1]

# --- Plot ---
fig, ax1 = plt.subplots()

# Line: weighted mean acceptable price
ax1.plot(grouped.index, grouped['mean'], marker='o', color='blue')
ax1.set_xlabel('Frequency of private vehicle use in the zone', fontsize=14)
ax1.set_ylabel('Average acceptability', fontsize=14, color='blue')
ax1.tick_params(axis='both', labelsize=12)
ax1.tick_params(axis='y', labelcolor='blue')

# Bars: share of respondents
ax2 = ax1.twinx()
ax2.bar(grouped.index, grouped['share'], alpha=0.3, color='gray')
ax2.tick_params(axis='y', labelsize=12, labelcolor='black')
ax2.set_ylabel('Share of respondents (%)', color='black', fontsize=14)

plt.show()




x = df_reg.acceptability
y = df_reg.acceptable_price

# Define bins
bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]
df_reg['bin'] = pd.cut(x, bins)

# Compute mean and count per bin
grouped = df_reg.groupby('bin')[y.name].agg(['mean', 'count'])
grouped = grouped[grouped['count'] >= 5]
bin_centers = [interval.mid for interval in grouped.index]

# Line plot of mean
fig, ax1 = plt.subplots()
ax1.plot(bin_centers, grouped['mean'], marker='o', color='blue')
ax1.set_xlabel('Negative experienced impact')
ax1.set_ylabel('Average Acceptability_10', color='blue')

# Secondary axis for counts
ax2 = ax1.twinx()
ax2.bar(bin_centers, grouped['count'], width=(bins[1]-bins[0])*0.8, alpha=0.3, color='gray')
ax2.set_ylabel('Number of observations', color='gray')

plt.show()



x = -df_reg.log_absolute_loss
y = df_reg.acceptable_price
w = df_reg.PESAIX   # survey weights

# Define bins
bins = np.linspace(x.min(), x.max(), 10)
df_reg['bin'] = pd.cut(x, bins)

# Weighted mean function
def weighted_mean(series, weights):
    return np.average(series, weights=weights)

# Group by bin and calculate weighted stats
grouped = (
    df_reg.groupby('bin')
    .apply(lambda g: pd.Series({
        "mean": weighted_mean(g[y.name], g[w.name]),
        "share": g[w.name].sum() / w.sum() * 100  # % of total respondents
    }))
)

# Drop bins with too few weighted respondents (optional, e.g. <1% share)
grouped = grouped[grouped['share'] >= 1]

# Get bin centers for plotting
bin_centers = [interval.mid for interval in grouped.index]

# --- Plot ---
fig, ax1 = plt.subplots()

# Line plot: weighted mean acceptable toll
ax1.plot(bin_centers, grouped['mean'], marker='o', color='blue')
ax1.set_xlabel('Mobility loss due to the toll (log, €)', fontsize=14)
ax1.set_ylabel('Acceptable toll (€)', fontsize=14, color='blue')
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12, colors='blue')

# Bar plot: share of respondents (%)
ax2 = ax1.twinx()
ax2.bar(
    bin_centers,
    grouped['share'],
    width=(bins[1]-bins[0]),  # a bit narrower than bin width
    alpha=0.3,
    color='gray',
    edgecolor='black',
    align="center"
)
ax2.set_ylabel('Share of respondents (%)', color='black', fontsize=14)
ax2.tick_params(axis='y', labelsize=12, colors='black')

plt.show()





x = -df_reg.income_loss_relative
y = df_reg.acceptability

# Define bins
bins = np.linspace(x.min(), x.max(), 15)
df_reg['bin'] = pd.cut(x, bins)

# Compute mean and count per bin
grouped = df_reg.groupby('bin')[y.name].agg(['mean', 'count'])
grouped = grouped[grouped['count'] >= 10]
bin_centers = [interval.mid for interval in grouped.index]

# Line plot of mean
fig, ax1 = plt.subplots()
ax1.plot(bin_centers, grouped['mean'], marker='o', color='blue')
ax1.set_xlabel('Negative experienced impact')
ax1.set_ylabel('Average Acceptability_10', color='blue')

# Secondary axis for counts
ax2 = ax1.twinx()
ax2.bar(bin_centers, grouped['count'], width=(bins[1]-bins[0])*0.8, alpha=0.3, color='gray')
ax2.set_ylabel('Number of observations', color='gray')

plt.show()


x = -df_reg.log_absolute_loss
y = df_reg.acceptability
w = df_reg.PESAIX   # survey weights

# Define bins
bins = np.linspace(x.min(), x.max(), 10)
df_reg['bin'] = pd.cut(x, bins)

# Weighted mean function
def weighted_mean(series, weights):
    return np.average(series, weights=weights)

# Group by bin and calculate weighted stats
grouped = (
    df_reg.groupby('bin')
    .apply(lambda g: pd.Series({
        "mean": weighted_mean(g[y.name], g[w.name]),
        "share": g[w.name].sum() / w.sum() * 100  # percentage
    }))
)

# Optional: drop bins with very small shares
grouped = grouped[grouped['share'] >= 1]

# Get bin centers for plotting
bin_centers = [interval.mid for interval in grouped.index]

# --- Plot ---
fig, ax1 = plt.subplots()

# Line: weighted mean acceptability
ax1.plot(bin_centers, grouped['mean'], marker='o', color='blue')
ax1.set_xlabel('Mobility loss due to the toll (log, €)', fontsize=14)
ax1.set_ylabel('Average acceptability', fontsize=14, color='blue')
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12, colors='blue')

# Bars: share of respondents (%)
ax2 = ax1.twinx()
ax2.bar(
    bin_centers,
    grouped['share'],
    width=(bins[1]-bins[0]),
    alpha=0.3,
    color='gray',
    edgecolor='black',
    align="center"
)
ax2.set_ylabel('Share of respondents (%)', color='black', fontsize=14)
ax2.tick_params(axis='y', labelsize=12, colors='black')

plt.show()



x = df_reg.log_absolute_loss
y = df_reg.acceptable_price

# Define bins
bins = np.linspace(x.min(), x.max(), 10)
df_reg['bin'] = pd.cut(x, bins)

# Compute mean and count per bin
grouped = df_reg.groupby('bin')[y.name].agg(['mean', 'count'])
grouped = grouped[grouped['count'] >= 5]
bin_centers = [interval.mid for interval in grouped.index]

# Line plot of mean
fig, ax1 = plt.subplots()
ax1.plot(bin_centers, grouped['mean'], marker='o', color='blue')
ax1.set_xlabel('Negative experienced impact')
ax1.set_ylabel('Average Acceptability_10', color='blue')

# Secondary axis for counts
ax2 = ax1.twinx()
ax2.bar(bin_centers, grouped['count'], width=(bins[1]-bins[0])*0.8, alpha=0.3, color='gray')
ax2.set_ylabel('Number of observations', color='gray')

plt.show()









############### clean tables

df_reg, meta = pyreadstat.read_sav(path_data + '040023 En moviment pel clima 2025_V01 - còpia.sav')
    
df_reg = gpd.GeoDataFrame(df_reg, geometry = gpd.points_from_xy(df_reg.GEO_X, df_reg.GEO_Y), crs="EPSG:4326")
gdf = gpd.read_file(path_data + "income_loss.geojson")
df_reg = df_reg.to_crs(gdf.crs)
df_reg = gpd.sjoin(df_reg, gdf, predicate="within")

df_reg.rename(columns={'P18': 'acceptable_price'}, inplace=True)
df_reg = df_reg.loc[~np.isnan(df_reg.acceptable_price) & (df_reg.acceptable_price < 15)]


df_reg.rename(columns={'P22_4': 'acceptability'}, inplace=True)
df_reg = df_reg.loc[~np.isnan(df_reg.acceptability) & (df_reg.acceptability < 97)]

df_reg.rename(columns={'P21_3': 'climate_change'}, inplace=True)
df_reg = df_reg.loc[~np.isnan(df_reg.climate_change) & (df_reg.climate_change < 80)]
    
df_reg["log_absolute_loss"] = -np.log(-df_reg["income_loss_absolute"])

df_reg.rename(columns={'P21_2': 'quality_of_life'}, inplace=True)
df_reg = df_reg.loc[~np.isnan(df_reg.quality_of_life) & (df_reg.quality_of_life < 80)]

df_reg.rename(columns={'P21_1': 'congestion'}, inplace=True)
df_reg = df_reg.loc[~np.isnan(df_reg.congestion) & (df_reg.congestion < 80)]

df_reg.rename(columns={'P11': 'declared_impacts'}, inplace=True)
df_reg = df_reg.loc[(df_reg.declared_impacts != 3) & (df_reg.declared_impacts != 4),:]
df_reg.declared_impacts = (df_reg.declared_impacts == 1) * 1
plt.hist(df_reg.declared_impacts)

df_reg["impact_loss"] = df_reg["log_absolute_loss"]  * df_reg["declared_impacts"] 

df_reg = df_reg.loc[df_reg.acceptability < 5,:]

#model 1

X_raw = df_reg[["log_absolute_loss", "climate_change", "quality_of_life", 'congestion']]
acceptability_raw = df_reg["acceptability"].values.reshape(-1, 1)

scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X_raw)
X_scaled = pd.DataFrame(X_scaled, columns=X_raw.columns)

scaler_y = MinMaxScaler()
acceptability_scaled = scaler_y.fit_transform(acceptability_raw).flatten()

X_scaled = sm.add_constant(X_scaled)

model1 = sm.WLS(acceptability_scaled, X_scaled, weights=df_reg['PESAIX']).fit()
print(model1.summary())

#model 2

X_raw = df_reg[["declared_impacts", "climate_change", "quality_of_life", 'congestion']]
acceptability_raw = df_reg["acceptability"].values.reshape(-1, 1)

scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X_raw)
X_scaled = pd.DataFrame(X_scaled, columns=X_raw.columns)

scaler_y = MinMaxScaler()
acceptability_scaled = scaler_y.fit_transform(acceptability_raw).flatten()

X_scaled = sm.add_constant(X_scaled)

model2 = sm.WLS(acceptability_scaled, X_scaled, weights=df_reg['PESAIX']).fit()
print(model2.summary())


#model3

X_raw = df_reg[["log_absolute_loss", "climate_change", "quality_of_life", 'congestion']]
acceptable_price_raw = (df_reg["acceptable_price"].values.reshape(-1, 1))

scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X_raw)
X_scaled = pd.DataFrame(X_scaled, columns=X_raw.columns)

X_scaled = sm.add_constant(X_scaled)

model3 = sm.WLS(acceptable_price_raw, X_scaled, weights=df_reg['PESAIX']).fit()
print(model3.summary())

#model4

X_raw = df_reg[["declared_impacts", "climate_change", "quality_of_life", 'congestion']]
acceptable_price_raw = (df_reg["acceptable_price"].values.reshape(-1, 1))

scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X_raw)
X_scaled = pd.DataFrame(X_scaled, columns=X_raw.columns)

X_scaled = sm.add_constant(X_scaled)

model4 = sm.WLS(acceptable_price_raw, X_scaled, weights=df_reg['PESAIX']).fit()
print(model4.summary())





models = [model1, model2, model3, model4]  # your fitted models
stargazer = Stargazer(models)

# Make LaTeX table
latex_table = stargazer.render_latex()

print(latex_table)
