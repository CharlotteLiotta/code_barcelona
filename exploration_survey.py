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
os.environ["R5_JAR"] = "C:/Users/1738037/AppData/Local/miniforge3/envs/r5py/Lib/site-packages/r5py/data/r5-v6.8-all.jar"
jpype.startJVM(classpath=[os.environ["R5_JAR"]])
import datetime
import warnings
import pickle

from functions import *
from import_data import * # type: ignore
from calibration import * # type: ignore
from model import * # type: ignore
from plotting_tools import * # type: ignore
from import_transport import *
from policy_support import *

path_data = "../data_barcelona/"

gdf = gpd.read_file(path_data + "experienced_impact.geojson")

#Import df
df, meta = pyreadstat.read_sav(path_data + '040023 En moviment pel clima 2025_V01 - còpia.sav')
print(df.head())       # preview data
print(meta.column_names)  # variable names
print(meta.column_labels)


### ACCEPTABILITY

#acceptability
plt.hist(df.P15) #1=in favor, 2=against
plt.hist(df.P22_4.loc[df.P22_4 < 97])

#willingness to pay
plt.hist(df.P17[df.P17 < 80]) #price acceptable
plt.hist(df.P18[df.P18 < 20]) #price max

### ADRESS

df = gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df.GEO_X, df.GEO_Y), crs="EPSG:4326")
df.plot()

df = df.to_crs(gdf.crs)
df.loc[df.P22_4 > 96, "P22_4"] = np.nan
df_reg = gpd.sjoin(df, gdf, predicate="within")
averages = df_reg.groupby("index_right")["P22_4"].agg(lambda x: np.nanmean(x))
gdf["avg_acceptability"] = gdf.index.map(averages)
plot_with_missing(gdf, gdf["avg_acceptability"])

### PERCEIVED IMPACTS

#effectiveness
plt.hist(df.P21_1[df.P21_1 < 80]) #traffic
plt.hist(df.P21_2[df.P21_2 < 80]) #local pollution and quality of life
plt.hist(df.P21_3[df.P21_3 < 80]) #climate change

#others
plt.hist(df.P23[df.P23 < 80]) #institutional trust
plt.hist(df.P26_5[df.P26_5 < 80]) #ecological catastroph
plt.hist(df.P27_1[df.P27_1 < 80]) #eco-anxiety
plt.hist(df.P35[df.P35 < 80]) #ideology
plt.hist(df.P30[df.P30 < 6]) #education
plt.hist(df.P01) #gender
plt.hist(df.P02) #age

#well-being/perceived (P11, P12, P13, P14)
plt.hist(df.P11) #1=went to the zone in private vehicle (filtered)
plt.hist(df.P12) #1=everyday, 2=almost every day

df_reg = gpd.sjoin(df, gdf, predicate="within")
df_reg = df_reg.loc[:,["P15", "P22_4", "P21_1", "P21_2", "P21_3", "P23", "P26_3", "P27_1", 'P35', "P30", "P01", "P02", "P11", "P12", "experienced_impact", "transport_mode"]]
df_reg.columns = ["Acceptability", "Acceptability_10", "Traffic", "Quality_life", "Climate_change", "Trust", "Pro_envt", "Eco_anxiety", "Ideology", "Education", "Gender", "Age", "Wellbeing", "Wellbeing2", "experienced_impact", "transport_mode"]

df_reg = df_reg.loc[(df_reg.Acceptability_10 < 97) &
                    (df_reg.Traffic < 80) &
                    (df_reg.Quality_life < 80) &
                    (df_reg.Climate_change < 80) &
                    (df_reg.Trust < 80) &
                    (df_reg.Pro_envt < 80) &
                    (df_reg.Eco_anxiety < 80) &
                    (df_reg.Ideology < 80) &
                    (df_reg.Education < 6),:]

df_reg["Man"] = (df_reg["Gender"] == 1) * 1
df_reg["Age"] = 2025 - df_reg["Age"]
df_reg["Impact"] = (df_reg["Wellbeing"] == 1) * 1
#df_reg["Impact"] = ((df_reg["Wellbeing2"] == 1) | (df_reg["Wellbeing2"] == 2)) * 1

#X_raw = df_reg[["Impact", "Traffic", "Quality_life", "Climate_change", "Trust", "Pro_envt", "Eco_anxiety", "Ideology", "Education", "Man", "Age", "transport_mode", "transport_cost"]]
X_raw = df_reg[["experienced_impact", "Climate_change", "Trust", "Education", "Age"]]
y_raw = df_reg["Acceptability_10"].values.reshape(-1, 1)

scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X_raw)
X_scaled = pd.DataFrame(X_scaled, columns=X_raw.columns)


scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y_raw).flatten()

X_scaled = sm.add_constant(X_scaled)


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


pca = PCA()
X_pca = pca.fit_transform(X_raw)

# Explained variance
explained_var = pca.explained_variance_ratio_

# Plot cumulative explained variance
plt.plot(np.cumsum(explained_var), marker='o')
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()

# Correlations between original variables and principal components
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(len(pca.components_))],
    index=X_raw.columns
)

# Visualize correlation heatmap of variables with first PCs
fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.matshow(loadings.iloc[:, :8], cmap="coolwarm", vmin=-1, vmax=1)
plt.xticks(range(8), loadings.columns[:8], rotation=45, ha="left")
plt.yticks(range(len(loadings.index)), loadings.index)
plt.colorbar(cax, ax=ax)
plt.title("Loadings of variables on first 5 PCs", pad=20)
plt.show()


model_statsmodel = sm.OLS(y_scaled, X_scaled).fit()
print(model_statsmodel.summary())

#pro_envt and ideology?

















x = -df_reg.experienced_impact
y = df_reg.Acceptability_10

# Define bins
bins = np.linspace(x.min(), x.max(), 20)
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


