import numpy as np
import pandas as pd

path_data = "../data_barcelona/"

emf = pd.read_csv(path_data + "emef/Microdades Ús públic_EMEF2023_Desplaçaments (2).csv", sep = ";")
emf = emf.loc[(emf.TIPOL == 1) & (emf.V03A == 3),:]
emf = emf.loc[:,["ID", "DISTANCIA_ORTO_REC_R1", "COM_O2", "COM_D2", "V03G_R3"]]
emf = emf.groupby("ID").first()


# Convert to int (ignore errors) and count values
counts = emf.DISTANCIA_ORTO_REC_R1.astype(int, errors="ignore").value_counts().sort_index()

# Plot sorted bar chart
plt.bar(counts.index, counts.values)
plt.xlabel("DISTANCIA_ORTO_REC_R1")
plt.ylabel("Frequency")
plt.title("Sorted bar plot of DISTANCIA_ORTO_REC_R1")
plt.show()

# Convert to int (ignore errors) and count values
emf = emf.loc[emf.COM_O2 == 3]
counts = emf.DISTANCIA_ORTO_REC_R1.astype(int, errors="ignore").value_counts().sort_index()

# Plot sorted bar chart
plt.bar(counts.index, counts.values)
plt.xlabel("DISTANCIA_ORTO_REC_R1")
plt.ylabel("Frequency")
plt.title("Sorted bar plot of DISTANCIA_ORTO_REC_R1")
plt.show()




emf["indic"] = 1
emf.loc[:,["indic", "COM_D2"]].groupby("COM_D2").sum()