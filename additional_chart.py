import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel("../code_barcelona/simulation_results.xlsx")

# Clean
df = df.dropna(subset=['Unnamed: 0'])

custom_order = [
    "baseline",
    "exemption_residents",
    "exemption_low_income",
    "exemption_trips_inside_zone",
    "improvement_public_transport",
    "discount_public_transport",
    "increasing_knowledge"
]

df["scenario"] = pd.Categorical(df["scenario"], categories=custom_order, ordered=True)

df = df.sort_values("scenario")

n = 7
# Two-column layout
cols = 2
rows = (n + 1) // 2
plt.rcParams['font.size'] = 14

plot_order = [
    "tax_level",
    "emission_change",
    "change_qol_in_zone",
    "change_qol_out_zone",
    "utility_change_low",
    "utility_change_med",
    "utility_change_high"
]

custom_titles = {
    'tax_level': 'Toll per day (€)',
    'emission_change': 'Transport emissions',
    'change_qol_in_zone': 'Pollution in the tax zone (%)',
    'change_qol_out_zone': 'Pollution out of the tax zone (%)',
    'utility_change_low': 'Utility (low-income, %)',
    'utility_change_med': 'Utility (middle-income, %)',
    'utility_change_high': 'Utility (high-income, %)',
}

fig, axes = plt.subplots(rows, cols, figsize=(13, 3.5*rows), constrained_layout=True, sharex=True)
axes = axes.flatten()

colors = ['grey'] * 7  # default color
colors[0] = 'red'  # change the second bar

for ax, var in zip(axes, plot_order):
    sub = df[df['Unnamed: 0'] == var]
    ax.bar(sub['scenario'], sub['t20'], color=colors, width=0.5)
    ax.set_title(custom_titles.get(var, var))
    ax.tick_params(axis='x', rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")

for ax in axes:
    if len(ax.get_xticklabels()) > 0:
        ax.tick_params(axis="x", labelrotation=45)
        ax.label_outer()
        for label in ax.get_xticklabels():
            label.set_ha("right")
# Hide unused axes
#for ax in axes[len(variables):]:
#    ax.axis('off')
# 2-column layout, flattened into 'axes'

# Hide the unused subplot but keep its space
axes[7].set_visible(False)

# Apply rotation everywhere
for ax in axes[:7]:
    ax.tick_params(axis="x", labelrotation=45)
    ax.label_outer()# Force x-labels to appear on subplot #6 (index 5)

axes[5].tick_params(axis="x", labelbottom=True)
axes[5].label_outer()
# Add room for overflow
plt.subplots_adjust(bottom=0.05, hspace=0.01)
plt.tight_layout()
plt.show()