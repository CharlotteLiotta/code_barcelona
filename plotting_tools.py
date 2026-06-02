import numpy as np # type: ignore
import matplotlib as mpl
import matplotlib.pyplot as plt # type: ignore
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
from matplotlib.patches import Patch
import matplotlib.lines as mlines

def plot_spatial_price(gdf, values, include_missing = True):
    # --- prepare values ---
    gdf_proj = gdf.to_crs(epsg=32632).copy()   # keep projection if needed
    gdf_proj["value"] = values

    # Dissolve by municipality to get one polygon per municipality
    muni_gdf = gdf_proj.dissolve(by='NMUN', as_index=False)
    muni_gdf['centroid'] = muni_gdf.geometry.centroid
    muni_gdf['x'] = muni_gdf.centroid.x
    muni_gdf['y'] = muni_gdf.centroid.y

    # plotting
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    cmap_name = "RdYlGn"
    vmin, vmax = gdf_proj["value"].min(), gdf_proj["value"].max()
    #vmin = 0
    #vmax = 6
    
    missing = gdf_proj[gdf_proj["value"].isna()]
    present = gdf_proj[gdf_proj["value"].notna()]

    # First: plot missing polygons in grey
    missing.plot(color="lightgrey", edgecolor="white",
             linewidth=0.2, ax=ax, label="Missing data")
    present.plot(column="value",
                  cmap=cmap_name,
                  vmin=vmin, vmax=vmax,
                  linewidth=0, edgecolor="grey",
                  ax=ax)

    ax.set_axis_off()

    # improve rendering
    for coll in ax.collections:
        coll.set_antialiased(False)
        coll.set_alpha(0.7)

    for _, row in muni_gdf.loc[muni_gdf["NMUN"].isin(["Badalona", "Castelldefels", "Castellbisbal", "Sant Cugat del Vallès"]),:].iterrows():
        ax.text(row.x, row.y, row['NMUN'], fontsize=9, fontweight='bold', ha='center', va='center', color='black')
    
    
    city_border = muni_gdf[muni_gdf.ID.str[:5].isin(["08019", "08101", "08194"])]
    city_border.boundary.plot(ax=ax, color='black', linewidth=2, label = "Toll area")
    
    # continuous colorbar (no title)
    sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    ticks = np.linspace(vmin, vmax, 5)
    cbar.set_ticks(ticks)
    cbar.ax.set_yticklabels([f"{t:.1f}" for t in ticks], fontsize=14)  # show 1 decimal + %

    if include_missing == True:
        missing_patch = mpatches.Patch(facecolor="lightgrey", edgecolor="white", label="Simulated population = 0")
    
    toll_patch = mpatches.Patch(facecolor="none", edgecolor="black", linewidth=2, label="Toll area")

    if include_missing == True:
        ax.legend(handles=[missing_patch, toll_patch], loc="lower right")
    else:
        ax.legend(handles=[toll_patch], loc="lower right")

def plot_spatial_price_discrete(gdf, values, include_missing=True):
    # --- prepare values ---
    gdf_proj = gdf.to_crs(epsg=32632).copy()
    gdf_proj["value"] = values

    # Dissolve by municipality
    muni_gdf = gdf_proj.dissolve(by='NMUN', as_index=False)
    muni_gdf['centroid'] = muni_gdf.geometry.centroid
    muni_gdf['x'] = muni_gdf.centroid.x
    muni_gdf['y'] = muni_gdf.centroid.y

    # --- discrete color setup ---
    bounds = [0, 1, 2, 3, 4, 5, gdf_proj["value"].max() + 0.01]
    labels = ["<1", "1–2", "2–3", "3–4", "4–5", ">5"]
    n_bins = len(labels)

    base_cmap = plt.cm.get_cmap("RdYlGn", n_bins)
    colors = [base_cmap(i) for i in range(n_bins)]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bounds, ncolors=n_bins)

    # --- plot ---
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    missing = gdf_proj[gdf_proj["value"].isna()]
    present = gdf_proj[gdf_proj["value"].notna()]

    missing.plot(color="lightgrey", edgecolor="white", linewidth=0.2, ax=ax)
    present.plot(column="value", cmap=cmap, norm=norm,
                 linewidth=0, edgecolor="grey", ax=ax)

    ax.set_axis_off()

    for coll in ax.collections:
        coll.set_antialiased(False)
        coll.set_alpha(0.7)

    for _, row in muni_gdf.loc[muni_gdf["NMUN"].isin(
            ["Badalona", "Castelldefels", "Castellbisbal", "Sant Cugat del Vallès"]), :].iterrows():
        ax.text(row.x, row.y, row['NMUN'], fontsize=9, fontweight='bold',
                ha='center', va='center', color='black')

    city_border = muni_gdf[muni_gdf.ID.str[:5].isin(["08019", "08101", "08194"])]
    city_border.boundary.plot(ax=ax, color='black', linewidth=2)

    # --- legend: one patch per bin + extras ---
    legend_handles = []

    # Color bins (note: apply same alpha=0.7 as the map polygons)
    for label, color in zip(labels, colors):
        legend_handles.append(
            mpatches.Patch(facecolor=(*color[:3], 0.7), edgecolor="grey",
                           linewidth=0.5, label=label)
        )

    # Separator: toll area and missing
    legend_handles.append(
        mpatches.Patch(facecolor="none", edgecolor="black", linewidth=2, label="Toll area")
    )
    if include_missing:
        legend_handles.append(
            mpatches.Patch(facecolor="lightgrey", edgecolor="white",
                           label="Simulated \npopulation = 0")
        )

    ax.legend(handles=legend_handles, loc="lower right", fontsize=9,
              framealpha=0.8, edgecolor="grey",
                bbox_to_anchor=(1.05, 0))
    
def main_plot(save_tax, emission_change, change_qol_in_zone, change_qol_out_zone, utility_change_low, utility_change_med, utility_change_high):
    
    # --- Global formatting for academic figures ---
    mpl.rcParams['font.size'] = 10
    mpl.rcParams['axes.labelsize'] = 10
    mpl.rcParams['xtick.labelsize'] = 9
    mpl.rcParams['ytick.labelsize'] = 9
    mpl.rcParams['legend.fontsize'] = 9

    _, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True, constrained_layout=True)
    years = range(len(save_tax))

    # --- Panel 1 ---
    axes[0].plot(years, 2 * save_tax, color="black", linewidth=1.5)
    axes[0].set_ylabel('Toll per day (€)')

    # --- Panel 2 ---
    axes[1].plot(years, utility_change_low,  label="Low-income",    color="orange", linewidth=1.5)
    axes[1].plot(years, utility_change_med,  label="Middle-income", color="orangered",  linewidth=1.5)
    axes[1].plot(years, utility_change_high, label="High-income",   color="maroon",    linewidth=1.5)
    axes[1].set_ylabel("Median utility variation (%)")
    axes[1].legend(frameon=False, loc="lower right", fontsize=9)

    # --- Panel 3 ---
    axes[2].plot(years, emission_change,        label="Transport emissions", color="green", linewidth=1.5)
    axes[2].plot(years, change_qol_in_zone,     label="Pollution inside the tax zone",         color="navy",   linewidth=1.5)
    axes[2].plot(years, change_qol_out_zone,    label="Pollution outside of the tax zone",    color="cyan", linewidth=1.5)
    axes[2].set_ylabel("Mean variation (%)")
    axes[2].legend(frameon=False, loc="center right", fontsize=9)
    axes[-1].set_xlabel("Year")
    axes[-1].set_xticks(list(years)[::2])   # every 2 years

    #fig.tight_layout()
    plt.show()

def plot_base_map(gdf):
    
    gdf_proj = gdf.to_crs(epsg=32632).copy()  
    # Dissolve by municipality to get one polygon per municipality
    muni_gdf = gdf_proj.dissolve(by='NMUN', as_index=False)
    muni_gdf['centroid'] = muni_gdf.geometry.centroid
    muni_gdf['x'] = muni_gdf.centroid.x
    muni_gdf['y'] = muni_gdf.centroid.y


    fig, ax = plt.subplots(figsize=(8, 8))

    # Census tracts
    gdf_proj.plot(
        ax=ax, facecolor="white", edgecolor="grey", linewidth=0.6, label="Census tracts"
    )

    # Municipal boundaries
    muni_gdf.boundary.plot(
        ax=ax, edgecolor="black", linewidth=0.8, label="Municipal boundaries"
    )

    # Congestion-pricing zone
    city_border = muni_gdf[muni_gdf.ID.str[:5].isin(["08019", "08101", "08194"])]
    city_border.dissolve().plot(
        ax=ax, facecolor="none", edgecolor="red", linewidth=2.2, label="Toll area"
    )

    # Municipality labels
    for _, row in muni_gdf.loc[muni_gdf["NMUN"].isin(["Badalona", "Barcelona", "Castelldefels", "Castellbisbal", "Sant Cugat del Vallès"]),:].iterrows(): ax.text(row.x, row.y, row['NMUN'], fontsize=9, fontweight='bold', ha='center', va='center', color='black')

    # ----- Legend -----
    # Handles for each layer
    tracts_handle = mpatches.Patch(facecolor="white", edgecolor="grey", label="Census tracts")
    muni_handle   = mlines.Line2D([], [], color="black", linewidth=0.8, label="Municipal boundaries")
    toll_handle   = mlines.Line2D([], [], color="red", linewidth=1.2, label="Toll area")

    ax.legend(handles=[tracts_handle, muni_handle, toll_handle], loc="lower right")

    # Aesthetics
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

def plot_base_map_with_land_cover(gdf, land_cover_plot, color_dict):
    # Ensure same CRS
    gdf_proj = gdf.to_crs(epsg=32632).copy()
    land_cover_proj = land_cover_plot.to_crs(gdf_proj.crs)

    # Dissolve by municipality
    muni_gdf = gdf_proj.dissolve(by='NMUN', as_index=False)
    muni_gdf['centroid'] = muni_gdf.geometry.centroid
    muni_gdf['x'] = muni_gdf.centroid.x
    muni_gdf['y'] = muni_gdf.centroid.y

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(right=0.75)
    # --- Land cover first, with some transparency ---
    land_cover_proj.plot(
        ax=ax,
        column="recat",
        categorical=True,
        color=land_cover_proj["recat"].map(color_dict),
        legend=True,
        #alpha=0.6,   # makes boundaries visible through polygons
        linewidth=0
    )

    # Census tracts - slightly darker and thinner lines
    gdf_proj.boundary.plot(
        ax=ax,
        edgecolor="black",
        linewidth=0.1,
        label="Census tracts"
    )

    # Municipal boundaries
    muni_gdf.boundary.plot(
        ax=ax, edgecolor="black", linewidth=0.8, label="Municipal boundaries"
    )

    # Toll area - use a distinct color to avoid clashing with land cover
    city_border = muni_gdf[muni_gdf.ID.str[:5].isin(["08019", "08101", "08194"])]
    city_border.dissolve().plot(
        ax=ax,
        facecolor="none",
        edgecolor="red",
        linewidth=3,
        label="Toll area"
    )

    # Municipality labels
    for _, row in muni_gdf.loc[muni_gdf["NMUN"].isin([
        "Barcelona",
        "Sant Adrià de Besòs",
        "Hospitalet de Llobregat, L'"])].iterrows():

        label = row['NMUN']

        # Fix names and line breaks
        if label == "Hospitalet de Llobregat, L'":
            label = "L'Hospitalet\nde Llobregat"

        if label == "Sant Adrià de Besòs":
            label = "Sant Adrià\nde Besòs"

        ax.text(
            row.x,
            row.y,
            label,
            fontsize=9,
            fontweight='bold',
            ha='center',
            va='center',
            color='black'
        )
        
    for _, row in muni_gdf.loc[muni_gdf["NMUN"].isin(
        ["Begues"]), :].iterrows():
        ax.text(row.x, row.y, "Garraf massif", fontsize=9, fontweight='bold',
                ha='center', va='center', color='black')
        
    for _, row in muni_gdf.loc[muni_gdf["NMUN"].isin(
        ["Sant Cugat del Vallès"]), :].iterrows():
        ax.text(row.x + 2000, row.y - 1500, "Collserola massif", fontsize=9, fontweight='bold',
                ha='center', va='center', color='black')
        
    

    # --- Legend: combine land-cover legend with manual handles ---
    # First get land-cover handles
    # --- Land-cover handles ---
    land_cover_handles = [
        mpatches.Patch(color=color_dict[cat], label=cat) for cat in color_dict
]

    # --- Boundary handles ---
    tracts_handle = mlines.Line2D([], [], color="black", linewidth=0.4, label="Census tracts")
    muni_handle   = mlines.Line2D([], [], color="black", linewidth=0.8, label="Municipal boundaries")
    toll_handle   = mlines.Line2D([], [], color="red", linewidth=3.0, label="Toll area")
    boundary_handles = [tracts_handle, muni_handle, toll_handle]

    # --- Create sublegends ---
    legend1 = ax.legend(handles=land_cover_handles, title="Land cover", bbox_to_anchor=(1.4, 0.26))
    legend2 = ax.legend(handles=boundary_handles, title="Administrative boundaries", bbox_to_anchor=(1.4, 0.45))

    # Make the first legend title bold
    legend1.get_title().set_fontweight('bold')
    legend2.get_title().set_fontweight('bold')

    # Add first legend back to the axes
    ax.add_artist(legend1)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

def plot_acceptability_price(df_reg):
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

def plot_hist_survey(df_reg, var, xlabel):

    plt.figure(figsize=(8, 5))
    bins = np.arange(-0.5, 11.5, 1)
    if var == "acceptable_price": 
        plt.hist(
        df_reg[var] * 2,
        weights=(df_reg["PESAIX"] / df_reg["PESAIX"].sum()) * 100,
        bins=bins,
        color='#1f77b4',
        alpha=0.85,
        edgecolor='black'  # clearer bar separation
        )
    else:
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

def plot_mobility_loss(x,y, df_reg):
    df_reg = df_reg.copy(deep=True)
    label_x = x
    label_y = y
    x = df_reg[x]
    y = df_reg[y]
    w = df_reg.PESAIX   # survey weights

    # Define bins
    bins = np.linspace(x.min(), x.max(), 20)
    df_reg.loc[:, 'bin'] = pd.cut(x, bins)

    # Weighted mean function
    def weighted_mean(series, weights):
        wsum = weights.sum()
        if wsum == 0:
            return np.nan
        return np.average(series, weights=weights)

    # Group by bin and calculate weighted stats
    grouped = (df_reg.groupby('bin', observed = False).apply(lambda g: pd.Series({"mean": weighted_mean(g[y.name], g[w.name]), "share": g[w.name].sum() / w.sum() * 100 # % of total respondents
                                                                }),
            include_groups=False))

    # Drop bins with too few weighted respondents (optional, e.g. <1% share)
    grouped = grouped[grouped['share'] >= 1]

    # Get bin centers for plotting
    bin_centers = [interval.mid for interval in grouped.index]

        # --- Plot ---
    fig, ax1 = plt.subplots()

    # Line plot: weighted mean acceptable toll
    ax1.plot(bin_centers, grouped['mean'], marker='o', color='blue')
    #ax1.set_xlabel(label_x, fontsize=14)
    ax1.set_xlabel("Welfare score", fontsize=14)
    #ax1.set_ylabel(label_y, fontsize=14, color='blue')
    ax1.set_ylabel("Acceptable price", fontsize=14, color='blue')
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
        align="center")
    ax2.set_ylabel('Share of respondents (%)', color='black', fontsize=14)
    ax2.tick_params(axis='y', labelsize=12, colors='black')

    plt.show()

def plot_mode_shares(mode_shares_lvl, income_levels, MAX_YEAR):
    values_0 = [100*mode_shares_lvl[level][0] for level in income_levels]
    values_20 = [100*mode_shares_lvl[level][MAX_YEAR - 1] for level in income_levels]
    x = np.arange(len(income_levels))
    width = 0.35
    plt.figure()
    plt.bar(x - width/2, values_0, width, label="Year 0", color = "#4C72B0")
    plt.bar(x + width/2, values_20, width, label="Year " + str(MAX_YEAR - 1), color = "#DD8452")
    plt.xticks(x, ["Low-income", "Middle-income", "High-income"])
    plt.ylabel("Share of public transport users (%)")
    plt.legend()
    plt.show()
    plt.close()


def plot_vkm(avg_vkm_in_zone_lvl, avg_vkm_out_zone_lvl, MAX_YEAR, income_levels):

    in_0  = [avg_vkm_in_zone_lvl[lvl][0]  for lvl in income_levels]
    in_20 = [avg_vkm_in_zone_lvl[lvl][MAX_YEAR - 1] for lvl in income_levels]
    out_0  = [avg_vkm_out_zone_lvl[lvl][0]  for lvl in income_levels]
    out_20 = [avg_vkm_out_zone_lvl[lvl][MAX_YEAR - 1] for lvl in income_levels]

    x = np.arange(len(income_levels))
    width = 0.35

    plt.figure()
    plt.bar(x - width/2, in_0,  width, label="Inside the tax zone (year 0)",  color="#4C72B0")
    plt.bar(x - width/2, out_0, width, bottom=in_0, label="Outside of the tax zone (year 0)", color="#9ecae9")
    plt.bar(x + width/2, in_20,  width, label="Inside the tax zone (year "+str(MAX_YEAR-1)+")",  color="#DD8452")
    plt.bar(x + width/2, out_20, width, bottom=in_20, label="Outside of the tax zone (year "+str(MAX_YEAR-1)+")", color="#fdd0a2")
    plt.xticks(x,  ["Low-income", "Middle-income", "High-income"])
    plt.ylabel("Average vehicle-km driven")
    plt.legend(loc = "lower right")
    plt.show()
    plt.close()


def plot_living_commuting_pattern(live_and_work_in_toll_lvl, live_out_and_work_out_lvl, live_in_toll_and_work_out_lvl, live_out_and_work_in_toll_lvl, MAX_YEAR, income_levels):

    lw_in_0  = [100 * live_and_work_in_toll_lvl[l][0]  for l in income_levels]
    lw_in_20 = [100 * live_and_work_in_toll_lvl[l][MAX_YEAR - 1] for l in income_levels]

    lw_out_0  = [100 * live_out_and_work_out_lvl[l][0]  for l in income_levels]
    lw_out_20 = [100 * live_out_and_work_out_lvl[l][MAX_YEAR - 1] for l in income_levels]

    in_out_0  = [100 * live_in_toll_and_work_out_lvl[l][0]  for l in income_levels]
    in_out_20 = [100 * live_in_toll_and_work_out_lvl[l][MAX_YEAR - 1] for l in income_levels]

    out_in_0  = [100 * live_out_and_work_in_toll_lvl[l][0]  for l in income_levels]
    out_in_20 = [100 * live_out_and_work_in_toll_lvl[l][MAX_YEAR - 1] for l in income_levels]

    x = np.arange(len(income_levels))
    width = 0.32
    # Muted academic palette
    colors = {
        "lw_in":  "#5B7C99",  # slate blue
        "lw_out": "#8C9A5B",  # muted olive
        "in_out": "#C2A878",  # sand
        "out_in": "#B07A8F"   # dusty rose
    }

    plt.figure()
    fig, ax = plt.subplots(figsize=(10, 6))
    edge_col = "0.3"
    edge_lw = 0.6

    # --- Year 0 ---
    bottom_0 = np.zeros(len(income_levels))
    pos_0 = x - width/2

    for vals, key in zip(
        [lw_in_0, lw_out_0, in_out_0, out_in_0],
        ["lw_in", "lw_out", "in_out", "out_in"]):
        
        ax.bar(pos_0, vals, width,
                bottom=bottom_0,
                color=colors[key],
                edgecolor=edge_col,
                linewidth=edge_lw)
        
        bottom_0 += vals

    # --- Year 20 ---
    bottom_20 = np.zeros(len(income_levels))
    pos_20 = x + width/2

    for vals, key in zip(
        [lw_in_20, lw_out_20, in_out_20, out_in_20],
        ["lw_in", "lw_out", "in_out", "out_in"]):
    
        ax.bar(pos_20, vals, width,
            bottom=bottom_20,
            color=colors[key],
            edgecolor=edge_col,
            linewidth=edge_lw)
        bottom_20 += vals

    # Year labels above bars
    offset = 0.02
    for i in range(len(income_levels)):
        ax.text(pos_0[i],  bottom_0[i]  + offset, "Year 0",  ha="center", va="bottom", fontsize=14)
        ax.text(pos_20[i], bottom_20[i] + offset, "Year " + str(MAX_YEAR - 1), ha="center", va="bottom", fontsize=14)

    ax.set_xticks(x,  ["Low-income", "Middle-income", "High-income"])
    ax.set_ylabel("Share (%)", fontsize=14)
    ax.set_ylim(0, max(max(bottom_0), max(bottom_20)) + 0.08)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    legend_elements = [
        Patch(facecolor=colors["lw_in"],  edgecolor=edge_col, label="Live and work inside the tax zone"),
        Patch(facecolor=colors["lw_out"], edgecolor=edge_col, label="Live and work outside the tax zone"),
        Patch(facecolor=colors["in_out"], edgecolor=edge_col, label="Live inside and work outside the zone"),
        Patch(facecolor=colors["out_in"], edgecolor=edge_col, label="Live outside and work inside the zone"),
    ]

    ax.legend(
        handles=legend_elements,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),  # 0.5 = center horizontally, -0.12 = below plot
        ncol=2, fontsize=14  # one column per category
    )
    fig.tight_layout()
    plt.show()
    plt.close()


def compute_change_costs(avg_tcost_lvl, avg_rent_lvl, avg_dsize_lvl, MAX_YEAR, income_levels):

    indicators = ["Average generalized travel cost", "Average rent per m2", "Average dwelling size"]

    # Compute % change from year 0 to 20
    change_tcost = [
        (avg_tcost_lvl[l][MAX_YEAR-1] - avg_tcost_lvl[l][0]) / avg_tcost_lvl[l][0] * 100
        for l in income_levels
    ]

    change_rent = [
        (avg_rent_lvl[l][MAX_YEAR-1] - avg_rent_lvl[l][0]) / avg_rent_lvl[l][0] * 100
        for l in income_levels
    ]

    change_dsize = [
        (avg_dsize_lvl[l][MAX_YEAR-1] - avg_dsize_lvl[l][0]) / avg_dsize_lvl[l][0] * 100
        for l in income_levels
    ]

    changes = np.array([change_tcost, change_rent, change_dsize])

    x = np.arange(len(income_levels))
    width = 0.25

    colors = ["#4C72B0", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(8,5))

    for i in range(len(indicators)):
        ax.bar(
            x + (i-1)*width,
            changes[i],
            width,
            color=colors[i],
            edgecolor="0.3",
            label=indicators[i]
        )

    ax.set_xticks(x)
    ax.set_xticklabels(income_levels, fontsize=13)

    ax.set_ylabel("Change between year 0 and " + str(MAX_YEAR - 1) + " (%)", fontsize=13)

    ax.tick_params(axis='y', labelsize=13)

    ax.legend(frameon=False, fontsize=13)

    fig.tight_layout()
    plt.show()
    plt.close()

def plot_employment(gdf, employment_centers, var):
    base = gdf.plot(color='lightgrey', edgecolor='white', linewidth=0.3, figsize=(10, 10))

    # Overlay: points with size proportional to a column (e.g., 'population')
    employment_centers.plot(
        ax=base,
        markersize = var,
        #markersize=employment_centers['employment'] * 0.001,  # adjust scale_factor
        #markersize=employment_centers.merge(employed_results, left_on = "cluster", right_on = "to_id")["weighted_employed"] * 0.001,  # adjust scale_factor
        #markersize= ARRAY_WAGE* 0.5,  # adjust scale_factor
        color='red',
        alpha=0.6)
    
    # Annotate each subcenter
    for idx, row in employment_centers.iterrows():
        base.annotate(
            str(row['cluster']+1),
            xy=(row.geometry.x, row.geometry.y),
            xytext=(3, 3),  # offset
            textcoords="offset points",
            fontsize=16, 
            color='black',
            fontweight='bold'
    )
    
    plt.axis("off")

    plt.show()


def plot_with_missing(gdf, var):
    """ Map a variable with missing values in grey """

    gdf.plot(
    column=var,
    legend=True,
    missing_kwds={
        "color": "lightgrey",
        "label": "Missing data"
    })

def map_calibration(gdf, var1, var2, title):
    """ Compare data and calibration - Map """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    cmap = "viridis"
    # Plot var1
    gdf.plot(var1, cmap=cmap, ax=ax1, legend=True)
    ax1.set_title("Calibration")

    # Plot var2
    gdf.plot(var2, cmap=cmap, ax=ax2, legend=True)
    ax2.set_title("Data")

    for ax in [ax1, ax2]:
        ax.set_axis_off()
    plt.title(title)
    plt.tight_layout()
    plt.show()

def scatter_calibration(gdf, var1, var2, title):
    """ Compare data and calibration - Scatter plot """

    plt.scatter(gdf["distance_center"], var1, label = "Calibration", s = 0.5)
    plt.scatter(gdf["distance_center"], var2, label = "Data", s = 0.5)
    plt.ylim(0,np.nanmax(var2))
    plt.title(title)
    plt.legend()

def plot_density(gdf):
    gdf["density"] = gdf["pop"] / gdf["area"]
    gdf["distance_bin"] = gdf["distance_center"].round().astype(int)
    agg = gdf.groupby("distance_bin").agg({"pop": "sum", "area": "sum"}).reset_index()
    agg["mean_density"] = agg["pop"] / agg["area"]
    plt.scatter(gdf["distance_center"], gdf["density"], s=1, alpha=0.3, label="Données individuelles")
    plt.plot(agg["distance_bin"], agg["mean_density"], color='red', linewidth=2, label="Densité moyenne par km")
    plt.xlabel("Distance au centre-ville (km)")
    plt.ylabel("Densité de population (hab/km²)")
    plt.legend()
    plt.show()

def compare_var(gdf, n):
    gdf["n"] = n
    gdf["density_pop"] = gdf["pop"] / gdf["area"]
    gdf["density_n"] = gdf["n"] / gdf["area"]

    # Bin by distance
    gdf["distance_bin"] = gdf["distance_center"].round().astype(int)

    # Aggregate by bin for both population sources
    agg = gdf.groupby("distance_bin").agg(
        pop=("pop", "sum"),
        n=("n", "sum"),
        area=("area", "sum")
        ).reset_index()

    # Compute mean densities
    agg["mean_density_pop"] = agg["pop"] / agg["area"]
    agg["mean_density_n"] = agg["n"] / agg["area"]

    # Plot
    plt.figure(figsize=(8, 5))

    # Individual points (optional, can be noisy)
    plt.scatter(gdf["distance_center"], gdf["density_pop"], color='red', s=1, alpha=0.3, label="Densité individuelle (pop)")
    plt.scatter(gdf["distance_center"], gdf["density_n"], s=1, alpha=0.3, label="Densité individuelle (n)", color='blue')

    # Aggregated lines
    plt.plot(agg["distance_bin"], agg["mean_density_pop"], color='red', linewidth=2, label="Densité moyenne (pop)")
    plt.plot(agg["distance_bin"], agg["mean_density_n"], color='blue', linewidth=2, label="Densité moyenne (n)")

    plt.xlabel("Distance au centre-ville (km)")
    plt.ylabel("Densité de population (hab/km²)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return agg

def compare_rent_or_size(gdf, var_data, var_simul, weighting, yaxis):
    gdf["simul"] = var_simul

    # Bin by distance
    gdf["distance_bin"] = gdf["distance_center"].round().astype(int)

    if weighting == 0:
        # Aggregate by bin for both population sources
        agg = gdf.groupby("distance_bin").agg(
            data=(var_data, "mean"),
            simul=("simul", "mean"),
            ).reset_index()
        
    elif weighting == 1:
        gdf["weighted_data"] = gdf[var_data] * gdf["pop"]
        gdf["weighted_simul"] = gdf["simul"] * gdf["pop"]

        agg = gdf.groupby("distance_bin").agg(
            data=("weighted_data", "sum"),
            simul=("weighted_simul", "sum"),
            pop=("pop", "sum")
            ).reset_index()

        # Compute mean densities
        agg["data"] = agg["data"] / agg["pop"]
        agg["simul"] = agg["simul"] / agg["pop"]

    # Plot
    plt.figure(figsize=(8, 5))

    # Individual points (optional, can be noisy)
    plt.scatter(gdf["distance_center"], gdf[var_data], s=1, alpha=0.3, label="Data", color='red')
    plt.scatter(gdf["distance_center"], gdf["simul"], s=1, alpha=0.3, label="Simulation", color='blue')

    # Aggregated lines
    plt.plot(agg["distance_bin"], agg["data"], color='red', linewidth=2, label="Data")
    plt.plot(agg["distance_bin"], agg["simul"], color='blue', linewidth=2, label="Simulation")

    plt.xlabel("Distance to city center (km)")
    plt.ylabel(yaxis)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return agg

def compute_weighted_mean_opinions(var, opinion_distance_matrix, N):
    weighted_mean_opinion = np.zeros(opinion_distance_matrix.shape[1])
    weighted_mean_opinion[np.nansum(opinion_distance_matrix, 0) > 0] = np.nansum(opinion_distance_matrix * var.reshape(N, 1), 0)[np.nansum(opinion_distance_matrix, 0) > 0] / np.nansum(opinion_distance_matrix, 0)[np.nansum(opinion_distance_matrix, 0) > 0]
    weighted_mean_opinion[np.nansum(opinion_distance_matrix, 0) == 0] = np.nan
    return weighted_mean_opinion

def plot_tax_suppport(save_tax, save_median_support):

    from matplotlib.ticker import FormatStrFormatter
    fig, ax1 = plt.subplots(figsize=(8, 6))  # make figure wider

    # Primary axis: tax
    color = 'tab:red'
    ax1.set_xlabel('Time (year)', fontsize=14)
    ax1.set_ylabel('Toll per entry (€)', color=color, fontsize=14)
    ax1.plot(save_tax[1:], color=color, linewidth=1.5)
    ax1.tick_params(axis='y', labelcolor=color)

    # Secondary axis: median support
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Median acceptability (%)', color=color, fontsize=14)
    ax2.plot(save_median_support[1:] * 100, color=color, linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)

    # Format y-axis to show 1 decimal place
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    plt.tight_layout()
    plt.show()

def plot_scores(save_score_emissions, save_score_qol, save_score_welfare):

    colors = plt.get_cmap("tab10").colors  
    plt.figure(figsize=(8, 6))

    # Plot lines (all same style, distinct colors)
    plt.plot(save_score_emissions[1:],#/save_score_emissions[1],
         label="Emissions", linewidth=2, color=colors[0])

    plt.plot(np.nanmean(save_score_qol[:,1:], 0), #/np.nanmean(save_score_qol[:,1], 0),
         label="Local pollution", linewidth=2, color=colors[1])

    #plt.plot(np.nanmean(save_score_congestion[:,1:], 0), #/np.nanmean(save_score_congestion[:,1], 0),
    #     label="Congestion", linewidth=2, color=colors[2])

    plt.plot(np.nanmedian(save_score_welfare[:,1:], 0), #/np.nanmean(save_score_welfare[:,1], 0),
         label="Utility", linewidth=2, color=colors[3])

    # Labels and title
    plt.xlabel("Time (years)", fontsize=14)
    plt.ylabel("Normalized score (base year = 1)", fontsize=14)

    # Legend
    plt.legend(fontsize=14, loc="best")


    # Tight layout
    plt.tight_layout()
    plt.show()


def plot_spatial_opinions(gdf, values):
    # --- prepare values ---
    gdf_proj = gdf.to_crs(epsg=32632).copy()   # keep projection if needed
    gdf_proj["value"] = values

    # Dissolve by municipality to get one polygon per municipality
    muni_gdf = gdf_proj.dissolve(by='NMUN', as_index=False)
    muni_gdf['centroid'] = muni_gdf.geometry.centroid
    muni_gdf['x'] = muni_gdf.centroid.x
    muni_gdf['y'] = muni_gdf.centroid.y

    # plotting
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    cmap_name = "cividis"
    vmin, vmax = gdf_proj["value"].min(), gdf_proj["value"].max()

    gdf_proj.plot(column="value",
                  cmap=cmap_name,
                  vmin=vmin, vmax=vmax,
                  linewidth=0.2, edgecolor="grey",
                  ax=ax)

    ax.set_axis_off()

    # improve rendering
    for coll in ax.collections:
        coll.set_antialiased(False)

    for _, row in muni_gdf.loc[muni_gdf["NMUN"].isin(["Badalona", "Castelldefels", "Castellbisbal", "Sant Cugat del Vallès"]),:].iterrows():
        ax.text(row.x, row.y, row['NMUN'], fontsize=9, fontweight='bold', ha='center', va='center', color='black')
    
    
    city_border = muni_gdf[muni_gdf.ID.str[:5].isin(["08019", "08101", "08194"])]
    city_border.boundary.plot(ax=ax, color='black', linewidth=2, label = "Toll area")
    

    # continuous colorbar (no title)
    sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    ticks = np.linspace(vmin, vmax, 5)
    cbar.set_ticks(ticks)
    cbar.ax.set_yticklabels([f"{t:.1f}%" for t in ticks], fontsize=14)  # show 1 decimal + %
    ax.legend()
    # title
    ax.set_title("Average opinions (year 19)", fontsize=14)

def plot_change_pop_line(gdf, save_population):
    gdf = gdf.copy()
    gdf["population0"] = save_population[:, 0]
    gdf["population19"] = save_population[:, 19]
    bins = np.arange(0, gdf["distance_center"].max() + 2, 2)
    gdf["distance_bin"] = pd.cut(gdf["distance_center"], bins=bins)
    pop_by_bin0 = gdf.groupby("distance_bin", observed=False)["population0"].sum()
    pop_by_bin19 = gdf.groupby("distance_bin", observed=False)["population19"].sum()
    pop_by_bin0.plot(kind="line", figsize=(10,5), label = "0")
    pop_by_bin19.plot(kind="line", figsize=(10,5), label = "19")
    plt.legend()
    plt.ylabel("Population")
    plt.xlabel("Distance to city center (km)")
    plt.title("Population by distance bins")
    plt.show()

def plot_change_population_custom(MAX_YEAR, gdf, save_population,
                                  bins=[-100, -50, -25, 0, 25, 50, 75, 700],
                                  cmap_name="RdBu_r", alpha=0.6):
    """
    Plot population change on a map with:
    - user-defined bins
    - colors sampled from a colormap (BlRd_r)
    - NaN/Inf in grey
    - transparency (alpha)
    - discrete legend with %
    """

    # --- prepare data ---
    gdf_proj = gdf.to_crs(epsg=32632).copy()
    gdf_proj["value"] = np.nan
    gdf_proj.loc[save_population[:, 0]>0, "value"] = 100 * (save_population[save_population[:, 0]>0, MAX_YEAR-1] - save_population[save_population[:, 0]>0, 0]) / save_population[save_population[:, 0]>0, 0]
    
    print("min", np.nanmin(gdf_proj["value"]))
    print("max", np.nanmax(gdf_proj["value"]))

    # Dissolve by municipality
    gdf_proj["Year 0"]= save_population[:, 0]
    gdf_proj["Year 19"]= save_population[:, MAX_YEAR-1]
    muni_bar = gdf_proj.loc[:,["NMUN", "Year 0", "Year 19"]].groupby("NMUN").sum()
    muni_bar = muni_bar.sort_values('Year 0', ascending=False)
    muni_bar[['Year 0', 'Year 19']].plot(
        kind='bar',
        figsize=(10, 6))

    plt.xlabel('')
    plt.ylabel('Nb of workers')
    plt.legend()
    plt.tight_layout()
    plt.show()

    muni_bar["diff"] = 100 * (muni_bar["Year 19"] - muni_bar["Year 0"]) / muni_bar["Year 0"]
    muni_bar[['diff']].plot(
        kind='bar',
        figsize=(10, 6), legend=False)
    
    plt.xlabel('')
    plt.ylabel('Change (%)')
    #plt.legend()
    plt.tight_layout()
    plt.show()

    muni_gdf = gdf_proj.dissolve(by='NMUN', as_index=False)
    muni_gdf['centroid'] = muni_gdf.geometry.centroid
    muni_gdf['x'] = muni_gdf.centroid.x
    muni_gdf['y'] = muni_gdf.centroid.y

    # Separate valid and invalid values
    valid = gdf_proj[np.isfinite(gdf_proj["value"])]
    invalid = gdf_proj[~np.isfinite(gdf_proj["value"])]

    # --- discrete colormap from user-defined cmap ---
    n_bins = len(bins) - 1
    cmap = plt.get_cmap(cmap_name)
    # sample colors evenly across the colormap
    colors = [cmap(i/(n_bins-1)) for i in range(n_bins)]
    discrete_cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bins, discrete_cmap.N)

    # --- figure ---
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # Plot invalid polygons first (grey)
    if len(invalid) > 0:
        p_invalid = invalid.plot(color="lightgrey", edgecolor="white", linewidth=0.5, ax=ax)
        for coll in p_invalid.collections:
            coll.set_alpha(alpha)
            coll.set_antialiased(False)

    # Plot valid polygons with discrete colors
    p_valid = valid.plot(column="value", cmap=discrete_cmap, norm=norm,
                         linewidth=0.01, alpha = 0.6, edgecolor="grey", ax=ax)
    for coll in p_valid.collections:
        coll.set_alpha(alpha)
        coll.set_antialiased(False)

    ax.set_axis_off()

    # --- municipality labels ---
    for _, row in muni_gdf.loc[
        muni_gdf["NMUN"].isin(["Badalona", "Castelldefels", "Castellbisbal", "Sant Cugat del Vallès"])
    ].iterrows():
        ax.text(row.x, row.y, row['NMUN'], fontsize=9, fontweight='bold',
                ha='center', va='center', color='black')

    # --- overlay city borders ---
    city_border = muni_gdf[muni_gdf.ID.str[:5].isin(["08019", "08101", "08194"])]
    city_border.boundary.plot(ax=ax, color='black', linewidth=2)

    # --- discrete legend with % ---
    patches = [mpatches.Patch(facecolor=colors[i],
                              edgecolor='grey',
                              label=f"{bins[i]:.1f}% to {bins[i+1]:.1f}%")
               for i in range(n_bins)]

    if len(invalid) > 0:
        patches.append(mpatches.Patch(facecolor='lightgrey', edgecolor='white', label='Missing / invalid'))

    patches.append(mpatches.Patch(facecolor='none', edgecolor='black', linewidth=2, label='Toll area'))

    ax.legend(handles=patches,
          loc='center left',       # reference point of the legend
          bbox_to_anchor=(0.8, 0.2),  # (x, y) relative to axes
          fontsize=11)
          
    plt.tight_layout()
    plt.show()


def plot_transport_cost(gdf, group):

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # Plot
    gdf.plot(
        column="transport_cost" + group,
        legend=True,
        ax=ax,
        cmap="YlOrRd"  # good perceptually uniform palette
    )

    # Remove axes
    ax.set_axis_off()

    # Title
    #ax.set_title("Calibrated transport costs", fontsize=14)

    # Adjust legend font size
    cbar = ax.get_figure().axes[-1]   # legend axis is added at the end
    cbar.tick_params(labelsize=14)
    cbar.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y)}€'))
    plt.show

def plot_transport_mode(gdf, group):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # Plot
    gdf.plot(
        column="transport_mode" + group,
        legend=True,
        ax=ax,
        cmap="cividis"  # good perceptually uniform palette
    )

    # Remove axes
    ax.set_axis_off()

    # Title
    #ax.set_title("Calibrated share of public transport users", fontsize=14)

    # Adjust legend font size
    cbar = ax.get_figure().axes[-1]  # colorbar axis
    cbar.tick_params(labelsize=14)
    cbar.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y*100)}%'))
    plt.show()


def plot_amenities(gdf):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # Plot
    gdf.plot(
        column="amenities",
        legend=True,
        ax=ax,
        cmap="RdYlGn"  # good perceptually uniform palette
    )

    # Remove axes
    ax.set_axis_off()

    # Title
    ax.set_title("Calibrated amenities", fontsize=14)

    # Adjust legend font size
    cbar = ax.get_figure().axes[-1]  # colorbar axis
    cbar.tick_params(labelsize=14)
    plt.show()

def plot_calib_housing(gdf_here, fitted):
    
    df = gdf_here.copy()
    df['dens_obs'] = np.exp(df['log_h'])
    df['dens_fit'] = np.exp(fitted)
    df = df.sort_values('distance_center')

    # --- Bin distances to compute quartiles ---
    nbins = 40
    df['bin'] = pd.qcut(df['distance_center'], nbins, duplicates='drop')

    quartiles = df.groupby('bin', observed=False).agg(
        dist_median=('distance_center','median'),
        q25_obs=('dens_obs', lambda x: np.percentile(x,25)),
        q75_obs=('dens_obs', lambda x: np.percentile(x,75)),
        q25_fit=('dens_fit', lambda x: np.percentile(x,25)),
        q75_fit=('dens_fit', lambda x: np.percentile(x,75))
        ).reset_index()

    
    # --- Smooth central trend (optional LOESS) ---
    loess_obs = lowess(df['dens_obs'], df['distance_center'], frac=0.3)
    loess_fit = lowess(df['dens_fit'], df['distance_center'], frac=0.3)

    # --- Plot ---
    plt.figure(figsize=(8,8))
    plt.rcParams.update({
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
    })
    # Scatter points
    plt.scatter(df['distance_center'], df['dens_obs'], s=2, alpha=0.3, color='steelblue', label='Observed')
    plt.scatter(df['distance_center'], df['dens_fit'], s=2, alpha=0.3, color='darkorange', label='Fitted')

    # Interquartile shading
    plt.fill_between(quartiles['dist_median'],
                        quartiles['q25_obs'], quartiles['q75_obs'],
                        color='steelblue', alpha=0.2, label='Interquartile range (data)')
    plt.fill_between(quartiles['dist_median'],
                        quartiles['q25_fit'], quartiles['q75_fit'],
                        color='darkorange', alpha=0.2, label='Interquartile range (fitted)')

    # Central smooth trends
    plt.plot(loess_obs[:,0], loess_obs[:,1], color='steelblue', lw=2, label='Central trend (observed)')
    plt.plot(loess_fit[:,0], loess_fit[:,1], color='darkorange', lw=2, label='Central trend (fitted)')

    # Labels
    plt.xlabel("Distance to city center (km)")
    plt.ylabel("Housing density")
    plt.legend()
    plt.show()

def print_maps(gdf, n, q, R):
    map_calibration(gdf, n, gdf["pop"] , "Population")
    map_calibration(gdf, q, gdf["size"], "Dwelling size per capita")
    map_calibration(gdf, R, gdf["rent_m2"], "Rent per m2")
    map_calibration(gdf, 1000000 * n / gdf["urb_area"], 1000000 * gdf["pop"] / gdf["urb_area"], "Population density")

def print_scatterplots(gdf, n, q, R):
    scatter_calibration(gdf, n, gdf["pop"] , "Population")
    scatter_calibration(gdf, q, gdf["size"], "Dwelling size per capita")
    scatter_calibration(gdf, R, gdf["rent_m2"], "Rent per m2")
    scatter_calibration(gdf, n * q / gdf["urb_area"], gdf["pop"] * gdf["size"] / gdf["urb_area"], "Housing")
    scatter_calibration(gdf, 1000000 * n / gdf["urb_area"], 1000000 * gdf["pop"] / gdf["urb_area"], "Population density")

def plot_line_charts(gdf, n, q, R):
    agg = compare_rent_or_size(gdf, "size", q, 1, "Avg dwelling size")
    agg = compare_rent_or_size(gdf, "rent_m2", R, 1, "Rents per sqm")
    agg = compare_var(gdf, n)

    plt.plot(agg["distance_bin"], agg["mean_density_pop"], color='red', linewidth=2, label="Data")
    plt.plot(agg["distance_bin"], agg["mean_density_n"], color='blue', linewidth=2, label="Simulation")
    plt.xlabel("Distance to city center (km)")
    plt.ylabel("Population density (hab/km²)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def compare_shares_in_tax_zone(gdf, n_group, houses_in_toll_area):
    print("LOW")
    print("Estimated share in tax zone:", round(100 * np.nansum(n_group["LOW"].loc[gdf.ID.isin(houses_in_toll_area)]) / np.nansum(n_group["LOW"])), "%")
    print("Actual share in tax zone:", round(100 * np.nansum(gdf.pop_LOW.loc[gdf.ID.isin(houses_in_toll_area)]) / np.nansum(n_group["LOW"])), "%")

    print("MED")
    print("Estimated share in tax zone:", round(100 * np.nansum(n_group["MED"].loc[gdf.ID.isin(houses_in_toll_area)]) / np.nansum(n_group["MED"])), "%")
    print("Actual share in tax zone:", round(100 * np.nansum(gdf.pop_MED.loc[gdf.ID.isin(houses_in_toll_area)]) / np.nansum(n_group["MED"])), "%")

    print("HIGH")
    print("Estimated share in tax zone:", round(100 * np.nansum(n_group["HIGH"].loc[gdf.ID.isin(houses_in_toll_area)]) / np.nansum(n_group["HIGH"])), "%")
    print("Actual share in tax zone:", round(100 * np.nansum(gdf.pop_HIGH.loc[gdf.ID.isin(houses_in_toll_area)]) / np.nansum(n_group["HIGH"])), "%")


def plot_distance_distrib_check(income_levels, travel_matrix, level):

    for lvl in income_levels:
        travel_matrix[f"pop_{lvl}"] *= travel_matrix[f"proba_center_{lvl}"]

    bins = np.array([0, 0.5, 2, 5, 10, 50])
    labels = [f"{i}km" for i in bins[:-1]]
    travel_matrix['distance_bin'] = pd.cut(travel_matrix['distance_car'] / 1000, bins=bins, labels=labels, right=False)

    pop_by_bin = {
        lvl: travel_matrix.groupby('distance_bin', observed=True)[f'pop_{lvl}'].sum()
        for lvl in income_levels
    }

    pop_by_bin[level].plot(kind='bar', figsize=(8, 4))
    plt.ylabel("Population")
    plt.xlabel("Distance to center")
    plt.title("Population by distance category")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def print_moving(save_population_lvl, income_levels, N):
    moving = {}
    for lvl in income_levels:
        moving[lvl] = 0
        for i in range(19):
            moving[lvl] = moving[lvl] + np.nansum(np.abs(save_population_lvl[lvl][:, i+1] - save_population_lvl[lvl][:, i]))/2
        print(lvl, 100 * (moving[lvl] / N[lvl]),  "% moving relative to the population")

