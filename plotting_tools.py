import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter

def plot_employment(gdf, employment_centers, var):
    base = gdf.plot(color='lightgrey', edgecolor='white', figsize=(10, 10))

    # Overlay: points with size proportional to a column (e.g., 'population')
    employment_centers.plot(
        ax=base,
        markersize = var,
        #markersize=employment_centers['employment'] * 0.001,  # adjust scale_factor
        #markersize=employment_centers.merge(employed_results, left_on = "cluster", right_on = "to_id")["weighted_employed"] * 0.001,  # adjust scale_factor
        #markersize= ARRAY_WAGE* 0.5,  # adjust scale_factor
        color='red',
        alpha=0.6)

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

def compare_rent_or_size(gdf, var_data, var_simul, weighting):
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
    plt.scatter(gdf["distance_center"], gdf["simul"], s=1, alpha=0.3, label="Calib", color='blue')

    # Aggregated lines
    plt.plot(agg["distance_bin"], agg["data"], color='red', linewidth=2, label="Data")
    plt.plot(agg["distance_bin"], agg["simul"], color='blue', linewidth=2, label="Calib")

    plt.xlabel("Distance au centre-ville (km)")
    #plt.ylabel("Rents per sqm")
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

def plot_scores(save_score_emissions, save_score_qol, save_score_congestion, save_score_welfare):

    colors = plt.get_cmap("tab10").colors  
    plt.figure(figsize=(8, 6))

    # Plot lines (all same style, distinct colors)
    plt.plot(save_score_emissions[1:]/save_score_emissions[1],
         label="Emissions", linewidth=2, color=colors[0])

    plt.plot(np.nanmedian(save_score_qol[:,1:], 0)/np.nanmedian(save_score_qol[:,1], 0),
         label="Quality of life", linewidth=2, color=colors[1])

    plt.plot(np.nanmedian(save_score_congestion[:,1:], 0)/np.nanmedian(save_score_congestion[:,1], 0),
         label="Congestion", linewidth=2, color=colors[2])

    plt.plot(np.nanmedian(save_score_welfare[:,1:], 0)/np.nanmedian(save_score_welfare[:,1], 0),
         label="Welfare", linewidth=2, color=colors[3])

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

    # plotting
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    cmap_name = "cividis"
    vmin, vmax = gdf_proj["value"].min(), gdf_proj["value"].max()

    gdf_proj.plot(column="value",
                  cmap=cmap_name,
                  vmin=vmin, vmax=vmax,
                  linewidth=0.2, edgecolor="black",
                  ax=ax)

    ax.set_axis_off()

    # improve rendering
    for coll in ax.collections:
        coll.set_antialiased(False)

    # continuous colorbar (no title)
    sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    ticks = np.linspace(vmin, vmax, 5)
    cbar.set_ticks(ticks)
    cbar.ax.set_yticklabels([f"{t:.1f}%" for t in ticks], fontsize=14)  # show 1 decimal + %

    # title
    ax.set_title("Average opinions (year 19)", fontsize=14)

def plot_change_population(gdf, save_population):

    # --- prepare data ---
    gdf_proj = gdf.to_crs(epsg=32632).copy()  # keep projection if needed
    gdf_proj["value"] = (save_population[:,19] - save_population[:,0]) #.astype(int)  # discrete

    print(sum(gdf_proj["value"]))
    print(sum(np.abs(gdf_proj["value"]))/2)
    # --- figure ---
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # --- discrete diverging colormap ---
    cmap_name = "RdBu_r"
    bounds = np.arange(-3.5, 4, 1)               # edges for discrete categories -3..3
    # plot
    gdf_proj.plot(column="value",
                  cmap=cmap_name,
                  linewidth=0.01,
                  edgecolor="grey",
                  ax=ax, legend = True)

    # remove axes
    ax.set_axis_off()

    cbar = ax.get_figure().axes[-1]
    cbar.tick_params(labelsize=14)

    # improve rendering for vector output
    for coll in ax.collections:
        coll.set_antialiased(False)

    # --- title ---
    ax.set_title("Changes in population (Year 19 - Year 0)", fontsize=14)

    plt.tight_layout()

    plt.show()

def plot_transport_cost(gdf):

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # Plot
    gdf.plot(
        column="transport_cost",
        legend=True,
        ax=ax,
        cmap="YlOrRd"  # good perceptually uniform palette
    )

    # Remove axes
    ax.set_axis_off()

    # Title
    ax.set_title("Calibrated transport costs", fontsize=14)

    # Adjust legend font size
    cbar = ax.get_figure().axes[-1]   # legend axis is added at the end
    cbar.tick_params(labelsize=14)
    plt.show

def plot_transport_mode(gdf):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # Plot
    gdf.plot(
        column="transport_mode",
        legend=True,
        ax=ax,
        cmap="cividis"  # good perceptually uniform palette
    )

    # Remove axes
    ax.set_axis_off()

    # Title
    ax.set_title("Calibrated share of public transport users \n Average: 53%", fontsize=14)

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