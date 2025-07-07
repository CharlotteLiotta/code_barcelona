import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

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