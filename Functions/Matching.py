import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
from matplotlib import gridspec
import mplhep as hep
import seaborn as sns
import pandas as pd
from matplotlib.colors import Normalize
import copy
import math
import textwrap
from pocket_coffea.parameters.lumi import lumi


def inital_distributions_plot(datasets, bins=50):
    """
    Function to plot the variables from the datasets.
    """
    # Handle the case when a single DataFrame is passed
    if isinstance(datasets, pd.DataFrame):
        datasets = {'Dataset': datasets}

    # Extract column names from the first dataset
    first_key = next(iter(datasets))
    df_first = datasets[first_key]
    num_variables = len(df_first.columns)

    num_rows = math.ceil(math.sqrt(num_variables))
    num_cols = math.ceil(num_variables / num_rows)
    
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(18, 16))
    fig.suptitle(f"Distribution of {num_variables} Variables", fontsize=18, fontweight="bold")

    axes = axes.flatten()

    for i, column in enumerate(df_first.columns):  
        ax = axes[i]

        # Plot each dataset on the same axes for comparison
        for dataset_name, dataset_df in datasets.items():
            dataset_df[column].hist(ax=ax, bins=bins, alpha=0.7, label=f'{dataset_name}', edgecolor='black')

        # Set titles and labels for clarity
        ax.set_title(column, fontsize=12)
        ax.set_xlabel('Value', fontsize=8)
        ax.set_ylabel('Frequency', fontsize=8)
        ax.tick_params(axis='both', labelsize=8)
        ax.legend(fontsize=6, loc='best')

    # Hide empty subplots (if any)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout to fit title and prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def setup_plot():
    """
    Helper function to set up a consistent plot style for all plots,
    using a manual CMS-style label.
    
    Returns:
        fig, ax: The figure and axis objects.
    """
    # Use CMS style
    hep.style.use("CMS")
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Manually add CMS-style label at the top left (adjust coords as needed)
    ax.text(
        0, 1.02,
        "Private work (CMS simulation)", fontsize=18,
        style='italic', transform=ax.transAxes
    )

    return fig, ax


def stacked_hist(datasets, column_name, year,xlim_upper=None, bins=100, drop_zeros=False, xlim_lower=0):
    """
    Plots a stacked histogram for the specified column from the datasets.
    """
    if isinstance(datasets, pd.DataFrame):
        datasets = {'Dataset': datasets}
        
    fig, ax = setup_plot()

    data_for_stack = []
    labels = list(datasets.keys())
    global_max = -float('inf')  # For automatic xlim_upper

    for name, df in datasets.items():
        if column_name in df.columns:
            data = df[column_name].dropna()
            if drop_zeros:
                data = data[data != 0]
            # Clip by lower bound only for now
            data = data[data >= xlim_lower]
            if xlim_upper is None:
                if not data.empty:
                    global_max = max(global_max, data.max())
            else:
                data = data[data <= xlim_upper]
            data_for_stack.append(data)
        else:
            print(f"Warning: Column '{column_name}' not found in dataset '{name}'")

    # Set xlim_upper automatically if it was None
    if xlim_upper is None:
        xlim_upper = global_max if global_max != -float('inf') else xlim_lower + 1  # fallback to prevent crash

    # Plot the stacked histogram
    plt.hist(data_for_stack, bins=bins, stacked=True, label=labels, edgecolor=None, alpha=1)

    plt.xlabel(column_name, fontsize=16)
    plt.ylabel("Counts", fontsize=16)
    plt.xlim([xlim_lower, xlim_upper])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)

    plt.legend(title=None, fontsize=12, loc='upper right')
    plt.grid(True)

    if year in lumi:
        lumi_display = f"{lumi[year]['tot'] * 1e-3:.1f} fb⁻¹ (13TeV)"
    else:
        lumi_display = f"Unknown Lumi for {year}"

    plt.text(0.99, 1.01, lumi_display, transform=ax.transAxes, fontsize=18, verticalalignment='bottom', 
             horizontalalignment='right', style='italic', color='black')

    plt.tight_layout()
    plt.show()


def heat_map(datasets, var1, var2, xlim, ylim, year="2018 (13TeV)", bins=100):
    """
    Plots a 2D heatmap where each bin along var1 (x-axis) is normalized to 1 over var2 (y-axis).
    Highlights the bin with the maximum normalized density.

    Parameters:
    - datasets: dict of DataFrames
    - var1: str, x-axis variable
    - var2: str, y-axis variable
    - xlim, ylim: tuples for plot limits
    - year: str, annotation on plot
    - bins: int, number of bins in each dimension
    """

    # Stack all data together
    x = np.concatenate([df[var1] for df in datasets.values()])
    y = np.concatenate([df[var2] for df in datasets.values()])

    # Create 2D histogram
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[xlim, ylim])

    # Normalize each column (each bin of var1) to sum to 1
    hist_normalized = hist / np.maximum(hist.sum(axis=1, keepdims=True), 1e-9)


    # Plot
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(10, 8))
    hep.cms.text("Private Work", ax=ax, fontsize=16)

    X, Y = np.meshgrid(yedges, xedges)  # transpose for imshow-like orientation
    pcm = ax.pcolormesh(X, Y, hist_normalized, cmap="Blues", norm=Normalize(vmin=0, vmax=hist_normalized.max()))

    # Adjust colorbar
    cbar = plt.colorbar(pcm, ax=ax)
    cbar.set_label("Normalized Density per $p_{T}$ Bin", fontsize=16)
    cbar.ax.tick_params(labelsize=14) 
    
    ax.text(0.99, 1, year, transform=ax.transAxes, fontsize=17,
            verticalalignment='bottom', horizontalalignment='right',
            style='italic', color='black')

    ax.set_xlim(*ylim)  # Note: Y-axis variable is on horizontal
    ax.set_ylim(*xlim)  # X-axis variable is on vertical
    ax.set_xlabel(f"{var2} (GeV)", fontsize=16)
    ax.set_ylabel(f"{var1} (GeV)", fontsize=16)

    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()

def heat_map1(datasets, var1, var2, xlim, ylim, year="2018 (13TeV)", bins=80):
    """
    Plots a 2D heatmap where each bin along var1 (x-axis, quantile bins) is normalized to 1 over var2 (y-axis, linear bins).
    Drops zero values from both variables.
    
    Parameters:
    - datasets: dict or DataFrame
    - var1: str, x-axis variable (quantile bins)
    - var2: str, y-axis variable (linear bins)
    - xlim, ylim: tuples for axis limits
    - year: str, annotation on plot
    - bins: int, number of bins in each dimension
    """

    # Support both dict and single DataFrame
    if isinstance(datasets, dict):
        x = np.concatenate([df[var1].dropna() for df in datasets.values()])
        y = np.concatenate([df[var2].dropna() for df in datasets.values()])
    else:
        x = datasets[var1].dropna().values
        y = datasets[var2].dropna().values

    # Drop zero values from both arrays
    nonzero_mask = (x != 0) & (y != 0)
    x = x[nonzero_mask]
    y = y[nonzero_mask]

    # Quantile-based bins for var1 (x-axis)
    quantile_bins = np.quantile(x, np.linspace(0, 1, bins + 1))

    # Linear bins for var2 (y-axis)
    linear_bins = np.linspace(y.min(), y.max(), bins + 1)

    # Create 2D histogram
    hist, xedges, yedges = np.histogram2d(x, y, bins=[quantile_bins, linear_bins])

    # Normalize each column (bin of var1) to sum to 1
    hist_normalized = hist / np.maximum(hist.sum(axis=1, keepdims=True), 1e-9)

    # Plot
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(10, 8))
    hep.cms.text("Private Work", ax=ax, fontsize=16)

    X, Y = np.meshgrid(yedges, xedges)
    pcm = ax.pcolormesh(X, Y, hist_normalized, cmap="Blues", norm=Normalize(vmin=0, vmax=hist_normalized.max()))

    cbar = plt.colorbar(pcm, ax=ax)
    cbar.set_label("Normalized Density per Quantile $p_{T}$ Bin", fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    ax.text(0.99, 1, year, transform=ax.transAxes, fontsize=17,
            verticalalignment='bottom', horizontalalignment='right',
            style='italic', color='black')

    ax.set_xlim(ylim)  # var2
    ax.set_ylim(xlim)  # var1
    ax.set_xlabel(f"{var2} (GeV)", fontsize=16)
    ax.set_ylabel(f"{var1} (GeV)", fontsize=16)
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def eff_plot(df, var, total_num, bins, upper_lim, year="2018 (13 TeV)"):
    """
    Plot the selection or generator-level matching efficiency as a function of transverse momentum.
    
    Parameters:
    - df: DataFrame (Semileptonic dataset)
    - var: str
    - total_num: int, total number of events used to normalise the efficiency
    - upper_limit: tuples for axis limits
    """

    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(8, 6))
    hep.cms.text("Private Work", ax=ax, fontsize=16)

    # Fields
    pt_field = f"{var}_pt"
    mass_field = f"{var}_mass"

    # Select non-zero entries
    pt = [pt for m, pt in zip(df[mass_field], df[pt_field]) if m > 0 and pt > 0]

    # Histogram
    arranges = np.linspace(0, np.max(pt), bins)
    counts, bin_edges = np.histogram(pt, bins=arranges)
    efficiency = counts / total_num
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Determine label based on var name
    eff_label = "Gen-Match Efficiency" if "Matched" in var else "Selection Efficiency"

    # Plot efficiency
    ax.plot(bin_centers, efficiency, marker="o", label=eff_label)
    
    # Find maximum efficiency point
    max_idx = np.argmax(efficiency)
    max_pt = bin_centers[max_idx]
    max_eff = efficiency[max_idx]

    # Annotate max efficiency in plot corner
    ax.text(0.98, 0.9,
            f"Max efficiency: ({max_pt:.1f}, {max_eff:.3f})",
            transform=ax.transAxes,
            fontsize=12,
            ha='right',
            va='top',
            color='red')

    # Axes labels
    ax.set_xlabel(f"$p_{{T}}^{{{var}}}$ [GeV]", fontsize=14)
    ax.set_ylabel("Efficiency", fontsize=14)

    # Ticks and limits
    ax.set_xlim(np.min(pt), upper_lim)
    ax.tick_params(axis='both', labelsize=12)

    # Grid and legend
    ax.grid(True)
    ax.legend(fontsize=12, loc='upper right', frameon=False)

    # Year label
    ax.text(0.99, 1.01, year,
            transform=ax.transAxes,
            fontsize=16,
            style='italic',
            ha='right',
            va='bottom')

    plt.tight_layout()
    plt.show()

    






    








    




