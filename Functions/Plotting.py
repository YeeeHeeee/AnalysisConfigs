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

def get_lumi_display(year):
    """
    Returns a formatted luminosity string for a given year or Run period.

    Parameters:
        year (str): The data-taking year or 'Run2'/'Run3'.
        lumi (dict): Dictionary containing luminosity values with 'tot' key.

    Returns:
        str: Formatted luminosity string.
    """
    run2_years = ["2016_PreVFP", "2016_PostVFP", "2017", "2018"]
    run3_years = ["2022", "2022_postEE", "2023"]

    if year in run2_years and year in lumi and 'tot' in lumi[year]:
        lumi_value = lumi[year]['tot'] * 1e-3
        return rf"${lumi_value:.1f}\,\mathrm{{fb}}^{{-1}}$ (13 TeV)"

    elif year in run3_years and year in lumi and 'tot' in lumi[year]:
        lumi_value = lumi[year]['tot'] * 1e-3
        return rf"${lumi_value:.1f}\,\mathrm{{fb}}^{{-1}}$ (13.6 TeV)"

    elif year == "Run2":
        total_lumi = sum(lumi[y]['tot'] for y in run2_years if y in lumi and 'tot' in lumi[y]) * 1e-3
        return rf"${total_lumi:.1f}\,\mathrm{{fb}}^{{-1}}$ (13 TeV)"

    elif year == "Run3":
        total_lumi = sum(lumi[y]['tot'] for y in run3_years if y in lumi and 'tot' in lumi[y]) * 1e-3
        return rf"${total_lumi:.1f}\,\mathrm{{fb}}^{{-1}}$ (13.6 TeV)"
    
    else:
        return f"Unknown Lumi for {year}"


def stacked_hist(datasets, column_name, year, xlim_upper=None, bins=100, drop_zeros=False, xlim_lower=0):
    """
    Plots a stacked histogram for the specified column from the datasets, with CMS styling.
    """
    # Convert single DataFrame to a dictionary if necessary
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

    lumi_display = get_lumi_display(year)

    # Add year and luminosity annotation to the top right
    plt.text(0.99, 1.01, lumi_display, transform=ax.transAxes, fontsize=18, verticalalignment='bottom', 
             horizontalalignment='right', style='italic', color='black')

    plt.tight_layout()
    plt.show()

def heat_map(datasets, var1, var2, xlim, ylim, year, bins=100):
    """
    Plots a 2D heatmap where each bin along var1 (x-axis, quantile bins) is normalized to 1 over var2 (y-axis, linear bins).
    Drops zero values from both variables.
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
    fig, ax = setup_plot()

    X, Y = np.meshgrid(yedges, xedges)
    pcm = ax.pcolormesh(X, Y, hist_normalized, cmap="Blues", norm=Normalize(vmin=0, vmax=hist_normalized.max()))

    cbar = plt.colorbar(pcm, ax=ax)
    cbar.set_label("Normalized Density per Quantile $p_{T}$ Bin", fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    lumi_display = get_lumi_display(year)

    # Add year and luminosity annotation to the top right
    plt.text(0.99, 1.01, lumi_display, transform=ax.transAxes, fontsize=18, verticalalignment='bottom', 
             horizontalalignment='right', style='italic', color='black')

    ax.set_xlim(ylim)  # var2
    ax.set_ylim(xlim)  # var1
    ax.set_xlabel(f"{var2}", fontsize=16)
    ax.set_ylabel(f"{var1}", fontsize=16)
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def eff_plot(df, var1, var2, bins=10, year="2018"):
    """
    Plot matching efficiency as a function of gen-level transverse momentum.
    
    Efficiency is defined as:
        (# gen objects with a match in var1) / (# total gen objects) in each pt bin.
    """

    # Set style
    fig, ax = setup_plot()

    # Field names
    pt_gen = df[f"{var2}_pt"]
    pt_matched = df[f"{var1}_pt"]

    # Mask for valid gen entries
    gen_mask = pt_gen > 0
    pt_gen = pt_gen[gen_mask]
    pt_matched = pt_matched[gen_mask]

    # Define matched (nonzero pt) entries
    matched_mask = pt_matched > 0

    # Histogram bins based on gen pt range
    min_pt = pt_gen.min()
    max_pt = pt_gen.max()
    bin_edges = np.linspace(min_pt, max_pt, bins + 1)

    # Bin gen entries
    gen_counts, _ = np.histogram(pt_gen, bins=bin_edges)

    # Bin only those gen entries which have a match
    matched_counts, _ = np.histogram(pt_gen[matched_mask], bins=bin_edges)

    # Efficiency
    efficiency = np.divide(
        matched_counts, gen_counts,
        out=np.zeros_like(matched_counts, dtype=float),
        where=gen_counts > 0
    )
    uncertainty = np.sqrt(efficiency * (1 - efficiency) / gen_counts)
    uncertainty[gen_counts == 0] = 0

    # Bin centers and widths
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = (bin_edges[1:] - bin_edges[:-1]) / 2

    # Plot
    ax.errorbar(
        bin_centers, efficiency, yerr=uncertainty, xerr=bin_widths,
        fmt='o', label="Gen-Match Efficiency", capsize=2
    )
    ax.plot(bin_centers, efficiency, linestyle='-', color='C0')

    # Annotate max
    if np.any(efficiency):
        max_idx = np.argmax(efficiency)
        ax.text(
            0.03, 0.92,
            f"Max efficiency at ({bin_centers[max_idx]:.1f}, {efficiency[max_idx]:.1f})",
            transform=ax.transAxes,
            fontsize=14,
            ha='left',
            va='top',
            color='red'
        )

    ax.set_xlabel(r"$p_T^{\mathrm{gen}}$", fontsize=16)
    ax.set_ylabel("Efficiency", fontsize=14)
    ax.grid(True)
    ax.legend(fontsize=12, loc='upper left', frameon=False)

    lumi_display = get_lumi_display(year)

    # Add year and luminosity annotation to the top right
    plt.text(0.99, 1.01, lumi_display, transform=ax.transAxes, fontsize=18, verticalalignment='bottom', 
             horizontalalignment='right', style='italic', color='black')


    plt.tight_layout()
    plt.show()





    








    




