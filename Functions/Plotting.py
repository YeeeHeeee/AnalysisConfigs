import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
from matplotlib import gridspec
import mplhep as hep
import pandas as pd
from matplotlib.colors import Normalize
import copy
import math
import textwrap
import os
from pocket_coffea.parameters.lumi import lumi

hep.style.use("CMS")

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


def plot_stacked_histogram_with_ratio(
    data_hist, 
    stack_hist_dict, 
    bin_edges, 
    data_name='Data', 
    xlabel="",
    ylabel="Events",
    name="fig", 
    data_errors=None, 
    stack_hist_errors=None, 
    stack_hist_errors_asym=None,
    use_stat_err=False,
    axis_text="",
    top_space=1.2,
    draw_ratio=True,
    colours = {},
    include_fraction=False,
    line_hist_dict={},
    line_colours={},
    cms_label="Work in progress",
    lumi_label=None
  ):
  """
  Plot a stacked histogram with a ratio plot.

  Parameters
  ----------
  data_hist : array-like
      Histogram values for the data.
  stack_hist_dict : dict
      Dictionary of histogram values for stacked components.
  bin_edges : array-like
      Bin edges for the histograms.
  data_name : str, optional
      Label for the data histogram (default is 'Data').
  xlabel : str, optional
      Label for the x-axis (default is '').
  ylabel : str, optional
      Label for the y-axis (default is 'Events').
  name : str, optional
      Name of the output plot file without extension (default is 'fig').
  data_errors : array-like, optional
      Errors for the data histogram (default is None).
  stack_hist_errors : array-like, optional
      Errors for the stacked histograms (default is None).
  use_stat_err : bool, optional
      If True, use statistical errors for the data and stacked histograms (default is False).
  axis_text : str, optional
      Text to be displayed on the top left corner of the plot (default is '').
  """

  if draw_ratio:
    if include_fraction:
      fig, (ax1, ax1p5, ax2) = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [2.5, 1, 1]})
    else:
      fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
  else:
    fig, ax1 = plt.subplots()

  bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2  # Compute bin centers

  if data_hist is not None:
    data_hist = data_hist.astype(np.float64)
  for k, v in stack_hist_dict.items():
    stack_hist_dict[k] = v.astype(np.float64)

  total_stack_hist = np.sum(list(stack_hist_dict.values()), axis=0)

  if data_hist is not None:
    if data_errors is None:
      data_errors = 0*data_hist
  if stack_hist_errors is None and stack_hist_errors_asym is None:
    stack_hist_errors = 0*total_stack_hist   

  if use_stat_err:
    if data_hist is not None:
      data_errors = np.sqrt(data_hist)
    stack_hist_errors = np.sqrt(total_stack_hist)
    stack_hist_errors_asym = None

  # Plot the histograms on the top pad
  for ind, (k, v) in enumerate(stack_hist_dict.items()):
    if ind == 0:
      bottom = None
    elif bottom is None:
      bottom = copy.deepcopy(stack_hist_dict[list(stack_hist_dict.keys())[ind-1]])
    else:
      bottom += copy.deepcopy(stack_hist_dict[list(stack_hist_dict.keys())[ind-1]])
    ax1.bar(
       bin_edges[:-1], 
       v, 
       bottom=bottom,
       width=np.diff(bin_edges), 
       align='edge', 
       alpha=1.0, 
       label=k, 
       color=colours[k], 
       edgecolor=None
      )

  step_edges = np.append(bin_edges,2*bin_edges[-1]-bin_edges[-2])
  summed_stack_hist = np.zeros(len(total_stack_hist))
  for k, v in stack_hist_dict.items():
    summed_stack_hist += v
    step_histvals = np.append(np.insert(summed_stack_hist,0,0.0),0.0)
    ax1.step(step_edges, step_histvals, color='black')


  for k, v in line_hist_dict.items():
    v_step = np.append(np.insert(v,0,0.0),0.0)
    ax1.step(step_edges, v_step, color=line_colours[k], label=k, linewidth=2.0)

  ax1.set_xlim([bin_edges[0],bin_edges[-1]])
  if data_hist is not None:
    ax1.set_ylim([0.0,top_space*max(np.maximum(data_hist,total_stack_hist))])
  else:
    ax1.set_ylim([0.0,top_space*max(total_stack_hist)])


  if stack_hist_errors_asym is None:
    ax1.fill_between(bin_edges[:],np.append(total_stack_hist,total_stack_hist[-1])-np.append(stack_hist_errors,stack_hist_errors[-1]),np.append(total_stack_hist,total_stack_hist[-1])+np.append(stack_hist_errors,stack_hist_errors[-1]),color="gray",alpha=0.3,step='post',label="Uncertainty")
  else:
    ax1.fill_between(bin_edges[:],np.append(total_stack_hist,total_stack_hist[-1])-np.append(stack_hist_errors_asym["down"],stack_hist_errors_asym["down"][-1]),np.append(total_stack_hist,total_stack_hist[-1])+np.append(stack_hist_errors_asym["up"],stack_hist_errors_asym["up"][-1]),color="gray",alpha=0.3,step='post',label="Uncertainty")


  if data_hist is not None:
    # Plot the other histogram as markers with error bars
    ax1.errorbar(bin_centers, data_hist, yerr=data_errors, fmt='o', label=data_name, color="black")

  # Get the current handles and labels of the legend
  handles, labels = ax1.get_legend_handles_labels()

  # Reverse the order of handles and labels
  handles = handles[::-1]
  labels = labels[::-1]

  legend = ax1.legend(handles, labels, loc='upper right', fontsize=18, bbox_to_anchor=(0.9, 0.88), bbox_transform=plt.gcf().transFigure, frameon=True, framealpha=1, facecolor='white', edgecolor="white")

  # Set legend width and wrap text manually
  legend.get_frame().set_linewidth(0)  # Remove legend box border
  legend.get_frame().set_facecolor('none')  # Make legend background transparent
  legend.get_frame().set_edgecolor('none')  # Make legend edge transparent

  max_label_length = 22  # Adjust the maximum length of each legend label
  for text in legend.get_texts():
    text.set_text(textwrap.fill(text.get_text(), max_label_length))

  ax1.set_ylabel(ylabel)
  hep.cms.text(cms_label,ax=ax1, fontsize=22)

  if lumi_label is not None:
    ax1.text(1.0, 1.0, lumi_label,
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax1.transAxes)

  ax1.text(0.03, 0.96, axis_text, transform=ax1.transAxes, va='top', ha='left')

  if not draw_ratio:
    ax1.set_xlabel(xlabel)


  if draw_ratio and include_fraction:
    # Create a new axis for the fraction plot
    ax1p5.set_ylabel('Fraction')
    ax1p5.set_ylim([0, 1.0])

    # Calculate the fraction of each component in the stack
    total_stack_hist_for_fraction = np.sum(list(stack_hist_dict.values()), axis=0)
    fractions = {}
    for k, v in stack_hist_dict.items():
      fractions[k] = v / total_stack_hist_for_fraction

    # Plot the histograms on the top pad
    for ind, (k, v) in enumerate(fractions.items()):
      if ind == 0:
        bottom = None
      elif bottom is None:
        bottom = copy.deepcopy(fractions[list(fractions.keys())[ind-1]])
      else:
        bottom += copy.deepcopy(fractions[list(fractions.keys())[ind-1]])
      ax1p5.bar(
        bin_edges[:-1], 
        v, 
        bottom=bottom,
        width=np.diff(bin_edges), 
        align='edge', 
        alpha=1.0, 
        label=k, 
        color=colours[k], 
        edgecolor=None
        )

    summed_fraction_hist = np.zeros(len(total_stack_hist))
    for k, v in fractions.items():
      summed_fraction_hist += v
      step_fraction_histvals = np.append(np.insert(summed_fraction_hist,0,0.0),0.0)
      ax1p5.step(step_edges, step_fraction_histvals, color='black')


  if draw_ratio:

    # Compute the ratio of the histograms
    zero_indices = np.where(total_stack_hist <= 0)
    for i in zero_indices: total_stack_hist[i] = 1.0

    if data_hist is not None:
      ratio = np.divide(data_hist,total_stack_hist)
      ratio_errors_2 = np.divide(data_errors,total_stack_hist)

    if stack_hist_errors_asym is None:
      ratio_errors_1 = np.divide(stack_hist_errors,total_stack_hist)
    else:
      ratio_errors_1_up = np.divide(stack_hist_errors_asym["up"],total_stack_hist)
      ratio_errors_1_down = np.divide(stack_hist_errors_asym["down"],total_stack_hist)

    for i in zero_indices:
      if data_hist is not None:
        ratio[i] = 0.0
        ratio_errors_2[i] = 0.0
      if stack_hist_errors_asym is None:
        ratio_errors_1[i] = 0.0
      else:
        ratio_errors_1_up[i] = 0.0
        ratio_errors_1_down[i] = 0.0

    if data_hist is not None:
      # Plot the ratio on the bottom pad
      ax2.errorbar(bin_centers, ratio, fmt='o', yerr=ratio_errors_2, label=data_name, color="black")

    for k, v in line_hist_dict.items():
      v = v/total_stack_hist
      v_step = np.append(np.insert(v,0,0.0),0.0)
      ax2.step(step_edges, v_step, color=line_colours[k], label=k, linewidth=2.0)

    ax2.axhline(y=1, color='black', linestyle='--')  # Add a horizontal line at ratio=1
    if stack_hist_errors_asym is None:
      ax2.fill_between(bin_edges,1-np.append(ratio_errors_1,ratio_errors_1[-1]),1+np.append(ratio_errors_1,ratio_errors_1[-1]),color="gray",alpha=0.3,step='post')
    else:
      ax2.fill_between(bin_edges,1-np.append(ratio_errors_1_down,ratio_errors_1_down[-1]),1+np.append(ratio_errors_1_up,ratio_errors_1_up[-1]),color="gray",alpha=0.3,step='post')

    ax2.set_xlabel(xlabel)
    ax2.set_ylabel('Ratio')
    ax2.set_ylim([0.5,1.5])
    ax2.xaxis.get_major_formatter().set_useOffset(False)

  # Adjust spacing between subplots
  plt.subplots_adjust(hspace=0.1, left=0.15)

  # Show the plot
  print("Created "+name+".pdf")
  os.makedirs(os.path.dirname(name+".pdf"), exist_ok=True)
  plt.savefig(name+".png")
  plt.savefig(name+".pdf")
  plt.close()


def plot_histograms_with_ratio(
  hists,
  hist_uncerts,
  hist_names,
  bins,
  xlabel = "",
  ylabel="Events",
  name="histogram_with_ratio",    
  ratio_range = [0.5,1.5],  
  colours = ["black","blue", "red", "orange", "purple", "brown", "pink", "gray"],
  cms_label="Work in progress",
  axis_text=None
):

  fig, ax= plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3,1]})

  hep.cms.text(cms_label,ax=ax[0])

  denom = np.array([v if v !=0 else 1.0 for v in hists[0]])

  # draw histograms
  legend = {}
  for ind in range(len(hists)):

    colour = colours[ind]
    hist = hists[ind]
    hist_uncert = hist_uncerts[ind]
    hist_name = hist_names[ind]
    colour = colours[ind]

    ax[0].plot(bins[:-1], hist, label=hist_name, color=colour, linestyle="-", drawstyle="steps-mid")

    # add uncertainty
    if hist_uncert is not None:
      ax[0].fill_between(
        bins,
        np.append(hist,hist[-1])-np.append(hist_uncert,hist_uncert[-1]),
        np.append(hist,hist[-1])+np.append(hist_uncert,hist_uncert[-1]),
        color=colour,
        alpha=0.2,
        step='mid'
        )

    # y label
    ax[0].set_ylabel(ylabel)

    # legend
    handles, labels = ax[0].get_legend_handles_labels()
    handles = handles[::-1]
    labels = labels[::-1]
    legend = ax[0].legend(handles, labels, loc='upper right', fontsize=18, bbox_to_anchor=(0.9, 0.88), bbox_transform=plt.gcf().transFigure, frameon=True, framealpha=1, facecolor='white', edgecolor="white")
    legend.get_frame().set_linewidth(0)  # Remove legend box border
    legend.get_frame().set_facecolor('none')  # Make legend background transparent
    legend.get_frame().set_edgecolor('none')  # Make legend edge transparent
    max_label_length = 20  # Adjust the maximum length of each legend label
    for text in legend.get_texts():
      text.set_text(textwrap.fill(text.get_text(), max_label_length))

    # draw ratios
    ax[-1].plot(bins[:-1], hist/denom, color=colour, linestyle="-", drawstyle="steps-mid")

    # add uncertainty ro ratio
    if hist_uncert is not None:
      ratio = hist/denom
      ratio_uncert = hist_uncert/denom
      ax[-1].fill_between(bins,np.append(ratio,ratio[-1])-np.append(ratio_uncert,ratio_uncert[-1]),np.append(ratio,ratio[-1])+np.append(ratio_uncert,ratio_uncert[-1]),color=colour,alpha=0.2,step='mid')

  # ratio labels
  ax[-1].axhline(y=1, color=colours[0], linestyle='--')  # Add a horizontal line at ratio=1
  ax[-1].set_xlabel(xlabel)
  ax[-1].set_ylabel('Ratio')
  ax[-1].set_ylim([ratio_range[0],ratio_range[1]])
  ax[-1].xaxis.get_major_formatter().set_useOffset(False)

  # Draw axis text
  if axis_text is not None:
    ax[0].text(0.03, 0.96, axis_text, transform=ax[0].transAxes, va='top', ha='left', fontsize=18)

  # Adjust spacing between subplots
  plt.subplots_adjust(hspace=0.1, left=0.15)

  # Show the plot
  print("Created "+name+".pdf")
  directory = os.path.dirname(name + ".pdf")
  if directory:
    os.makedirs(directory, exist_ok=True)
  plt.savefig(name+".pdf")
  plt.savefig(name+".png")
  plt.close()    








    




