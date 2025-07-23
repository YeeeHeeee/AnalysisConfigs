import argparse
import copy
import fnmatch
import glob
import matplotlib.pylab as plt
import os
import pandas as pd
import numpy as np
import mplhep as hep
import textwrap
from tabulate import tabulate
hep.style.use("CMS")

parser = argparse.ArgumentParser()
parser.add_argument('--input', "-i", help='The input folder of the parquet files', type=str, default="output_merged_v3")
parser.add_argument('--output', "-o", help='The output plot directory', type=str, default="./")
parser.add_argument('--var', help='The name of the variable', type=str, default=None)
parser.add_argument('--sel', help='A selection to apply', type=str, default=None)
parser.add_argument('--bins', help='The name of the variable', type=str, default="auto")
parser.add_argument('--year', help='The name of the year', type=str, default="all")
parser.add_argument('--cms-label', help='The cms label for the plot', type=str, default="Work in progress")
parser.add_argument('--num-bins', help='The number of bins if bins=auto', type=int, default=50)
parser.add_argument('--calculate', help='Calculate the variable', type=str, default=None)
parser.add_argument('--include-fraction', help='Include the fractions of process in plot', action='store_true', default=False)
parser.add_argument('--scale', help='Comma separate list of key and the scalings', type=str, default=None)
parser.add_argument('--xlabel', help='X label for plot. If none will use var', type=str, default=None)
parser.add_argument('--normalise', help='Normalise the MC to data', action='store_true', default=False)

args = parser.parse_args()

if args.year == "all":
  wildcard = "*"
  lumi_label = "$200\ fb^{-1}\ (13,13.6\ TeV)$"
elif args.year == "run2":
  wildcard = ["*2016_PreVFP*", "*2016_PostVFP*", "*2017*", "*2018*"]
  lumi_label = "$138\ fb^{-1}\ (13\ TeV)$"
elif args.year == "run3":
  wildcard = ["*2022_preEE*", "*2022_postEE*", "*2023_preBPix*", "*2023_postBPix*"]
  lumi_label = "$61.9\ fb^{-1}\ (13.6\ TeV)$"
elif args.year == "2016_PreVFP":
  wildcard = "*2016_PreVFP*"
  lumi_label = "$19.6\ fb^{-1}\ (13\ TeV)$"
elif args.year == "2016_PostVFP":
  wildcard = "*2016_PostVFP*"
  lumi_label = "$17.0\ fb^{-1}\ (13\ TeV)$"
elif args.year == "2017":
  wildcard = "*2017*"
  lumi_label = "$41.5\ fb^{-1}\ (13\ TeV)$"
elif args.year == "2018":
  wildcard = "*2018*"
  lumi_label = "$59.8\ fb^{-1}\ (13\ TeV)$"
elif args.year == "2022_preEE":
  wildcard = "*2022_preEE*"
  lumi_label = "$7.98\ fb^{-1}\ (13.6\ TeV)$"
elif args.year == "2022_postEE":
  wildcard = "*2022_postEE*"
  lumi_label = "$26.7\ fb^{-1}\ (13.6\ TeV)$"
elif args.year == "2023_preBPix":
  wildcard = "*2023_preBPix*"
  lumi_label = "$17.8\ fb^{-1}\ (13.6\ TeV)$"
elif args.year == "2023_postBPix":
  wildcard = "*2023_postBPix*"
  lumi_label = "$9.45\ fb^{-1}\ (13.6\ TeV)$"
else:
  raise ValueError(f"Unknown year: {args.year}. Please specify a valid year.")

groups = {
  "Data": ["DATA_*.parquet"],
  #"TT": ["TTToSemiLeptonic_*.parquet", "TTToHadronic_*.parquet", "TTTo2L2Nu_*.parquet", "TTMtt*.parquet"],
  "TT$\\rightarrow$LNu2Q": ["TTToSemiLeptonic_*.parquet","TTMtt*.parquet"],
  "TT$\\rightarrow$Other": ["TTToHadronic_*.parquet","TTTo2L2Nu_*.parquet","TTMtt*.parquet"],
  #"TTToSemiLeptonic": ["TTToSemiLeptonic171p5_*.parquet"],
  #"TTToHadronic": ["TTToHadronic171p5_*.parquet"],
  #"TTTo2L2Nu": ["TTTo2L2Nu171p5_*.parquet"],
  "ST": ["ST_*.parquet"],
  "WJ": ["WJetsToLNu_*.parquet","WJetsToLNuHT*.parquet"],
  "Other": ["QCD_Mu*.parquet","QCD_bcToE*.parquet", "DY*.parquet", "WW*.parquet", "WZ*.parquet", "ZZ*.parquet"],

  #"QCD": ["QCD_Mu*.parquet","QCD_bcToE*.parquet"],
  #"DY" : ["DY*.parquet"],
  #"VV" : ["WW*.parquet", "WZ*.parquet", "ZZ*.parquet"],
}

colours = {
  #"TT": "blue",
  "TT$\\rightarrow$LNu2Q": "blue",
  "TT$\\rightarrow$Other": "orange",
  "WJ": "red",
  #"QCD": "cyan",
  #"DY": "magenta",
  "ST": "brown",
  #"VV": "gray",
  "Other": "cyan",
}

group_selection = {
  "TT$\\rightarrow$Other" : "(GenTT_count_l == 0) | (GenTT_count_l == 2)",
  "TT$\\rightarrow$LNu2Q" : "GenTT_count_l == 1",
}

if isinstance(wildcard, str):
  wildcard = [wildcard]
files = []
for w in wildcard:
  files += glob.glob(f"{args.input}/{w}.parquet")
files = list(set(files))
if len(files) == 0:
  raise FileNotFoundError(f"No parquet files found in {args.input}")

if args.var is None:
  raise ValueError("Please specify a variable with --var")

if "(" in args.bins:
  bin_vals = args.bins.split("(")[1].split(")")[0].split(",")
  bins = np.arange(float(bin_vals[0]), float(bin_vals[1]), float(bin_vals[2]))
elif "[" in args.bins:
  bins = np.array([float(i) for i in args.bins.split("[")[1].split("]")[0].split(',')])
elif args.bins == "auto":
  bins = args.num_bins

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
    include_fraction=False
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
  hep.cms.text(args.cms_label,ax=ax1, fontsize=22)

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
    zero_indices = np.where(total_stack_hist == 0)
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


scale_factors = {}
if args.scale is not None:
  scale_factors = {}
  for s in args.scale.split(","):
    key, value = s.split(":")
    scale_factors[key] = float(value)

hists = {}
hists_squared = {}
n_events = {}
n_positive = {}
n_negative = {}
first = True
for f in files:
  df = pd.read_parquet(f)

  if args.sel is not None:
    df = df.query(args.sel)

  if args.calculate is not None:
    df.loc[:,args.var] = df.eval(args.calculate)

  for k, v in groups.items():
    for fn in v:
      if fnmatch.fnmatch(f.split("/")[-1], fn):

        if k in group_selection.keys():
          df = df.query(group_selection[k])

        hist, bins = np.histogram(df.loc[:,args.var], bins=bins, weights=df.loc[:,"weight"], density=False)
        hist_squared, bins = np.histogram(df.loc[:,args.var], bins=bins, weights=df.loc[:,"weight"]**2, density=False)
        n = len(df)
        n_pos = len(df[df.loc[:,"weight"] >= 0])
        n_neg = len(df[df.loc[:,"weight"] < 0])

        if k not in hists:
          hists[k] = hist
          hists_squared[k] = hist_squared
          n_events[k] = n
          n_positive[k] = n_pos
          n_negative[k] = n_neg
        else:
          hists[k] += hist
          hists_squared[k] += hist_squared
          n_events[k] += n
          n_positive[k] += n_pos
          n_negative[k] += n_neg

if "Data" in hists:
  data_hist = hists["Data"]
  data_uncert = np.sqrt(hists_squared["Data"])
  data_name = "Data"
else:
  data_hist = None
  data_uncert = None
  data_name = None

hists = {k: hists[k] for k in list(groups.keys())[::-1] if k in hists and k != "Data"}
hists_squared = {k: v for k, v in hists_squared.items() if k in hists and k != "Data"}

if args.normalise and data_hist is not None:
  total_data = np.sum(data_hist)
  total_sim = np.sum(np.array(list(hists.values())))
  for k in hists:
    if k != "Data":
      hists[k] = hists[k] * (total_data / total_sim)
      hists_squared[k] = hists_squared[k] * (total_data / total_sim)**2

order = list(hists.keys())
for k,v in scale_factors.items():
  if k in hists:
    hists[f"{v} x {k}"] = hists[k] * v
    hists_squared[f"{v} x {k}"] = hists_squared[k] * v**2
    n_events[f"{v} x {k}"] = n_events[k]
    n_positive[f"{v} x {k}"] = n_positive[k]
    n_negative[f"{v} x {k}"] = n_negative[k]
    colours[f"{v} x {k}"] = colours[k]
    order[order.index(k)] = f"{v} x {k}"
    del hists[k]
    del hists_squared[k]
    del n_events[k]
    del n_positive[k]
    del n_negative[k]
    del colours[k]
hists = {k: hists[k] for k in order if k in hists}
hists_squared = {k: hists_squared[k] for k in order if k in hists_squared}
uncerts = np.sqrt(np.sum(np.array(list(hists_squared.values())), axis=0))

RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m" 
RESET = "\033[0m"
tabulated_data = [["Group", "Sum of Weights", "Number of Events", "Positive Weight Fraction"]]
for k, v in hists.items():
  tabulated_data.append([
    f"{GREEN}{k}{RESET}", 
    f"{round(np.sum(v),2)} +/- {round(np.sqrt(np.sum(hists_squared[k])),2)}", 
    int(n_events[k]), 
    f"{round(100*n_positive[k]/n_events[k],2)}%" if n_events[k] > 0 else "N/A"
  ])
tabulated_data.append([
  f"{RED}Total Pred{RESET}", 
  f"{round(np.sum(np.array(list(hists.values()))),2)} +/- {round(np.sqrt(np.sum([np.sum(hists_squared[k]) for k in hists])),2)}", 
  int(np.sum([n_events[k] for k in hists if k!="data"])), 
  f"{round(100*np.sum([n_positive[k] for k in hists if k!='data'])/np.sum([n_events[k] for k in hists if k!='data']),2)}%" if np.sum([n_events[k] for k in hists if k!='data']) > 0 else "N/A"
])
if data_hist is not None:
  tabulated_data.append([
    f"{BLUE}Data{RESET}", 
    f"{round(np.sum(data_hist),2)} +/- {round(np.sqrt(np.sum(data_uncert**2)),2)}", 
    int(n_events["Data"]),
    f"{round(100*n_positive['Data']/n_events['Data'],2)}%" if n_events["Data"] > 0 else "N/A"  
  ])
print(tabulate(tabulated_data[1:], headers=tabulated_data[0], tablefmt="fancy_grid"))

plot_stacked_histogram_with_ratio(
  data_hist,
  hists,
  bins,
  data_name=data_name,
  xlabel=args.var if args.xlabel is None else rf"{args.xlabel}",
  ylabel="Events",
  name=os.path.join(args.output, f"distribution_{args.var}_{args.year}"),
  data_errors=data_uncert,
  stack_hist_errors=uncerts,
  stack_hist_errors_asym=None,
  use_stat_err=False,
  axis_text="",
  top_space=1.2,
  draw_ratio=True,
  colours=colours,
  include_fraction=args.include_fraction
)