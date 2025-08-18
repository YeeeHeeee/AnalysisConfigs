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
import yaml
import uproot
import boost_histogram as bh
from tabulate import tabulate
import importlib.util
from Functions.Plotting import plot_stacked_histogram_with_ratio, plot_histograms_with_ratio
from Functions.rebinning import find_rebinning, rebin_histogram
hep.style.use("CMS")

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', "-c", help='Config of run parameters', type=str, default="params/plotting_extra_mass_bw.yaml")
parser.add_argument('--input', "-i", help='The input folder of the parquet files', type=str, default="output_merged_v3")
parser.add_argument('--output', "-o", help='The output plot directory', type=str, default="./")
parser.add_argument('--var', help='The name of the variable', type=str, default=None)
parser.add_argument('--pre-sel', help='A pre-selection before any recalculations', type=str, default=None)
parser.add_argument('--sel', help='A selection to apply', type=str, default=None)
parser.add_argument('--bins', help='The name of the variable', type=str, default=None)
parser.add_argument('--year', help='The name of the year', type=str, default="all")
parser.add_argument('--cms-label', help='The cms label for the plot', type=str, default="Work in progress")
parser.add_argument('--num-bins', help='The number of bins if bins=auto', type=int, default=50)
parser.add_argument('--calculate', help='Calculate the variable', type=str, default=None)
parser.add_argument('--include-fraction', help='Include the fractions of process in plot', action='store_true', default=False)
parser.add_argument('--scale', help='Comma separate list of key and the scalings', type=str, default=None)
parser.add_argument('--xlabel', help='X label for plot. If none will use var', type=str, default=None)
parser.add_argument('--normalise', help='Normalise the MC to data', action='store_true', default=False)
parser.add_argument('--weight', help='The weight to apply to the histograms', type=str, default="weight")
parser.add_argument('--write', help='Write histogram to root file', action='store_true', default=False)
parser.add_argument('--syst', help='Process systematics', action='store_true', default=False)
parser.add_argument('--plot-syst-variation', help='Plot the systematic variations', action='store_true', default=False)
parser.add_argument('--rebin', help='Rebin the histogram', action='store_true', default=False)
parser.add_argument('--rebin-fraction', help='The bin uncertainty fraction threshold', type=float, default=0.15)
parser.add_argument('--rebin-count', help='The bin count threshold', type=float, default=10)
parser.add_argument('--rebin-from', help='Data or MC', type=str, default="Data")
parser.add_argument('--rebin-bins', help='Comma separated list to rebin to if loading in histograms', type=str, default=None)
parser.add_argument('--norm-to-bin-width', help='Normalise to bin width', action='store_true', default=False)
parser.add_argument('--load-from-root', help='Root file to load histograms from', type=str, default=None)

args = parser.parse_args()

class GetHistograms:

  def __init__(
    self,
    input_folder, 
    cfg,
    var=None,
    sel=None,
    bins=None,
    num_bins=50,
    year="all",
    normalise=False,
    weight="weight",
    write=False,
    calculate=None,
    scale=None,
    syst=False,
    pre_sel=None,
    rebin=False,
    rebin_fraction=0.15,
    rebin_count=10,
    rebin_from="Data",
    load_from_root=None,
    rebin_bins=None
    ):
    self.input_folder = input_folder
    self.var = var
    self.sel = sel
    self.bins = bins
    self.num_bins = num_bins
    self.year = year
    self.normalise = normalise
    self.weight = weight
    self.write = write
    self.calculate = calculate
    self.scale = scale
    self.syst = syst
    self.pre_sel = pre_sel
    self.rebin = rebin
    self.rebin_fraction = rebin_fraction
    self.rebin_count = rebin_count
    self.rebin_from = rebin_from
    self.load_from_root = load_from_root
    self.rebin_bins = rebin_bins

    # Load the config file
    if ".yaml" in cfg:
      with open(cfg, 'r') as f:
        self.cfg = yaml.safe_load(f)
    elif ".py" in cfg:
      spec = importlib.util.spec_from_file_location("config", cfg)
      module = importlib.util.module_from_spec(spec)
      spec.loader.exec_module(module)
      self.cfg = module.config

    self._set_missing_cfg_args()
    self._get_year_wildcard()
    self._get_total_groups()
    self._get_files()
    self._get_scale_factors()
    self._get_bins()

    store_columns = [
      "hists",
      "hists_per_file",
      "hists_per_group",
      "hists_squared",
      "hists_squared_per_file",
      "hists_squared_per_group",
      "n_events",
      "n_events_per_file",
      "n_events_per_group",
      "n_positive",
      "n_positive_per_file",
      "n_positive_per_group",
      "n_negative",
      "n_negative_per_file",
      "n_negative_per_group"
    ]

    self.stores = {i: {k:{} for k in store_columns} for i in ["Nom", "Up", "Down"]}

    if self.load_from_root:
      self.root_file = uproot.open(self.load_from_root)

  def _set_missing_cfg_args(self):

    # Update var and bins if provided
    if self.var is not None:
      if self.bins is None:
        self.cfg["variables"] = {self.var: self.cfg["variables"][self.var]}
      else:
        self.cfg["variables"] = {self.var: self.bins}
      if self.calculate is not None:
        self.cfg["calculate"] = {self.var: self.calculate}
      
    if len(self.cfg["variables"].keys()) == 0:
      raise ValueError("Please specify the variables in the config file")

    # Add missing args
    fill_with_empty = [
      "write_translate"
      "other_groups",
      "plot_extra", 
      "plot_extra_colours",
      "group_selection", 
      "variables",
      "calculate",
      "systematics"
    ]
    for arg in fill_with_empty:
      if arg not in self.cfg:
        self.cfg[arg] = {}

  def _get_year_wildcard(self):
    if self.year == "all":
      self.wildcard = "*"
    elif self.year == "run2":
      self.wildcard = ["*2016_PreVFP*", "*2016_PostVFP*", "*2017*", "*2018*"]
    elif self.year == "run3":
      self.wildcard = ["*2022_preEE*", "*2022_postEE*", "*2023_preBPix*", "*2023_postBPix*"]
    elif self.year in ["2016_PreVFP", "2016_PostVFP", "2017", "2018", "2022_preEE", "2022_postEE", "2023_preBPix", "2023_postBPix"]:
      self.wildcard = f"*{self.year}*"
    else:
      raise ValueError(f"Unknown year: {self.year}. Please specify a valid year.")

  def _get_total_groups(self):

    # Get total groups
    self.total_groups = copy.deepcopy(self.cfg["groups"])
    for k, v in self.cfg["other_groups"].items():
      self.total_groups[k] = v

  def _get_files(self):

    if isinstance(self.wildcard, str):
      self.wildcard = [self.wildcard]
    self.files = []
    for w in self.wildcard:
      self.files += glob.glob(f"{self.input_folder}/{w}.parquet")
    self.files = sorted(list(set(self.files)))
    if len(self.files) == 0:
      raise FileNotFoundError(f"No parquet files found in {self.input_folder}")

  def _get_scale_factors(self):
    self.scale_factors = {}
    if self.scale is not None:
      for s in self.scale.split(","):
        key, value = s.split(":")
        self.scale_factors[key] = float(value)

  def _check_file_in_groups(self, f):

    match_found = False
    for k, v in self.total_groups.items():
      for fn in v:
        if fnmatch.fnmatch(f.split("/")[-1], fn):
          match_found = True
          group = k
          break
      if match_found:
        break

    if not match_found:
      return None
    else:
      return group

  def _get_file_names(self, f):

    file_name = f.split("/")[-1].split(".parquet")[0]    
    split_year_name = file_name.split("_")
    for i in range(len(split_year_name)-1, -1, -1):
      if split_year_name[i].isdigit():
        era_name = "_".join(split_year_name[i:])
        break
    file_name_minus_era = file_name.replace(f"_{era_name}", "")

    return file_name, era_name, file_name_minus_era

  def _write_root_files(self):

    self.root_files = {}
    for var in self.cfg["variables"].keys():
      root_name = os.path.join(args.output, f"datacard_{var}.root")
      if os.path.exists(root_name):
        os.remove(root_name)
      os.makedirs(os.path.dirname(root_name), exist_ok=True)
      self.root_files[var] = uproot.recreate(root_name)

      for hist_name, var_name in {"hists":"hists_squared", "hists_per_file":"hists_squared_per_file", "hists_per_group":"hists_squared_per_group"}.items():
        for i in ["Nom", "Up", "Down"]:

          if hist_name not in self.stores[i].keys(): continue
          if var not in self.stores[i][hist_name].keys(): continue
          for file_name, hist in self.stores[i][hist_name][var].items():
            h_bh = bh.Histogram(bh.axis.Variable(copy.deepcopy(self.bin_store[var])), storage=bh.storage.Weight())
            h_bh.view().value[:] = copy.deepcopy(hist)
            h_bh.view().variance[:] = copy.deepcopy(self.stores[i][var_name][var][file_name])

            if file_name in self.cfg["write_translate"].keys():
              file_name = self.cfg["write_translate"][file_name]

            file_name = file_name.replace("(","").replace(")","").replace("[","").replace("]","").replace(",","_").replace(".","").replace(" ","_")

            if i == "Nom":
              write_name = file_name
            else:
              write_name = f"{file_name}{i}"

            self.root_files[var][write_name] = copy.deepcopy(h_bh)

  def _get_bins(self):

    self.bin_store = {}
    for var in self.cfg["variables"].keys():
      bins = self.cfg["variables"][var]
      if "(" in bins:
        bin_vals = bins.split("(")[1].split(")")[0].split(",")
        bins = np.arange(float(bin_vals[0]), float(bin_vals[1]), float(bin_vals[2]))
      elif "[" in bins:
        bins = np.array([float(i) for i in bins.split("[")[1].split("]")[0].split(',')])
      elif bins == "auto":
        bins = args.num_bins
      self.bin_store[var] = bins

  def _get_histograms(self, df, calculate, selection, file_name, group, total_name, save_to="Nom", group_name=None):

    if self.load_from_root is None:
      if len(df) == 0: return

      # Calculate the columns
      for calc in calculate:
        for col_name, func in calc.items():
          df.loc[:, col_name] = df.eval(func)

      # Apply selection if provided
      for sel in selection:
        df = df.query(sel)

      if len(df) == 0: return

    for var in self.cfg["variables"].keys():

      if self.load_from_root is None:
        # Mask valid entries
        valid = df["weight"].notna() & df[var].notna()

      # Set up stores
      for store_name, store_dict in self.stores[save_to].items():
        if var not in store_dict.keys():
          self.stores[save_to][store_name][var] = {}

      # Calculate the histograms
      if self.load_from_root is None:
        hist, bins = np.histogram(df.loc[valid,var], bins=self.bin_store[var], weights=df.loc[valid,"weight"], density=False)
        if isinstance(self.bin_store[var], int):
          self.bin_store[var] = copy.deepcopy(bins)
        hist_squared, _ = np.histogram(df.loc[valid,var], bins=self.bin_store[var], weights=df.loc[valid,"weight"]**2, density=False)
        n = len(df)
        n_pos = len(df[df.loc[:,"weight"] >= 0])
        n_neg = len(df[df.loc[:,"weight"] < 0])
      else:
        root_file_name = file_name.replace("(","").replace(")","").replace("[","").replace("]","").replace(",","_").replace(".","").replace(" ","_")
        if save_to in ["Up","Down"]:
          root_file_name += save_to
        root_hist = self.root_file.get(root_file_name)
        if root_hist is None: continue
        hist, bins = root_hist.to_numpy()
        hist_squared = root_hist.variances()
        n = 0
        n_pos = 0
        n_neg = 0

      # Scale hists
      if group_name in self.scale_factors.keys():
        print(f"Scaling {file_name} by {self.scale_factors[group_name]}")
        hist *= self.scale_factors[group_name]
        hist_squared *= self.scale_factors[group_name]**2

      # Save to dictionaries
      if group_name is None or group_name in self.cfg["groups"].keys():
        if total_name not in self.stores[save_to]["hists"][var]:
          self.stores[save_to]["hists"][var][total_name] = copy.deepcopy(hist)
          self.stores[save_to]["hists_squared"][var][total_name] = copy.deepcopy(hist_squared)
          self.stores[save_to]["n_events"][var][total_name] = copy.deepcopy(n)
          self.stores[save_to]["n_positive"][var][total_name] = copy.deepcopy(n_pos)
          self.stores[save_to]["n_negative"][var][total_name] = copy.deepcopy(n_neg)
        else:
          self.stores[save_to]["hists"][var][total_name] += hist
          self.stores[save_to]["hists_squared"][var][total_name] += hist_squared
          self.stores[save_to]["n_events"][var][total_name] += n
          self.stores[save_to]["n_positive"][var][total_name] += n_pos
          self.stores[save_to]["n_negative"][var][total_name] += n_neg
      if group not in self.stores[save_to]["hists_per_group"][var]:
        self.stores[save_to]["hists_per_group"][var][group] = copy.deepcopy(hist)
        self.stores[save_to]["hists_squared_per_group"][var][group] = copy.deepcopy(hist_squared)
        self.stores[save_to]["n_events_per_group"][var][group] = copy.deepcopy(n)
        self.stores[save_to]["n_positive_per_group"][var][group] = copy.deepcopy(n_pos)
        self.stores[save_to]["n_negative_per_group"][var][group] = copy.deepcopy(n_neg)
      else:
        self.stores[save_to]["hists_per_group"][var][group] += hist
        self.stores[save_to]["hists_squared_per_group"][var][group] += hist_squared
        self.stores[save_to]["n_events_per_group"][var][group] += n
        self.stores[save_to]["n_positive_per_group"][var][group] += n_pos
        self.stores[save_to]["n_negative_per_group"][var][group] += n_neg
      if file_name not in self.stores[save_to]["hists_per_file"][var]:
        self.stores[save_to]["hists_per_file"][var][file_name] = copy.deepcopy(hist)
        self.stores[save_to]["hists_squared_per_file"][var][file_name] = copy.deepcopy(hist_squared)
        self.stores[save_to]["n_events_per_file"][var][file_name] = copy.deepcopy(n)
        self.stores[save_to]["n_positive_per_file"][var][file_name] = copy.deepcopy(n_pos)
        self.stores[save_to]["n_negative_per_file"][var][file_name] = copy.deepcopy(n_neg)
      else:
        self.stores[save_to]["hists_per_file"][var][file_name] += hist
        self.stores[save_to]["hists_squared_per_file"][var][file_name] += hist_squared
        self.stores[save_to]["n_events_per_file"][var][file_name] += n
        self.stores[save_to]["n_positive_per_file"][var][file_name] += n_pos
        self.stores[save_to]["n_negative_per_file"][var][file_name] += n_neg

    return df

  def _add_nom_to_syst_groups(self, shift_name, group, file_name, syst_name):

    for var in self.cfg["variables"].keys():
      for store_name, store_dict in self.stores[shift_name].items():
        if var not in store_dict.keys():
          self.stores[shift_name][store_name][var] = {}
      if file_name not in self.stores["Nom"]["hists_per_file"][var].keys(): 
        continue
      if f"{group}_{syst_name}" not in self.stores[shift_name]["hists_per_group"][var]:
        self.stores[shift_name]["hists_per_group"][var][f"{group}_{syst_name}"] = copy.deepcopy(self.stores["Nom"]["hists_per_file"][var][file_name])
        self.stores[shift_name]["hists_squared_per_group"][var][f"{group}_{syst_name}"] = copy.deepcopy(self.stores["Nom"]["hists_squared_per_file"][var][file_name])
        self.stores[shift_name]["n_events_per_group"][var][f"{group}_{syst_name}"] = copy.deepcopy(self.stores["Nom"]["n_events_per_file"][var][file_name])
        self.stores[shift_name]["n_positive_per_group"][var][f"{group}_{syst_name}"] = copy.deepcopy(self.stores["Nom"]["n_positive_per_file"][var][file_name])
        self.stores[shift_name]["n_negative_per_group"][var][f"{group}_{syst_name}"] = copy.deepcopy(self.stores["Nom"]["n_negative_per_file"][var][file_name])
      else:
        self.stores[shift_name]["hists_per_group"][var][f"{group}_{syst_name}"] += self.stores["Nom"]["hists_per_file"][var][file_name]
        self.stores[shift_name]["hists_squared_per_group"][var][f"{group}_{syst_name}"] += self.stores["Nom"]["hists_squared_per_file"][var][file_name]
        self.stores[shift_name]["n_events_per_group"][var][f"{group}_{syst_name}"] += self.stores["Nom"]["n_events_per_file"][var][file_name]
        self.stores[shift_name]["n_positive_per_group"][var][f"{group}_{syst_name}"] += self.stores["Nom"]["n_positive_per_file"][var][file_name]
        self.stores[shift_name]["n_negative_per_group"][var][f"{group}_{syst_name}"] += self.stores["Nom"]["n_negative_per_file"][var][file_name]
      if group in self.cfg["groups"].keys():
        if syst_name not in self.stores[shift_name]["hists"][var]:
          self.stores[shift_name]["hists"][var][syst_name] = copy.deepcopy(self.stores["Nom"]["hists_per_file"][var][file_name])
          self.stores[shift_name]["hists_squared"][var][syst_name] = copy.deepcopy(self.stores["Nom"]["hists_squared_per_file"][var][file_name])
          self.stores[shift_name]["n_events"][var][syst_name] = copy.deepcopy(self.stores["Nom"]["n_events_per_file"][var][file_name])
          self.stores[shift_name]["n_positive"][var][syst_name] = copy.deepcopy(self.stores["Nom"]["n_positive_per_file"][var][file_name])
          self.stores[shift_name]["n_negative"][var][syst_name] = copy.deepcopy(self.stores["Nom"]["n_negative_per_file"][var][file_name])
        else:
          self.stores[shift_name]["hists"][var][syst_name] += self.stores["Nom"]["hists_per_file"][var][file_name]
          self.stores[shift_name]["hists_squared"][var][syst_name] += self.stores["Nom"]["hists_squared_per_file"][var][file_name]
          self.stores[shift_name]["n_events"][var][syst_name] += self.stores["Nom"]["n_events_per_file"][var][file_name]
          self.stores[shift_name]["n_positive"][var][syst_name] += self.stores["Nom"]["n_positive_per_file"][var][file_name]
          self.stores[shift_name]["n_negative"][var][syst_name] += self.stores["Nom"]["n_negative_per_file"][var][file_name]
    
  def _rebin_histograms(self):

    for var in self.cfg["variables"].keys():
      old_bins = copy.deepcopy(self.bin_store[var])
      if self.rebin_bins is None:
        new_bins = find_rebinning(self.stores["Nom"]["hists"][var][f"Total_{self.rebin_from}"], old_bins, bin_threshold=self.rebin_count, bin_uncert_frac_threshold=self.rebin_fraction)
      else:
        new_bins = np.array([float(x) for x in self.rebin_bins.split(",")])
      self.bin_store[var] = copy.deepcopy(new_bins)
      for shift_name in ["Nom", "Up", "Down"]:
        for hist_name, var_name in {"hists":"hists_squared", "hists_per_file":"hists_squared_per_file", "hists_per_group":"hists_squared_per_group"}.items():
          if var not in self.stores[shift_name][hist_name].keys(): continue
          for group, hist in self.stores[shift_name][hist_name][var].items():
            hist, uncert = rebin_histogram(hist, old_bins, new_bins, uncert=self.stores[shift_name][var_name][var][group]**0.5)
            self.stores[shift_name][hist_name][var][group] = copy.deepcopy(hist)
            self.stores[shift_name][var_name][var][group] = uncert**2

  def Run(self):

    # Loop over files
    for f in self.files:

      # Check if file is in groups
      k = self._check_file_in_groups(f)
      if k is None: continue

      # Get the file names
      file_name, era_name, file_name_minus_era = self._get_file_names(f)

      print(f"Processing {file_name} for group {k}")

      # Read the parquet file
      if self.load_from_root is None:
        df = pd.read_parquet(f)
        if len(df) == 0: continue
        # Apply pre-selection if provided
        if self.pre_sel is not None:
          df = df.query(self.pre_sel)
      else:
        df = None

      # Get nominal histograms and return nominal df
      calculate = [{"weight": self.weight}]
      calculate += [self.cfg["calculate"]] if "calculate" in self.cfg else []
      sel = [self.sel] if self.sel is not None else []
      sel += [self.cfg["group_selection"][k]] if k in self.cfg["group_selection"].keys() else []
      _ = self._get_histograms(copy.deepcopy(df), calculate, sel, file_name, k, "Total_MC" if "Data" not in k else "Total_Data", save_to="Nom", group_name=k)

      # Get systemtics histograms
      if "systematics" in self.cfg and "Data" not in k and self.syst:
        for syst_name, syst_info in self.cfg["systematics"].items():
          for shift_name, syst_val in {"Down":-1.0,"Up":1.0}.items():
            if file_name_minus_era in syst_info["files"] and era_name in syst_info["years"]:
              syst_calculate = [{syst_name: syst_val}]
              syst_calculate += [self.cfg["calculate"]] if "calculate" in self.cfg else []
              syst_calculate += [syst_info["functions"]]
              _ = self._get_histograms(copy.deepcopy(df), syst_calculate, sel, f"{file_name}_{syst_name}", f"{k}_{syst_name}", syst_name, save_to=shift_name, group_name=k)
            else:
              self._add_nom_to_syst_groups(shift_name, k, file_name, syst_name)

    # Rebin
    if self.rebin:
      self._rebin_histograms()

    # Write root files
    if self.write and self.load_from_root is None:
      self._write_root_files()


if args.var is None and args.rebin:
  raise ValueError("Please specify a variable to rebin. Use --var to specify the variable.")

# Make histograms
gh = GetHistograms(
  input_folder=args.input,
  cfg=args.cfg,
  var=args.var,
  sel=args.sel,
  bins=args.bins,
  num_bins=args.num_bins,
  year=args.year,
  normalise=args.normalise,
  weight=args.weight,
  write=args.write,
  calculate=args.calculate,
  scale=args.scale,
  syst=args.syst,
  pre_sel=args.pre_sel,
  rebin=args.rebin,
  rebin_count=args.rebin_count,
  rebin_fraction=args.rebin_fraction,
  rebin_from=args.rebin_from,
  load_from_root=args.load_from_root,
  rebin_bins=args.rebin_bins
)
gh.Run()


# Run plotting
for var in gh.cfg["variables"].keys():

  # Get data
  if "Total_Data" not in gh.stores["Nom"]["hists"][var].keys():
    data_hist = None
    data_uncert = None
    data_name = None
    total_data = None
  else:
    data_hist = gh.stores["Nom"]["hists"][var]["Total_Data"]
    data_uncert = np.sqrt(gh.stores["Nom"]["hists_squared"][var]["Total_Data"])
    data_name = "Data"
    total_data = np.sum(data_hist)

  # Get MC histograms
  hists = {k: gh.stores["Nom"]["hists_per_group"][var][k] for k in list(gh.cfg["groups"].keys())[::-1] if k != "Data"}
  hists_squared = {k: gh.stores["Nom"]["hists_squared_per_group"][var][k] for k in gh.cfg["groups"].keys() if k != "Data"}
  total_mc = np.sum(gh.stores["Nom"]["hists"][var]["Total_MC"])
  extra_hists = {}
  for k in list(gh.cfg["plot_extra"].keys())[::-1]:
    extra_hists[k] = copy.deepcopy(gh.stores["Nom"]["hists"][var]["Total_MC"])
    for v in gh.cfg["plot_extra"][k]:
      extra_hists[k] += gh.stores["Nom"]["hists_per_group"][var][v]
    if k in gh.cfg["plot_extra_subtract"].keys():
      for v in gh.cfg["plot_extra_subtract"][k]:
        extra_hists[k] -= gh.stores["Nom"]["hists_per_group"][var][v]

  # Get n events and positive weights
  n_events = {k: gh.stores["Nom"]["n_events_per_group"][var][k] for k in gh.cfg["groups"].keys()}
  n_positive = {k: gh.stores["Nom"]["n_positive_per_group"][var][k] for k in gh.cfg["groups"].keys()}
  n_negative = {k: gh.stores["Nom"]["n_negative_per_group"][var][k] for k in gh.cfg["groups"].keys()}

  # Normalise histograms if required
  if args.normalise and data_hist is not None:
    for k in hists:
      hists[k] = hists[k] * (total_data / total_mc)
      hists_squared[k] = hists_squared[k] * (total_data / total_mc)**2

  # Get uncertainties
  if args.syst and len(gh.cfg["systematics"].keys()) > 0:
    stack_hist_errors = None
    up_variance = gh.stores["Nom"]["hists_squared"][var]["Total_MC"]
    down_variance = gh.stores["Nom"]["hists_squared"][var]["Total_MC"]
    nom_hist = gh.stores["Nom"]["hists"][var]["Total_MC"]
    for syst_name in gh.cfg["systematics"].keys():
      up_shift = np.zeros_like(nom_hist)
      down_shift = np.zeros_like(nom_hist)
      if syst_name in gh.stores["Up"]["hists"][var].keys():
        up_shift = gh.stores["Up"]["hists"][var][f"{syst_name}"] - nom_hist
      if syst_name in gh.stores["Down"]["hists"][var].keys():
        down_shift = gh.stores["Down"]["hists"][var][f"{syst_name}"] - nom_hist
      up_hist = np.array([max(max(down_shift[i], up_shift[i]),0.0) for i in range(len(down_variance))])
      down_hist = np.array([min(min(down_shift[i], up_shift[i]),0.0) for i in range(len(down_variance))])
      up_variance += up_hist**2
      down_variance += down_hist**2
    stack_hist_errors_asym = {
      "up": np.sqrt(up_variance),
      "down": np.sqrt(down_variance)
    }
  else:
    stack_hist_errors = np.sqrt(gh.stores["Nom"]["hists_squared"][var]["Total_MC"])
    stack_hist_errors_asym = None


  # Make table
  RED = "\033[91m"
  GREEN = "\033[92m"
  BLUE = "\033[94m" 
  RESET = "\033[0m"
  tabulated_data = [["Group", "Sum of Weights", "Number of Events", "Positive Weight Fraction"]]
  print(f"Table for {var} for {args.year}")
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

  lumi_labels = {
    "all" : "$200\ fb^{-1}\ (13,13.6\ TeV)$",
    "run2" : "$138\ fb^{-1}\ (13\ TeV)$",
    "run3" : "$61.9\ fb^{-1}\ (13.6\ TeV)$",
    "2016_PreVFP" : "$19.6\ fb^{-1}\ (13\ TeV)$",
    "2016_PostVFP" : "$17.0\ fb^{-1}\ (13\ TeV)$",
    "2017" : "$41.5\ fb^{-1}\ (13\ TeV)$",
    "2018" : "$59.8\ fb^{-1}\ (13\ TeV)$",
    "2022_preEE" : "$7.98\ fb^{-1}\ (13.6\ TeV)$",
    "2022_postEE" : "$26.7\ fb^{-1}\ (13.6\ TeV)$",
    "2023_preBPix" : "$17.8\ fb^{-1}\ (13.6\ TeV)$",
    "2023_postBPix" : "$9.45\ fb^{-1}\ (13.6\ TeV)$"
  }


  # Normalise to bin width
  ylabel = "Events"
  if args.norm_to_bin_width:
    bins_widths = np.array([gh.bin_store[var][ind+1] - gh.bin_store[var][ind] for ind in range(len(gh.bin_store[var])-1)])
    data_hist /= bins_widths
    for k, v in hists.items():
      hists[k] /= bins_widths
    if data_uncert is not None:
      data_uncert /= bins_widths
    for k, v in extra_hists.items():
      extra_hists[k] /= bins_widths
    if stack_hist_errors is not None:
      stack_hist_errors /= bins_widths
    if stack_hist_errors_asym is not None:
      stack_hist_errors_asym["up"] /= bins_widths
      stack_hist_errors_asym["down"] /= bins_widths

    ylabel = "Events / GeV"

  # Make plot
  plot_stacked_histogram_with_ratio(
    data_hist,
    hists,
    gh.bin_store[var],
    data_name=data_name,
    xlabel=var if var not in gh.cfg["translate"] else gh.cfg["translate"][var],
    ylabel=ylabel,
    name=os.path.join(args.output, f"distribution_{var}_{args.year}"),
    data_errors=data_uncert,
    stack_hist_errors=stack_hist_errors,
    stack_hist_errors_asym=stack_hist_errors_asym,
    use_stat_err=False,
    axis_text="",
    top_space=1.2,
    draw_ratio=True,
    colours=gh.cfg["colours"],
    include_fraction=args.include_fraction,
    line_hist_dict=extra_hists,
    line_colours=gh.cfg["plot_extra_colours"],
    cms_label=args.cms_label,
    lumi_label=lumi_labels[args.year]
  )


  # Plot systematic variations if required
  if args.syst and len(gh.cfg["systematics"].keys()) > 0 and args.plot_syst_variation:
    for syst_name in gh.cfg["systematics"].keys():
      for group in gh.cfg["groups"].keys():
        if group == "Data": continue
        if group not in gh.stores["Nom"]["hists_per_group"][var].keys(): continue
        if f"{group}_{syst_name}" not in gh.stores["Up"]["hists_per_group"][var].keys(): continue
        if f"{group}_{syst_name}" not in gh.stores["Down"]["hists_per_group"][var].keys(): continue

        # Get the up and down histograms
        up_hist = gh.stores["Up"]["hists_per_group"][var][f"{group}_{syst_name}"]
        down_hist = gh.stores["Down"]["hists_per_group"][var][f"{group}_{syst_name}"]
        nom_hist = gh.stores["Nom"]["hists_per_group"][var][group]

        group_name = group.replace("(","").replace(")","").replace("[","").replace("]","").replace(",","_").replace(".","").replace(" ","_")

        plot_histograms_with_ratio(
          [nom_hist, up_hist, down_hist],
          [None, None, None],
          ["Nominal", "Up", "Down"],
          gh.bin_store[var],
          xlabel=var if var not in gh.cfg["translate"] else gh.cfg["translate"][var],
          ylabel="Events",
          name=f"{args.output}/systematic_variation_{syst_name}_{group_name}_{var}_{args.year}",
          ratio_range=[0.9,1.1],
          axis_text=f"{group} {syst_name}"
        )
