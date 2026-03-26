import argparse
import copy
import fnmatch
import glob
import gc
import matplotlib.pylab as plt
import os
import re
import pandas as pd
import numpy as np
import mplhep as hep
import yaml
import uproot
import boost_histogram as bh
from tabulate import tabulate
import importlib.util
import sys
from pathlib import Path
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
parser.add_argument('--scale-to', help='Comma separate list of key to scale and the key to scale to', type=str, default=None)
parser.add_argument('--xlabel', help='X label for plot. If none will use var', type=str, default=None)
parser.add_argument('--normalise', help='Normalise the MC to data', action='store_true', default=False)
parser.add_argument('--weight', help='The weight to apply to the histograms', type=str, default="weight")
parser.add_argument('--write', help='Write histogram to root file', action='store_true', default=False)
parser.add_argument('--syst', help='Process systematics', action='store_true', default=False)
parser.add_argument('--plot-syst-variation', help='Plot the systematic variations', action='store_true', default=False)
parser.add_argument('--rebin', help='Rebin the histogram', action='store_true', default=False)
parser.add_argument('--rebin-fraction', help='The bin uncertainty fraction threshold', type=float, default=0.1)
parser.add_argument('--rebin-count', help='The bin count threshold', type=float, default=10)
parser.add_argument('--rebin-from', help='Data or MC', type=str, default="Data")
parser.add_argument('--rebin-bins', help='Comma separated list to rebin to if loading in histograms', type=str, default=None)
parser.add_argument('--norm-groups-to-data', help='Comma separated list of groups to normalise to data', type=str, default=None)
parser.add_argument('--norm-to-bin-width', help='Normalise to bin width', action='store_true', default=False)
parser.add_argument('--load-from-root', help='Root file to load histograms from', type=str, default=None)
parser.add_argument('--specific-histogram', help='Specific histogram to make', type=str, default=None)
parser.add_argument('--submit', help='Submit all histograms to the batch', action='store_true', default=False)
parser.add_argument('--hadd', help='Submit all histograms to the batch', action='store_true', default=False)
parser.add_argument('--points-per-job', help='Number of points per job', type=int, default=20)
parser.add_argument('--write-after-load', help='Write histogram to root file after loading from root', action='store_true', default=False)
parser.add_argument('--extra-name', help='Extra name for root output', type=str, default=None)


args = parser.parse_args()

def MakeCommandOptions(args):
  parsers = []
  for attr, value in vars(args).items():
    if value is None or (value is False and isinstance(value,bool)):
      continue
    attr_name = attr.replace("_","-")
    if value is True and isinstance(value,bool):
      parsers.append(f'--{attr_name}')
    elif isinstance(value, str):
      if attr_name in ["input","output","cfg"]:
        value = os.path.abspath(value)
      parsers.append(f'--{attr_name}="{value}"')
    else:
      parsers.append(f'--{attr_name}={value}')
  return parsers

parsers = MakeCommandOptions(args)
if '--submit' in parsers:
  parsers.remove('--submit')
command = f"python3 {os.path.abspath(__file__)} {' '.join(parsers)}"


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
    rebin_bins=None,
    specific_histogram=None,
    command=None,
    submit=False,
    points_per_job=20,
    write_after_load=False,
    output = "./",
    extra_name=None,
    norm_groups_to_data=None,
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
    self.specific_histogram = specific_histogram
    self.command = command
    self.submit = submit
    self.write_after_load = write_after_load
    self.output = output
    self.extra_name = extra_name
    self.norm_groups_to_data = norm_groups_to_data.split(",") if norm_groups_to_data is not None else []

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
      "sum_wt",
      "sum_wt_per_file",
      "sum_wt_per_group",
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
    self.cmd_store = []
    self.n_per_job = points_per_job
    self.submit_ind = 0
    self.command_ind = 0
    self.files_groups_ran = {}

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
      self.eras = ["2016_PreVFP", "2016_PostVFP", "2017", "2018", "2022_preEE", "2022_postEE", "2023_preBPix", "2023_postBPix"]
    elif self.year == "run2":
      self.wildcard = ["*2016_PreVFP*", "*2016_PostVFP*", "*2017*", "*2018*"]
      self.eras = ["2016_PreVFP", "2016_PostVFP", "2017", "2018"]
    elif self.year == "2223":
      self.wildcard = ["*2022_preEE*", "*2022_postEE*", "*2023_preBPix*", "*2023_postBPix*"]
      self.eras = ["2022_preEE", "2022_postEE", "2023_preBPix", "2023_postBPix"]
    elif self.year == "run3":
      self.wildcard = ["*2022_preEE*", "*2022_postEE*", "*2023_preBPix*", "*2023_postBPix*", "*2024*"]
      self.eras = ["2022_preEE", "2022_postEE", "2023_preBPix", "2023_postBPix", "2024"]
    elif self.year in ["2016_PreVFP", "2016_PostVFP", "2017", "2018", "2022_preEE", "2022_postEE", "2023_preBPix", "2023_postBPix", "2024"]:
      self.wildcard = f"*{self.year}*"
      self.eras = [self.year]
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
    
    # If file in group twice, duplicate it
    new_files = []
    for group_files in self.total_groups.values():
      for gf in group_files:
        for file in self.files:
          file_name = file.split("/")[-1]
          if "*" not in gf:
            if file_name in gf:
              new_files.append(file)
          elif fnmatch.fnmatch(file_name, gf):
            new_files.append(file)
      
    self.files = copy.deepcopy(new_files)

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

          if k not in self.files_groups_ran.keys():
            self.files_groups_ran[k] = []

          if f in self.files_groups_ran[k]: continue

          match_found = True
          group = k

          self.files_groups_ran[k].append(f)
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
      if self.extra_name is None:
        root_name = os.path.join(self.output, f"datacard_{var}.root")
      else:
        root_name = os.path.join(self.output, f"datacard_{var}_{self.extra_name}.root")
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

    # Close files
    for var in self.root_files.keys():
      self.root_files[var].close()

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
        bins = self.num_bins
      self.bin_store[var] = bins

  def _get_tokens(self, input):

    tokens = re.findall(r"[A-Za-z_]\w*", input)
    reserved = {"and", "or", "not", "cos", "sin", "sinh", "cosh", "tanh", "abs", "exp", "sqrt", "arctan2", "arcsinh", "arccosh", "arctanh", "log", "log10"}
    return [t for t in tokens if t not in reserved]

  def _get_histograms(self, df, calculate, selection, file_name, group, total_name, save_to="Nom", group_name=None, function_to_apply=None, metadata_for_function={}):

    # Calculate the histograms
    if self.load_from_root is None:
      if df.empty: return

      for calc in calculate:
        for col_name, func in calc.items():
          if isinstance(func, (int, float)):
            df[col_name] = float(func)
          else:
            tokens = self._get_tokens(func)
            local_dict = {
              "cos": np.cos,
              "sin": np.sin,
              "sinh": np.sinh,
              "sqrt": np.sqrt,
              "arctan2": np.arctan2,
              "arcsinh": np.arcsinh,
              "log": np.log,
            }
            for k in tokens:
              local_dict[k] = df[k]
            df[col_name] = eval(func, {"np": np}, local_dict)

      # Apply function to apply
      if function_to_apply is not None:
        df = function_to_apply(df, metadata={"file_name": file_name, "group": group, **metadata_for_function})

      # Apply selection if provided
      for sel in selection:
        df = df.query(sel)

      if df.empty: return

    for var in self.cfg["variables"].keys():

      if self.load_from_root is None:
        # Mask valid entries
        valid = df["weight"].notna() & df[var].notna()

      # Set up stores
      for store_name, store_dict in self.stores[save_to].items():
        if var not in store_dict.keys():
          self.stores[save_to][store_name][var] = {}

      if self.load_from_root is None:
        hist, bins = np.histogram(df.loc[valid,var], bins=self.bin_store[var], weights=df.loc[valid,"weight"], density=False)
        if isinstance(self.bin_store[var], int):
          self.bin_store[var] = copy.deepcopy(bins)
        hist_squared, _ = np.histogram(df.loc[valid,var], bins=self.bin_store[var], weights=df.loc[valid,"weight"]**2, density=False)
        w = df["weight"].to_numpy()
        n = w.size
        n_pos = (w >= 0).sum()
        n_neg = (w < 0).sum()
        sum_wt = w.sum(dtype=float)
      else:
        root_file_name = copy.deepcopy(file_name)
        if save_to in ["Up","Down"]:
          root_file_name += save_to
        root_hist = self.root_file.get(root_file_name)
        if root_hist is None: continue
        hist, bins = root_hist.to_numpy()
        hist_squared = root_hist.variances()
        n = 0
        n_pos = 0
        n_neg = 0
        sum_wt = np.sum(hist)

      # Scale hists
      if group_name in self.scale_factors.keys():
        #print(f"Scaling {file_name} by {self.scale_factors[group_name]}")
        hist *= self.scale_factors[group_name]
        hist_squared *= self.scale_factors[group_name]**2

      # Save to dictionaries
      if self.specific_histogram is None:
        if group_name is None or group_name in self.cfg["groups"].keys():
          if total_name not in self.stores[save_to]["hists"][var]:
            self.stores[save_to]["hists"][var][total_name] = copy.deepcopy(hist)
            self.stores[save_to]["hists_squared"][var][total_name] = copy.deepcopy(hist_squared)
            self.stores[save_to]["n_events"][var][total_name] = copy.deepcopy(n)
            self.stores[save_to]["n_positive"][var][total_name] = copy.deepcopy(n_pos)
            self.stores[save_to]["n_negative"][var][total_name] = copy.deepcopy(n_neg)
            self.stores[save_to]["sum_wt"][var][total_name] = copy.deepcopy(sum_wt)
          else:
            self.stores[save_to]["hists"][var][total_name] += hist
            self.stores[save_to]["hists_squared"][var][total_name] += hist_squared
            self.stores[save_to]["n_events"][var][total_name] += n
            self.stores[save_to]["n_positive"][var][total_name] += n_pos
            self.stores[save_to]["n_negative"][var][total_name] += n_neg
            self.stores[save_to]["sum_wt"][var][total_name] += sum_wt
        if group not in self.stores[save_to]["hists_per_group"][var]:
          self.stores[save_to]["hists_per_group"][var][group] = copy.deepcopy(hist)
          self.stores[save_to]["hists_squared_per_group"][var][group] = copy.deepcopy(hist_squared)
          self.stores[save_to]["n_events_per_group"][var][group] = copy.deepcopy(n)
          self.stores[save_to]["n_positive_per_group"][var][group] = copy.deepcopy(n_pos)
          self.stores[save_to]["n_negative_per_group"][var][group] = copy.deepcopy(n_neg)
          self.stores[save_to]["sum_wt_per_group"][var][group] = copy.deepcopy(sum_wt)
        else:
          self.stores[save_to]["hists_per_group"][var][group] += hist
          self.stores[save_to]["hists_squared_per_group"][var][group] += hist_squared
          self.stores[save_to]["n_events_per_group"][var][group] += n
          self.stores[save_to]["n_positive_per_group"][var][group] += n_pos
          self.stores[save_to]["n_negative_per_group"][var][group] += n_neg
          self.stores[save_to]["sum_wt_per_group"][var][group] += sum_wt
      if file_name not in self.stores[save_to]["hists_per_file"][var]:
        self.stores[save_to]["hists_per_file"][var][file_name] = copy.deepcopy(hist)
        self.stores[save_to]["hists_squared_per_file"][var][file_name] = copy.deepcopy(hist_squared)
        self.stores[save_to]["n_events_per_file"][var][file_name] = copy.deepcopy(n)
        self.stores[save_to]["n_positive_per_file"][var][file_name] = copy.deepcopy(n_pos)
        self.stores[save_to]["n_negative_per_file"][var][file_name] = copy.deepcopy(n_neg)
        self.stores[save_to]["sum_wt_per_file"][var][file_name] = copy.deepcopy(sum_wt)
      else:
        self.stores[save_to]["hists_per_file"][var][file_name] += hist
        self.stores[save_to]["hists_squared_per_file"][var][file_name] += hist_squared
        self.stores[save_to]["n_events_per_file"][var][file_name] += n
        self.stores[save_to]["n_positive_per_file"][var][file_name] += n_pos
        self.stores[save_to]["n_negative_per_file"][var][file_name] += n_neg
        self.stores[save_to]["sum_wt_per_file"][var][file_name] += sum_wt

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
        self.stores[shift_name]["sum_wt_per_group"][var][f"{group}_{syst_name}"] = copy.deepcopy(self.stores["Nom"]["sum_wt_per_file"][var][file_name])
      else:
        self.stores[shift_name]["hists_per_group"][var][f"{group}_{syst_name}"] += self.stores["Nom"]["hists_per_file"][var][file_name]
        self.stores[shift_name]["hists_squared_per_group"][var][f"{group}_{syst_name}"] += self.stores["Nom"]["hists_squared_per_file"][var][file_name]
        self.stores[shift_name]["n_events_per_group"][var][f"{group}_{syst_name}"] += self.stores["Nom"]["n_events_per_file"][var][file_name]
        self.stores[shift_name]["n_positive_per_group"][var][f"{group}_{syst_name}"] += self.stores["Nom"]["n_positive_per_file"][var][file_name]
        self.stores[shift_name]["n_negative_per_group"][var][f"{group}_{syst_name}"] += self.stores["Nom"]["n_negative_per_file"][var][file_name]
        self.stores[shift_name]["sum_wt_per_group"][var][f"{group}_{syst_name}"] += self.stores["Nom"]["sum_wt_per_file"][var][file_name]
      if group in self.cfg["groups"].keys():
        if syst_name not in self.stores[shift_name]["hists"][var]:
          self.stores[shift_name]["hists"][var][syst_name] = copy.deepcopy(self.stores["Nom"]["hists_per_file"][var][file_name])
          self.stores[shift_name]["hists_squared"][var][syst_name] = copy.deepcopy(self.stores["Nom"]["hists_squared_per_file"][var][file_name])
          self.stores[shift_name]["n_events"][var][syst_name] = copy.deepcopy(self.stores["Nom"]["n_events_per_file"][var][file_name])
          self.stores[shift_name]["n_positive"][var][syst_name] = copy.deepcopy(self.stores["Nom"]["n_positive_per_file"][var][file_name])
          self.stores[shift_name]["n_negative"][var][syst_name] = copy.deepcopy(self.stores["Nom"]["n_negative_per_file"][var][file_name])
          self.stores[shift_name]["sum_wt"][var][syst_name] = copy.deepcopy(self.stores["Nom"]["sum_wt_per_file"][var][file_name])
        else:
          self.stores[shift_name]["hists"][var][syst_name] += self.stores["Nom"]["hists_per_file"][var][file_name]
          self.stores[shift_name]["hists_squared"][var][syst_name] += self.stores["Nom"]["hists_squared_per_file"][var][file_name]
          self.stores[shift_name]["n_events"][var][syst_name] += self.stores["Nom"]["n_events_per_file"][var][file_name]
          self.stores[shift_name]["n_positive"][var][syst_name] += self.stores["Nom"]["n_positive_per_file"][var][file_name]
          self.stores[shift_name]["n_negative"][var][syst_name] += self.stores["Nom"]["n_negative_per_file"][var][file_name]
          self.stores[shift_name]["sum_wt"][var][syst_name] += self.stores["Nom"]["sum_wt_per_file"][var][file_name]
    
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

  def _norm_groups_to_data(self):

    for var in self.cfg["variables"].keys():

      data_integral = self.stores["Nom"]["sum_wt"][var]["Total_Data"]
      mc_integral = self.stores["Nom"]["sum_wt"][var]["Total_MC"]

      group_integral = 0.0
      for group in self.norm_groups_to_data:
        group_integral += self.stores["Nom"]["sum_wt_per_group"][var][group]

      #group_integral = np.sum(total_group_hist)
      norm_factor = (data_integral - mc_integral + group_integral) / group_integral 
      post_scale_factors = {}
      for group in self.norm_groups_to_data:
        post_scale_factors[group] = norm_factor

      # Loop over files
      self.files_groups_ran = {}
      for f in self.files:

        # Check if file is in groups
        k = self._check_file_in_groups(f)

        if k is None: continue
        if k not in self.norm_groups_to_data: continue

        # Get the file names
        file_name, era_name, file_name_minus_era = self._get_file_names(f)
        total_name = "Total_MC" if "Data" not in k else "Total_Data"

        scale_factor = post_scale_factors[k]

        # Do Nom
        save_to = "Nom"
        if var not in self.stores[save_to]["hists_per_file"]: continue
        if file_name not in self.stores[save_to]["hists_per_file"][var]: continue

        # Do totals by subtractin old and adding new
        self.stores[save_to]["hists"][var][total_name] += (self.stores[save_to]["hists_per_file"][var][file_name] * (scale_factor - 1.0))
        self.stores[save_to]["hists_squared"][var][total_name] += self.stores[save_to]["hists_squared_per_file"][var][file_name] * (scale_factor**2 - 1.0)
        self.stores[save_to]["sum_wt"][var][total_name] += self.stores[save_to]["sum_wt_per_file"][var][file_name] * (scale_factor - 1.0)

        # Do total systematics
        if "systematics" in self.cfg and "Data" not in k and self.syst:
          for syst_name, syst_info in self.cfg["systematics"].items():
            for shift in ["Down", "Up"]:
              if var not in self.stores[shift]["hists_per_file"]: continue
              file_key = f"{file_name}_{syst_name}"
              if file_key not in self.stores[shift]["hists_per_file"][var]: continue

              self.stores[shift]["hists"][var][syst_name] += self.stores[shift]["hists_per_file"][var][file_key] * (scale_factor - 1.0)
              self.stores[shift]["hists_squared"][var][syst_name] += self.stores[shift]["hists_squared_per_file"][var][file_key] * (scale_factor**2 - 1.0)
              self.stores[shift]["sum_wt"][var][syst_name] += self.stores[shift]["sum_wt_per_file"][var][file_key] * (scale_factor - 1.0)

        # Scale groups
        self.stores["Nom"]["hists_per_group"][var][k] += self.stores[save_to]["hists_per_file"][var][file_name] * (scale_factor - 1.0)
        self.stores["Nom"]["hists_squared_per_group"][var][k] += self.stores[save_to]["hists_squared_per_file"][var][file_name] * (scale_factor**2 - 1.0)
        self.stores["Nom"]["sum_wt_per_group"][var][k] += self.stores[save_to]["sum_wt_per_file"][var][file_name] * (scale_factor - 1.0)

        # Do systematics groups
        if "systematics" in self.cfg and "Data" not in k and self.syst:
          for syst_name, syst_info in self.cfg["systematics"].items():
            for shift in ["Down", "Up"]:
              if var not in self.stores[shift]["hists_per_file"]: continue
              file_key = f"{file_name}_{syst_name}"
              if file_key not in self.stores[shift]["hists_per_file"][var]: continue
              self.stores[shift]["hists_per_group"][var][f"{k}_{syst_name}"] += self.stores[shift]["hists_per_file"][var][file_key] * (scale_factor - 1.0)
              self.stores[shift]["hists_squared_per_group"][var][f"{k}_{syst_name}"] += self.stores[shift]["hists_squared_per_file"][var][file_key] * (scale_factor**2 - 1.0)
              self.stores[shift]["sum_wt_per_group"][var][f"{k}_{syst_name}"] += self.stores[shift]["sum_wt_per_file"][var][file_key] * (scale_factor - 1.0)

        # Do Nom
        self.stores[save_to]["hists_per_file"][var][file_name] = self.stores[save_to]["hists_per_file"][var][file_name] * scale_factor
        self.stores[save_to]["hists_squared_per_file"][var][file_name] = self.stores[save_to]["hists_squared_per_file"][var][file_name] * (scale_factor**2)
        self.stores[save_to]["sum_wt_per_file"][var][file_name] = self.stores[save_to]["sum_wt_per_file"][var][file_name] * scale_factor

        # Do Up and Down
        if "systematics" in self.cfg and "Data" not in k and self.syst:
          for syst_name, syst_info in self.cfg["systematics"].items():
            for save_to in ["Down", "Up"]:
              if era_name not in self.eras: continue
              file_key = f"{file_name}_{syst_name}"
              if var not in self.stores[save_to]["hists_per_file"]: continue
              if file_key not in self.stores[save_to]["hists_per_file"][var]: continue
              self.stores[save_to]["hists_per_file"][var][file_key] = self.stores[save_to]["hists_per_file"][var][file_key] * scale_factor
              self.stores[save_to]["hists_squared_per_file"][var][file_key] = self.stores[save_to]["hists_squared_per_file"][var][file_key] * (scale_factor**2)
              self.stores[save_to]["sum_wt_per_file"][var][file_key] = self.stores[save_to]["sum_wt_per_file"][var][file_key] * scale_factor


  def _scale_to(self):

    for var in self.cfg["variables"].keys():

      # Loop over files
      self.files_groups_ran = {}
      for f in self.files:

        # Check if file is in groups
        k = self._check_file_in_groups(f)

        if k is None: continue

        # Check whether need to use
        if k not in self.cfg["scale_to"].keys(): continue

        # Get the file names
        file_name, era_name, file_name_minus_era = self._get_file_names(f)

        total_name = "Total_MC" if "Data" not in k else "Total_Data"

        # Get scale factor
        if isinstance(self.cfg["scale_to"][k], str):
          scale_to_files = [self.cfg["scale_to"][k]]
        else:
          scale_to_files = self.cfg["scale_to"][k]

        numerator = 0.0
        for scale_to_file in scale_to_files:
          if scale_to_file in self.stores["Nom"]["sum_wt_per_group"][var].keys():
            numerator += self.stores["Nom"]["sum_wt_per_group"][var][scale_to_file]
        scale_factor = numerator / self.stores["Nom"]["sum_wt_per_group"][var][k]

        # Do Nom
        save_to = "Nom"
        if var not in self.stores[save_to]["hists_per_file"]:
          continue
        if file_name not in self.stores[save_to]["hists_per_file"][var]:
          continue

        nom_shift_hist = self.stores[save_to]["hists_per_file"][var][file_name] * (scale_factor - 1.0)
        nom_shift_hist_squared = self.stores[save_to]["hists_squared_per_file"][var][file_name] * (scale_factor**2 - 1.0)

        if var in self.stores[save_to]["hists_per_file"]:
          if file_name in self.stores[save_to]["hists_per_file"][var]:
            self.stores[save_to]["hists_per_file"][var][file_name] += nom_shift_hist
            self.stores[save_to]["hists_squared_per_file"][var][file_name] += nom_shift_hist_squared

        # Do Up and Down
        if "systematics" in self.cfg and "Data" not in k and self.syst:
          for syst_name, syst_info in self.cfg["systematics"].items():
            for save_to in ["Down", "Up"]:
              if era_name not in self.eras: continue
              file_key = f"{file_name}_{syst_name}"

              if var not in self.stores[save_to]["hists_per_file"]:
                continue
              if file_key not in self.stores[save_to]["hists_per_file"][var]:
                continue

              group_name = f"{k}_{syst_name}"

              shift_hist = self.stores[save_to]["hists_per_file"][var][file_key] * (scale_factor - 1.0)
              shift_hist_squared = self.stores[save_to]["hists_squared_per_file"][var][file_key] * (scale_factor**2 - 1.0)

              if var in self.stores[save_to]["hists_per_file"]:
                if file_key in self.stores[save_to]["hists_per_file"][var]:
                  self.stores[save_to]["hists_per_file"][var][file_key] += shift_hist
                  self.stores[save_to]["hists_squared_per_file"][var][file_key] += shift_hist_squared

      # Scale groups
      for k in self.cfg["scale_to"].keys():

        if k not in self.stores["Nom"]["hists_per_group"][var].keys(): continue
        
        if isinstance(self.cfg["scale_to"][k], str):
          scale_to_files = [self.cfg["scale_to"][k]]
        else:
          scale_to_files = self.cfg["scale_to"][k]

        numerator = 0.0
        for scale_to_file in scale_to_files:
          if scale_to_file in self.stores["Nom"]["sum_wt_per_group"][var].keys():
            numerator += self.stores["Nom"]["sum_wt_per_group"][var][scale_to_file]

        scale_factor = numerator / self.stores["Nom"]["sum_wt_per_group"][var][k]

        # Total
        if k in self.cfg["groups"].keys():
          total_name = "Total_MC" if "Data" not in k else "Total_Data"
          if total_name in self.stores["Nom"]["hists"][var].keys():
            self.stores["Nom"]["hists"][var][total_name] += self.stores["Nom"]["hists_per_group"][var][k] * (scale_factor - 1.0)
            self.stores["Nom"]["hists_squared"][var][total_name] += self.stores["Nom"]["hists_squared_per_group"][var][k] * (scale_factor**2 - 1.0)

        # Nom group
        self.stores["Nom"]["hists_per_group"][var][k] *= scale_factor
        self.stores["Nom"]["hists_squared_per_group"][var][k] *= scale_factor**2

        # Up and Down
        for save_to in ["Up", "Down"]:
          if "systematics" in self.cfg and "Data" not in k and self.syst:
            for syst_name, syst_info in self.cfg["systematics"].items():
              group_name = f"{k}_{syst_name}"
              if group_name in self.stores[save_to]["hists_per_group"][var].keys():

                # Total
                if k in self.cfg["groups"].keys():
                  if syst_name in self.stores[save_to]["hists"][var].keys():
                    self.stores[save_to]["hists"][var][syst_name] += self.stores[save_to]["hists_per_group"][var][group_name] * (scale_factor - 1.0)
                    self.stores[save_to]["hists_squared"][var][syst_name] += self.stores[save_to]["hists_squared_per_group"][var][group_name] * (scale_factor**2 - 1.0)

                # Group
                self.stores[save_to]["hists_per_group"][var][group_name] *= scale_factor
                self.stores[save_to]["hists_squared_per_group"][var][group_name] *= scale_factor**2


  def _get_needed_columns(self, k):

    cols = []
    calc_cols = []
    cols += self._get_tokens(self.weight)
    #if self.sel is not None:
    #  cols += self._get_tokens(self.sel)
    if self.pre_sel is not None:
      cols += self._get_tokens(self.pre_sel)
    if "extra_columns" in self.cfg:
      for col in self.cfg["extra_columns"]:
        cols += [col]
    if self.cfg["calculate"] is not None:
      for k, v in self.cfg["calculate"].items():
        cols += [v1 for v1 in self._get_tokens(v) if v1 not in calc_cols]
        if k not in cols:
          calc_cols += [k]
    if "systematics" in self.cfg and "Data" not in k and self.syst:
      for syst_name, syst_info in self.cfg["systematics"].items():
        calc_cols += [syst_name]
        for func_key, func in syst_info["string_functions"].items():
          cols += [v1 for v1 in self._get_tokens(func) if v1 not in calc_cols]
          if func_key not in cols:
            calc_cols += [func_key]

    for var in self.cfg["variables"].keys():
      if var not in calc_cols:
        cols += [var]
    return sorted(list(set(cols)))


  def _CreateJob(self, cmd_list, job_name, delete_job=True):
    dir_name = "/".join(job_name.split("/")[:-1])
    if not os.path.exists(dir_name):
      os.makedirs(dir_name)
    if os.path.exists(job_name) and delete_job: os.system(f'rm {job_name}')
    for cmd in cmd_list:
      prep_cmd = cmd.replace('"','\\"')
      prep_cmd = prep_cmd.replace('$','\\$')
      os.system(f'echo "{prep_cmd}" >> {job_name}')
    os.system(f'chmod +x {job_name}' % vars())
    if delete_job:
      print("Created job:",job_name)
    else:
      print("Adding to:",job_name)


  def _submit(self, specific_histogram_name):

    self.cmd_store.append(specific_histogram_name)

    #self.cmd_store.append(f"{self.command} --specific-histogram {specific_histogram_name}")
    job_name = os.path.abspath(os.path.join(self.output, f"job.sh"))

    if self.command_ind == 0:
      # make .sh file
      # file directory
      pp = os.path.abspath(__file__).split("AnalysisConfigs")[0]+"AnalysisConfigs"
    
      commands = [
        "#!/bin/bash",
        "export XRD_RUNFORKHANDLER=1",
        "export MALLOC_TRIM_THRESHOLD_=0",
        f"export PYTHONPATH=\"{pp}:$PYTHONPATH\"",
      ]
      self._CreateJob(commands, job_name, delete_job=True)

    self.command_ind += 1


    if len(self.cmd_store) >= self.n_per_job:

      commands = [f"if [ $1 -eq {self.submit_ind} ]; then"]
      #commands += [f"   {cmd}" for cmd in self.cmd_store]
      commands += [f"   {self.command} --specific-histogram {','.join(self.cmd_store)} --extra-name={self.submit_ind}"]
      commands += ["fi"]
      self._CreateJob(commands, job_name, delete_job=False)
      self.submit_ind += 1
      self.cmd_store = []


  def _submit_sweep(self):

    job_name = os.path.abspath(os.path.join(self.output, f"job.sh"))
    if len(self.cmd_store) > 0:
      commands = [f"if [ $1 -eq {self.submit_ind} ]; then"]
      #commands += [f"   {cmd}" for cmd in self.cmd_store]
      commands += [f"   {self.command} --specific-histogram {','.join(self.cmd_store)} --extra-name={self.submit_ind}"]
      commands += ["fi"]
      self._CreateJob(commands, job_name, delete_job=False)
      self.submit_ind += 1
      self.cmd_store = []

    sub = [
      f"Executable = {job_name}",
      f"Error = {job_name.replace('.sh', '_$(ClusterId).$(ProcId).err')}",
      f"Output = {job_name.replace('.sh', '_$(ClusterId).$(ProcId).out')}",
      f"Log = {job_name.replace('.sh', '_$(ClusterId).log')}",
      "MY.SendCredential = True",
      "MY.SingularityImage = \"/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/pocketcoffea:lxplus-el9-576bd3cd\"",
      "+JobFlavour = \"longlunch\"",
      "RequestCpus = 1",
      "RequestMemory = 2GB",
      "arguments = $(ProcId)",
      "should_transfer_files = YES",
      "when_to_transfer_output = ON_EXIT",
      "on_exit_remove = (ExitBySignal == False) && (ExitCode == 0)",
      "max_retries = 10",
      "requirements = Machine =!= LastRemoteHost",
      f"queue {self.submit_ind}",        
    ]
    sub_name = job_name.replace('.sh', '.sub')
    self._CreateJob(sub, sub_name, delete_job=True)

    # submit job
    print(f'Submitting job: {sub_name}')
    os.system(f'condor_submit {sub_name}')


  def Run(self):

    # Load function to apply
    if "function_to_apply" in self.cfg and self.cfg["function_to_apply"] is not None and not self.submit:
      path = self.cfg["function_to_apply"][0]   # full .py file path
      func_name = self.cfg["function_to_apply"][1]
      module_name = Path(path).stem  # plotting_extra_mass
      spec = importlib.util.spec_from_file_location(module_name, path)
      module = importlib.util.module_from_spec(spec)
      sys.modules[module_name] = module
      spec.loader.exec_module(module)
      func = getattr(module, func_name)
    else:
      func = None


    # Loop over files
    previous_file = None
    previous_year = None
    file_names_run = []
    file_names_ext_nums = {}
    self.files_groups_ran = {}
    for f in self.files:

      # Check if file is in groups
      k = self._check_file_in_groups(f)
      if k is None: continue

      # Get the file names
      file_name, era_name, file_name_minus_era = self._get_file_names(f)

      root_file_name = file_name.replace("(","").replace(")","").replace("[","").replace("]","").replace(",","_").replace(".","").replace(" ","_")
      if self.specific_histogram is not None:
        run_file = False
        for specific_hist in self.specific_histogram:
          if root_file_name in specific_hist:
            run_file = True
        if not run_file:
          continue

      # Check if we should just load all columns
      if "all_columns" in self.cfg and self.cfg["all_columns"]:
        needed_columns = None
      else:
        needed_columns = self._get_needed_columns(k)

      print(f"Processing {file_name} for group {k}")

      # Read the parquet file
      if not self.submit:
        if self.load_from_root is None:

          loaded_df = pd.read_parquet(f, columns=needed_columns)

          if loaded_df.empty: continue
          # Apply pre-selection if provided
          if self.pre_sel is not None:
            df = loaded_df.query(self.pre_sel)
          else:
            df = loaded_df.copy()
          del loaded_df
          gc.collect()

          # Apply function
          if func is not None:
            length_df = len(df)
            events_at_once = 10**5
            if length_df < events_at_once:
              df = func(df, metadata={"file_name": file_name, "group": k, "era": era_name, "cfg": self.cfg})
            else:
              df_list = []
              for i in range(0, length_df, events_at_once):
                df_chunk = df.iloc[i:i+events_at_once].copy()
                df_chunk = func(df_chunk, metadata={"file_name": file_name, "group": k, "era": era_name, "cfg": self.cfg})
                df_list.append(df_chunk)
              del df
              gc.collect()
              df = pd.concat(df_list)

        else:
          df = None

      # Check if file_name has already been run (for multiple eras in same file)
      if file_name in file_names_run:
        if file_name not in file_names_ext_nums.keys():
          file_names_ext_nums[file_name] = 0
        file_names_ext_nums[file_name] += 1
        file_name_ext = f"{file_name}_ext_{file_names_ext_nums[file_name]}"
      else:
        file_name_ext = file_name
      file_names_run.append(file_name)

      # Get nominal histograms and return nominal df
      if "DATA" not in file_name:
        calculate = [{"weight": self.weight}]
      else:
        calculate = [{"weight": "1.0"}]
      calculate += [self.cfg["calculate"]] if "calculate" in self.cfg else []
      sel = [self.sel] if self.sel is not None else []
      sel += [self.cfg["group_selection"][k]] if k in self.cfg["group_selection"].keys() else []
      if self.specific_histogram is None or root_file_name in self.specific_histogram:

        if self.submit:
          self._submit(specific_histogram_name=root_file_name)
        else:
          print("Running nominal")
          _ = self._get_histograms(df.copy() if df is not None else None, calculate, sel, file_name_ext, k, "Total_MC" if "Data" not in k else "Total_Data", save_to="Nom", group_name=k)

      # Get systemtics histograms
      if "systematics" in self.cfg and "Data" not in k and self.syst:
        for syst_name, syst_info in self.cfg["systematics"].items():
          for shift_name, syst_val in {"Down":-1.0,"Up":1.0}.items():
            if era_name not in self.eras: continue
            if file_name_minus_era in syst_info["files"] and era_name in syst_info["years"]:

              root_syst_file_name = f"{root_file_name}_{syst_name}{shift_name}"
              if self.specific_histogram is not None:
                run_syst = False
                for specific_hist in self.specific_histogram:
                  if root_syst_file_name in specific_hist:
                    run_syst = True
                if not run_syst:
                  continue

              print(f"Running systematic {syst_name}{shift_name}")
              syst_calculate = [{syst_name: syst_val}]
              syst_calculate += [self.cfg["calculate"]] if "calculate" in self.cfg else []
              function_to_apply = None
              if "string_functions" in syst_info:
                syst_calculate += [syst_info["string_functions"]]
              elif "function" in syst_info:
                path = syst_info["function"][0]   # full .py file path
                func_name = syst_info["function"][1]
                module_name = Path(path).stem  # plotting_extra_mass
                spec = importlib.util.spec_from_file_location(module_name, path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                function_to_apply = getattr(module, func_name)
              if self.submit:
                self._submit(specific_histogram_name=root_syst_file_name)
              else: 
                _ = self._get_histograms(df.copy() if df is not None else None, syst_calculate, sel, f"{file_name_ext}_{syst_name}", f"{k}_{syst_name}", syst_name, save_to=shift_name, group_name=k, function_to_apply=function_to_apply, metadata_for_function={"syst_name": syst_name, "shift_name": shift_name, "syst_val": syst_val, "era_name": era_name})

            else:

              if self.specific_histogram is None and not self.submit:
                self._add_nom_to_syst_groups(shift_name, k, file_name_ext, syst_name)


    if self.specific_histogram is None and not self.submit:

      # Normalise groups to data
      if self.norm_groups_to_data is not None:
        self._norm_groups_to_data()

      # Do scale to
      if "scale_to" in self.cfg and self.cfg["scale_to"] is not None:
        self._scale_to()

      # Rebin
      if self.rebin:
        self._rebin_histograms()

    # Write root files
    if not self.submit:
      if (self.write and self.load_from_root is None) or (self.write_after_load and self.load_from_root is not None):
        self._write_root_files()

    if self.submit:
      self._submit_sweep()



if args.var is None and args.rebin:
  raise ValueError("Please specify a variable to rebin. Use --var to specify the variable.")

specific_histogram = None
if args.specific_histogram is not None:
  if "," in args.specific_histogram:
    specific_histogram = args.specific_histogram.split(",")
  else:
    specific_histogram = [args.specific_histogram]  

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
  rebin_bins=args.rebin_bins,
  specific_histogram=specific_histogram,
  submit=args.submit,
  command=command,
  points_per_job=args.points_per_job,
  write_after_load=args.write_after_load,
  output=args.output,
  extra_name=args.extra_name,
  norm_groups_to_data=args.norm_groups_to_data,
)

# hadd
if args.hadd:

  out_files = {}
  for var in gh.cfg["variables"].keys():

    # Find matching ROOT files
    in_file_paths = sorted(glob.glob(f"{args.output}/datacard_{var}_*.root"))
    out_file_path = f"{args.output}/datacard_{var}.root"
    print(f"[INFO] Merging {len(in_file_paths)} files into {out_file_path}")

    # Initiate merged histograms
    os.makedirs(args.output, exist_ok=True)
    out_files[var] = uproot.recreate(out_file_path)

    merged_hists = {}
    # Open first file to initialize histogram structure
    for path in in_file_paths:
      with uproot.open(path) as f0:
        for name, obj in f0.items():
          vals, edges = obj.to_numpy()
          variance = obj.variances()

          h_bh = bh.Histogram(bh.axis.Variable(copy.deepcopy(edges)), storage=bh.storage.Weight())
          h_bh.view().value[:] = copy.deepcopy(vals)
          h_bh.view().variance[:] = copy.deepcopy(variance)
          out_files[var][name.split(";")[0]] = copy.deepcopy(h_bh)

    out_files[var].close()
    print(f"[OK] Wrote merged file: {out_file_path}")
  exit(0)

gh.Run()

if args.specific_histogram is not None:
  exit(0)

if args.submit:
  exit(0)

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
  hists = {k: copy.deepcopy(gh.stores["Nom"]["hists_per_group"][var][k]) for k in list(gh.cfg["groups"].keys())[::-1] if k != "Data"}
  hists_squared = {k: copy.deepcopy(gh.stores["Nom"]["hists_squared_per_group"][var][k]) for k in gh.cfg["groups"].keys() if k != "Data"}
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
    up_variance = copy.deepcopy(gh.stores["Nom"]["hists_squared"][var]["Total_MC"])
    down_variance = copy.deepcopy(gh.stores["Nom"]["hists_squared"][var]["Total_MC"])
    nom_hist = copy.deepcopy(gh.stores["Nom"]["hists"][var]["Total_MC"])
    for syst_name in gh.cfg["systematics"].keys():

      for syst_era in gh.cfg["systematics"][syst_name]["years"]:
        if syst_era in gh.eras:
          break
      else:
        continue

      up_shift = np.zeros_like(nom_hist)
      down_shift = np.zeros_like(nom_hist)
      if syst_name in gh.stores["Up"]["hists"][var].keys():
        up_shift = copy.deepcopy(gh.stores["Up"]["hists"][var][f"{syst_name}"]) - nom_hist
      if syst_name in gh.stores["Down"]["hists"][var].keys():
        down_shift = copy.deepcopy(gh.stores["Down"]["hists"][var][f"{syst_name}"]) - nom_hist
      up_hist = np.array([max(max(down_shift[i], up_shift[i]),0.0) for i in range(len(down_variance))])
      down_hist = np.array([min(min(down_shift[i], up_shift[i]),0.0) for i in range(len(down_variance))])

      print(syst_name, "up hist sum:", np.sum(up_hist), "down hist sum:", np.sum(down_hist))
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
    "all" : "$309\ fb^{-1}\ (13,13.6\ TeV)$",
    "run2" : "$138\ fb^{-1}\ (13\ TeV)$",
    "run3" : "$171\ fb^{-1}\ (13.6\ TeV)$",
    "2016_PreVFP" : "$19.6\ fb^{-1}\ (13\ TeV)$",
    "2016_PostVFP" : "$17.0\ fb^{-1}\ (13\ TeV)$",
    "2017" : "$41.5\ fb^{-1}\ (13\ TeV)$",
    "2018" : "$59.8\ fb^{-1}\ (13\ TeV)$",
    "2022_preEE" : "$7.98\ fb^{-1}\ (13.6\ TeV)$",
    "2022_postEE" : "$26.7\ fb^{-1}\ (13.6\ TeV)$",
    "2023_preBPix" : "$18.1\ fb^{-1}\ (13.6\ TeV)$",
    "2023_postBPix" : "$9.69\ fb^{-1}\ (13.6\ TeV)$",
    "2223" : "$62.4\ fb^{-1}\ (13.6\ TeV)$",
    "2024" : "$109.0\ fb^{-1}\ (13.6\ TeV)$",
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
    lumi_label=lumi_labels[args.year],
    uncertainty_label="Stat. Uncertainty" if not args.syst else "Stat. + Syst. Uncertainty"
  )

  
  # Plot systematic variations if required
  if args.syst and len(gh.cfg["systematics"].keys()) > 0 and args.plot_syst_variation:
    for syst_name in gh.cfg["systematics"].keys():

      # Check if the era of the systematic is relevant
      for syst_era in gh.cfg["systematics"][syst_name]["years"]:
        if syst_era in gh.eras:
          break
      else:
        continue

      for group in gh.total_groups.keys():
        if group == "Data": continue
        if group not in gh.stores["Nom"]["hists_per_group"][var].keys(): continue
        if f"{group}_{syst_name}" not in gh.stores["Up"]["hists_per_group"][var].keys(): continue
        if f"{group}_{syst_name}" not in gh.stores["Down"]["hists_per_group"][var].keys(): continue

        # Get the up and down histograms
        up_hist = copy.deepcopy(gh.stores["Up"]["hists_per_group"][var][f"{group}_{syst_name}"])
        down_hist = copy.deepcopy(gh.stores["Down"]["hists_per_group"][var][f"{group}_{syst_name}"])
        nom_hist = copy.deepcopy(gh.stores["Nom"]["hists_per_group"][var][group])

        if args.norm_to_bin_width:
          bins_widths = np.array([gh.bin_store[var][ind+1] - gh.bin_store[var][ind] for ind in range(len(gh.bin_store[var])-1)])
          up_hist = up_hist / bins_widths
          down_hist = down_hist / bins_widths
          nom_hist = nom_hist / bins_widths

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


# Plot different top mass (just 169.5, 172.5, 175.5) if available, in the same way as systematic variations
nom_name = "TT (172.5 GeV)"
up_name = "TT (171.5 GeV)"
down_name = "TT (173.5 GeV)"
if nom_name in gh.total_groups.keys() and up_name in gh.total_groups.keys() and down_name in gh.total_groups.keys():
  for var in gh.cfg["variables"].keys():
    if nom_name not in gh.stores["Nom"]["hists_per_group"][var].keys(): continue
    if up_name not in gh.stores["Nom"]["hists_per_group"][var].keys(): continue
    if down_name not in gh.stores["Nom"]["hists_per_group"][var].keys(): continue

    # Get the up and down histograms
    nom_hist = copy.deepcopy(gh.stores["Nom"]["hists_per_group"][var][nom_name])
    up_hist = copy.deepcopy(gh.stores["Nom"]["hists_per_group"][var][up_name])
    down_hist = copy.deepcopy(gh.stores["Nom"]["hists_per_group"][var][down_name])

    if args.norm_to_bin_width:
      bins_widths = np.array([gh.bin_store[var][ind+1] - gh.bin_store[var][ind] for ind in range(len(gh.bin_store[var])-1)])
      up_hist = up_hist / bins_widths
      down_hist = down_hist / bins_widths
      nom_hist = nom_hist / bins_widths

    plot_histograms_with_ratio(
      [nom_hist, up_hist, down_hist],
      [None, None, None],
      ["172.5 GeV", "171.5 GeV", "173.5 GeV"],
      gh.bin_store[var],
      xlabel=var if var not in gh.cfg["translate"] else gh.cfg["translate"][var],
      ylabel="Events",
      name=f"{args.output}/top_mass_variation_{var}_{args.year}",
      ratio_range=[0.9,1.1],
      axis_text="Top Mass Variation"
    )
