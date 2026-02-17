import argparse
import copy
import glob
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import os
from Functions.Plotting import plot_histograms_with_ratio


parser = argparse.ArgumentParser()
parser.add_argument('--input', "-i", help='The input folder of the parquet files', type=str, default="output_merged_v3/*.parquet")
parser.add_argument('--output', "-o", help='The output folder', type=str, default="output.parquet")
parser.add_argument('--mass-to', help='Mass to convert to', type=str, default="172.5")
parser.add_argument('--yield-input', help='The input folder of the yield', type=str, default="output_merged_v3/TTToSemiLeptonic_*.parquet")
parser.add_argument('--vars', help='Variables to plot', type=str, default="GenTop1_mass")
parser.add_argument('--plot-output', help='The output folder for plots', type=str, default="output/")
parser.add_argument('--file-ext', help='Name at end of file', type=str, default="o")
parser.add_argument('--ignore-fraction', help='Ignore sample if effective events fraction < fraction', type=float, default=0.1)
args = parser.parse_args()

if "," not in args.mass_to:
  mass_to = [float(args.mass_to)]
else:
  mass_to = [float(m) for m in args.mass_to.split(",")]

# Get files
input_files = args.input.split(",")
files = []
for f in input_files:
  if "*" in f:
    files += glob.glob(f)
  else:
    files += [f]
files = list(set(files))
files = sorted(files)

print("Input files:", files)

# Get the yield files
input_yield_files = args.yield_input.split(",")
yield_files = []
for f in input_yield_files:
  if "*" in f:
    yield_files += glob.glob(f)
  else:
    yield_files += [f]
yield_files = list(set(yield_files))
yield_files = sorted(yield_files)

print("Yield files:", yield_files)

# Get the masses from
masses_from = []
mass_files = {}
for f in files:
  if "166p5" in f:
    masses_from.append(166.5)
  elif "169p5" in f:
    masses_from.append(169.5)
  elif "171p5" in f:
    masses_from.append(171.5)
  elif "173p5" in f:
    masses_from.append(173.5)
  elif "175p5" in f:
    masses_from.append(175.5)
  elif "178p5" in f:
    masses_from.append(178.5)
  else:
    masses_from.append(172.5)

  if masses_from[-1] not in mass_files.keys():
    mass_files[masses_from[-1]] = []
  mass_files[masses_from[-1]].append(copy.deepcopy(f))


def BW(s, l=1.32, m=172.5):
  s = np.asarray(s)  # ensure it's a numpy array
  k = 1
  return k / ((s - m**2)**2 + (m*l)**2)

def TopQuarkWidth(m):
  return 0.0270*m - 3.3455

def ApplyBWReweight(df, mf=172.5, mi=172.5, gen_mass="GenTop1_mass", gen_mass_other="GenTop2_mass"):
  l_f = TopQuarkWidth(mf)
  l_i = TopQuarkWidth(mi)
  
  # Vectorized
  w1 = BW(df[gen_mass]**2, l=l_f, m=mf) / BW(df[gen_mass]**2, l=l_i, m=mi)
  df["weight"] *= w1
  
  if gen_mass_other is not None:
    w2 = BW(df[gen_mass_other]**2, l=l_f, m=mf) / BW(df[gen_mass_other]**2, l=l_i, m=mi)
    df["weight"] *= w2
      
  return df


#def BW(s, l=1.32, m=172.5):
#  """
#  Calculate the Breit-Wigner distribution.
#  Args:
#      s (float): The mass squared.
#      l (float): The width of the top quark.
#      m (float): The mass of the top quark.
#  Returns:
#      float: The Breit-Wigner distribution.
#  """
#  # Calculate the Breit-Wigner distribution
#  k = 1
#  return k/((s-(m**2))**2 + (m*l)**2)
#
#def TopQuarkWidth(m):
#  """
#  Calculate the width of the top quark.
#  Args:
#      m (float): The mass of the top quark.
#  Returns:
#      float: The width of the top quark.
#  """
#  # Calculate the width of the top quark
#  return (0.0270*m) - 3.3455
#
#def ApplyBWReweight(df, mf=172.5, lf=1.32, mi=172.5, li=1.32, gen_mass="GenTop1_mass", gen_mass_other="GenTop2_mass"):
#  """
#  Apply the BW reweighting to the dataframe.
#  Args:
#      df (pd.DataFrame): The input dataframe.
#      m (float): The mass of the top quark.
#      l (float): The width of the top quark. 
#  """
#  # Apply the BW reweighting
#  df.loc[:,"weight"] *= BW(df.loc[:,gen_mass]**2,l=TopQuarkWidth(mf), m=mf)/BW(df.loc[:,gen_mass]**2,l=TopQuarkWidth(mi), m=mi) 
#  if gen_mass_other is not None:
#    df.loc[:,"weight"] *= BW(df.loc[:,gen_mass_other]**2,l=TopQuarkWidth(mf), m=mf)/BW(df.loc[:,gen_mass_other]**2,l=TopQuarkWidth(mi), m=mi) 
#  return df


# Get the yield
total_yield = 0.0
for f in yield_files:
  print("Getting yield from file:", f)
  df = pd.read_parquet(f)
  total_yield += np.sum(df.loc[:, "weight"])

print("Total yield:", total_yield)

sum_wts = {mt:{mf:0.0 for mf in mass_files.keys()} for mt in mass_to}
sum_wts_squared = {mt:{mf:0.0 for mf in mass_files.keys()} for mt in mass_to}
for mass_from, mf in mass_files.items():
  for f in mf:
    df = pd.read_parquet(f)
    for mt in mass_to:
      tmp_df = ApplyBWReweight(df.copy(), mf=mt, mi=mass_from)
      sum_wts[mt][mass_from] += np.sum(tmp_df["weight"])
      sum_wts_squared[mt][mass_from] += np.sum(tmp_df["weight"]**2)

print("Sum of weights:", sum_wts)
print("Sum of weights squared:", sum_wts_squared)

# Get fractions
sum_opt_wts = {}
eff_events = {}
eff_events_fraction = {}
total_sum_opt_wts = {}
rescaled_sum_opt_wts = {}
for mt in mass_to:

  # eff event
  eff_events[mt] = {m : (sum_wts[mt][m]**2) / sum_wts_squared[mt][m] if sum_wts_squared[mt][m] > 0 else 0.0 for m in mass_files.keys()}
  total_eff_events = np.sum(list(eff_events[mt].values()))
  eff_events_fraction[mt] = {m : eff_events[mt][m] / total_eff_events if total_eff_events > 0 else 0.0 for m in mass_files.keys()}
  print(f"Effective events for mass {mt}:")
  print(eff_events[mt])

  # Opt weights
  sum_opt_wts[mt] = {m : sum_wts[mt][m] / sum_wts_squared[mt][m] if sum_wts_squared[mt][m] > 0 and eff_events_fraction[mt][m] > args.ignore_fraction else 0.0 for m in mass_files.keys()}
  total_sum_opt_wts[mt] = np.sum([sum_opt_wts[mt][m] * sum_wts[mt][m] for m in mass_files.keys()])
  rescaled_sum_opt_wts[mt] = {m : i * total_yield / total_sum_opt_wts[mt] for m, i in sum_opt_wts[mt].items()}

print("Fractions:", rescaled_sum_opt_wts)

# Make the samples
for mt in mass_to:
  for mf, mfiles in mass_files.items():
    for f in mfiles:
      fn = f"{args.output}/TT_{str(mt).replace('.', 'p')}_from_{f.split('/')[-1].split('.')[0]}{args.file_ext}.parquet"
      if os.path.isfile(fn):
        raise FileExistsError(f"File {fn} already exists. Please choose a different output file name.")
      if not os.path.exists(os.path.dirname(fn)):
        os.makedirs(os.path.dirname(fn))

sum_wts_after = 0.0
sum_wts_after_squared = 0.0
files_made = {mt: [] for mt in mass_to}
for mf, mfiles in mass_files.items():
  for f in mfiles:
    print(f"Processing file: {f}")
    df = pd.read_parquet(f)

    for mt in mass_to:

      tmp_df = ApplyBWReweight(df.copy(), mf=mt, mi=mf)
      tmp_df["weight"] *= rescaled_sum_opt_wts[mt][mf]

      w = tmp_df["weight"].to_numpy()

      if np.sum(w) == 0:
        print(f"Skipping file {f} for mass {mt} due to zero weights after reweighting.")
        continue

      sum_wts_after += w.sum(dtype=float)
      sum_wts_after_squared += np.sum(w**2)

      table = pa.Table.from_pandas(tmp_df, preserve_index=False)
      fn = f"{args.output}/TT_{str(mt).replace('.', 'p')}_from_{f.split('/')[-1].split('.')[0]}{args.file_ext}.parquet"
      files_made[mt].append(fn)
      print("Creating file:", fn)
      pq.write_table(table, fn, compression='snappy')

      ## Write the dataframe to parquet
      #table = pa.Table.from_pandas(tmp_df, preserve_index=False)
      #fn = file_name.replace("MASSTO", str(mt).replace(".", "p"))
      #if os.path.isfile(fn):
      #  combined_table = pa.concat_tables([pq.read_table(fn), table])
      #  pq.write_table(combined_table, fn, compression='snappy')
      #else:
      #  print("Creating file:", fn)
      #  pq.write_table(table, fn, compression='snappy') 


# Collect
for mt in mass_to:
  print(f"Combining files for mass {mt}")
  file_name = f"{args.output}/TT_{str(mt).replace('.', 'p')}{args.file_ext}.parquet"
  if os.path.isfile(file_name):
    raise FileExistsError(f"File {file_name} already exists. Please choose a different output file name.")
  tables = [pq.read_table(f) for f in files_made[mt]]
  combined_table = pa.concat_tables(tables)
  pq.write_table(combined_table, file_name, compression='snappy')

  # Removing collected files
  for f in files_made[mt]:
    print("Removing file:", f)
    os.remove(f)

"""

variables = args.vars.split(",")

for mt in mass_to:

  fn = f"{args.output}/TT_{str(mt).replace('.', 'p')}{args.file_ext}.parquet"

  # Get the histogram snew samples
  hists_new = {}
  hist_uncerts_new = {}
  bins_new = {}
  for var in variables:
    df = pd.read_parquet(fn, columns=[var, "weight"])
    df = df.query(f"{var} >= {df[var].quantile(0.05)} and {var} <= {df[var].quantile(0.95)}")
    hist, bins = np.histogram(df[var], bins=30, weights=df["weight"])
    bins_new[var] = copy.deepcopy(bins)
    hist_uncert = np.sqrt(np.histogram(df[var], bins=bins_new[var], weights=df["weight"]**2)[0])
    hists_new[var] = copy.deepcopy(hist)
    hist_uncerts_new[var] = copy.deepcopy(hist_uncert)

  # Get the histograms for the nominal samples
  hists_nom = {}
  hist_uncerts_nom = {}
  for mass, file in zip(masses_from, files):
    if mass not in hists_nom:
      hists_nom[mass] = {}
      hist_uncerts_nom[mass] = {}
    df = pd.read_parquet(file)
    df = ApplyBWReweight(df, mf=mt, mi=mass)
    for var in variables:
      hist = np.histogram(df[var], bins=bins_new[var], weights=df["weight"])[0]
      hist_uncert = np.sqrt(np.histogram(df[var], bins=bins_new[var], weights=df["weight"]**2)[0])

      if var not in hists_nom[mass]:
        hists_nom[mass][var] = copy.deepcopy(hist)
        hist_uncerts_nom[mass][var] = copy.deepcopy(hist_uncert)
      else:
        hists_nom[mass][var] += copy.deepcopy(hist)
        hist_uncerts_nom[mass][var] = (hist_uncerts_nom[mass][var]**2 + hist_uncert**2)**0.5

  # Plot the histograms
  for var in variables:
    for mass in list(set(masses_from)):
      mass_from_name = str(mass).replace(".", "p")
      mass_to_name = str(mt).replace(".", "p")
      if mass_from_name != mass_to_name:
        legend_name = rf"{mass} to {mt}"
      else:
        legend_name = f"Nominal sample for {mass}"

      plot_histograms_with_ratio(
        [hists_new[var], hists_nom[mass][var]],
        [hist_uncerts_new[var], hist_uncerts_nom[mass][var]],
        [f"BW sample for {mt}", legend_name],
        bins_new[var],
        xlabel=var,
        ylabel="Events",
        name=f"{args.plot_output}/bw_reweighting_from_{mass_from_name}_to_{mass_to_name}_{var}{args.file_ext}",
        ratio_range=[0.9,1.1],
      )
      plot_histograms_with_ratio(
        [hists_new[var]/np.sum(hists_new[var]), hists_nom[mass][var]/np.sum(hists_nom[mass][var])],
        [hist_uncerts_new[var]/np.sum(hists_new[var]), hist_uncerts_nom[mass][var]/np.sum(hists_nom[mass][var])],
        [f"BW sample for {mt}", legend_name],
        bins_new[var],
        xlabel=var,
        ylabel="Density",
        name=f"{args.plot_output}/bw_reweighting_density_from_{mass_from_name}_to_{mass_to_name}_{var}{args.file_ext}",
        ratio_range=[0.9,1.1],
      )
"""