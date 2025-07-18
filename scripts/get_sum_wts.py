import argparse
import copy
import glob
import pandas as pd
import numpy as np
from tabulate import tabulate

parser = argparse.ArgumentParser()
parser.add_argument('--input', "-i", help='The input folder of the parquet files', type=str, default="output_merged_v3/*.parquet")
args = parser.parse_args()

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

# Get sum of weights
sum_wts_dict = {}
sum_wts_squared_dict = {}
n_events_dict = {}
n_positive_wts_dict = {}
eff_events_dict = {}
total_sum_wts = 0.0
for f in files:
  df = pd.read_parquet(f)
  sum_wts = np.sum(df.loc[:, "weight"])
  sum_wts_squared = np.sum(df.loc[:, "weight"]**2)
  total_sum_wts += sum_wts
  sum_wts_dict[f] = sum_wts*1.0
  sum_wts_squared_dict[f] = sum_wts_squared*1.0
  n_events_dict[f] = len(df)
  n_positive_wts_dict[f] = len(df.loc[df["weight"] > 0.0])
  if sum_wts_squared > 0:
    eff_events_dict[f] = sum_wts**2 / sum_wts_squared
  else:
    eff_events_dict[f] = 0.0

# Make table
RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m" 
RESET = "\033[0m"
tabulated_data = [["File", "Sum of Weights", "N Events", "Postive Weight Fraction", "Effective Events"]]
for k, v in sum_wts_dict.items():
  tabulated_data.append([
    f"{GREEN}{k}{RESET}", 
    round(float(v),2),
    n_events_dict[k],
    round(float(n_positive_wts_dict[k]) / n_events_dict[k], 2) if n_events_dict[k] > 0 else 0.0,
    round(float(eff_events_dict[k]), 2) if eff_events_dict[k] > 0 else 0.0
  ])
tabulated_data.append([
  f"{BLUE}Total{RESET}", 
  round(float(total_sum_wts),2),
  sum(n_events_dict.values()),
  round(float(sum(n_positive_wts_dict.values())) / sum(n_events_dict.values()), 2) if sum(n_events_dict.values()) > 0 else 0.0,
  round(float(total_sum_wts**2 / sum(sum_wts_squared_dict.values())), 2) if sum(sum_wts_squared_dict.values()) > 0 else 0.0
])
print(tabulate(tabulated_data[1:], headers=tabulated_data[0], tablefmt="fancy_grid"))
