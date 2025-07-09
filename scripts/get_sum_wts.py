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

# Get sum of weights
sum_wts_dict = {}
total_sum_wts = 0.0
for f in files:
  df = pd.read_parquet(f)
  sum_wts = np.sum(df.loc[:, "weight"])
  total_sum_wts += sum_wts
  sum_wts_dict[f] = sum_wts*1.0

# Make table
RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m" 
RESET = "\033[0m"
tabulated_data = [["File", "Sum of Weights"]]
for k, v in sum_wts_dict.items():
  tabulated_data.append([
    f"{GREEN}{k}{RESET}", 
    round(float(v),2)
  ])
tabulated_data.append([
  f"{BLUE}Total{RESET}", 
  round(float(total_sum_wts),2)
])
print(tabulate(tabulated_data[1:], headers=tabulated_data[0], tablefmt="fancy_grid"))
