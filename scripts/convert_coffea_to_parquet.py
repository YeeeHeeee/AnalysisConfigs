import argparse
import fnmatch
import glob
import os

import coffea.util as cu
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

parser = argparse.ArgumentParser()
parser.add_argument('--input', "-i", help='The input coffea file', type=str, default="output_merged_v3/output_all.coffea")
parser.add_argument('--output', "-o", help='The output parquet directory', type=str, default="output_merged_v3")
parser.add_argument('--overwrite', help='Overwrite the output', action='store_true')
parser.add_argument('--verbose', help='Print the dataframes', action='store_true')
parser.add_argument('--collect-only', help='Collecting the split parquet files', action='store_true')
parser.add_argument('--weight', help='The formula for the weight', type=str, default="weight")
parser.add_argument('--norm-weight', help='The formula for the weight to normalise the sum of weights to', type=str, default=None)
parser.add_argument('--norm-files', help='The files to get the weight to normalise the sum of weights to', type=str, default=None)
args = parser.parse_args()

# Files created
files_created = []
year_names = []

# Get files
input_files = args.input.split(",")
files = []
for f in input_files:
  if "*" in f:
    files += glob.glob(f)
  else:
    files += [f]
files = list(set(files))

# If number at end of file (minus .coffea), sort by that
files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]) if x.split("_")[-1].split(".")[0].isdigit() else 0)

if not args.collect_only:

  # Get sum of gen weights
  sum_wts_dict = {}
  for file in files:

    print(f"Processing file for sum of gen weights: {file}")

    # Check if the file exists
    if not os.path.exists(file):
      print(f"File {file} does not exist.")
      continue

    # Load the file
    hist_dict = cu.load(file)

    # Get the sum of gen weights
    if "sum_genweights" in hist_dict:
      for year_name in hist_dict["sum_genweights"].keys():
        if year_name not in sum_wts_dict:
          sum_wts_dict[year_name] = 0.0
        sum_wts_dict[year_name] += hist_dict["sum_genweights"][year_name]




  # Get the sum of weights and of normalisation weights
  if args.norm_weight is not None:

    norm_sum_wts_dict = {}
    norm_nom_sum_wts_dict = {}

    for file in files:

      print(f"Processing file for sum of weights and normalisation weights: {file}")

      # Check if the file exists
      if not os.path.exists(file):
        print(f"File {file} does not exist.")
        continue

      # Load the file
      hist_dict = cu.load(file)

      # Start coffea key loop
      for overall_name in hist_dict["columns"].keys():
        for year_name in hist_dict["columns"][overall_name].keys():
          for variation_name in hist_dict["columns"][overall_name][year_name].keys():

            if variation_name != "baseline": continue

            if args.norm_files is not None:
              for_norm = False
              for norm_file in args.norm_files.split(","):
                if "*" in norm_file:
                  if fnmatch.fnmatch(year_name, norm_file):
                    for_norm = True
                  elif norm_file == year_name:
                    for_norm = True
              if not for_norm:
                continue

            if "DATA" in year_name:
              continue

            data_dict = {}
            for column_name, arr in hist_dict["columns"][overall_name][year_name][variation_name]["nominal"].items():
              if len(arr.value.shape) == 1:
                data_dict[column_name] = arr.value
              else:
                for i in range(arr.value.shape[1]):
                  data_dict[f"{column_name}_{i+1}"] = arr.value[:,i]
            df = pd.DataFrame(data_dict)

            df.loc[:,"norm_weight"] = df.eval(args.norm_weight)/sum_wts_dict[year_name]
            df.loc[:,"weight"] = df.eval(args.weight)/sum_wts_dict[year_name]

            sum_norm_wts = np.sum(df["norm_weight"])
            sum_wts = np.sum(df["weight"])

            # Find the era
            split_year_name = year_name.split("_")
            for i in range(len(split_year_name)-1, -1, -1):
              if split_year_name[i].isdigit():
                era = "_".join(split_year_name[i:])
                break

            if era not in norm_sum_wts_dict:
              norm_sum_wts_dict[era] = 0.0
            norm_sum_wts_dict[era] += sum_norm_wts

            if era not in norm_nom_sum_wts_dict:
              norm_nom_sum_wts_dict[era] = 0.0
            norm_nom_sum_wts_dict[era] += sum_wts

    # Change sum_wts_dict to be the normalisation weights if specified
    for year_name in sum_wts_dict.keys():
      split_year_name = year_name.split("_")
      for i in range(len(split_year_name)-1, -1, -1):
        if split_year_name[i].isdigit():
          era = "_".join(split_year_name[i:])
          break
      if era in norm_nom_sum_wts_dict and era in norm_sum_wts_dict:
        sum_wts_dict[year_name] *= norm_nom_sum_wts_dict[era] / norm_sum_wts_dict[era]





# Get datasets
write_inds = {}

for file in files:

  print(f"Processing file for dataset: {file}")

  # Check if the file exists
  if not os.path.exists(file):
    print(f"File {file} does not exist.")
    continue

  # Load the file
  hist_dict = cu.load(file)

  # Start coffea key loop
  for overall_name in hist_dict["columns"].keys():
    for year_name in hist_dict["columns"][overall_name].keys():
      for variation_name in hist_dict["columns"][overall_name][year_name].keys():
          
        # Check if the year name is already in the list
        if year_name not in year_names:
          year_names.append(year_name)

        # Make file name
        if variation_name == "baseline":
          base_name = f"{args.output}/{year_name}"
        else:
          base_name = f"{args.output}/{year_name}_{variation_name}"

        if base_name not in write_inds:
          write_inds[base_name] = 0
        else:
          write_inds[base_name] += 1

        file_name = f"{base_name}_{write_inds[base_name]}.parquet"

        # Check if the file already exists
        if not args.collect_only:
          if file_name not in files_created:
            files_created.append(file_name)
            if os.path.exists(file_name):
              if not args.overwrite:
                  print(f"File {file_name} already exists. Use --overwrite to overwrite.")
                  continue
              else:
                os.remove(file_name)

          # Make directory if it doesn't exist
          os.makedirs(os.path.dirname(file_name), exist_ok=True)

          # Make the data dictionary/dataframe
          data_dict = {}
          for column_name, arr in hist_dict["columns"][overall_name][year_name][variation_name]["nominal"].items():
            if len(arr.value.shape) == 1:
              data_dict[column_name] = arr.value
            else:
              for i in range(arr.value.shape[1]):
                data_dict[f"{column_name}_{i+1}"] = arr.value[:,i]
          df = pd.DataFrame(data_dict)

          if "DATA" not in year_name:
            df.loc[:,"weight"] = df.eval(args.weight)
            df.loc[:,"weight"] /= sum_wts_dict[year_name]
          else:
            df.loc[:,"weight"] = 1.0

          # Print the dataframe if verbose
          if args.verbose:
            print(f"Dataframe for {file_name}:")
            print(df)

          # Write the dataframe to parquet
          table = pa.Table.from_pandas(df, preserve_index=False)
          if os.path.isfile(file_name):
            print("File already exists:", file_name)
            print("Clear files and try again")
            print("Exiting")
            exit()
          else:
            print("Creating file:", file_name)
            pq.write_table(table, file_name, compression='snappy')


# Merge files
for key, value in write_inds.items():
  
  if os.path.isfile(f"{key}.parquet"):
    print("File already exists:", f"{key}.parquet")
    print("Skipping ...")
    continue

  print("Combining files for:", key)
  tables = [pq.read_table(f"{key}_{i}.parquet") for i in range(value + 1)]
  combined = pa.concat_tables(tables)
  pq.write_table(combined, f"{key}.parquet", compression='snappy')

  # remove individual files
  for i in range(value + 1):
    os.remove(f"{key}_{i}.parquet")