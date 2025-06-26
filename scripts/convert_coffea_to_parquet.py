import argparse
import glob
import os

import coffea.util as cu
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

parser = argparse.ArgumentParser()
parser.add_argument('--input', "-i", help='The input coffea file', type=str, default="output_merged_v3/output_all.coffea")
parser.add_argument('--output', "-o", help='The output parquet directory', type=str, default="output_merged_v3")
parser.add_argument('--overwrite', help='Overwrite the output', action='store_true')
parser.add_argument('--verbose', help='Print the dataframes', action='store_true')
args = parser.parse_args()

# Files created
files_created = []

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
for file in files:

  print(f"Processing file: {file}")

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

# Get datasets
for file in files:

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
          
          # Make file name
          if variation_name == "baseline":
            file_name = f"{args.output}/{year_name}.parquet"
          else:
            file_name = f"{args.output}/{year_name}_{variation_name}.parquet"

          # Check if the file already exists
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
          for column_name, arr in hist_dict["columns"][overall_name][year_name][variation_name].items():
            if len(arr.value.shape) == 1:
              data_dict[column_name] = arr.value
            else:
              for i in range(arr.value.shape[1]):
                data_dict[f"{column_name}_{i+1}"] = arr.value[:,i]
          df = pd.DataFrame(data_dict)

          if "DATA" not in year_name:
            # Add the sum of weights
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
            combined_table = pa.concat_tables([pq.read_table(file_name), table])
            pq.write_table(combined_table, file_name, compression='snappy')
          else:
            print("Creating file:", file_name)
            pq.write_table(table, file_name, compression='snappy')
