import argparse
import copy
import fnmatch
import glob
import os
import re
import pandas as pd
import numpy as np
import coffea.util as cu


parser = argparse.ArgumentParser()
parser.add_argument('--input', help='The input folder of the coffea files with the histograms using the weight', type=str, default=None)
parser.add_argument('--input-w2', help='The input folder of the coffea files with the histograms using the weight squared', type=str, default=None)
parser.add_argument('--output-file', help='The output of python file created with the weights function', type=str, default="./stitching_weights.py")
parser.add_argument('--output-name', help='The name of the stored function', type=str, default="StitchingWeights")
parser.add_argument('--category-conversion', help='A dictionary formatting string to convert the category into a string', type=str, default=None)
parser.add_argument('--extra-sel', help='The string of extra selection in the function', type=str, default=None)
parser.add_argument('--dataset', help='Boolean saying whether the inputs area a dataset instead of a histogram.', action='store_true', default=False)
parser.add_argument('--dataset-bins', help='Comma separated list of the string of selection in each bin', type=str, default=None)
parser.add_argument('--remove-large-outlier-weights', help='Remove large outlier weights.', action='store_true', default=False)
args = parser.parse_args()

# Check if input is provided
if args.input is None:
  raise ValueError("Please provide an input file with the --input argument.")

# Setup category conversion
category_conversion = None
if args.category_conversion is not None:
  category_conversion = {item.split(":")[0]: item.split(":")[1] for item in args.category_conversion.split(",")}

# Setup dataset bins
dataset_bins = None
if args.dataset_bins is not None:
  dataset_bins = args.dataset_bins.split(",")

# Get files
input_files_w = args.input.split(",")
input_files_w2 = args.input_w2.split(",") if args.input_w2 is not None else []
files_w = []
for f in input_files_w:
  if "*" in f:
    files_w += glob.glob(f)
  else:
    files_w += [f]
files_w = list(set(files_w))
files_w2 = []
for f in input_files_w2:
  if "*" in f:
    files_w2 += glob.glob(f)
  else:
    files_w2 += [f]
files_w2 = list(set(files_w2))
  
# Get storage type
storage_type = "variables"
if args.dataset:
  storage_type = "columns"

# Initiate storage
hists = {}
hists_w2 = {}
normalisations = {}
first = True

# Loop through the input files
for file in files_w:

  # Open weighted files
  hist_dict = cu.load(file)

  # Loop through coffea keys
  for overall_name in hist_dict[storage_type].keys():
    for year_name in hist_dict[storage_type][overall_name].keys():
      for variation_name in hist_dict[storage_type][overall_name][year_name].keys():

        store_name = copy.deepcopy(variation_name)
        if args.dataset:
          store_name = copy.deepcopy(year_name)

        # Check if the variation is already in the hists dictionary
        if store_name in hists:
          raise ValueError(f"{store_name} already exists in the hists dictionary. Please check the input files.")

        # Setup histogram storage
        hists[store_name] = []

        if args.dataset:

          data_dict = {}
          for column_name, arr in hist_dict["columns"][overall_name][year_name][variation_name].items():
            if len(arr.value.shape) == 1:
              data_dict[column_name] = arr.value
            else:
              for i in range(arr.value.shape[1]):
                data_dict[f"{column_name}_{i+1}"] = arr.value[:,i]
          df = pd.DataFrame(data_dict)

          if args.remove_large_outlier_weights:
            mean_wt = np.mean(df.loc[:,"weight"])
            std_wt = np.std(df.loc[:,"weight"])
            n_events_before = len(df)
            df = df[(df.loc[:,"weight"] <= (mean_wt + (10*std_wt)))]
            print(f"Removed {n_events_before-len(df)}/{n_events_before} events in {store_name}")

          hists[store_name] = []
          hists_w2[store_name] = []
          if first:
            bin_sel = []
          for b in dataset_bins:
            df_sel = b.replace("events.","")
            df_sel = re.sub(r'(?<!\d)\.(?!\d)', '_', df_sel)
            sliced_df = df.query(df_sel)
            hists[store_name].append(float(np.sum(sliced_df.loc[:,"weight"])))
            hists_w2[store_name].append(float(np.sum(sliced_df.loc[:,"weight"]**2)))
          bin_sel = copy.deepcopy(dataset_bins)

        else:

          # Setup bins
          if first:
            bins = hist_dict["variables"][overall_name][year_name][variation_name].axes[2].edges
            bin_sel = []

          # Loop through categories
          for cat_ind, hist in enumerate(hist_dict["variables"][overall_name][year_name][variation_name].values()):

            # Append histogram values
            hists[store_name] += list(hist[0])

            # Make selection string
            if first:
              var_name = hist_dict["variables"][overall_name][year_name][variation_name].axes[2].name
              cat_name = hist_dict["variables"][overall_name][year_name][variation_name].axes[0][cat_ind]
              if cat_name == "baseline":
                bin_sel += [f"(events.{var_name}>={bins[ind]}) & (events.{var_name}<{bins[ind+1]})" for ind in range(len(bins)-1)]
              else:
                if category_conversion is None:
                  raise ValueError("Category conversion is not set, please provide a category conversion string.")
                if cat_name not in category_conversion:
                  raise ValueError(f"Category {cat_name} not found in category conversion dictionary.")
                bin_sel += [f"(events.{var_name}>={bins[ind]}) & (events.{var_name}<{bins[ind+1]}) & ({category_conversion[cat_name]})" for ind in range(len(bins)-1)]

        # Convert to numpy array
        hists[store_name] = np.array(hists[store_name])

        # Normalise the histogram
        normalisations[store_name] = np.sum(hists[store_name])
        if (normalisations[store_name] > 1.01 or normalisations[store_name] < 0.99) and not args.dataset:
          print(f"Warning: Histogram {store_name} normalisation is not 1.0, sum is {normalisations[store_name]}. Check this behaviour is expected.")
        hists[store_name] = hists[store_name] / normalisations[store_name]
        if args.dataset:
          hists_w2[store_name] = hists_w2[store_name] / (normalisations[store_name]**2)

        # Flip the first bool
        if first:
          first = False
          

# Make weights squared histograms
if not args.dataset:

  if args.input_w2 is None:

    # Assume no weights
    hists_w2 = hists

  else:

    # Load histograms in for w2
    for file in files_w2:
      hist_dict = cu.load(file)
      for overall_name in hist_dict["variables"].keys():
        for year_name in hist_dict["variables"][overall_name].keys():
          for variation_name in hist_dict["variables"][overall_name][year_name].keys():
            if variation_name not in hists.keys(): continue
            if variation_name in hists_w2:
              raise ValueError(f"Variation {variation_name} already exists in the hists_w2 dictionary. Please check the input files.")
            hists_w2[variation_name] = []
            for cat_ind, hist in enumerate(hist_dict["variables"][overall_name][year_name][variation_name].values()):
              hists_w2[variation_name] += list(hist[0]/hist_dict["sum_genweights"][variation_name]) # The division is need because of how the normalisation is dealt with
            hists_w2[variation_name] = np.array(hists_w2[variation_name])
            hists_w2[variation_name] = hists_w2[variation_name] / normalisations[variation_name]**2

  # Check all keys in hists are in hists_w2
  for k in hists.keys():
    if k not in hists_w2:
      raise ValueError(f"Histogram {k} not found in hists_w2. Please check the input files.")


# Split by years
years_in_files = []
years = ["2016_PreVFP", "2016_PostVFP", "2017", "2018", "2022_preEE", "2022_postEE", "2023_preBPix", "2023_postBPix"]
hists_in_years = {
  year: {k: v for k, v in hists.items() if k.endswith(year)}
  for year in years
  if any(k.endswith(year) for k in hists)
}
hists_w2_in_years = {
  year: {k: v for k, v in hists_w2.items() if k.endswith(year)}
  for year in years
  if any(k.endswith(year) for k in hists_w2)
}

# Print the initial histograms for the year
for year in hists_in_years.keys():
  print()
  print(f"Histograms for year {year}:")
  for k, v in hists_in_years[year].items():
    if isinstance(v, dict):
      for ind, (k1, v1) in enumerate(v.items()):
        print(f"{k} - {bin_sel[ind]}: {list(v1)}")
    else:
      print(f"{k}: {list(v)}")

# Print the squared histograms for the year
for year in hists_w2_in_years.keys():
  print()
  print(f"Squared histograms for year {year}:")
  for k, v in hists_w2_in_years[year].items():
    if isinstance(v, dict):
      for ind, (k1, v1) in enumerate(v.items()):
        print(f"{k} - {bin_sel[ind]}: {list(v1)}")
    else:
      print(f"{k}: {list(v)}")


# Print the number of effective events for the year
for year in hists_in_years.keys():
  print()
  print(f"Effective events for year {year}:")
  for k, v in hists_in_years[year].items():
    if isinstance(v, dict):
      for ind, (k1, v1) in enumerate(v.items()):
        numerator = np.array([v1[i]**2 if hists_w2_in_years[year][k][i] != 0 else 0.0 for i in range(len(v1))])
        denominator = np.array([hists_w2_in_years[year][k][i] if hists_w2_in_years[year][k][i] != 0 else 1.0 for i in range(len(hists_w2_in_years[year][k]))])
        eff_events = numerator / denominator
        print(f"{k} - {bin_sel[ind]}: {list(eff_events)}")
    else:
      numerator = np.array([v[i]**2 if hists_w2_in_years[year][k][i] != 0 else 0.0 for i in range(len(v))])
      denominator = np.array([hists_w2_in_years[year][k][i] if hists_w2_in_years[year][k][i] != 0 else 1.0 for i in range(len(hists_w2_in_years[year][k]))])
      eff_events = numerator / denominator
      print(f"{k}: {list(eff_events)}")

# Initiate storage
nominal_names = {}
file_bins = {}
scalers = {}

# Loop through years
for year, year_hists in hists_in_years.items():

  # Initiate per year storage
  nominal_names[year] = None
  file_bins[year] = {}
  scalers[year] = {}

  # Find the nominal histogram and get the bin ind of each file
  for variation_name, hist in year_hists.items():

    # Normalise thie histogram
    norm_hist = hist / np.sum(hist)

    # Check if more than one bin is filled
    if np.max(norm_hist) < 0.995:

      # Check if the nominal histogram is already set
      if nominal_names[year] is None:
        nominal_names[year] = variation_name
      else:
        raise ValueError(f"More than one histogram with more than one bin filled found for year {year}: {nominal_names[year]} and {variation_name}")
      
    else:

      # Link the file to a bin
      bin_ind = np.argmax(norm_hist)
      if bin_ind in file_bins[year]:
        raise ValueError(f"More than one histogram with bin {bin_ind} filled found for year {year}: {file_bins[year][bin_ind]} and {variation_name}")
      file_bins[year][bin_ind] = variation_name

  if nominal_names[year] is None:
    raise ValueError(f"No nominal histogram found for year {year}")
  
  # Loop through the bins
  for b in range(len(hists[nominal_names[year]])):

    # Make sure the nominal name is in the scalers dictionary
    if nominal_names[year] not in scalers[year]:
      scalers[year][nominal_names[year]] = {}

    # Check if this bin has a file
    if b not in file_bins[year]:

      # Set nominal scaling to 1.0
      scalers[year][nominal_names[year]][b] = 1.0

    else:

      # Scale by sum weights over sum of weights squared for optimal stats
      if hists_w2[nominal_names[year]][b] > 0.0:
        scalers[year][nominal_names[year]][b] = hists[nominal_names[year]][b] / hists_w2[nominal_names[year]][b]
      else:
        scalers[year][nominal_names[year]][b] = 0.0
      if hists_w2[file_bins[year][b]][b] > 0.0:
        scalers[year][file_bins[year][b]] = hists[file_bins[year][b]][b] / hists_w2[file_bins[year][b]][b]
      else:
        scalers[year][file_bins[year][b]] = 0.0
      
      # Get total scale
      total_scale = (scalers[year][nominal_names[year]][b] * hists[nominal_names[year]][b]) + (scalers[year][file_bins[year][b]] * hists[file_bins[year][b]][b])

      # Normalise back to the nominal histogram
      scalers[year][nominal_names[year]][b] = scalers[year][nominal_names[year]][b] * hists[nominal_names[year]][b] / total_scale
      scalers[year][file_bins[year][b]] = scalers[year][file_bins[year][b]] * hists[nominal_names[year]][b] / total_scale
      del total_scale

  # Print the scalers for the year
  print()
  print(f"Scalers for year {year}:")
  for k, v in scalers[year].items():
    if isinstance(v, dict):
      for ind, (k1, v1) in enumerate(v.items()):
        print(f"{k} - {bin_sel[ind]}: {v1}")
    else:
      print(f"{k}: {v}")


  # Compare the histograms and effective events
  total_hist = np.array([scalers[year][nominal_names[year]][b] * hists_in_years[year][nominal_names[year]][b] for b in range(len(hists_in_years[year][nominal_names[year]]))])
  total_hist_squared = np.array([(scalers[year][nominal_names[year]][b]**2) * hists_w2_in_years[year][nominal_names[year]][b] for b in range(len(hists_w2_in_years[year][nominal_names[year]]))])
  for k in hists_in_years[year].keys():
    if k == nominal_names[year]: continue
    total_hist += scalers[year][k] * hists_in_years[year][k]
    total_hist_squared += ((scalers[year][k]**2) * hists_w2_in_years[year][k])
  print()
  print(f"Nominal histogram for year {year}:")
  print(list(hists_in_years[year][nominal_names[year]]))
  print(f"New histogram for year {year}:")
  print(list(total_hist))
  print(f"Nominal effective events for year {year}:")
  numerator = np.array([hists_in_years[year][nominal_names[year]][i]**2 if hists_w2_in_years[year][nominal_names[year]][i] != 0 else 0.0 for i in range(len(hists_in_years[year][nominal_names[year]]))])
  denominator = np.array([hists_w2_in_years[year][nominal_names[year]][i] if hists_w2_in_years[year][nominal_names[year]][i] != 0 else 1.0 for i in range(len(hists_w2_in_years[year][nominal_names[year]]))])
  nom_eff_events = numerator / denominator
  print(list(nom_eff_events))
  print(f"New effective events for year {year}:")
  numerator = np.array([total_hist[i]**2 if total_hist_squared[i] != 0 else 0.0 for i in range(len(total_hist))])
  denominator = np.array([total_hist_squared[i] if total_hist_squared[i] != 0 else 1.0 for i in range(len(total_hist_squared))])
  new_eff_events = numerator / denominator
  print(list(new_eff_events))
  print(f"Effective events ratio for year {year}:") 
  print(new_eff_events / nom_eff_events)


# Begin writing the python file
write_file = [
  "from pocket_coffea.lib.weights import WeightLambda",
  "import numpy as np",
  "",
]
write_file += ["def stitching_func(params, metadata, events, size, shape_variations):"]

# apply extra selection if provided
if args.extra_sel is not None:
  extra_sel_str = f" & ({args.extra_sel})"
  not_extra_sel = f"~({args.extra_sel})"
else:
  extra_sel_str = ""
  not_extra_sel = None


# Nominal sample
for year, scale_files in scalers.items():
  nominal_name_minus_year = nominal_names[year].replace(f"_{year}", "")
  write_file += [f'  if metadata["year"] == "{year}" and metadata["sample"] == "{nominal_name_minus_year}":']
  write_file += [f'    return np.select(']
  write_file += [f'      condlist=[' ]
  for cut_str in bin_sel:
    write_file += [f'        {cut_str}{extra_sel_str},']
  if not_extra_sel is not None:
    write_file += [f'        {not_extra_sel},']
  write_file += [f'      ],']
  write_file += [f'      choicelist=[' ]
  for val in scale_files[nominal_names[year]].values():
    write_file += [f'        np.ones(len(events)) * {val},']
  if not_extra_sel is not None:
    write_file += [f'        np.ones(len(events)),']
  write_file += [f'      ],']
  write_file += [f'      default=0.0']
  write_file += [f'    )']

# Split samples
for year, scale_files in scalers.items():
  for bin_ind in range(len(bin_sel)):
    if bin_ind not in file_bins[year]: continue
    name = file_bins[year][bin_ind]
    name_minus_year = name.replace(f"_{year}", "")
    write_file += [f'  elif metadata["year"] == "{year}" and metadata["sample"] == "{name_minus_year}":']
    if args.extra_sel is not None:
      write_file += [f'    return np.select(']
      write_file += [f'      condlist=[' ]
      write_file += [f'        {args.extra_sel},']
      write_file += [f'        {not_extra_sel},']
      write_file += [f'      ],']
      write_file += [f'      choicelist=[' ]
      write_file += [f'        np.ones(len(events)) * {scale_files[name]},']
      write_file += [f'        np.ones(len(events)),']
      write_file += [f'      ],']
      write_file += [f'      default=0.0']
      write_file += [f'    )']
    else:
      write_file += [f'    return np.ones(len(events)) * {scale_files[name]}']

# Default case
write_file += ['  return np.ones(len(events))']

write_file += [
  '',
  'wl_func = WeightLambda.wrap_func(',
  f'    name="{args.output_name}",',
  '    function=stitching_func,',
  '    has_variations=False',
  ')',
  '',
  f'{args.output_name} = [wl_func]',
]

with open(f"{args.output_file}", "w") as f:
  for line in write_file:
    f.write(line + "\n")
print(f"Created file {args.output_file}")