import json


file_name = "Datasets/mc_definitions.json"

with open(file_name, "r") as f:
  mc_definitions = json.load(f)


run_years = {
  "Run 2": ["2016_PreVFP", "2016_PostVFP", "2017", "2018"],
  "Run 3": ["2022_preEE", "2022_postEE", "2023_preBPix", "2023_postBPix", "2024"]
}

for run, years in run_years.items():
  
  for yr in years:

    print()

    lines = []
    lines += ["\\begin{adjustwidth}{-1in}{-1in}"]
    lines += ["\\begin{table}[hbtp]"]
    lines += ["  \\begin{center}"]
    lines += ["  \\resizebox{\\textwidth}{!}{"]
    lines += ["  \\begin{tabular}{|l|l|l|}"]
    lines += ["  \\hline"]
    lines += ["  Dataset Description & Dataset Name & XS[pb] $\\times$ BR \\\\"]
    lines += ["  \\hline"]

    first = True

    for sample, details in mc_definitions.items():
      for file_details in details["files"]:
        if file_details["metadata"]["year"] != yr: continue

        xsec = file_details["metadata"]["xsec"]
        for das_name in file_details["das_names"]:

          if first:
            global_tag_name = "_".join("-".join(das_name.split("/")[2].split("-")[:-1]).split("_")[:-1])
            global_tag_name_tex = global_tag_name.replace("_", r"\_")
            lines += [f"  \\multicolumn{{3}}{{|c|}}{{Simulated Samples for {yr.replace('_', ' ')} run, * = {global_tag_name_tex}}} \\\\"]
            lines += ["  \\hline"]            
            first = False

          das_name = das_name.replace(global_tag_name, "*")

          das_name_tex = das_name.replace("_", r"\_")
          sample_name_tex = sample.replace("_", r"\_")

          lines += [f"  {sample_name_tex} & \\texttt{{{das_name_tex}}} & {xsec} \\\\"]

          #print(sample, das_name, xsec)

    lines += ["  \\hline"]

    lines += ["  \\end{tabular}"]
    lines += ["  }"]
    lines += ["  \\end{center}"]
    lines += [f"  \\caption{{List of simulated samples for {yr.replace('_', ' ')} used for modelling.}}"]
    lines += [f"  \\label{{tab:mc_{yr}}}"]
    lines += ["\\end{table}"]
    lines += ["\\end{adjustwidth}"]

    print("\n".join(lines))
