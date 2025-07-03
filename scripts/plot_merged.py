import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input', "-i", help='The input folder of the parquet files', type=str, default="output_merged_v3")
parser.add_argument('--output', "-o", help='The output plot directory', type=str, default="./")
parser.add_argument('--sel', help='A selection to apply', type=str, default=None)
args = parser.parse_args()


years = ["run2", "run3", "all", "2016_PreVFP", "2016_PostVFP", "2017", "2018", "2022_preEE", "2022_postEE", "2023_preBPix", "2023_postBPix"]

variables = {
  "FatJet_mass" : "(50,300,5)",
  "FatJet_msoftdrop" : "(50,300,5)",
  "FatJet_pt" : "(500,800,10)",
  "FatJet_eta" : "(-2.5,2.5,0.1)",
  "FatJet_phi" : "(-3.2,3.2,0.1)",
  "FatJet_tau1" : "(0,0.8,0.02)",
  "FatJet_tau2" : "(0,0.6,0.01)",
  "FatJet_tau3" : "(0,0.5,0.01)",
  "FatJet_tau4" : "(0,0.4,0.01)",
  "FatJet_btagDeepB" : "(0,1,0.02)",
  "LeptonSave_leptonType" : "[-0.5,0.5,1.5]",
  "LeptonSave_pt" : "(50,400,10)",
  "MET_pt" : "(0,400,10)",
  "MET_phi" : "(-3.2,3.2,0.1)",
  "SubJet1_mass" : "(0,300,5)",
  "SubJet1_pt" : "(200,600,10)",
  "SubJet1_eta" : "(-2.5,2.5,0.1)",
  "SubJet1_phi" : "(-3.2,3.2,0.1)",
  "SubJet1_btagDeepB" : "(0,1,0.02)",
  "SubJet2_mass" : "(0,300,5)",
  "SubJet2_pt" : "(0,400,10)",
  "SubJet2_eta" : "(-2.5,2.5,0.1)",
  "SubJet2_phi" : "(-3.2,3.2,0.1)",
  "SubJet2_btagDeepB" : "(0,1,0.02)",
}

for year in years:
  for var, bins in variables.items():
    cmd = f"python3 scripts/plot_from_parquet.py --input={args.input} --output={args.output} --var={var} --bins='{bins}' --year={year}"
    if args.sel is not None:
      cmd += f" --sel='{args.sel}'"
    os.system(cmd)