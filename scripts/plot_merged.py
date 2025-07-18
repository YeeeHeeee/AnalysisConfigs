import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input', "-i", help='The input folder of the parquet files', type=str, default="output_merged_v3")
parser.add_argument('--output', "-o", help='The output plot directory', type=str, default="./")
parser.add_argument('--sel', help='A selection to apply', type=str, default=None)
parser.add_argument('--extra-args', help='Extra args to provide', type=str, default=None)
parser.add_argument('--year', help='Comma separated list of years', type=str, default="all,run2,run3,2016_PreVFP,2016_PostVFP,2017,2018,2022_preEE,2022_postEE,2023_preBPix,2023_postBPix")
parser.add_argument('--variable', help='Comma separated list of years', type=str, default=None)
args = parser.parse_args()

years = args.year.split(",")

variables = {
  "FatJet_mass" : "(50,300,5)",
  "FatJet_msoftdrop" : "(50,300,5)",
  "FatJet_pt" : "(500,800,10)",
  "FatJet_eta" : "(-2.5,2.5,0.1)",
  "FatJet_phi" : "(-3.2,3.2,0.2)",
  "FatJet_tau1" : "(0,0.7,0.02)",
  "FatJet_tau2" : "(0,0.4,0.01)",
  "FatJet_tau3" : "(0,0.2,0.005)",
  "FatJet_tau4" : "(0,0.2,0.005)",
  "FatJet_btagDeepB" : "(0,1,0.02)",
  "FatJet_tau32" : "(0,1,0.02)",
  "FatJet_tau21" : "(0,1,0.02)",
  "FatJet_tau31" : "(0,1,0.02)",
  "LeptonSave_leptonType" : "[-0.5,0.5,1.5]",
  "LeptonSave_pt" : "(50,400,10)",
  "MET_pt" : "(0,400,10)",
  "MET_phi" : "(-3.2,3.2,0.1)",
  "SubJet1_mass" : "(0,200,5)",
  "SubJet1_pt" : "(200,700,10)",
  "SubJet1_eta" : "(-2.5,2.5,0.1)",
  "SubJet1_phi" : "(-3.2,3.2,0.1)",
  "SubJet1_btagDeepB" : "(0,1,0.02)",
  "SubJet1_tau1" : "(0,0.7,0.02)",
  "SubJet1_tau2" : "(0,0.4,0.01)",
  "SubJet1_tau3" : "(0,0.2,0.005)",
  "SubJet1_tau4" : "(0,0.2,0.005)",
  "SubJet1_tau32" : "(0,1,0.02)",
  "SubJet1_tau21" : "(0,1,0.02)",
  "SubJet1_tau31" : "(0,1,0.02)",
  "SubJet2_mass" : "(0,150,5)",
  "SubJet2_pt" : "(0,400,10)",
  "SubJet2_eta" : "(-2.5,2.5,0.1)",
  "SubJet2_phi" : "(-3.2,3.2,0.2)",
  "SubJet2_btagDeepB" : "(0,1,0.02)",
  "SubJet2_tau1" : "(0,0.7,0.02)",
  "SubJet2_tau2" : "(0,0.4,0.01)",
  "SubJet2_tau3" : "(0,0.2,0.005)",
  "SubJet2_tau4" : "(0,0.2,0.005)",
  "SubJet2_tau32" : "(0,1,0.02)",
  "SubJet2_tau21" : "(0,1,0.02)",
  "SubJet2_tau31" : "(0,1,0.02)",
  "JetLepton_ptrel" : "(0,400,10)",
  "JetLepton_deltaR" : "(0,5,0.1)",
  "LeptonMET_mt" : "(0,400,10)",
}

calculate = {
  "FatJet_tau32" : "FatJet_tau3 / FatJet_tau2",
  "FatJet_tau21" : "FatJet_tau2 / FatJet_tau1",
  "FatJet_tau31" : "FatJet_tau3 / FatJet_tau1",
  "SubJet1_tau32" : "SubJet1_tau3 / SubJet1_tau2",
  "SubJet1_tau21" : "SubJet1_tau2 / SubJet1_tau1",
  "SubJet1_tau31" : "SubJet1_tau3 / SubJet1_tau1",
  "SubJet2_tau32" : "SubJet2_tau3 / SubJet2_tau2",
  "SubJet2_tau21" : "SubJet2_tau2 / SubJet2_tau1",
  "SubJet2_tau31" : "SubJet2_tau3 / SubJet2_tau1",
}


translate = {
  "FatJet_mass" : r"$m_{rec}^{AK8}$ (GeV)",
  "FatJet_msoftdrop" : r"$m_{rec}^{AK8, softdrop}$ (GeV)",
  "FatJet_pt" : r"$p_{T}^{AK8}$ (GeV)",
  "FatJet_eta" : r"$\eta^{AK8}$",
  "FatJet_phi" : r"$\phi^{AK8}$",
  "FatJet_tau1" : r"$\tau_{1}^{AK8}$",
  "FatJet_tau2" : r"$\tau_{2}^{AK8}$",
  "FatJet_tau3" : r"$\tau_{3}^{AK8}$",
  "FatJet_tau4" : r"$\tau_{4}^{AK8}$",
  "FatJet_btagDeepB" : r"b-tagging score (AK8)",
  "FatJet_tau32" : r"$\tau_{32}^{AK8}$",
  "FatJet_tau21" : r"$\tau_{21}^{AK8}$",
  "FatJet_tau31" : r"$\tau_{31}^{AK8}$",
  "LeptonSave_leptonType" : r"Lepton Type ($e=0$, $\mu=1$)",
  "LeptonSave_pt" : r"$p_{T}^{lep}$ (GeV)",
  "MET_pt" : r"MET (GeV)",
  "MET_phi" : r"$\phi^{MET}$",
  "SubJet1_mass" : r"$m_{rec}^{AK8, S1}$ (GeV)",
  "SubJet1_pt" : r"$p_{T}^{AK8, S1}$ (GeV)",
  "SubJet1_eta" : r"$\eta^{AK8, S1}$",
  "SubJet1_phi" : r"$\phi^{AK8, S1}$",
  "SubJet1_btagDeepB" : r"b-tagging score (AK8, S1)",
  "SubJet1_tau1" : r"$\tau_{1}^{AK8, S1}$",
  "SubJet1_tau2" : r"$\tau_{2}^{AK8, S1}$",
  "SubJet1_tau3" : r"$\tau_{3}^{AK8, S1}$",
  "SubJet1_tau4" : r"$\tau_{4}^{AK8, S1}$",
  "SubJet1_tau32" : r"$\tau_{32}^{AK8, S1}$",
  "SubJet1_tau21" : r"$\tau_{21}^{AK8, S1}$",
  "SubJet1_tau31" : r"$\tau_{31}^{AK8, S1}$",
  "SubJet2_mass" : r"$m_{rec}^{AK8, S2}$ (GeV)",
  "SubJet2_pt" : r"$p_{T}^{AK8, S2}$ (GeV)",
  "SubJet2_eta" : r"$\eta^{AK8, S2}$",
  "SubJet2_phi" : r"$\phi^{AK8, S2}$",
  "SubJet2_btagDeepB" : r"b-tagging score (AK8, S2)",
  "SubJet2_tau1" : r"$\tau_{1}^{AK8, S2}$",
  "SubJet2_tau2" : r"$\tau_{2}^{AK8, S2}$",
  "SubJet2_tau3" : r"$\tau_{3}^{AK8, S2}$",
  "SubJet2_tau4" : r"$\tau_{4}^{AK8, S2}$",
  "SubJet2_tau32" : r"$\tau_{32}^{AK8, S2}$",
  "SubJet2_tau21" : r"$\tau_{21}^{AK8, S2}$",
  "SubJet2_tau31" : r"$\tau_{31}^{AK8, S2}$",
  "JetLepton_ptrel" : r"$p_{T}^{rel}$(lep, Next AK4 jet) (GeV)",
  "JetLepton_deltaR" : r"$\Delta R$(lep, Next AK4 jet)",
  "LeptonMET_mt" : r"$m_{T}^{lep, MET}$ (GeV)",
}

# get current path
acdir = os.path.dirname(os.path.realpath(__file__)).split("AnalysisConfigs")[0]

for year in years:
  for var, bins in variables.items():

    if var in translate:
      xlabel = translate[var]
    else:
      xlabel = var

    if args.variable is not None and args.variable != var: continue
    cmd = f"python3 {acdir}/AnalysisConfigs/scripts/plot_from_parquet.py --input={args.input} --output={args.output} --var={var} --bins='{bins}' --year={year} --xlabel='{xlabel}'"
    if var in calculate:
      cmd += f" --calculate='{calculate[var]}'"
    if args.sel is not None:
      cmd += f" --sel='{args.sel}'"
    if args.extra_args is not None:
      cmd += f" {args.extra_args}"
    os.system(cmd)