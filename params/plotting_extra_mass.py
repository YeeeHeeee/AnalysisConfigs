import os

import numpy as np


#file_loc = os.getcwd()
#ac_loc = file_loc.split("/Merged")[0]
ac_loc = "/afs/cern.ch/work/g/guttley/private/top_reco/AnalysisConfigs"


use_bw_files = False
get_extra_masses = True
get_extra_extra_masses = False
use_boosted = True
split_merged = True

ttbar_files = [
  "TTToSemiLeptonic",
  "TTTo2L2Nu",
  "TTToHadronic",
]
if use_boosted:
  ttbar_files += [
    "TTMtt700To1000",
    "TTMtt1000",
  ]


ttbar_split_masses = [
  #166.5,
  169.5,
  #171.5,
  #173.5,
  175.5,
  #178.5,
]
ttbar_bw_extra_split_masses = [
  #170.5,
  #171.0,
  #172.0,
  172.5,
  #173.0,
  #174.0,
  #174.5,
]

ttbar_bw_extra_extra_split_masses = [
  #171.6,
  #171.8,
  #172.2,
  #172.4,
  #172.6,
  #172.8,
  #173.2,
  #173.4,
]

if get_extra_extra_masses:
  ttbar_bw_extra_split_masses += ttbar_bw_extra_extra_split_masses

ttbar_split_mass_files = []
for mass in [str(m).replace(".","p") for m in ttbar_split_masses]:
  ttbar_split_mass_files += [
    f"TTToSemiLeptonic{mass}",
    f"TTTo2L2Nu{mass}",
    f"TTToHadronic{mass}",
  ]

ttbar_bw_files = []
for mass in [str(m).replace(".","p") for m in ttbar_split_masses+ttbar_bw_extra_split_masses]:
  ttbar_bw_files += [f"TT_{mass}"]

all_ttbar_files = ttbar_files + ttbar_split_mass_files + ttbar_bw_files

st_files = [
  "ST_t_channel_top",
  "ST_t_channel_antitop",
  "ST_s_channel",
  "ST_s_channel_top",
  "ST_s_channel_antitop",
  "ST_tW_antitop",
  "ST_tW_top"
]

top_files = all_ttbar_files + st_files

wjets_files = [
  "WJetsToLNu",
  "WJetsToLNuHT70To100",
  "WJetsToLNuHT100To200",
  "WJetsToLNuHT200To400",
  "WJetsToLNuHT400To600",
  "WJetsToLNuHT600To800",
  "WJetsToLNuHT800To1200",
  "WJetsToLNuHT1200To2500",
  "WJetsToLNuHT2500",
  "WJetsToLNuHT40To100MLNu0To120",
  "WJetsToLNuHT100To400MLNu0To120",
  "WJetsToLNuHT400To800MLNu0To120",
  "WJetsToLNuHT800To1500MLNu0To120",
  "WJetsToLNuHT1500To2500MLNu0To120",
  "WJetsToLNuHT2500MLNu0To120",
  "WJetsToLNuHT40To100MLNu120",
  "WJetsToLNuHT100To400MLNu120",
  "WJetsToLNuHT400To800MLNu120",
  "WJetsToLNuHT800To1500MLNu120",
  "WJetsToLNuHT1500To2500MLNu120",
  "WJetsToLNuHT2500MLNu120",  
  #"WJetsToLNuMLNu120To200",
  #"WJetsToLNuMLNu200To400",
  #"WJetsToLNuMLNu400To800",
  #"WJetsToLNuMLNu800To1500",
  #"WJetsToLNuMLNu1500To2500",
  #"WJetsToLNuMLNu2500To4000",
  #"WJetsToLNuMLNu4000To6000",
  #"WJetsToLNuMLNu6000",
  "WJetsToLNu1J",
  "WJetsToLNu2J",
  "WJetsToLNu3J",
  "WJetsToLNu4J",
]

other_files = [
  "QCD_MuEnriched_Pt15To20",
  "QCD_MuEnriched_Pt20To30",
  "QCD_MuEnriched_Pt30To50",
  "QCD_MuEnriched_Pt50To80",
  "QCD_MuEnriched_Pt80To120",
  "QCD_MuEnriched_Pt120To170",
  "QCD_MuEnriched_Pt170To300",
  "QCD_MuEnriched_Pt300To470",
  "QCD_MuEnriched_Pt470To600",
  "QCD_MuEnriched_Pt600To800",
  "QCD_MuEnriched_Pt800To1000",
  "QCD_MuEnriched_Pt1000",
  "QCD_bcToE_Pt15To20",
  "QCD_bcToE_Pt20To30",
  "QCD_bcToE_Pt30To80",
  "QCD_bcToE_Pt80To170",
  "QCD_bcToE_Pt170To250",
  "QCD_bcToE_Pt250",
  #"QCD_EMEnriched_Pt15To20",
  #"QCD_EMEnriched_Pt20To30",
  #"QCD_EMEnriched_Pt30To50",
  #"QCD_EMEnriched_Pt50To80",
  #"QCD_EMEnriched_Pt80To120",
  #"QCD_EMEnriched_Pt120To170",
  #"QCD_EMEnriched_Pt170To300",
  #"QCD_EMEnriched_Pt300",
  #"QCD_EMEnriched_Pt30To80",
  #"QCD_EMEnriched_Pt80To170",
  #"QCD_EMEnriched_Pt170To250",
  #"QCD_EMEnriched_Pt250",
  "WW",
  "WZ",
  "ZZ",
  "DY",
]

all_files = top_files + other_files + wjets_files

extra_columns = ["LeptonSave_eta", "LeptonSave_phi", "LeptonSave_mass", "GenWeights_isr1fsr0p5", "GenWeights_isr1fsr2"]
if not split_merged:
  groups = {
    'Data': ['DATA_*.parquet'],
    'TT (172.5 GeV)': [f'{f}_20*.parquet' for f in ttbar_files] if not use_bw_files else ['TT_172p5_*_chunk_*.parquet'],
    'ST': [f'{f}_*.parquet' for f in st_files],
    'WJ': [f'{f}_*.parquet' for f in wjets_files],
    #'Other': [f'{f}_*.parquet' for f in other_files],
  }
  group_selection = {}
else:
  groups = {
    'Data': ['DATA_*.parquet'],
    'TT Merged (172.5 GeV)': [f'{f}_20*.parquet' for f in ttbar_files] if not use_bw_files else ['TT_172p5_*_chunk_*.parquet'],
    'TT Unmerged (172.5 GeV)': [f'{f}_20*.parquet' for f in ttbar_files] if not use_bw_files else ['TT_172p5_*_chunk_*.parquet'],
    'ST': [f'{f}_*.parquet' for f in st_files],
    'WJ': [f'{f}_*.parquet' for f in wjets_files],
    #'Other': [f'{f}_*.parquet' for f in other_files],
  }
  group_selection = {
    'TT Merged (172.5 GeV)': 'MergingInfo_FatJet_TopDecaysMerged > 0.5',
    'TT Unmerged (172.5 GeV)': 'MergingInfo_FatJet_TopDecaysMerged < 0.5',
  }
  extra_columns += ['MergingInfo_FatJet_TopDecaysMerged']

scale_to = {}

if get_extra_masses:
  extra_masses = [i for i in ttbar_split_masses]
  if use_bw_files:
    extra_masses += ttbar_bw_extra_split_masses
  other_groups = {}
  for mass in extra_masses:
    other_groups[f'TT ({mass} GeV)'] = [f'TTToSemiLeptonic{str(mass).replace(".","p")}_*.parquet', f'TTToHadronic{str(mass).replace(".","p")}_*.parquet', f'TTTo2L2Nu{str(mass).replace(".","p")}_*.parquet'] if not use_bw_files else [f'TT_{str(mass).replace(".","p")}_*chunk_*.parquet']
    if not split_merged:
      scale_to[f'TT ({mass} GeV)'] = "TT (172.5 GeV)"
    else:
      scale_to[f'TT ({mass} GeV)'] = ["TT Merged (172.5 GeV)", "TT Unmerged (172.5 GeV)"]

else:
  other_groups = {}

if not split_merged:
  colours = {
    'TT (172.5 GeV)': 'blue',
    'WJ': 'red',
    'ST': 'brown',
    'VV': 'gray',
    'Other': 'cyan',
  }
else:
  colours = {
    'TT Merged (172.5 GeV)': 'blue',
    'TT Unmerged (172.5 GeV)': 'purple',
    'WJ': 'red',
    'ST': 'brown',
    'VV': 'gray',
    'Other': 'cyan',
  }


if get_extra_masses:
  plot_extra = {
    'Total (169.5 GeV)': ['TT (169.5 GeV)'],
    'Total (175.5 GeV)': ['TT (175.5 GeV)'],
  }
  if not split_merged:
    plot_extra_subtract = {
      'Total (169.5 GeV)': ['TT (172.5 GeV)'],
      'Total (175.5 GeV)': ['TT (172.5 GeV)'],
    }
  else:
    plot_extra_subtract = {
      'Total (169.5 GeV)': ['TT Merged (172.5 GeV)', 'TT Unmerged (172.5 GeV)'],
      'Total (175.5 GeV)': ['TT Merged (172.5 GeV)', 'TT Unmerged (172.5 GeV)'],
    }
  plot_extra_colours = {
    'Total (169.5 GeV)': 'orange',
    'Total (175.5 GeV)': 'green',
  }
else:
  plot_extra = {}
  plot_extra_subtract = {}
  plot_extra_colours = {}

variables = {
  'FatJet_mass' : '(50,320,5)',
  'FatJet_msoftdrop' : '(50,320,5)',
  'FatJet_pt' : '(400,800,10)',
  'FatJet_eta' : '(-2.5,2.5,0.1)',
  'FatJet_phi' : '(-3.2,3.2,0.2)',
  'FatJet_tau1' : '(0,0.7,0.02)',
  'FatJet_tau2' : '(0,0.4,0.01)',
  'FatJet_tau3' : '(0,0.2,0.005)',
  'FatJet_tau4' : '(0,0.2,0.005)',
  'FatJet_btagDeepB' : '(0,1,0.02)',
  'FatJet_tau32' : '(0,1,0.02)',
  'FatJet_tau21' : '(0,1,0.02)',
  'FatJet_tau31' : '(0,1,0.02)',
  'LeptonSave_leptonType' : '[-0.5,0.5,1.5]',
  'LeptonSave_pt' : '(20,400,10)',
  'LeptonSave_RelIso' : '(0,0.5,0.01)',
  'MET_pt' : '(0,400,10)',
  "nu_E" : '(0,400,10)',
  'MET_phi' : '(-3.2,3.2,0.1)',
  'SubJet1_mass' : '(0,200,5)',
  'SubJet1_pt' : '(200,700,10)',
  'SubJet1_eta' : '(-2.5,2.5,0.1)',
  'SubJet1_phi' : '(-3.2,3.2,0.1)',
  'SubJet1_btagDeepB' : '(0,1,0.02)',
  'SubJet1_tau1' : '(0,0.7,0.02)',
  'SubJet1_tau2' : '(0,0.4,0.01)',
  'SubJet1_tau3' : '(0,0.2,0.005)',
  'SubJet1_tau4' : '(0,0.2,0.005)',
  'SubJet1_tau32' : '(0,1,0.02)',
  'SubJet1_tau21' : '(0,1,0.02)',
  'SubJet1_tau31' : '(0,1,0.02)',
  'SubJet1_n2b1' : '(0,0.5,0.02)',
  'SubJet1_n3b1' : '(1,4,0.1)',
  'SubJet2_mass' : '(0,150,5)',
  'SubJet2_pt' : '(0,400,10)',
  'SubJet2_eta' : '(-2.5,2.5,0.1)',
  'SubJet2_phi' : '(-3.2,3.2,0.2)',
  'SubJet2_btagDeepB' : '(0,1,0.02)',
  'SubJet2_tau1' : '(0,0.7,0.02)',
  'SubJet2_tau2' : '(0,0.4,0.01)',
  'SubJet2_tau3' : '(0,0.2,0.005)',
  'SubJet2_tau4' : '(0,0.2,0.005)',
  'SubJet2_tau32' : '(0,1,0.02)',
  'SubJet2_tau21' : '(0,1,0.02)',
  'SubJet2_tau31' : '(0,1,0.02)',
  'SubJet2_n2b1' : '(0,0.5,0.02)',
  'SubJet2_n3b1' : '(1,4,0.1)',
  'CombinedSubJets_mass' : '(50,320,5)',
  'CombinedSubJets_pt' : '(400,800,10)',
  'CombinedSubJets_eta' : '(-2.5,2.5,0.1)',
  'CombinedSubJets_phi' : '(-3.2,3.2,0.2)',
  'CombinedSubJets_mass_W_rescaled' : '(50,320,5)',
  'CombinedSubJets_pt_W_rescaled' : '(400,800,10)',
  'CombinedSubJets_eta_W_rescaled' : '(-2.5,2.5,0.1)',
  'CombinedSubJets_phi_W_rescaled' : '(-3.2,3.2,0.2)',
  'JetLepton_ptrel' : '(0,400,10)',
  'JetLepton_deltaR' : '(0,5,0.1)',
  'ClosestJetWithLeptonRemoved_ptrel' : '(0,400,10)',
  'ClosestJetWithLeptonRemoved_deltaR' : '(0,5,0.1)',
  'ClosestJetWithoutLeptonRemoved_ptrel' : '(0,400,10)',
  'ClosestJetWithoutLeptonRemoved_deltaR' : '(0,5,0.1)',
  'LeptonMET_mt' : '(0,400,10)',
  'BJetLep_pt' : '(0,500,10)',
  'BJetLep_eta' : '(-2.5,2.5,0.1)',
  'BJetLep_phi' : '(-3.2,3.2,0.1)',
  'BJetLep_mass' : '(0,200,5)',
  'BJetLep_btagDeepFlavB' : '(0,1,0.02)',
  'LeptonicTop_mass' : '(20,250,5)',
  'LeptonicTop_pt' : '(0,600,20)',
  'LeptonicTopPlusMET_mass' : '(20,400,5)',
  'LeptonicTopPlusMET_pt' : '(0,600,20)',
  'LeptonicTop_Wconstraint_mass' : '(100,300,5)',
  'LeptonicTop_Wconstraint_pt' : '(0,600,20)',
  'ClosestJetToLepton_mass' : '(0,200,5)',
  'ClosestJetToLepton_pt' : '(0,200,10)',
  'ClosestJetToLepton_eta' : '(-2.5,2.5,0.1)',
  'ClosestJetToLepton_phi' : '(-3.2,3.2,0.1)',
}

translate = {
  'FatJet_mass' : '$m_{rec}^{AK8}$ (GeV)',
  'FatJet_msoftdrop' : '$m_{rec}^{AK8, softdrop}$ (GeV)',
  'FatJet_pt' : '$p_{T}^{AK8}$ (GeV)',
  'FatJet_eta' : '$\\eta^{AK8}$',
  'FatJet_phi' : '$\\phi^{AK8}$',
  'FatJet_tau1' : '$\\tau_{1}^{AK8}$',
  'FatJet_tau2' : '$\\tau_{2}^{AK8}$',
  'FatJet_tau3' : '$\\tau_{3}^{AK8}$',
  'FatJet_tau4' : '$\\tau_{4}^{AK8}$',
  'FatJet_btagDeepB' : 'b tagging score (AK8)',
  'FatJet_tau32' : '$\\tau_{32}^{AK8}$',
  'FatJet_tau21' : '$\\tau_{21}^{AK8}$',
  'FatJet_tau31' : '$\\tau_{31}^{AK8}$',
  'LeptonSave_leptonType' : 'Lepton Type ($e=0$, $\mu=1$)',
  'LeptonSave_pt' : '$p_{T}^{lep}$ (GeV)',
  'MET_pt' : 'MET (GeV)',
  'MET_phi' : '$\phi^{MET}$',
  'SubJet1_mass' : '$m_{rec}^{AK8, S1}$ (GeV)',
  'SubJet1_pt' : '$p_{T}^{AK8, S1}$ (GeV)',
  'SubJet1_eta' : '$\\eta^{AK8, S1}$',
  'SubJet1_phi' : '$\\phi^{AK8, S1}$',
  'SubJet1_btagDeepB' : 'b tagging score (AK8, S1)',
  'SubJet1_tau1' : '$\\tau_{1}^{AK8, S1}$',
  'SubJet1_tau2' : '$\\tau_{2}^{AK8, S1}$',
  'SubJet1_tau3' : '$\\tau_{3}^{AK8, S1}$',
  'SubJet1_tau4' : '$\\tau_{4}^{AK8, S1}$',
  'SubJet1_tau32' : '$\\tau_{32}^{AK8, S1}$',
  'SubJet1_tau21' : '$\\tau_{21}^{AK8, S1}$',
  'SubJet1_tau31' : '$\\tau_{31}^{AK8, S1}$',
  'SubJet2_mass' : '$m_{rec}^{AK8, S2}$ (GeV)',
  'SubJet2_pt' : '$p_{T}^{AK8, S2}$ (GeV)',
  'SubJet2_eta' : '$\\eta^{AK8, S2}$',
  'SubJet2_phi' : '$\\phi^{AK8, S2}$',
  'SubJet2_btagDeepB' : 'b tagging score (AK8, S2)',
  'SubJet2_tau1' : '$\\tau_{1}^{AK8, S2}$',
  'SubJet2_tau2' : '$\\tau_{2}^{AK8, S2}$',
  'SubJet2_tau3' : '$\\tau_{3}^{AK8, S2}$',
  'SubJet2_tau4' : '$\\tau_{4}^{AK8, S2}$',
  'SubJet2_tau32' : '$\\tau_{32}^{AK8, S2}$',
  'SubJet2_tau21' : '$\\tau_{21}^{AK8, S2}$',
  'SubJet2_tau31' : '$\\tau_{31}^{AK8, S2}$',
  'CombinedSubJets_mass' : '$m_{rec}^{AK8, S1+S2}$ (GeV)',
  'CombinedSubJets_pt' : '$p_{T}^{AK8, S1+S2}$ (GeV)',
  'CombinedSubJets_eta' : '$\\eta^{AK8, S1+S2}$',
  'CombinedSubJets_phi' : '$\\phi^{AK8, S1+S2}$',
  'CombinedSubJets_mass_W_rescaled' : '$m_{rec}^{AK8, S1+S2, W-rescaled}$ (GeV)',
  'CombinedSubJets_pt_W_rescaled' : '$p_{T}^{AK8, S1+S2, W-rescaled}$ (GeV)',
  'CombinedSubJets_eta_W_rescaled' : '$\\eta^{AK8, S1+S2, W-rescaled}$',
  'CombinedSubJets_phi_W_rescaled' : '$\\phi^{AK8, S1+S2, W-rescaled}$',
  'BJetLep_pt' : '$p_{T}^{lep b jet}$ (GeV)',
  'BJetLep_eta' : '$\\eta^{lep b jet}$',
  'BJetLep_phi' : '$\\phi^{lep b jet}$',
  'BJetLep_mass' : '$m_{rec}^{b jet}$ (GeV)',
  'BJetLep_btagDeepFlavB' : 'b tagging score (lep b jet)',
  'JetLepton_ptrel' : '$p_{T}^{rel}$(lep, Next AK4 jet) (GeV)',
  'JetLepton_deltaR' : '$\\Delta R$(lep, Next AK4 jet)',
  'ClosestJetWithLeptonRemoved_ptrel' : '$p_{T}^{rel}$(lep, Next AK4 jet) (GeV)',
  'ClosestJetWithLeptonRemoved_deltaR' : '$\\Delta R$(lep, Next AK4 jet)',
  'LeptonMET_mt' : '$m_{T}^{lep, MET}$ (GeV)',
}


all_years = [
  "2016_PreVFP",
  "2016_PostVFP",
  "2017",
  "2018",
  "2022_preEE",
  "2022_postEE",
  "2023_preBPix",
  "2023_postBPix",
]

extra_columns += list(variables.keys())


# Ideal
#jec_uncert = {
#  "AbsoluteMPFBias": {"Correlation" : 1},
#  "AbsoluteScale": {"Correlation" : 1},
#  "AbsoluteStat": {"Correlation" : 0},
#  "FlavorQCD": {"Correlation" : 1},
#  "Fragmentation": {"Correlation" : 1},
#  "PileUpDataMC": {"Correlation" : 0.5},
#  "PileUpPtBB": {"Correlation" : 0.5},
#  "PileUpPtEC1": {"Correlation" : 0.5},
#  "PileUpPtEC2": {"Correlation" : 0.5},
#  "PileUpPtHF": {"Correlation" : 0.5},
#  "PileUpPtRef": {"Correlation" : 0.5},
#  "RelativeFSR": {"Correlation" : 0.5},
#  "RelativePtBB": {"Correlation" : 0.5},
#  "RelativePtEC1": {"Correlation" : 0},
#  "RelativePtEC2": {"Correlation" : 0},
#  "RelativePtHF": {"Correlation" : 0.5},
#  "RelativeBal": {"Correlation" : 0.5},
#  "RelativeSample": {"Correlation" : 0},
#  "RelativeStatEC": {"Correlation" : 0},
#  "RelativeStatFSR": {"Correlation" : 0},
#  "RelativeStatHF": {"Correlation" : 0},
#  "SinglePionECAL": {"Correlation" : 1},
#  "SinglePionHCAL": {"Correlation" : 1},
#  "TimePtEta": {"Correlation" : 0},
#}

calculate = {}
if not use_boosted:
  if "weight" in calculate:
    calculate["weight"] = f"({calculate['weight']})/(Extra_TTTo2L2NuRun2Stitching*Extra_TTToSemiLeptonicRun2Stitching*Extra_TTToHadronicRun2Stitching)"
  else:
    calculate.update({
      "weight" : "weight/(Extra_TTTo2L2NuRun2Stitching*Extra_TTToSemiLeptonicRun2Stitching*Extra_TTToHadronicRun2Stitching)"
    })

# Add systematics
systematics = {}

# Jet energy uncertainties
jec_uncert = {
  # Simple JEC correlation
  "AbsoluteMPFBias": {"Correlation" : 1},
  "AbsoluteScale": {"Correlation" : 1},
  "AbsoluteStat": {"Correlation" : 0},
  "FlavorQCD": {"Correlation" : 1},
  "Fragmentation": {"Correlation" : 1},
  "PileUpDataMC": {"Correlation" : 1},
  "PileUpPtBB": {"Correlation" : 1},
  "PileUpPtEC1": {"Correlation" : 1},
  "PileUpPtEC2": {"Correlation" : 1},
  "PileUpPtHF": {"Correlation" : 1},
  "PileUpPtRef": {"Correlation" : 1},
  "RelativeFSR": {"Correlation" : 1},
  "RelativeJEREC1": {"Correlation" : 0},
  "RelativeJEREC2": {"Correlation" : 0},
  "RelativeJERHF": {"Correlation" : 1},
  "RelativePtBB": {"Correlation" : 1},
  "RelativePtEC1": {"Correlation" : 0},
  "RelativePtEC2": {"Correlation" : 0},
  "RelativePtHF": {"Correlation" : 1},
  "RelativeBal": {"Correlation" : 1},
  "RelativeSample": {"Correlation" : 0},
  "RelativeStatEC": {"Correlation" : 0},
  "RelativeStatFSR": {"Correlation" : 0},
  "RelativeStatHF": {"Correlation" : 0},
  "SinglePionECAL": {"Correlation" : 1},
  "SinglePionHCAL": {"Correlation" : 1},
  "TimePtEta": {"Correlation" : 0},
  # Flavour
  "FlavorPureGluon" : {"Correlation" : 1},
  "FlavorPureQuark" : {"Correlation" : 1},
  "FlavorPureCharm" : {"Correlation" : 1},
  "FlavorPureBottom" : {"Correlation" : 1},
  # JER
  "JER_eta_lt_1p93": {"Correlation" : 1},
  "JER_eta_1p93_to_2p5": {"Correlation" : 1},
  "JER_eta_2p5_to_3p0_pt_0_to_50": {"Correlation" : 1},
  "JER_eta_2p5_to_3p0_pt_gt_50": {"Correlation" : 1},
  "JER_eta_3p0_to_5p0_pt_0_to_50": {"Correlation" : 1},
  "JER_eta_3p0_to_5p0_pt_gt_50": {"Correlation" : 1}

}
for name, info in jec_uncert.items():
  if info["Correlation"] == 1:
    corr_years = [all_years]
    syst_names = [name]
    scalings = 1.0
  elif info["Correlation"] == 0:
    corr_years = [[yr] for yr in all_years]
    syst_names = [f"{name}_{yr}" for yr in all_years]
    scalings = 1.0
  elif info["Correlation"] == 0.5:
    corr_years = [all_years] + [[yr] for yr in all_years]
    syst_names = [name] + [f"{name}_{yr}" for yr in all_years]
    scalings = 0.5
  for ind in range(len(syst_names)):
    if "FlavorPure" not in syst_names[ind]:
      systematics[syst_names[ind]] = {
        "function": [f"{ac_loc}/params/plotting_extra_mass.py","btm_jec"],
        "files": all_files,
        "years": corr_years[ind],
      }
    else:
      systematics[syst_names[ind]] = {
        "function": [f"{ac_loc}/params/plotting_extra_mass.py","btm_jec"],
        "files": all_ttbar_files,
        "years": corr_years[ind],
      }      

# Luminosity uncertainties
lumi_uncerts = {
  "lumi_13TeV_1516_l" : ["2016_PreVFP", "2016_PostVFP"],
  "lumi_13TeV_151617_l" : ["2016_PreVFP", "2016_PostVFP", "2017"],
  "lumi_13TeV_15161718_l" : ["2016_PreVFP", "2016_PostVFP", "2017", "2018"],
  "lumi_13p6TeV_2223_l" : ["2022_preEE", "2022_postEE", "2023_preBPix", "2023_postBPix"],
  "lumi_13p6TeV_23_l" : ["2023_preBPix", "2023_postBPix"],
}
for name, uncert in lumi_uncerts.items():
  systematics[name] = {
    "function" : [f"{ac_loc}/params/plotting_extra_mass.py","lumi_uncertainty"],
    'files' : all_files,
    'years' : uncert,
  }

# Top pt reweighting uncertainty
systematics["top_pt_uncert"] = {
  "function": [f"{ac_loc}/params/plotting_extra_mass.py","top_pt_uncertainty"],
  "files": all_ttbar_files,
  "years": all_years,
}

# Renormalisation/factorisation uncertainties
systematics["renormalisation_scale_uncertainty_ttbar"] = {
  "function": [f"{ac_loc}/params/plotting_extra_mass.py","renormalisation_scale_uncertainty"],
  "files": all_ttbar_files,
  "years": all_years,
}
systematics["renormalisation_scale_uncertainty_st"] = {
  "function": [f"{ac_loc}/params/plotting_extra_mass.py","renormalisation_scale_uncertainty"],
  "files": st_files,
  "years": all_years,
}
systematics["factorisation_scale_uncertainty_ttbar"] = {
  "function": [f"{ac_loc}/params/plotting_extra_mass.py","factorisation_scale_uncertainty"],
  "files": all_ttbar_files,
  "years": all_years,
}
systematics["factorisation_scale_uncertainty_st"] = {
  "function": [f"{ac_loc}/params/plotting_extra_mass.py","factorisation_scale_uncertainty"],
  "files": st_files,
  "years": all_years,
}

# ISR/FSR uncertainties
systematics['isr_ttbar'] = {
  "function": [f"{ac_loc}/params/plotting_extra_mass.py","isr_uncertainty"],
  "files": all_ttbar_files,
  "years": all_years,
}
systematics['isr_st'] = {
  "function": [f"{ac_loc}/params/plotting_extra_mass.py","isr_uncertainty"],
  "files": st_files,
  "years": all_years,
}
# FSR is hard because we have already shifted


#########

write_translate = {
  "Data": "data_obs",
}



config = {
  "groups": groups,
  "other_groups": other_groups,
  "colours": colours,
  "group_selection": group_selection,
  "plot_extra": plot_extra,
  "plot_extra_subtract": plot_extra_subtract,
  "plot_extra_colours": plot_extra_colours,
  "variables": variables,
  "translate": translate,
  "systematics": systematics,
  "write_translate": write_translate,
  "scale_to": scale_to,
  "extra_columns": extra_columns,
  "all_columns": False,
  "function_to_apply": [f"{ac_loc}/params/plotting_extra_mass.py","df_processing"],
  "calculate": calculate,
}



### Functions ###


def CombineObjects(obj1, obj2):

  # Convert everything to arrays if not already
  pt1, eta1, phi1, m1 = obj1["pt"], obj1["eta"], obj1["phi"], obj1["mass"]
  pt2, eta2, phi2, m2 = obj2["pt"], obj2["eta"], obj2["phi"], obj2["mass"]

  # Compute 4-momenta
  px = pt1 * np.cos(phi1) + pt2 * np.cos(phi2)
  py = pt1 * np.sin(phi1) + pt2 * np.sin(phi2)
  pz = pt1 * np.sinh(eta1) + pt2 * np.sinh(eta2)
  e  = np.sqrt(m1**2 + pt1**2 * np.cosh(eta1)**2) + np.sqrt(m2**2 + pt2**2 * np.cosh(eta2)**2)

  # Compute final kinematics
  pt   = np.sqrt(px**2 + py**2)
  mass = np.sqrt(np.maximum(e**2 - px**2 - py**2 - pz**2, 0))
  eta  = 0.5 * np.log((e + pz) / np.maximum(e - pz, 1e-12))
  phi  = np.arctan2(py, px)

  return {"pt": pt, "eta": eta, "phi": phi, "mass": mass}


def AsymLogNormal(nu, kp=1.2, km=0.8, q=0.5):

  nu = np.asarray(nu)

  out = np.empty_like(nu, dtype=float)

  mask_pos = nu >= q
  mask_neg = nu < -q
  mask_mid = ~(mask_pos | mask_neg)

  if isinstance(kp, float):
    kp = np.full_like(nu, kp, dtype=float)
  if isinstance(km, float):
    km = np.full_like(nu, km, dtype=float)

  # nu >= q
  out[mask_pos] = np.exp(nu[mask_pos] * np.log(kp[mask_pos]))

  # nu < -q
  out[mask_neg] = np.exp(-nu[mask_neg] * np.log(km[mask_neg]))

  # -q <= nu < q
  nu_m = nu[mask_mid]
  out[mask_mid] = np.exp(
      nu_m * (
          (np.log(km[mask_mid]) + np.log(kp[mask_mid])) *
          (3 * nu_m**5 / (8 * q**5)
            - 5 * nu_m**3 / (4 * q**3)
            + 15 * nu_m / (8 * q))
          - np.log(km[mask_mid]) + np.log(kp[mask_mid])
      ) / 2
  )

  return out


def fsr_weight(
    df,
    fsr_value=None,
  ):

  if fsr_value is not None:
    df["log_fsr"] = np.log(fsr_value)
  
  nu = df["log_fsr"]/np.log(2)

  lower_clip = 0.25
  higher_clip = 4.0
  df["GenWeights_isr1fsr0p5"] = df["GenWeights_isr1fsr0p5"].clip(lower=lower_clip, upper=higher_clip)
  df["GenWeights_isr1fsr2"] = df["GenWeights_isr1fsr2"].clip(lower=lower_clip, upper=higher_clip)
  asymln = np.clip(AsymLogNormal(nu, kp=df["GenWeights_isr1fsr2"], km=df["GenWeights_isr1fsr0p5"]), 0.0, 10.0)
  df.loc[:, "weight"] *= asymln
  return df


def df_processing(df, metadata={}):

  # Make a copy of the dataframe to avoid modifying the original
  df = df.copy()

  # Add era name conversion
  era_name_conversion = {
    "2016_PreVFP": 0,
    "2016_PostVFP": 1,
    "2017": 2,
    "2018": 3,
    "2022_preEE": 4,
    "2022_postEE": 5,
    "2023_preBPix": 6,
    "2023_postBPix": 7,
  }
  df["era"] = -1
  for k, v in era_name_conversion.items():
    if "era_name" in metadata:
      if metadata["era_name"].startswith(k):
        df["era"] = v
        break

  # Calculate tau ratios
  for i in ['1','2','3']:
    for j in ['1','2','3']:
      if i <= j: continue
      df[f"FatJet_tau{i}{j}"] = df[f"FatJet_tau{i}"] / df[f"FatJet_tau{j}"]
      df[f"SubJet1_tau{i}{j}"] = df[f"SubJet1_tau{i}"] / df[f"SubJet1_tau{j}"]
      df[f"SubJet2_tau{i}{j}"] = df[f"SubJet2_tau{i}"] / df[f"SubJet2_tau{j}"]

  # Calculate leptonic top mass and pt
  vlepb = CombineObjects(
    {"pt": df["BJetLep_pt"], "eta": df["BJetLep_eta"], "phi": df["BJetLep_phi"], "mass": df["BJetLep_mass"]},
    {"pt": df["LeptonSave_pt"], "eta": df["LeptonSave_eta"], "phi": df["LeptonSave_phi"], "mass": df["LeptonSave_mass"]}
  )
  df = df.assign(
    LeptonicTop_pt = vlepb["pt"],
    LeptonicTop_eta = vlepb["eta"],
    LeptonicTop_phi = vlepb["phi"],
    LeptonicTop_mass = vlepb["mass"],
  )

  # Calculate combined subjets
  combined_subjets = CombineObjects(
    {"pt": df["SubJet1_pt"], "eta": df["SubJet1_eta"], "phi": df["SubJet1_phi"], "mass": df["SubJet1_mass"]},
    {"pt": df["SubJet2_pt"], "eta": df["SubJet2_eta"], "phi": df["SubJet2_phi"], "mass": df["SubJet2_mass"]}
  )
  df = df.assign(
    CombinedSubJets_pt = combined_subjets["pt"],
    CombinedSubJets_eta = combined_subjets["eta"],
    CombinedSubJets_phi = combined_subjets["phi"],
    CombinedSubJets_mass = combined_subjets["mass"],
  )

  # Add W rescaled top mass
  w_rescale = 80.379 / df["SubJet1_mass"]
  w_rescaled_subjets = CombineObjects(
    {"pt": df["SubJet1_pt"] * w_rescale, "eta": df["SubJet1_eta"], "phi": df["SubJet1_phi"], "mass": df["SubJet1_mass"] * w_rescale},
    {"pt": df["SubJet2_pt"], "eta": df["SubJet2_eta"], "phi": df["SubJet2_phi"], "mass": df["SubJet2_mass"]}
  )
  df.loc[:, "CombinedSubJets_mass_W_rescaled"] = w_rescaled_subjets["mass"]
  df.loc[:, "CombinedSubJets_pt_W_rescaled"] = w_rescaled_subjets["pt"]
  df.loc[:, "CombinedSubJets_eta_W_rescaled"] = w_rescaled_subjets["eta"]
  df.loc[:, "CombinedSubJets_phi_W_rescaled"] = w_rescaled_subjets["phi"]


  leptonic_top_plus_met = CombineObjects(
    {"pt": df["LeptonicTop_pt"], "eta": df["LeptonicTop_eta"], "phi": df["LeptonicTop_phi"], "mass": df["LeptonicTop_mass"]},
    {"pt": df["MET_pt"], "eta": 0, "phi": df["MET_phi"], "mass": 0}
  )
  df.loc[:, "LeptonicTopPlusMET_mass"] = leptonic_top_plus_met["mass"]
  df.loc[:, "LeptonicTopPlusMET_pt"] = leptonic_top_plus_met["pt"]

  # Add W constraint leptonic top mass
  MW = 80.379  # GeV
  lep_pt   = df["LeptonSave_pt"]
  lep_eta  = df["LeptonSave_eta"]
  lep_phi  = df["LeptonSave_phi"]
  lep_mass = df["LeptonSave_mass"]
  lep_px = lep_pt * np.cos(lep_phi)
  lep_py = lep_pt * np.sin(lep_phi)
  lep_pz = lep_pt * np.sinh(lep_eta)
  lep_E  = np.sqrt(lep_px**2 + lep_py**2 + lep_pz**2 + lep_mass**2)
  nu_px = df["MET_pt"] * np.cos(df["MET_phi"])
  nu_py = df["MET_pt"] * np.sin(df["MET_phi"])
  K = (
      (MW**2 - lep_mass**2) / 2
      + lep_px * nu_px
      + lep_py * nu_py
  )
  a = lep_E**2 - lep_pz**2
  disc = K**2 - a * (nu_px**2 + nu_py**2)
  sqrt_disc = np.sqrt(np.maximum(disc, 0))
  nu_pz_plus = (K * lep_pz + lep_E * sqrt_disc) / a
  nu_pz_minus = (K * lep_pz - lep_E * sqrt_disc) / a
  nu_pz = np.where(np.abs(nu_pz_plus) < np.abs(nu_pz_minus), nu_pz_plus, nu_pz_minus)
  nu_E = np.sqrt(nu_px**2 + nu_py**2 + nu_pz**2)
  b_px = df["BJetLep_pt"] * np.cos(df["BJetLep_phi"])
  b_py = df["BJetLep_pt"] * np.sin(df["BJetLep_phi"])
  b_pz = df["BJetLep_pt"] * np.sinh(df["BJetLep_eta"])
  b_E  = np.sqrt(b_px**2 + b_py**2 + b_pz**2 + df["BJetLep_mass"]**2)
  leptonic_top_Wconstraint_px = lep_px + nu_px + b_px
  leptonic_top_Wconstraint_py = lep_py + nu_py + b_py
  leptonic_top_Wconstraint_pz = lep_pz + nu_pz + b_pz
  leptonic_top_Wconstraint_E  = lep_E + nu_E + b_E
  df.loc[:, "LeptonicTop_Wconstraint_mass"] = np.sqrt(np.maximum(leptonic_top_Wconstraint_E**2 - leptonic_top_Wconstraint_px**2 - leptonic_top_Wconstraint_py**2 - leptonic_top_Wconstraint_pz**2, 0))
  df.loc[:, "LeptonicTop_Wconstraint_pt"] = np.sqrt(leptonic_top_Wconstraint_px**2 + leptonic_top_Wconstraint_py**2)

  df.loc[:, "nu_E"] = nu_E

  ## Apply FSR weight if specified
  #if "TT" in metadata.get("group", ""):
  #  df = fsr_weight(df, fsr_value=0.37)

  return df


def btm_jec(
    df, 
    metadata={},
    years=["2016_PreVFP","2016_PostVFP","2017","2018","2022_preEE","2022_postEE","2023_preBPix","2023_postBPix"], 
    include_b=True,
    include_b_syst=True
  ):

  # JEC and flavour uncertainties
  jec_uncert = {
    # jec
    "AbsoluteMPFBias": {"Correlation" : 1, "Type" : "corrFactor"},
    "AbsoluteScale": {"Correlation" : 1, "Type" : "corrFactor"},
    "AbsoluteStat": {"Correlation" : 0, "Type" : "corrFactor"},
    "FlavorQCD": {"Correlation" : 1, "Type" : "corrFactor"},
    "Fragmentation": {"Correlation" : 1, "Type" : "corrFactor"},
    "PileUpDataMC": {"Correlation" : 1, "Type" : "corrFactor"},
    "PileUpPtBB": {"Correlation" : 1, "Type" : "corrFactor"},
    "PileUpPtEC1": {"Correlation" : 1, "Type" : "corrFactor"},
    "PileUpPtEC2": {"Correlation" : 1, "Type" : "corrFactor"},
    "PileUpPtHF": {"Correlation" : 1, "Type" : "corrFactor"},
    "PileUpPtRef": {"Correlation" : 1, "Type" : "corrFactor"},
    "RelativeFSR": {"Correlation" : 1, "Type" : "corrFactor"},
    "RelativePtBB": {"Correlation" : 1, "Type" : "corrFactor"},
    "RelativePtEC1": {"Correlation" : 0, "Type" : "corrFactor"},
    "RelativePtEC2": {"Correlation" : 0, "Type" : "corrFactor"},
    "RelativePtHF": {"Correlation" : 1, "Type" : "corrFactor"},
    "RelativeBal": {"Correlation" : 1, "Type" : "corrFactor"},
    "RelativeSample": {"Correlation" : 0, "Type" : "corrFactor"},
    "RelativeStatEC": {"Correlation" : 0, "Type" : "corrFactor"},
    "RelativeStatFSR": {"Correlation" : 0, "Type" : "corrFactor"},
    "RelativeStatHF": {"Correlation" : 0, "Type" : "corrFactor"},
    "SinglePionECAL": {"Correlation" : 1, "Type" : "corrFactor"},
    "SinglePionHCAL": {"Correlation" : 1, "Type" : "corrFactor"},
    "TimePtEta": {"Correlation" : 0, "Type" : "corrFactor"},
    # flavour
    "FlavorPureGluon" : {"Correlation" : 1, "ObjectSelection" : "abs(MatchedGenJet_$OBJECT_partonFlavour)==21", "Type" : "corrFactor"},
    "FlavorPureQuark" : {"Correlation" : 1, "ObjectSelection" : "abs(MatchedGenJet_$OBJECT_partonFlavour)<=3", "Type" : "corrFactor"},
    "FlavorPureCharm" : {"Correlation" : 1, "ObjectSelection" : "abs(MatchedGenJet_$OBJECT_partonFlavour)==4", "Type" : "corrFactor"},
    "FlavorPureBottom" : {"Correlation" : 1, "ObjectSelection" : "abs(MatchedGenJet_$OBJECT_partonFlavour)==5", "Type" : "corrFactor"},
    # JER
    "JER_eta_lt_1p93": {"Correlation" : 1, "ObjectSelection": "$OBJECT_eta < 1.93", "Type" : "smearFactor"},
    "JER_eta_1p93_to_2p5": {"Correlation" : 1, "ObjectSelection": "($OBJECT_eta >= 1.93) & ($OBJECT_eta < 2.5)", "Type" : "smearFactor"},
    "JER_eta_2p5_to_3p0_pt_0_to_50": {"Correlation" : 1, "ObjectSelection": "($OBJECT_eta >= 2.5) & ($OBJECT_eta < 3.0) & ($OBJECT_pt < 50)", "Type" : "smearFactor"},
    "JER_eta_2p5_to_3p0_pt_gt_50": {"Correlation" : 1, "ObjectSelection": "($OBJECT_eta >= 2.5) & ($OBJECT_eta < 3.0) & ($OBJECT_pt >= 50)", "Type" : "smearFactor"},
    "JER_eta_3p0_to_5p0_pt_0_to_50": {"Correlation" : 1, "ObjectSelection": "($OBJECT_eta >= 3.0) & ($OBJECT_eta < 5.0) & ($OBJECT_pt < 50)", "Type" : "smearFactor"},
    "JER_eta_3p0_to_5p0_pt_gt_50": {"Correlation" : 1, "ObjectSelection": "($OBJECT_eta >= 3.0) & ($OBJECT_eta < 5.0) & ($OBJECT_pt >= 50)", "Type" : "smearFactor"}
  }

  # Apply JEC to jets
  added_columns = []
  for name, info in jec_uncert.items():

    if info["Correlation"] == 1:
      syst_names = [name]
      scalings = 1.0
    elif info["Correlation"] == 0:
      syst_names = [f"{name}_{yr}" for yr in years]
      scalings = 1.0
    elif info["Correlation"] == 0.5:
      syst_names = [name] + [f"{name}_{yr}" for yr in years]
      scalings = 0.5

    for ind in range(len(syst_names)):

      if "syst_name" in metadata:
        if syst_names[ind] != metadata["syst_name"]:
          continue

      #if syst_names[ind] not in nuisances:
      #  continue

      if df[syst_names[ind]].eq(0).all():
        continue

      # Initiate corrFactor syst columns
      df[f"SubJet1_shiftFactor_{syst_names[ind]}"] = 1.0
      df[f"SubJet2_shiftFactor_{syst_names[ind]}"] = 1.0
      added_columns.extend([f"SubJet1_shiftFactor_{syst_names[ind]}", f"SubJet2_shiftFactor_{syst_names[ind]}"])
      if include_b:
        df[f"BJetLep_shiftFactor_{syst_names[ind]}"] = 1.0
        added_columns.append(f"BJetLep_shiftFactor_{syst_names[ind]}")

      # Apply only to selected events if specified
      selected_indices = np.ones(len(df), dtype=bool)
      if "Selection" in info:
        selected_indices &= df.eval(info["Selection"])
      subjet1_indices = selected_indices.copy()
      subjet2_indices = selected_indices.copy()
      if include_b and include_b_syst:
        bjetlep_indices = selected_indices.copy()
      if "ObjectSelection" in info:
        subjet1_indices &= df.eval(info["ObjectSelection"].replace("$OBJECT", "SubJet1"))
        subjet2_indices &= df.eval(info["ObjectSelection"].replace("$OBJECT", "SubJet2"))
        if include_b and include_b_syst:
          bjetlep_indices &= df.eval(info["ObjectSelection"].replace("$OBJECT", "BJetLep"))

      # Get variations
      if info["Type"] == "corrFactor":

        corrFactor = lambda df, obj, indices, syst_name, name, scalings: 1.0 + (scalings*df.loc[indices,syst_name]*df.loc[indices,f"{obj}_corrFactor_{name}"]/df.loc[indices,f"{obj}_corrFactor"])

        df.loc[subjet1_indices, f"SubJet1_shiftFactor_{syst_names[ind]}"] = corrFactor(df, "SubJet1", subjet1_indices, syst_names[ind], name, scalings)
        df.loc[subjet2_indices, f"SubJet2_shiftFactor_{syst_names[ind]}"] = corrFactor(df, "SubJet2", subjet2_indices, syst_names[ind], name, scalings)
        if include_b and include_b_syst:
          df.loc[bjetlep_indices, f"BJetLep_shiftFactor_{syst_names[ind]}"] = corrFactor(df, "BJetLep", bjetlep_indices, syst_names[ind], name, scalings)

      elif info["Type"] == "smearFactor":
        
        smearFactor = lambda df, obj, indices, syst_name, scalings: 1.0 + ((df.loc[indices,syst_name]>=0) * scalings * df.loc[indices,syst_name] * (df.loc[indices, f"{obj}_smearFactor_up"]-df.loc[indices, f"{obj}_smearFactor"]) / df.loc[indices, f"{obj}_smearFactor"]) + ((df.loc[indices,syst_name]<0) * scalings * abs(df.loc[indices,syst_name]) * (df.loc[indices, f"{obj}_smearFactor_down"]-df.loc[indices, f"{obj}_smearFactor"]) / df.loc[indices, f"{obj}_smearFactor"])

        df.loc[subjet1_indices, f"SubJet1_shiftFactor_{syst_names[ind]}"] = smearFactor(df, "SubJet1", subjet1_indices, syst_names[ind], scalings)
        df.loc[subjet2_indices, f"SubJet2_shiftFactor_{syst_names[ind]}"] = smearFactor(df, "SubJet2", subjet2_indices, syst_names[ind], scalings)
        if include_b and include_b_syst:
          df.loc[bjetlep_indices, f"BJetLep_shiftFactor_{syst_names[ind]}"] = smearFactor(df, "BJetLep", bjetlep_indices, syst_names[ind], scalings)

      # Apply variations
      df["SubJet1_pt"] *= df[f"SubJet1_shiftFactor_{syst_names[ind]}"]
      df["SubJet2_pt"] *= df[f"SubJet2_shiftFactor_{syst_names[ind]}"]
      df["SubJet1_mass"] *= df[f"SubJet1_shiftFactor_{syst_names[ind]}"]
      df["SubJet2_mass"] *= df[f"SubJet2_shiftFactor_{syst_names[ind]}"]
      if include_b and include_b_syst:
        df["BJetLep_pt"] *= df[f"BJetLep_shiftFactor_{syst_names[ind]}"]
        df["BJetLep_mass"] *= df[f"BJetLep_shiftFactor_{syst_names[ind]}"]


  # Combine subjets
  v12 = CombineObjects(
    {"pt": df["SubJet1_pt"].values, "eta": df["SubJet1_eta"].values, "phi": df["SubJet1_phi"].values, "mass": df["SubJet1_mass"].values},
    {"pt": df["SubJet2_pt"].values, "eta": df["SubJet2_eta"].values, "phi": df["SubJet2_phi"].values, "mass": df["SubJet2_mass"].values}
  )

  df["CombinedSubJets_mass"] = v12["mass"]
  df["CombinedSubJets_pt"] = v12["pt"]

  if include_b:
    vlepb = CombineObjects(
      {"pt": df["BJetLep_pt"], "eta": df["BJetLep_eta"], "phi": df["BJetLep_phi"], "mass": df["BJetLep_mass"]},
      {"pt": df["LeptonSave_pt"], "eta": df["LeptonSave_eta"], "phi": df["LeptonSave_phi"], "mass": df["LeptonSave_mass"]}
    )
    df["LeptonicTop_mass"] = vlepb["mass"]
    df["LeptonicTop_pt"] = vlepb["pt"]

  return df


def lumi_uncertainty(
    df, 
    metadata={}
  ):
  lumi_uncert = {
    "lumi_13TeV_1516_l" : {"2016_PreVFP" : 1.0118, "2016_PostVFP" : 1.0118},
    "lumi_13TeV_151617_l" : {"2016_PreVFP" : 1.0004, "2016_PostVFP" : 1.0004, "2017" : 1.0055},
    "lumi_13TeV_15161718_l" : {"2016_PreVFP" : 1.0035, "2016_PostVFP" : 1.0035, "2017" : 1.0061, "2018" : 1.0084},
    "lumi_13p6TeV_2223_l" : {"2022_preEE" : 1.0138, "2022_postEE" : 1.0138, "2023_preBPix" : 1.0017, "2023_postBPix" : 1.0017},
    "lumi_13p6TeV_23_l" : {"2023_preBPix" : 1.0127, "2023_postBPix" : 1.0127},
  }
  for name, uncert in lumi_uncert.items():

    if "syst_name" in metadata:
      if name != metadata["syst_name"]:
        continue

    for era, factor in uncert.items():
      if "era_name" in metadata:
        if metadata["era_name"].startswith(era):
          df.loc[:, "weight"] *= (1 + (df[name] * (factor - 1) ))

  return df


def top_pt_uncertainty(
    df, 
    metadata={}
  ):
  asymln = AsymLogNormal(df["top_pt_uncert"], kp=df["Extra_TopPTReweighting"], km=1/df["Extra_TopPTReweighting"])
  df["weight"] *= asymln
  return df

def factorisation_scale_uncertainty(
    df, 
    metadata={}
  ):
  if metadata.get("group","").startswith("TT"):
    nui_name = "factorisation_scale_uncertainty_ttbar"
  else:
    nui_name = "factorisation_scale_uncertainty_st"

  # Make sure the weights are never 0, if goes negative them symmetrize
  df.loc[(df["GenWeights_muF2muR1"]<=0), "GenWeights_muF2muR1"] = 1/df["GenWeights_muF0p5muR1"]
  df.loc[(df["GenWeights_muF0p5muR1"]<=0), "GenWeights_muF0p5muR1"] = 1/df["GenWeights_muF2muR1"]

  # If still negative set to 1
  df.loc[(df["GenWeights_muF2muR1"]<=0), "GenWeights_muF2muR1"] = 1.0
  df.loc[(df["GenWeights_muF0p5muR1"]<=0), "GenWeights_muF0p5muR1"] = 1.0

  asymln = AsymLogNormal(df[nui_name], kp=df["GenWeights_muF2muR1"], km=df["GenWeights_muF0p5muR1"])
  df["weight"] *= asymln
  return df

def renormalisation_scale_uncertainty(
    df, 
    metadata={}
  ):
  if metadata.get("group","").startswith("TT"):
    nui_name = "renormalisation_scale_uncertainty_ttbar"
  else:
    nui_name = "renormalisation_scale_uncertainty_st"

  # Make sure the weights are never 0, if goes negative them symmetrize
  df.loc[(df["GenWeights_muF1muR2"]<=0), "GenWeights_muF1muR2"] = 1/df["GenWeights_muF1muR0p5"]
  df.loc[(df["GenWeights_muF1muR0p5"]<=0), "GenWeights_muF1muR0p5"] = 1/df["GenWeights_muF1muR2"]

  # If still negative set to 1
  df.loc[(df["GenWeights_muF1muR2"]<=0), "GenWeights_muF1muR2"] = 1.0
  df.loc[(df["GenWeights_muF1muR0p5"]<=0), "GenWeights_muF1muR0p5"] = 1.0

  asymln = AsymLogNormal(df[nui_name], kp=df["GenWeights_muF1muR2"], km=df["GenWeights_muF1muR0p5"])
  df["weight"] *= asymln
  return df

def isr_uncertainty(
  df,
  metadata={}
):
  if metadata.get("group","").startswith("TT"):
    nui_name = "isr_ttbar"
  else:
    nui_name = "isr_st"

  # Make sure the weights are never 0, if goes negative them symmetrize
  df.loc[(df["GenWeights_isr2fsr1"]<=0), "GenWeights_isr2fsr1"] = 1/df["GenWeights_isr0p5fsr1"]
  df.loc[(df["GenWeights_isr0p5fsr1"]<=0), "GenWeights_isr0p5fsr1"] = 1/df["GenWeights_isr2fsr1"]

  # If still negative set to 1
  df.loc[(df["GenWeights_isr2fsr1"]<=0), "GenWeights_isr2fsr1"] = 1.0
  df.loc[(df["GenWeights_isr0p5fsr1"]<=0), "GenWeights_isr0p5fsr1"] = 1.0

  asymln = AsymLogNormal(df[nui_name], kp=df["GenWeights_isr2fsr1"], km=df["GenWeights_isr0p5fsr1"])
  df["weight"] *= asymln
  return df