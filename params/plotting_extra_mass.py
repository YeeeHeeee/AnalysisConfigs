use_bw_files = True
get_extra_masses = True

ttbar_files = [
  "TTToSemiLeptonic",
  "TTTo2L2Nu",
  "TTToHadronic",
  "TTMtt700To1000",
  "TTMtt1000",
]

ttbar_split_mass_files = []
for mass in ["166p5", "169p5", "171p5", "173p5", "175p5", "178p5"]:
  ttbar_split_mass_files += [
    f"TTToSemiLeptonic{mass}",
    f"TTTo2L2Nu{mass}",
    f"TTToHadronic{mass}",
  ]

ttbar_bw_files = []
for mass in ["166p5", "169p5", "171p5", "172p5", "173p5", "175p5", "178p5"]:
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

groups = {
  'Data': ['DATA_*.parquet'],
  'TT (172.5 GeV)': [f'{f}_*.parquet' for f in ttbar_files] if not use_bw_files else ['TT_172p5_*.parquet'],
  'ST': [f'{f}_*.parquet' for f in st_files],
  'WJ': [f'{f}_*.parquet' for f in wjets_files],
  #'Other': [f'{f}_*.parquet' for f in other_files],
}

if get_extra_masses:
  other_groups = {
    "TT (166.5 GeV)": [f'TTToSemiLeptonic166p5_*.parquet', f'TTToHadronic166p5_*.parquet', f'TTTo2L2Nu166p5_*.parquet'] if not use_bw_files else ['TT_166p5_*.parquet'],
    "TT (169.5 GeV)": [f'TTToSemiLeptonic169p5_*.parquet', f'TTToHadronic169p5_*.parquet', f'TTTo2L2Nu169p5_*.parquet'] if not use_bw_files else ['TT_169p5_*.parquet'],
    "TT (171.5 GeV)": [f'TTToSemiLeptonic171p5_*.parquet', f'TTToHadronic171p5_*.parquet', f'TTTo2L2Nu171p5_*.parquet'] if not use_bw_files else ['TT_171p5_*.parquet'],
    "TT (173.5 GeV)": [f'TTToSemiLeptonic173p5_*.parquet', f'TTToHadronic173p5_*.parquet', f'TTTo2L2Nu173p5_*.parquet'] if not use_bw_files else ['TT_173p5_*.parquet'],
    "TT (175.5 GeV)": [f'TTToSemiLeptonic175p5_*.parquet', f'TTToHadronic175p5_*.parquet', f'TTTo2L2Nu175p5_*.parquet'] if not use_bw_files else ['TT_175p5_*.parquet'],
    "TT (178.5 GeV)": [f'TTToSemiLeptonic178p5_*.parquet', f'TTToHadronic178p5_*.parquet', f'TTTo2L2Nu178p5_*.parquet'] if not use_bw_files else ['TT_178p5_*.parquet'],
  }
  if use_bw_files:
    other_groups["TT (170.5 GeV)"] = ['TT_170p5_*.parquet']
    other_groups["TT (174.5 GeV)"] = ['TT_174p5_*.parquet']

else:
  other_groups = {}

colours = {
  'TT (172.5 GeV)': 'blue',
  'WJ': 'red',
  'ST': 'brown',
  'VV': 'gray',
  'Other': 'cyan',
}

group_selection = {}

if get_extra_masses:
  plot_extra = {
    'Total (169.5 GeV)': ['TT (169.5 GeV)'],
    'Total (175.5 GeV)': ['TT (175.5 GeV)'],
  }
  plot_extra_subtract = {
    'Total (169.5 GeV)': ['TT (172.5 GeV)'],
    'Total (175.5 GeV)': ['TT (172.5 GeV)'],
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
  'FatJet_pt' : '(500,800,10)',
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
  'LeptonSave_pt' : '(50,400,10)',
  'MET_pt' : '(0,400,10)',
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
  'CombinedSubJets_mass' : '(50,320,5)',
  'CombinedSubJets_pt' : '(500,800,10)',
  'CombinedSubJets_eta' : '(-2.5,2.5,0.1)',
  'CombinedSubJets_phi' : '(-3.2,3.2,0.2)',
  'JetLepton_ptrel' : '(0,400,10)',
  'JetLepton_deltaR' : '(0,5,0.1)',
  'LeptonMET_mt' : '(0,400,10)',
  'BJetLep_pt' : '(0,400,10)',
  'BJetLep_eta' : '(-2.5,2.5,0.1)',
  'BJetLep_phi' : '(-3.2,3.2,0.1)',
  'BJetLep_mass' : '(0,200,5)',
  'BJetLep_btagDeepFlavB' : '(0,1,0.02)',
}

calculate = {
  'FatJet_tau32' : 'FatJet_tau3 / FatJet_tau2',
  'FatJet_tau21' : 'FatJet_tau2 / FatJet_tau1',
  'FatJet_tau31' : 'FatJet_tau3 / FatJet_tau1',
  'SubJet1_tau32' : 'SubJet1_tau3 / SubJet1_tau2',
  'SubJet1_tau21' : 'SubJet1_tau2 / SubJet1_tau1',
  'SubJet1_tau31' : 'SubJet1_tau3 / SubJet1_tau1',
  'SubJet2_tau32' : 'SubJet2_tau3 / SubJet2_tau2',
  'SubJet2_tau21' : 'SubJet2_tau2 / SubJet2_tau1',
  'SubJet2_tau31' : 'SubJet2_tau3 / SubJet2_tau1',
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
  'BJetLep_pt' : '$p_{T}^{lep b jet}$ (GeV)',
  'BJetLep_eta' : '$\\eta^{lep b jet}$',
  'BJetLep_phi' : '$\\phi^{lep b jet}$',
  'BJetLep_mass' : '$m_{rec}^{b jet}$ (GeV)',
  'BJetLep_btagDeepFlavB' : 'b tagging score (lep b jet)',
  'JetLepton_ptrel' : '$p_{T}^{rel}$(lep, Next AK4 jet) (GeV)',
  'JetLepton_deltaR' : '$\\Delta R$(lep, Next AK4 jet)',
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

# Simple
jec_uncert = {
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
}

systematics = {}

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
    systematics[syst_names[ind]] = {
      "functions": {
        "FatJet_pt": f"FatJet_pt * (1 + ({scalings}*{syst_names[ind]}*FatJet_corrFactor_{name}/FatJet_corrFactor))",
        "FatJet_mass": f"FatJet_mass * (1 + ({scalings}*{syst_names[ind]}*FatJet_corrFactor_{name}/FatJet_corrFactor))",
        "FatJet_msoftdrop": f"FatJet_msoftdrop * (1 + ({scalings}*{syst_names[ind]}*FatJet_corrFactor_{name}/FatJet_corrFactor))",
        "SubJet1_pt": f"SubJet1_pt * (1 + ({scalings}*{syst_names[ind]}*SubJet1_corrFactor_{name}/SubJet1_corrFactor))",
        "SubJet1_mass": f"SubJet1_mass * (1 + ({scalings}*{syst_names[ind]}*SubJet1_corrFactor_{name}/SubJet1_corrFactor))",
        "SubJet2_pt": f"SubJet2_pt * (1 + ({scalings}*{syst_names[ind]}*SubJet2_corrFactor_{name}/SubJet2_corrFactor))",
        "SubJet2_mass": f"SubJet2_mass * (1 + ({scalings}*{syst_names[ind]}*SubJet2_corrFactor_{name}/SubJet2_corrFactor))",
        "SubJet1_px": "SubJet1_pt * cos(SubJet1_phi)",
        "SubJet1_py": "SubJet1_pt * sin(SubJet1_phi)",
        "SubJet1_pz": "SubJet1_pt * sinh(SubJet1_eta)",
        "SubJet1_E": "sqrt(SubJet1_px**2 + SubJet1_py**2 + SubJet1_pz**2 + SubJet1_mass**2)",
        "SubJet2_px": "SubJet2_pt * cos(SubJet2_phi)",
        "SubJet2_py": "SubJet2_pt * sin(SubJet2_phi)",
        "SubJet2_pz": "SubJet2_pt * sinh(SubJet2_eta)",
        "SubJet2_E": "sqrt(SubJet2_px**2 + SubJet2_py**2 + SubJet2_pz**2 + SubJet2_mass**2)",
        "CombinedSubJets_px": "(SubJet1_px + SubJet2_px)",
        "CombinedSubJets_py": "(SubJet1_py + SubJet2_py)",
        "CombinedSubJets_pz": "(SubJet1_pz + SubJet2_pz)",
        "CombinedSubJets_E": "(SubJet1_E + SubJet2_E)",
        "CombinedSubJets_eta": "arcsinh(CombinedSubJets_pz / sqrt(CombinedSubJets_px**2 + CombinedSubJets_py**2))",
        "CombinedSubJets_phi": "arctan2(CombinedSubJets_py, CombinedSubJets_px)",
        "CombinedSubJets_pt": "( (CombinedSubJets_px)**2 + (CombinedSubJets_py)**2 ) ** 0.5",
        "CombinedSubJets_mass": "( (CombinedSubJets_E)**2 - (CombinedSubJets_px)**2 - (CombinedSubJets_py)**2 - (CombinedSubJets_pz)**2 ) ** 0.5",
      },
      "files": all_files,
      "years": corr_years[ind],
    }

  write_translate = {
    "Data": "data_obs",
  }

  #"isr_ttbar":
  #  "functions": 
  #    "weight": "weight *(((isr_ttbar>=0)*(isr_ttbar*GenWeights_isr2fsr1)) + ((isr_ttbar<0)*(isr_ttbar*GenWeights_isr2fsr1)))"
  #  "files": *ttbar_files
  #  "years": *all_years

  #"isr_st":
  #  "functions": 
  #    "weight": "weight*(((isr_st>=0)*(isr_st*GenWeights_isr2fsr1)) + ((isr_st<0)*(isr_st*GenWeights_isr2fsr1)))"
  #  "files": *st_files
  #  "years": *all_years

  #"fsr":
  #  "functions": 
  #    "weight": "weight*(((fsr>=0)*(fsr*GenWeights_isr1fsr2)) + ((fsr<0)*(fsr*GenWeights_isr1fsr0p5)))"
  #  "files": *top_files
  #  "years": *all_years

  #"factorisation_scale":
  #  "functions": 
  #    "weight": "weight*(((factorisation_scale>=0)*(factorisation_scale*GenWeights_muF2muR1)) + ((factorisation_scale<0)*(factorisation_scale*GenWeights_muF0p5muR1)))"
  #  "files": *top_files
  #  "years": *all_years

  #"renormalisation_scale":
  #  "functions": 
  #    "weight": "weight*(((renormalisation_scale>=0)*(renormalisation_scale*GenWeights_muF1muR2)) + ((renormalisation_scale<0)*(renormalisation_scale*GenWeights_muF1muR0p5)))"
  #  "files": *top_files
  #  "years": *all_years


config = {
  "groups": groups,
  "other_groups": other_groups,
  "colours": colours,
  "group_selection": group_selection,
  "plot_extra": plot_extra,
  "plot_extra_subtract": plot_extra_subtract,
  "plot_extra_colours": plot_extra_colours,
  "variables": variables,
  "calculate": calculate,
  "translate": translate,
  "systematics": systematics,
  "write_translate": write_translate,
}