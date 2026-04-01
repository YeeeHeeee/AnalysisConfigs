from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_definition import Cut
from pocket_coffea.lib.cut_functions import get_nObj_min, get_HLTsel, get_nPVgood, goldenJson, eventFlags
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import *
import workflow
from workflow import ttBaseProcessor_merge
from pocket_coffea.lib.weights.common import common_weights
from pocket_coffea.lib.columns_manager import ColOut, ColumnsManager, column_accumulator

# Functions to be used in the workflow
from Functions.WJetsRun2StitchingWeights import WJetsRun2Stitching
from Functions.WJetsRun3StitchingWeights import WJetsRun3Stitching
from Functions.TTTo2L2NuRun2StitchingWeights import TTTo2L2NuRun2Stitching
from Functions.TTToSemiLeptonicRun2StitchingWeights import TTToSemiLeptonicRun2Stitching
from Functions.TTToHadronicRun2StitchingWeights import TTToHadronicRun2Stitching
from Functions.TopPTReweighting import TopPTReweighting
from Functions.LeptonScaleFactors import SF_ele_custom, SF_mu_custom
from Functions.Prefiring import Prefiring
from Functions.BtaggingWeightScaleFactors import BTagWeightCorrection
from Functions.BtaggingShapeScaleFactors import BTagShapeCorrection
from Functions.jec_config import nom_jec_variations

# Register custom modules in cloudpickle to propagate them to dask workers
import cloudpickle
import cut
import Functions.WJetsRun2StitchingWeights
import Functions.WJetsRun3StitchingWeights
import Functions.TTTo2L2NuRun2StitchingWeights
import Functions.TTToSemiLeptonicRun2StitchingWeights
import Functions.TTToHadronicRun2StitchingWeights
import Functions.TopPTReweighting
import Functions.LeptonScaleFactors
import Functions.Prefiring
import Functions.BtaggingWeightScaleFactors
import Functions.BtaggingShapeScaleFactors
import Functions.JetsCom
import Functions.Leptons
import Functions.jec_config
import Functions.corrections
import Functions.met_xy_correction
import Functions.jet_veto_maps
cloudpickle.register_pickle_by_value(workflow)
cloudpickle.register_pickle_by_value(cut)
cloudpickle.register_pickle_by_value(Functions.WJetsRun2StitchingWeights)
cloudpickle.register_pickle_by_value(Functions.WJetsRun3StitchingWeights)
cloudpickle.register_pickle_by_value(Functions.TTTo2L2NuRun2StitchingWeights)
cloudpickle.register_pickle_by_value(Functions.TTToSemiLeptonicRun2StitchingWeights)
cloudpickle.register_pickle_by_value(Functions.TTToHadronicRun2StitchingWeights)
cloudpickle.register_pickle_by_value(Functions.TopPTReweighting)
cloudpickle.register_pickle_by_value(Functions.LeptonScaleFactors)
cloudpickle.register_pickle_by_value(Functions.Prefiring)
cloudpickle.register_pickle_by_value(Functions.BtaggingWeightScaleFactors)
cloudpickle.register_pickle_by_value(Functions.BtaggingShapeScaleFactors)
cloudpickle.register_pickle_by_value(Functions.JetsCom)
cloudpickle.register_pickle_by_value(Functions.Leptons)
cloudpickle.register_pickle_by_value(Functions.jec_config)
cloudpickle.register_pickle_by_value(Functions.corrections)
cloudpickle.register_pickle_by_value(Functions.met_xy_correction)
cloudpickle.register_pickle_by_value(Functions.jet_veto_maps)

from cut import *
import os
localdir = os.path.dirname(os.path.abspath(__file__))

# Loading default parameters
from pocket_coffea.parameters import defaults
default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir+"/params")

parameters = defaults.merge_parameters_from_files(default_parameters,
                                                  f"{localdir}/../params/object_preselection.yaml",
                                                  f"{localdir}/../params/triggers.yaml",
                                                  f"{localdir}/../params/btag_corrections.yaml",
                                                  f"{localdir}/../params/lumi.yaml",
                                                  f"{localdir}/../params/pc_jet_calibration.yaml",
                                                  f"{localdir}/../params/btagging.yaml",
                                                  f"{localdir}/../params/met_corrections.yaml",
                                                  f"{localdir}/../params/jet_veto_maps.yaml",
                                                  f"{localdir}/../params/event_flags.yaml",
                                                  f"{localdir}/../params/lepton_scale_factors.yaml",
                                                  f"{localdir}/../params/pileup.yaml",
                                                  f"{localdir}/../params/electron_trigger_run2.yaml",
                                                  update=True)


common_weights = common_weights + WJetsRun2Stitching + WJetsRun3Stitching + TopPTReweighting + SF_ele_custom + SF_mu_custom + Prefiring
common_weights = common_weights + TTTo2L2NuRun2Stitching + TTToSemiLeptonicRun2Stitching + TTToHadronicRun2Stitching
common_weights = common_weights + BTagWeightCorrection + BTagShapeCorrection


data_samples = [
    "DATA_SingleMuon",
    "DATA_SingleEle"
]
ttbar_samples = [
    "TTToSemiLeptonic",
    "TTTo2L2Nu",
    "TTToHadronic",
    "TTMtt700To1000",
    "TTMtt1000",
]
ttbar_mass_samples = [
    "TTToSemiLeptonic166p5",
    "TTToSemiLeptonic169p5",
    "TTToSemiLeptonic171p5",
    "TTToSemiLeptonic173p5",
    "TTToSemiLeptonic175p5",
    "TTToSemiLeptonic178p5",
    "TTTo2L2Nu166p5",
    "TTTo2L2Nu169p5",
    "TTTo2L2Nu171p5",
    "TTTo2L2Nu173p5",
    "TTTo2L2Nu175p5",
    "TTTo2L2Nu178p5",
    "TTToHadronic166p5",
    "TTToHadronic169p5",
    "TTToHadronic171p5",
    "TTToHadronic173p5",
    "TTToHadronic175p5",
    "TTToHadronic178p5",
]
ttbar_modelling_samples = [
    "TTToSemiLeptonic_CR1",
    "TTToSemiLeptonic_CR2",
    "TTToSemiLeptonic_hdamp_Up",
    "TTToSemiLeptonic_hdamp_Down",
    "TTToSemiLeptonic_ue_Up",
    "TTToSemiLeptonic_ue_Down",
    "TTToSemiLeptonic_ERDOn"
]
wjet_samples = [
    #"WJetsToLNu",
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
    #"WJetsToLNuMLNu120To200", # 2024 samples not used
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
st_samples = [
    "ST_t_channel_top",
    "ST_t_channel_antitop",
    "ST_s_channel",
    "ST_s_channel_top",
    "ST_s_channel_antitop",
    "ST_tW_antitop",
    "ST_tW_top",
]
qcd_samples = [
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
    "QCD_EMEnriched_Pt15To20",
    "QCD_EMEnriched_Pt20To30",
    "QCD_EMEnriched_Pt30To50",
    "QCD_EMEnriched_Pt50To80",
    "QCD_EMEnriched_Pt80To120",
    "QCD_EMEnriched_Pt120To170",
    "QCD_EMEnriched_Pt170To300",
    "QCD_EMEnriched_Pt300",
    "QCD_EMEnriched_Pt30To80",
    "QCD_EMEnriched_Pt80To170",
    "QCD_EMEnriched_Pt170To250",
    "QCD_EMEnriched_Pt250",
]
other_samples = [
    "WW",
    "WZ",
    "ZZ",
    "DY",
]

jec_store = [
    "pt_raw",
    "mass_raw",
    "corrFactor",
    "smearFactor",
    "smearFactor_up",
    "smearFactor_down",
] + [f"corrFactor_{i}" for i in nom_jec_variations]

cfg = Configurator(
    parameters = parameters,
    datasets = {
        "jsons": [
            f"{localdir}/../Datasets/signals_MC_ttbar.json",
            f"{localdir}/../Datasets/backgrounds_MC_ttbar.json",
            f"{localdir}/../Datasets/DATA_SingleMuon.json",
            f"{localdir}/../Datasets/DATA_SingleEle.json",
        ],
        "filter" : {
            "samples": ttbar_samples + ttbar_mass_samples + ttbar_modelling_samples + wjet_samples + st_samples + qcd_samples + other_samples + data_samples,
            #"samples" : ttbar_samples + data_samples,
            "samples_exclude" : [],
            "year": ['2016_PreVFP', '2016_PostVFP', '2017', '2018', '2022_preEE', '2022_postEE', '2023_preBPix', '2023_postBPix', '2024']
            #"year": ["2016_PreVFP"]
        },
        "subsamples": {
            'DATA_SingleEle'  : {
                'DATA_SingleEle' : [get_HLTsel(primaryDatasets=["SingleEle"])]
            },
            'DATA_SingleMuon' : {
                'DATA_SingleMuon' : [get_HLTsel(primaryDatasets=["SingleMuon"]), get_HLTsel(primaryDatasets=["SingleEle"], invert=True)]
            },
        }
    },

    workflow = ttBaseProcessor_merge,

    skim = [
            get_nObj_min(1, 15.0, "Jet"),
            get_nObj_min(1, 300.0, "FatJet"), # Initial skim, have not set this to 350 as we have corrections to apply afterwards
            get_nPVgood(1), eventFlags, goldenJson,
            get_HLTsel(primaryDatasets=["SingleMuon", "SingleEle"]),
            ],
        
    preselections = [semileptonic_presel_merge],
    categories = {
        "baseline": [passthrough],
    },


    weights_classes = common_weights,
    
    weights = {
        "common": {
            "inclusive": ["genWeight","lumi","XS",
                          "pileup", "prefiring",
                          "sf_mu_id_custom", "sf_mu_iso_custom","sf_mu_trigger_custom",
                          "sf_ele_id_custom", "sf_ele_reco_custom","sf_ele_trigger_custom",
                          "WJetsRun2Stitching", "WJetsRun3Stitching",
                          "TTTo2L2NuRun2Stitching", "TTToSemiLeptonicRun2Stitching", "TTToHadronicRun2Stitching",
                          "TopPTReweighting",
                          "BTagWeightCorrection",
                          ],
            "bycategory" : {
            }
        },
        "bysample": {
        }
    },

    variations = {
        "weights": {
            "common": {
                "inclusive": [], 
                "bycategory" : {
                }
            },
        "bysample": {
        }    
        },
    },   
    columns = {
        "common": {
            "inclusive": [
                ColOut(
                    "MET",
                    ["pt", "phi"] + [f"corrFactor_{i}" for i in nom_jec_variations] + ["smearFactor_up", "smearFactor_down"],
                    flatten=False
                ),
                ColOut(
                    "GenTop_AK8",
                    ["pt", "eta", "phi", "mass"],
                    flatten=False
                ),
                ColOut(
                    "FatJet",
                    ["pt", "eta", "phi", "mass", 'n2b1', 'n3b1', 'jetId', 'nConstituents', 'btagDeepB', 'msoftdrop', 'tau1', 'tau2', 'tau3', 'tau4', 'msoftdrop_raw'] + jec_store,
                    flatten=False
                ),
                ColOut(
                    "SubJet1",
                    ['btagDeepB', 'eta', 'mass', 'n2b1', 'n3b1', 'phi', 'pt', 'rawFactor', 'tau1', 'tau2', 'tau3', 'tau4'] + jec_store,
                    flatten=False
                ),
                ColOut(
                    "SubJet2",
                    ['btagDeepB', 'eta', 'mass', 'n2b1', 'n3b1', 'phi', 'pt', 'rawFactor', 'tau1', 'tau2', 'tau3', 'tau4'] + jec_store,
                    flatten=False
                ),
                ColOut(
                    "CombinedSubJets",
                    ["pt", "eta", "phi", "mass"],
                    flatten=False
                ),
                ColOut(
                    "GenTop1",
                    ["pt", "eta", "phi", "mass"],
                    flatten=False
                ),
                ColOut(
                    "GenTop2",
                    ["pt", "eta", "phi", "mass"],
                    flatten=False
                ),
                ColOut(
                    "MatchedTop_AK8",
                    ["pt", "eta", "phi", "mass"],
                    flatten=False
                ),
                ColOut(
                    "BJetLep",
                    ["pt", "eta", "phi", "mass", "btagDeepFlavB"] + jec_store,
                    flatten=False
                ),
                ColOut(
                    "LeptonSave",
                    ["pt", "eta", "phi", "mass", "leptonType", "RelIso"],
                    flatten=False
                ),
                ColOut(
                    "LNu",
                    ["mass"],
                    flatten=False
                ),
                ColOut(
                    "LHE",
                    ["HT"],
                    flatten=False
                ),
                ColOut(
                    "JetLepton",
                    ["pt", "eta", "phi", "mass", "deltaR", "deltaPhi", "deltaEta","ptrel"],
                    flatten=False
                ),
                ColOut(
                    "ClosestJetToLepton",
                    ["pt", "eta", "phi", "mass"] + jec_store,
                    flatten=False
                ),
                ColOut(
                    "LeptonMET",
                    ["mt"],
                    flatten=False
                ),
                ColOut(
                    "GenTT",
                    ["count_l", "mass"],
                    flatten=False
                ),
                ColOut(
                    "MatchedGenJet_SubJet1",
                    ["partonFlavour"],
                    flatten=False
                ),
                ColOut(
                    "MatchedGenJet_SubJet2",
                    ["partonFlavour"],
                    flatten=False
                ),
                ColOut(
                    "MatchedGenJet_BJetLep",
                    ["partonFlavour"],
                    flatten=False
                ),
                ColOut(
                    "Extra",
                    ["BTagShapeCorrectionSubjets","BTagShapeCorrectionSubjets_down_hf","BTagShapeCorrectionSubjets_up_hf","BTagShapeCorrectionSubjets_down_lf","BTagShapeCorrectionSubjets_up_lf","BTagShapeCorrectionSubjets_down_hfstats1","BTagShapeCorrectionSubjets_up_hfstats1","BTagShapeCorrectionSubjets_down_hfstats2","BTagShapeCorrectionSubjets_up_hfstats2","BTagShapeCorrectionSubjets_down_lfstats1","BTagShapeCorrectionSubjets_up_lfstats1","BTagShapeCorrectionSubjets_down_lfstats2","BTagShapeCorrectionSubjets_up_lfstats2","BTagShapeCorrectionSubjets_down_cferr1","BTagShapeCorrectionSubjets_up_cferr1","BTagShapeCorrectionSubjets_down_cferr2","BTagShapeCorrectionSubjets_up_cferr2","BTagWeightCorrection_up","BTagWeightCorrection_down","BTagWeightCorrection_up_correlated","BTagWeightCorrection_down_correlated","WJetsRun2Stitching","WJetsRun3Stitching","TTTo2L2NuRun2Stitching","TTToSemiLeptonicRun2Stitching","TTToHadronicRun2Stitching","TopPTReweighting"] + [f"{wt}{var}" for wt in ["sf_ele_id_custom", "sf_ele_reco_custom", "sf_ele_trigger_custom", "sf_mu_id_custom", "sf_mu_iso_custom", "sf_mu_trigger_custom", "prefiring", "pileup"] for var in ["", "_up", "_down"]],
                    flatten=False
                ),
                ColOut(
                    "GenWeights",
                    ["isr2fsr1", "isr1fsr2", "isr0p5fsr1", "isr1fsr0p5", "muF0p5muR0p5", "muF1muR0p5", "muF2muR0p5", "muF0p5muR1", "muF1muR1", "muF2muR1", "muF0p5muR2", "muF1muR2", "muF2muR2", "pdf_max", "pdf_min", "pdf_rmse"],
                    flatten=False
                ),
                ColOut(
                    "GenTopHadronic",
                    ["pt", "eta", "phi", "mass"],
                    flatten=False
                ),
                ColOut(
                    "GenTopLeptonic",
                    ["pt", "eta", "phi", "mass"],
                    flatten=False
                ),   
                ColOut(
                    "GenWHadronic",
                    ["pt", "eta", "phi", "mass"],
                    flatten=False
                ),               
                ColOut(
                    "GenWLeptonic",
                    ["pt", "eta", "phi", "mass"],
                    flatten=False
                ),         
                ColOut(
                    "GenBQuarkHadronic",
                    ["pt", "eta", "phi", "mass"],
                    flatten=False
                ),    
                ColOut(
                    "GenBQuarkLeptonic",
                    ["pt", "eta", "phi", "mass"],
                    flatten=False
                ),    
                ColOut(
                    "GenLepton",
                    ["pt", "eta", "phi", "mass"],
                    flatten=False
                ),  
                ColOut(
                    "GenLightQuark1",
                    ["pt", "eta", "phi", "mass"],
                    flatten=False
                ),  
                ColOut(
                    "GenLightQuark2",
                    ["pt", "eta", "phi", "mass"],
                    flatten=False
                ),  
                ColOut(
                    "deltaR_Jet_Gen",
                    ["FatJet_GenTopHadronic","FatJet_GenWHadronic","FatJet_GenBQuarkHadronic","FatJet_GenLightQuark1","FatJet_GenLightQuark2","SubJet1_GenLightQuark1","SubJet1_GenLightQuark2","SubJet1_GenBQuarkHadronic","SubJet2_GenLightQuark1","SubJet2_GenLightQuark2","SubJet2_GenBQuarkHadronic"],
                    flatten=False
                ),
                ColOut(
                    "MergingInfo",
                    ["FatJet_TopDecaysMerged","FatJet_WDecaysMerged","SubJet1_WDecaysMerged","SubJet2_WDecaysMerged","SubJet1_BMerged","SubJet2_BMerged"],
                    flatten=False
                )
            ],
            "bycategory": {},
        },
        "bysample": {
        },
    },

   variables = {
    }
)

