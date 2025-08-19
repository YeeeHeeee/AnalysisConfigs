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
from Functions.LeptonScaleFactors import SF_ele_custom
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
                                                  update=True)


common_weights = common_weights + WJetsRun2Stitching + WJetsRun3Stitching + TopPTReweighting + SF_ele_custom + Prefiring
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
wjet_samples = [
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
            "samples": ttbar_samples + ttbar_mass_samples + wjet_samples + st_samples + qcd_samples + other_samples + data_samples,
            #"samples" : data_samples,
            #"samples": ttbar_samples + ttbar_mass_samples + wjet_samples + st_samples + qcd_samples + other_samples,
            "samples_exclude" : [],
            "year": ['2016_PreVFP', '2016_PostVFP', '2017', '2018', '2022_preEE', '2022_postEE', '2023_preBPix', '2023_postBPix']
            #"year" : ['2022_preEE']
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
<<<<<<< HEAD
            get_nObj_min(1, 400.0, "FatJet"), # Initial skim, have not set this to 500 as we have corrections to apply afterwards
=======
            get_nObj_min(1, 480.0, "FatJet"), # Initial skim, have not set this to 500 as we have corrections to apply afterwards
>>>>>>> 6915bdf (Adding new files tp config and tidied up workflow)
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
                          "sf_mu_id", "sf_mu_iso", 
                          #"sf_mu_trigger",
                          "sf_ele_id_custom", "sf_ele_reco", 
                          #"sf_ele_trigger_custom",
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
                    ["pt", "phi"],
                    flatten=False
                ),
                ColOut(
                    "GenTop_AK8",
                    ["pt", "eta", "phi", "mass"],
                    flatten=False
                ),
                ColOut(
                    "FatJet",
                    ["pt", "eta", "phi", "mass", 'jetId', 'nConstituents', 'btagDeepB', 'msoftdrop', 'tau1', 'tau2', 'tau3', 'tau4', 'msoftdrop_raw'] + jec_store,
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
                    ["pt", "eta", "phi", "mass"],
                    flatten=False
                ),
<<<<<<< HEAD
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
                    "ExtraWeights",
                    ["BTagShapeCorrectionSubjets","WJetsRun2Stitching","WJetsRun3Stitching","TTTo2L2NuRun2Stitching","TTToSemiLeptonicRun2Stitching","TTToHadronicRun2Stitching","TopPTReweighting"],
                    flatten=False
                ),
                ColOut(
                    "GenWeights",
                    ["isr2fsr1", "isr1fsr2", "isr0p5fsr1", "isr1fsr0p5", "muF0p5muR0p5", "muF1muR0p5", "muF2muR0p5", "muF0p5muR1", "muF2muR1", "muF0p5muR2", "muF1muR2", "muF2muR2"],
                    flatten=False
                ),
=======
>>>>>>> 6915bdf (Adding new files tp config and tidied up workflow)
            ],
            "bycategory": {},
        },
        "bysample": {
        },
    },

   variables = {
    }
)

