from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_definition import Cut
from pocket_coffea.lib.cut_functions import get_nObj_min, get_HLTsel, get_nPVgood, goldenJson, eventFlags
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import *
import workflow
from workflow import ttBaseProcessor_merge
from pocket_coffea.lib.weights.common import common_weights
from pocket_coffea.lib.columns_manager import ColOut, ColumnsManager, column_accumulator

# Register custom modules in cloudpickle to propagate them to dask workers
import cloudpickle
import cut
cloudpickle.register_pickle_by_value(workflow)
cloudpickle.register_pickle_by_value(cut)

from cut import *
import os
localdir = os.path.dirname(os.path.abspath(__file__))

from Functions.WJetsRun2StitchingWeights import WJetsRun2Stitching
from Functions.WJetsRun3StitchingWeights import WJetsRun3Stitching


# Loading default parameters
from pocket_coffea.parameters import defaults
default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir+"/params")

parameters = defaults.merge_parameters_from_files(default_parameters,
                                                  f"{localdir}/../params/object_preselection.yaml",
                                                  f"{localdir}/../params/triggers.yaml",
                                                  f"{localdir}/../params/plotting.yaml",
                                                  update=True)


common_weights = common_weights + WJetsRun2Stitching + WJetsRun3Stitching

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
            "samples": [
                "DATA_SingleMuon",
                "DATA_SingleEle",
                "TTToSemiLeptonic",
                "TTTo2L2Nu",
                "TTToHadronic",
                "WJetsToLNu",
                "WJetsToLNuHT7OTo100",
                "WJetsToLNuHT100To200",
                "WJetsToLNuHT200To400",
                "WJetsToLNuHT400To600",
                "WJetsToLNuHT600To800",
                "WJetsToLNuHT800To1200",
                "WJetsToLNuHT1200To2500",
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
                "ST_t_channel_top",
                "ST_t_channel_antitop",
            ],
            "samples_exclude" : [],
            "year": ['2016_PreVFP', '2016_PostVFP', '2017', '2018', '2022_preEE', '2022_postEE', '2023_preBPix', '2023_postBPix']
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

    skim = [get_nPVgood(1), eventFlags, goldenJson,
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
                          "pileup",
                          "sf_mu_id","sf_mu_iso",
                          "WJetsRun2Stitching", "WJetsRun3Stitching"
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
                "inclusive": ["pileup",
                                "sf_mu_id", "sf_mu_iso",
                              ], 
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
                    ["pt", "eta", "phi", "mass", 'jetId', 'nConstituents', 'btagDDBvLV2', 'btagDDCvBV2', 'btagDDCvLV2', 'btagDeepB', 'msoftdrop', 'tau1', 'tau2', 'tau3', 'tau4', 'lsf3'],
                    flatten=False
                ),
                ColOut(
                    "SubJet1",
                    ['btagDeepB', 'eta', 'mass', 'n2b1', 'n3b1', 'phi', 'pt', 'rawFactor', 'tau1', 'tau2', 'tau3', 'tau4'],
                    flatten=False
                ),
                ColOut(
                    "SubJet2",
                    ['btagDeepB', 'eta', 'mass', 'n2b1', 'n3b1', 'phi', 'pt', 'rawFactor', 'tau1', 'tau2', 'tau3', 'tau4'],
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
                    "BJet_HighestPt",
                    ["pt", "eta", "phi", "mass"],
                    flatten=False
                ),
                ColOut(
                    "BJet_ClosestToLepton",
                    ["pt", "eta", "phi", "mass"],
                    flatten=False
                ),
                ColOut(
                    "LeptonSave",
                    ["pt", "eta", "phi", "mass", "leptonType"],
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
            ],
            "bycategory": {},
        },
        "bysample": {
        },
    },

   variables = {
    }
)

