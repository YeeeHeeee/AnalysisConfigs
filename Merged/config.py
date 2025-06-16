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

# Loading default parameters
from pocket_coffea.parameters import defaults
default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir+"/params")

parameters = defaults.merge_parameters_from_files(default_parameters,
                                                  f"{localdir}/../params/object_preselection.yaml",
                                                  f"{localdir}/../params/triggers.yaml",
                                                  f"{localdir}/../params/plotting.yaml",
                                                  update=True)



cfg = Configurator(
    parameters = parameters,
    datasets = {
        "jsons": [f"{localdir}/../Datasets/signals_MC_ttbar.json",
                  f"{localdir}/../Datasets/backgrounds_MC_ttbar.json"],
        "filter" : {
            "samples": ["TTToSemiLeptonic",
                        "TTTo2L2Nu",
                        "TTToHadronic",
                        "WJetsToLNu"],

            "samples_exclude" : [],
            "year": ['2018','2016_PreVFP', '2016_PostVFP', '2017',
                '2022_preEE', '2022_postEE', '2023_preBPix', '2023_postBPix'], 
        }
    },

    workflow = ttBaseProcessor_merge,

    skim = [get_nPVgood(4), goldenJson], 
    
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
                                "sf_mu_id", "sf_mu_iso"
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
                # ColOut(
                #     "MET",
                #     ["pt", "phi", 
                #     'fiducialGenPhi', 'fiducialGenPt'],
                #     flatten=False
                # ),
                # Save the Gen-level data:
                ColOut(
                    "GenTop_AK8",
                    ["pt", "eta", "phi", "mass"],
                    flatten=False
                ),
                # Save the Reco data:
                ColOut(
                    "FatJet",
                    ["pt", "eta", "phi", "mass", 'jetId', 'nConstituents', 'subJetIdx1', 'subJetIdx2', 'btagDDBvLV2', 'btagDDCvBV2', 'btagDDCvLV2', 'btagDeepB', 'btagHbb', 'msoftdrop', 'tau1', 'tau2', 'tau3', 'tau4', 'lsf3', 'hadronFlavour', 'nBHadrons', 'nCHadrons', 'genJetAK8Idx', 'genJetAK8IdxG', 'subJetIdx1G', 'subJetIdx2G', 'subJetIdxG'],
                    flatten=False
                ),
                ColOut(
                    "SubJet1",
                    ['btagDeepB', 'eta', 'mass', 'n2b1', 'n3b1', 'phi', 'pt', 'rawFactor', 'tau1', 'tau2', 'tau3', 'tau4', 'hadronFlavour', 'nBHadrons', 'nCHadrons'],
                    flatten=False
                ),
                ColOut(
                    "SubJet2",
                    ['btagDeepB', 'eta', 'mass', 'n2b1', 'n3b1', 'phi', 'pt', 'rawFactor', 'tau1', 'tau2', 'tau3', 'tau4', 'hadronFlavour', 'nBHadrons', 'nCHadrons'],
                    flatten=False
                ),
                ColOut(
                    "CombinedSubJets",
                    ["pt", "eta", "phi", "mass"],
                    flatten=False
                ),
                # Save the matched data
                ColOut(
                    "MatchedTop_AK8",
                    ["pt", "eta", "phi", "mass"],
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

