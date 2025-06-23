from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_definition import Cut
from pocket_coffea.lib.cut_functions import get_nObj_min, get_HLTsel, get_nPVgood, goldenJson, eventFlags
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import *
import workflow
from workflow import ttBaseProcessor_GenInfo
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
        "jsons": [f"{localdir}/../Datasets/signals_MC_ttbar.json"],
        "filter" : {
            "samples": ["TTToSemiLeptonic"],
            "year": ['2018'] 
        }
    },

    workflow = ttBaseProcessor_GenInfo,

    skim = [get_nPVgood(4), goldenJson], 
    
    preselections = [gen_info_presel],
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
                "inclusive": [  "pileup",
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
                ColOut(
                    "LeptonGood",
                    [],
                    flatten=False,
                ),
                ColOut(
                    "LeptonPadded",
                    ["pt", "eta", "phi", "mass"],
                    flatten=False
                ), 
                ColOut(
                    "JetPadded",
                    ['pt', 'eta', 'phi', 'mass', 'btag'],
                    flatten=False
                ),
                ColOut(
                    "FatJetPadded",
                    ['pt', 'eta', 'phi', 'mass'],
                    flatten=False
                ),            
                ColOut(
                    "SubJet1Padded",
                    ['pt', 'eta', 'phi', 'mass'],
                    flatten=False
                ),  
                ColOut(
                    "SubJet2Padded",
                    ['pt', 'eta', 'phi', 'mass'],
                    flatten=False
                ),  
                ColOut(
                    "GenTop",
                    ['pt', 'eta', 'phi', 'mass', 'pdgId'],
                    flatten=False
                ),
                ColOut(
                    "GenTopHadronic",
                    ['pt', 'eta', 'phi', 'mass', 'pdgId'],
                    flatten=False
                ),
                ColOut(
                    "GenTopLeptonic",
                    ['pt', 'eta', 'phi', 'mass', 'pdgId'],
                    flatten=False
                ),
                ColOut(
                    "GenWHadronic",
                    ['pt', 'eta', 'phi', 'mass'],
                    flatten=False
                ),
                ColOut(
                    "GenWLeptonic",
                    ['pt', 'eta', 'phi', 'mass', 'pdgId'],
                    flatten=False
                ),
                ColOut(
                    "GenBQuarkHadronic",
                    ['pt', 'eta', 'phi', 'mass', 'pdgId'],  
                    flatten=False
                ),
                ColOut(
                    "GenBQuarkLeptonic",
                    ['pt', 'eta', 'phi', 'mass', 'pdgId'],
                    flatten=False
                ),
                ColOut(
                    "GenLepton",
                    ['pt', 'eta', 'phi', 'mass', 'pdgId'],
                    flatten=False
                ),
                ColOut(
                    "GenLightQuark",
                    ['pt', 'eta', 'phi', 'mass', 'pdgId'],
                    flatten=False
                ),
                ColOut(
                    "GenLeptonMatched_LeptonPadded",
                    ["matched"],
                    flatten=False
                ),
                ColOut(
                    "GenLightQuark1Matched_JetPadded",
                    ["matched"],
                    flatten=False
                ),
                ColOut(
                    "GenLightQuark2Matched_JetPadded",
                    ["matched"],
                    flatten=False
                ),
                ColOut(
                    "GenLightQuark1Matched_FatJetPadded",
                    ["matched"],
                    flatten=False
                ),
                ColOut(
                    "GenLightQuark2Matched_FatJetPadded",
                    ["matched"],
                    flatten=False
                ), 
                ColOut(
                    "GenLightQuark1Matched_SubJet1Padded",
                    ["matched"],
                    flatten=False
                ),
                ColOut(
                    "GenLightQuark2Matched_SubJet1Padded",
                    ["matched"],
                    flatten=False
                ),   
                ColOut(
                    "GenLightQuark1Matched_SubJet2Padded",
                    ["matched"],
                    flatten=False
                ),
                ColOut(
                    "GenLightQuark2Matched_SubJet2Padded",
                    ["matched"],
                    flatten=False
                ),  
                ColOut(
                    "GenBQuarkHadronicMatched_JetPadded",
                    ["matched"],
                    flatten=False
                ),            
                ColOut(
                    "GenBQuarkLeptonicMatched_JetPadded",
                    ["matched"],
                    flatten=False
                ),
                ColOut(
                    "GenBQuarkHadronicMatched_FatJetPadded",
                    ["matched"],
                    flatten=False
                ),
                ColOut(
                    "GenBQuarkLeptonicMatched_FatJetPadded",
                    ["matched"],
                    flatten=False
                ),
                ColOut(
                    "GenBQuarkHadronicMatched_SubJet1Padded",
                    ["matched"],
                    flatten=False
                ),
                ColOut(
                    "GenBQuarkLeptonicMatched_SubJet1Padded",
                    ["matched"],
                    flatten=False
                ),
                ColOut(
                    "GenBQuarkHadronicMatched_SubJet2Padded",
                    ["matched"],
                    flatten=False
                ),
                ColOut(
                    "GenBQuarkLeptonicMatched_SubJet2Padded",
                    ["matched"],
                    flatten=False
                ),
                ColOut(
                    "HighestPt_q1",
                    ['pt', 'eta', 'phi', 'mass'],
                    flatten=False
                ),
                ColOut(
                    "HighestPt_q2",
                    ['pt', 'eta', 'phi', 'mass'],
                    flatten=False
                ),
                ColOut(
                    "HighestPt_b",
                    ['pt', 'eta', 'phi', 'mass'],
                    flatten=False
                ),
                ColOut(
                    "HighestPt_W",
                    ['pt', 'eta', 'phi', 'mass'],
                    flatten=False
                ),
                ColOut(
                    "HighestPt_top",
                    ['pt', 'eta', 'phi', 'mass'],
                    flatten=False
                ),
                ColOut(
                    "HighestPt_gen_matched",
                    ['q1', 'q2', 'b'],
                    flatten=False
                ),
                ColOut(
                    "ClosestDR_q1",
                    ['pt', 'eta', 'phi', 'mass'],
                    flatten=False
                ),
                ColOut(
                    "ClosestDR_q2",
                    ['pt', 'eta', 'phi', 'mass'],
                    flatten=False
                ),
                ColOut(
                    "ClosestDR_b",
                    ['pt', 'eta', 'phi', 'mass'],
                    flatten=False
                ),
                ColOut(
                    "ClosestDR_W",
                    ['pt', 'eta', 'phi', 'mass'],
                    flatten=False
                ),
                ColOut(
                    "ClosestDR_top",
                    ['pt', 'eta', 'phi', 'mass'],
                    flatten=False
                ),
                ColOut(
                    "ClosestDR_gen_matched",
                    ['q1', 'q2', 'b'],
                    flatten=False
                ),
                ColOut(
                    "ClosestMass_q1",
                    ['pt', 'eta', 'phi', 'mass'],
                    flatten=False
                ),
                ColOut(
                    "ClosestMass_q2",
                    ['pt', 'eta', 'phi', 'mass'],
                    flatten=False
                ),
                ColOut(
                    "ClosestMass_b",
                    ['pt', 'eta', 'phi', 'mass'],
                    flatten=False
                ),
                ColOut(
                    "ClosestMass_W",
                    ['pt', 'eta', 'phi', 'mass'],
                    flatten=False
                ),
                ColOut(
                    "ClosestMass_top",
                    ['pt', 'eta', 'phi', 'mass'],
                    flatten=False
                ),
                ColOut(
                    "ClosestMass_gen_matched",
                    ['q1', 'q2', 'b'],
                    flatten=False
                ),
            ],
            "bycategory": {},
        },
        "bysample": {
        },
    },

   variables = {}
)

