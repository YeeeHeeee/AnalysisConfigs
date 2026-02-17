from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import *
import workflow_run3
from workflow_run3 import WJetsStitchingWorkflow
from pocket_coffea.lib.weights.common import common_weights
import numpy as np
from pocket_coffea.lib.cut_definition import Cut
from pocket_coffea.lib.columns_manager import ColOut

# Register custom modules in cloudpickle to propagate them to dask workers
import cloudpickle
cloudpickle.register_pickle_by_value(workflow_run3)

import os
localdir = os.path.dirname(os.path.abspath(__file__))

# Loading default parameters
from pocket_coffea.parameters import defaults
default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir+"/params")
parameters = defaults.merge_parameters_from_files(
    default_parameters, 
    f"{localdir}/../params/pc_jet_calibration.yaml",
    update=True)

cfg = Configurator(
    parameters = parameters,
    datasets = {
        "jsons": [f"{localdir}/../Datasets/backgrounds_MC_ttbar.json"],
        "filter" : {
            "samples": [
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
                "WJetsToLNu"
            ],
            "samples_exclude" : [],
            #"year": ['2022_preEE', '2022_postEE', '2023_preBPix', '2023_postBPix'],
            "year": ["2022_preEE"]
        }
    },
    workflow = WJetsStitchingWorkflow,
    skim = [], 
    preselections = [],
    categories = {},

    weights_classes = common_weights,
    
    weights = {
        "common": {
            "inclusive": ["genWeight"],
            "bycategory": {}
        },
        "bysample": {
        }
    },

    variations = {
        "weights": {
            "common": {
                "inclusive": [],
                "bycategory" : {}
            },
        "bysample": {
        }    
        },
    },   
    columns = {
        "common": {
            "inclusive": [
                ColOut(
                    "LHE",
                    ["HT"],
                    flatten=False
                ),
                ColOut(
                    "LNu",
                    ["mass"],
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

