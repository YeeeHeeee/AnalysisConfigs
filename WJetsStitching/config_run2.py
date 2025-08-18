from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import *
import workflow_run2
from workflow_run2 import WJetsStitchingWorkflow
from pocket_coffea.lib.weights.common import common_weights
import numpy as np
from pocket_coffea.lib.columns_manager import ColOut, ColumnsManager, column_accumulator

# Register custom modules in cloudpickle to propagate them to dask workers
import cloudpickle
cloudpickle.register_pickle_by_value(workflow_run2)

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
            "samples": ["WJetsToLNuHT7OTo100",
                        "WJetsToLNuHT100To200",
                        "WJetsToLNuHT200To400",
                        "WJetsToLNuHT400To600",
                        "WJetsToLNuHT600To800",
                        "WJetsToLNuHT800To1200",
                        "WJetsToLNuHT1200To2500",
                        "WJetsToLNu"],
            "samples_exclude" : [],
            "year": ['2016_PreVFP', '2016_PostVFP', '2017', '2018'], 
        }
    },
    workflow = WJetsStitchingWorkflow,
    skim = [], 
    preselections = [],
    categories = {
        "baseline": [passthrough],
    },

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
            ],
            "bycategory": {},
        },
        "bysample": {
        },
    },
    variables = {}
)

