from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import *
import workflow_run2
from workflow_run2 import WJetsStitchingWorkflow
from pocket_coffea.lib.weights.common import common_weights
from pocket_coffea.lib.weights import WeightLambda
import numpy as np

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
    
genWeightExtra = WeightLambda.wrap_func(
    name="genWeightExtra",
    function=lambda params, metadata, events, size, shape_variations:
            events.genWeight,
    has_variations=False
)

common_weights += [genWeightExtra]


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
            "inclusive": ["genWeight","genWeightExtra"],
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
            "inclusive": [],
            "bycategory": {},
        },
        "bysample": {
        },
    },
    variables = {
      "HT" : HistConf([Axis(coll="LHE", field="HT", bins=[0,70,100,200,400,600,800,1200,2500,5000], label="HT")]),
    }
)

