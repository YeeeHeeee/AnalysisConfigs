from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import *
import workflow
from workflow import TTStitchingWorkflow
from pocket_coffea.lib.weights.common import common_weights
from pocket_coffea.lib.weights import WeightLambda
import numpy as np

# Register custom modules in cloudpickle to propagate them to dask workers
import cloudpickle
import cut_hadronic

cloudpickle.register_pickle_by_value(workflow)
cloudpickle.register_pickle_by_value(cut_hadronic)

from cut_hadronic import *
import os
localdir = os.path.dirname(os.path.abspath(__file__))

# Loading default parameters
from pocket_coffea.parameters import defaults
default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir+"/params")
parameters = defaults.merge_parameters_from_files(default_parameters, update=True)

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
        "jsons": [f"{localdir}/../Datasets/signals_MC_ttbar.json"],
        "filter" : {
            "samples": [
              "TTToHadronic",
              "TTMtt700To1000",
              "TTMtt1000"
            ],
            "samples_exclude" : [],
            "year": ['2016_PreVFP', '2016_PostVFP', '2017', '2018'], 
        }
    },
    workflow = TTStitchingWorkflow,
    skim = [], 
    preselections = [hadronic_presel],
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
      "Mtt" : HistConf([Axis(coll="GenTT", field="mass", bins=[0,700,1000,5000], label="mass")]),
    }
)

