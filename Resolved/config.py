#export PYTHONPATH=..:$PYTHONPATH

from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_definition import Cut
from pocket_coffea.lib.cut_functions import get_nObj_min, get_HLTsel, get_nPVgood, goldenJson, eventFlags
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import *
import workflow
from workflow import ttBaseProcessor_res
from pocket_coffea.lib.weights.common import common_weights

from pocket_coffea.lib.columns_manager import ColOut, ColumnsManager, column_accumulator

# Register custom modules in cloudpickle to propagate them to dask workers
import cloudpickle
import Cut_func
cloudpickle.register_pickle_by_value(workflow)
cloudpickle.register_pickle_by_value(Cut_func)

from Cut_func import *
import os
localdir = os.path.dirname(os.path.abspath(__file__))

# Loading default parameters
from pocket_coffea.parameters import defaults
default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir+"/params")

parameters = defaults.merge_parameters_from_files(default_parameters,
                                                  f"{localdir}/params/object_preselection.yaml",
                                                  f"{localdir}/params/triggers.yaml",
                                                  f"{localdir}/params/plotting.yaml",
                                                  update=True)

cfg = Configurator(
    parameters = parameters,
    datasets = {
        "jsons": [f"{localdir}/Datasets/signals_MC_ttbar.json",],
        "filter" : {
            "samples": ["TTToSemiLeptonic",
                        "TTTo2L2Nu",
                        "TTToHadronic"],

            "samples_exclude" : [],
            "year": ['2018']
            # , '2016_PreVFP',  '2017', '2016_PostVFP'
        }
    },

    workflow = ttBaseProcessor_res,

    skim = [get_nPVgood(4), goldenJson], 
    
    preselections = [semileptonic_presel],
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
                    "MET",
                    ["pt", "phi", 
                    'fiducialGenPhi', 'fiducialGenPt'],
                    flatten=False
                ),
                # Save the Gen-level data:
                ColOut(
                    "Genjj",
                    ['pt', 'eta', 'phi', 'mass'],
                    flatten=False
                ),
                ColOut(
                    "Genbjj_deltaR",
                    ['pt', 'eta', 'phi', 'mass'],
                    flatten=False
                ),
                ColOut(
                    "Genbjj_deltaM",
                    ['pt', 'eta', 'phi', 'mass'],
                    flatten=False
                ),
                # Save the reco data:
                ColOut(
                    "jj",
                    ['pt', 'eta', 'phi', 'mass'],
                    flatten=False
                ),
                ColOut(
                    "bjj_deltaR",
                    ['pt', 'eta', 'phi', 'mass'],
                    flatten=False
                ),
                ColOut(
                    "bjj_deltaM",
                    ['pt', 'eta', 'phi', 'mass'],
                    flatten=False
                ),
                ColOut(
                    "Matchedbjj",
                    ['pt', 'eta', 'phi', 'mass'],
                    flatten=False
                ),
                # Save the matched data:
                ColOut(
                    "Matchedjj",
                    ['pt', 'eta', 'phi', 'mass'],
                    flatten=False
                ),
                ColOut(
                    "Matchedbjj_deltaR",
                    ['pt', 'eta', 'phi', 'mass'],
                    flatten=False
                ),
                ColOut(
                    "Matchedbjj_deltaM",
                    ['pt', 'eta', 'phi', 'mass'],
                    flatten=False
                )
            ],
            "bycategory": {},
        },
        "bysample": {
        },
    },
    
   variables = {
        # **muon_hists(coll="MuonGood", pos=0),
        # **ele_hists(coll="ElectronGood", pos=0),
        # **count_hist(name="nElectronGood", coll="ElectronGood",bins=3, start=0, stop=3),
        # **count_hist(name="nMuonGood", coll="MuonGood",bins=3, start=0, stop=3),
        # **count_hist(name="nJets", coll="JetGood",bins=8, start=0, stop=8),
        # **count_hist(name="nBJets", coll="BJetGood",bins=8, start=0, stop=8),
        # **jet_hists(coll="JetGood", pos=0),
        # **jet_hists(coll="JetGood", pos=1),
        # **jet_hists(coll="bjj", pos=0),
        # "mjj" : HistConf( [Axis(coll="jj", field="mass", bins=100, start=0, stop=200, label=r"$M_{jj}$ [GeV]")] ),
        # "mtop_1" : HistConf([Axis(coll="bjj", field="mass", bins=100, start=0, stop=700, label=r"$M_{top_1}$ [GeV]")]),
        # "Pt_1": HistConf([Axis(coll="bjj", field="pt", bins=100, start=0, stop=700, label=r"$P_{t1}$ [GeV]")]),
        # "mtop_2": HistConf([Axis(coll="bjj_output", field="mass", bins=100, start=0, stop=700, label=r"$M_{top_2}$ [GeV]")]),
        # "Pt_2": HistConf([Axis(coll="bjj_output", field="pt", bins=100, start=0, stop=700, label=r"$P_{t2}$ [GeV]")]),
        # # "mtop_AK8": HistConf([Axis(coll="FatJetGood", field="mass", bins=100, start=0, stop=700, label=r"$M_{AK8}$ [GeV]")]),
        # # "Pt_AK8": HistConf([Axis(coll="FatJetGood", field="pt", bins=100, start=0, stop=700, label=r"$P_{AK8}$ [GeV]")]),
        # "mtop_AK4": HistConf([Axis(coll="JetGood", field="mass", bins=100, start=0, stop=700, label=r"$M_{AK4}$ [GeV]")]),
        # "Pt_AK4": HistConf([Axis(coll="JetGood", field="pt", bins=100, start=0, stop=700, label=r"$P_{AK4}$ [GeV]")]),
    }
)
