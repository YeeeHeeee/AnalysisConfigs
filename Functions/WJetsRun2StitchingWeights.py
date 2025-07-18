from pocket_coffea.lib.weights import WeightLambda
import numpy as np

def stitching_func(params, metadata, events, size, shape_variations):
  if metadata["year"] in ["2016_PreVFP","2016_PostVFP","2017","2018"] and metadata["sample"] == "WJetsToLNu":
    return np.select(
      condlist=[
        ((events.LHE.HT>=0) & (events.LHE.HT<70)),
        ((events.LHE.HT>=70) & (events.LHE.HT<100)),
        ((events.LHE.HT>=100) & (events.LHE.HT<200)),
        ((events.LHE.HT>=200) & (events.LHE.HT<400)),
        ((events.LHE.HT>=400) & (events.LHE.HT<600)),
        ((events.LHE.HT>=600) & (events.LHE.HT<800)),
        ((events.LHE.HT>=800) & (events.LHE.HT<1200)),
        ((events.LHE.HT>=1200) & (events.LHE.HT<2500)),
        (events.LHE.HT>=2500),
      ],
      choicelist=[
        np.ones(len(events)) * 1.0,
        np.ones(len(events)) * 0.0,
        np.ones(len(events)) * 0.0,
        np.ones(len(events)) * 0.0,
        np.ones(len(events)) * 0.0,
        np.ones(len(events)) * 0.0,
        np.ones(len(events)) * 0.0,
        np.ones(len(events)) * 0.0,
        np.ones(len(events)) * 0.0,
      ],
      default=0.0
    )
  elif metadata["year"] in ["2016_PreVFP","2016_PostVFP","2017","2018"] and metadata["sample"] == "WJetsToLNuHT70To100":
    return np.ones(len(events)) * 0.024404987
  elif metadata["year"] in ["2016_PreVFP","2016_PostVFP","2017","2018"] and metadata["sample"] == "WJetsToLNuHT100To200":
    return np.ones(len(events)) * 0.026350586
  elif metadata["year"] in ["2016_PreVFP","2016_PostVFP","2017","2018"] and metadata["sample"] == "WJetsToLNuHT200To400":
    return np.ones(len(events)) * 0.007704949
  elif metadata["year"] in ["2016_PreVFP","2016_PostVFP","2017","2018"] and metadata["sample"] == "WJetsToLNuHT400To600":
    return np.ones(len(events)) * 0.001085757
  elif metadata["year"] in ["2016_PreVFP","2016_PostVFP","2017","2018"] and metadata["sample"] == "WJetsToLNuHT600To800":
    return np.ones(len(events)) * 0.000243105
  elif metadata["year"] in ["2016_PreVFP","2016_PostVFP","2017","2018"] and metadata["sample"] == "WJetsToLNuHT800To1200":
    return np.ones(len(events)) * 0.000158529
  elif metadata["year"] in ["2016_PreVFP","2016_PostVFP","2017","2018"] and metadata["sample"] == "WJetsToLNuHT1200To2500":
    return np.ones(len(events)) * 2.19116e-05
  elif metadata["year"] in ["2016_PreVFP","2016_PostVFP","2017","2018"] and metadata["sample"] == "WJetsToLNuHT2500":
    return np.ones(len(events)) * 4.998111069134869e-07
  return np.ones(len(events))

wl_func = WeightLambda.wrap_func(
    name="WJetsRun2Stitching",
    function=stitching_func,
    has_variations=False
)

WJetsRun2Stitching = [wl_func]
