from pocket_coffea.lib.weights import WeightLambda
import numpy as np

def stitching_func(params, metadata, events, size, shape_variations):
  if not (((events.count_l==0) & (events.count_nu==0))):
    return np.ones(len(events))
  if metadata["year"] == "2016_PreVFP" and metadata["sample"] == "TTToHadronic":
    return np.select(
      condlist=[
        (events.GenTT.mass>=0.0) & (events.GenTT.mass<700.0),
        (events.GenTT.mass>=700.0) & (events.GenTT.mass<1000.0),
        (events.GenTT.mass>=1000.0) & (events.GenTT.mass<5000.0),
      ],
      choicelist=[
        np.ones(len(events)) * 1.0,
        np.ones(len(events)) * 0.461003141713682,
        np.ones(len(events)) * 0.1871270830653978,
      ],
      default=0.0
    )
  if metadata["year"] == "2016_PostVFP" and metadata["sample"] == "TTToHadronic":
    return np.select(
      condlist=[
        (events.GenTT.mass>=0.0) & (events.GenTT.mass<700.0),
        (events.GenTT.mass>=700.0) & (events.GenTT.mass<1000.0),
        (events.GenTT.mass>=1000.0) & (events.GenTT.mass<5000.0),
      ],
      choicelist=[
        np.ones(len(events)) * 1.0,
        np.ones(len(events)) * 0.40004756456446977,
        np.ones(len(events)) * 0.19906366910198683,
      ],
      default=0.0
    )
  if metadata["year"] == "2017" and metadata["sample"] == "TTToHadronic":
    return np.select(
      condlist=[
        (events.GenTT.mass>=0.0) & (events.GenTT.mass<700.0),
        (events.GenTT.mass>=700.0) & (events.GenTT.mass<1000.0),
        (events.GenTT.mass>=1000.0) & (events.GenTT.mass<5000.0),
      ],
      choicelist=[
        np.ones(len(events)) * 1.0,
        np.ones(len(events)) * 0.5754425934467186,
        np.ones(len(events)) * 0.36004841705480783,
      ],
      default=0.0
    )
  if metadata["year"] == "2018" and metadata["sample"] == "TTToHadronic":
    return np.select(
      condlist=[
        (events.GenTT.mass>=0.0) & (events.GenTT.mass<700.0),
        (events.GenTT.mass>=700.0) & (events.GenTT.mass<1000.0),
        (events.GenTT.mass>=1000.0) & (events.GenTT.mass<5000.0),
      ],
      choicelist=[
        np.ones(len(events)) * 1.0,
        np.ones(len(events)) * 0.6946441552733406,
        np.ones(len(events)) * 0.4370848403395523,
      ],
      default=0.0
    )
  elif metadata["year"] == "2016_PreVFP" and metadata["sample"] == "TTMtt700To1000":
    return np.ones(len(events)) * 0.05085534652569182
  elif metadata["year"] == "2016_PreVFP" and metadata["sample"] == "TTMtt1000":
    return np.ones(len(events)) * 0.019485654047343735
  elif metadata["year"] == "2016_PostVFP" and metadata["sample"] == "TTMtt700To1000":
    return np.ones(len(events)) * 0.05659606320880067
  elif metadata["year"] == "2016_PostVFP" and metadata["sample"] == "TTMtt1000":
    return np.ones(len(events)) * 0.01922077362984597
  elif metadata["year"] == "2017" and metadata["sample"] == "TTMtt700To1000":
    return np.ones(len(events)) * 0.040057424186050676
  elif metadata["year"] == "2017" and metadata["sample"] == "TTMtt1000":
    return np.ones(len(events)) * 0.015338036230480523
  elif metadata["year"] == "2018" and metadata["sample"] == "TTMtt700To1000":
    return np.ones(len(events)) * 0.028799588947413912
  elif metadata["year"] == "2018" and metadata["sample"] == "TTMtt1000":
    return np.ones(len(events)) * 0.013497770657061326
  return np.ones(len(events))

wl_func = WeightLambda.wrap_func(
    name="TTRun2Stitching",
    function=stitching_func,
    has_variations=False
)

TTRun2Stitching = [wl_func]
