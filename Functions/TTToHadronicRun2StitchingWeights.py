from pocket_coffea.lib.weights import WeightLambda
import numpy as np

def stitching_func(params, metadata, events, size, shape_variations):
  if metadata["year"] == "2016_PreVFP" and metadata["sample"] == "TTToHadronic":
    return np.select(
      condlist=[
        (events.GenTT.mass>=0.0) & (events.GenTT.mass<700.0) & ((events.count_l==0) & (events.count_nu==0)),
        (events.GenTT.mass>=700.0) & (events.GenTT.mass<1000.0) & ((events.count_l==0) & (events.count_nu==0)),
        (events.GenTT.mass>=1000.0) & (events.GenTT.mass<5000.0) & ((events.count_l==0) & (events.count_nu==0)),
        ~((events.count_l==0) & (events.count_nu==0)),
      ],
      choicelist=[
        np.ones(len(events)) * 1.0,
        np.ones(len(events)) * 0.461003141713682,
        np.ones(len(events)) * 0.1871270830653978,
        np.ones(len(events)),
      ],
      default=0.0
    )
  if metadata["year"] == "2016_PostVFP" and metadata["sample"] == "TTToHadronic":
    return np.select(
      condlist=[
        (events.GenTT.mass>=0.0) & (events.GenTT.mass<700.0) & ((events.count_l==0) & (events.count_nu==0)),
        (events.GenTT.mass>=700.0) & (events.GenTT.mass<1000.0) & ((events.count_l==0) & (events.count_nu==0)),
        (events.GenTT.mass>=1000.0) & (events.GenTT.mass<5000.0) & ((events.count_l==0) & (events.count_nu==0)),
        ~((events.count_l==0) & (events.count_nu==0)),
      ],
      choicelist=[
        np.ones(len(events)) * 1.0,
        np.ones(len(events)) * 0.40004756456446977,
        np.ones(len(events)) * 0.19906366910198683,
        np.ones(len(events)),
      ],
      default=0.0
    )
  if metadata["year"] == "2017" and metadata["sample"] == "TTToHadronic":
    return np.select(
      condlist=[
        (events.GenTT.mass>=0.0) & (events.GenTT.mass<700.0) & ((events.count_l==0) & (events.count_nu==0)),
        (events.GenTT.mass>=700.0) & (events.GenTT.mass<1000.0) & ((events.count_l==0) & (events.count_nu==0)),
        (events.GenTT.mass>=1000.0) & (events.GenTT.mass<5000.0) & ((events.count_l==0) & (events.count_nu==0)),
        ~((events.count_l==0) & (events.count_nu==0)),
      ],
      choicelist=[
        np.ones(len(events)) * 1.0,
        np.ones(len(events)) * 0.5754425934467186,
        np.ones(len(events)) * 0.36004841705480783,
        np.ones(len(events)),
      ],
      default=0.0
    )
  if metadata["year"] == "2018" and metadata["sample"] == "TTToHadronic":
    return np.select(
      condlist=[
        (events.GenTT.mass>=0.0) & (events.GenTT.mass<700.0) & ((events.count_l==0) & (events.count_nu==0)),
        (events.GenTT.mass>=700.0) & (events.GenTT.mass<1000.0) & ((events.count_l==0) & (events.count_nu==0)),
        (events.GenTT.mass>=1000.0) & (events.GenTT.mass<5000.0) & ((events.count_l==0) & (events.count_nu==0)),
        ~((events.count_l==0) & (events.count_nu==0)),
      ],
      choicelist=[
        np.ones(len(events)) * 1.0,
        np.ones(len(events)) * 0.6946441552733406,
        np.ones(len(events)) * 0.4370848403395523,
        np.ones(len(events)),
      ],
      default=0.0
    )
  elif metadata["year"] == "2016_PreVFP" and metadata["sample"] == "TTMtt700To1000":
    return np.select(
      condlist=[
        (events.count_l==0) & (events.count_nu==0),
        ~((events.count_l==0) & (events.count_nu==0)),
      ],
      choicelist=[
        np.ones(len(events)) * 0.05085534652569182 * 0.86163059, # Extra correction
        np.ones(len(events)),
      ],
      default=0.0
    )
  elif metadata["year"] == "2016_PreVFP" and metadata["sample"] == "TTMtt1000":
    return np.select(
      condlist=[
        (events.count_l==0) & (events.count_nu==0),
        ~((events.count_l==0) & (events.count_nu==0)),
      ],
      choicelist=[
        np.ones(len(events)) * 0.019485654047343735 * 0.81828033,
        np.ones(len(events)),
      ],
      default=0.0
    )
  elif metadata["year"] == "2016_PostVFP" and metadata["sample"] == "TTMtt700To1000":
    return np.select(
      condlist=[
        (events.count_l==0) & (events.count_nu==0),
        ~((events.count_l==0) & (events.count_nu==0)),
      ],
      choicelist=[
        np.ones(len(events)) * 0.05659606320880067 * 0.94694766,
        np.ones(len(events)),
      ],
      default=0.0
    )
  elif metadata["year"] == "2016_PostVFP" and metadata["sample"] == "TTMtt1000":
    return np.select(
      condlist=[
        (events.count_l==0) & (events.count_nu==0),
        ~((events.count_l==0) & (events.count_nu==0)),
      ],
      choicelist=[
        np.ones(len(events)) * 0.01922077362984597 * 0.89159113,
        np.ones(len(events)),
      ],
      default=0.0
    )
  elif metadata["year"] == "2017" and metadata["sample"] == "TTMtt700To1000":
    return np.select(
      condlist=[
        (events.count_l==0) & (events.count_nu==0),
        ~((events.count_l==0) & (events.count_nu==0)),
      ],
      choicelist=[
        np.ones(len(events)) * 0.040057424186050676 * 0.92282622,
        np.ones(len(events)),
      ],
      default=0.0
    )
  elif metadata["year"] == "2017" and metadata["sample"] == "TTMtt1000":
    return np.select(
      condlist=[
        (events.count_l==0) & (events.count_nu==0),
        ~((events.count_l==0) & (events.count_nu==0)),
      ],
      choicelist=[
        np.ones(len(events)) * 0.015338036230480523 * 0.88054898,
        np.ones(len(events)),
      ],
      default=0.0
    )
  elif metadata["year"] == "2018" and metadata["sample"] == "TTMtt700To1000":
    return np.select(
      condlist=[
        (events.count_l==0) & (events.count_nu==0),
        ~((events.count_l==0) & (events.count_nu==0)),
      ],
      choicelist=[
        np.ones(len(events)) * 0.028799588947413912 * 0.8894473,
        np.ones(len(events)),
      ],
      default=0.0
    )
  elif metadata["year"] == "2018" and metadata["sample"] == "TTMtt1000":
    return np.select(
      condlist=[
        (events.count_l==0) & (events.count_nu==0),
        ~((events.count_l==0) & (events.count_nu==0)),
      ],
      choicelist=[
        np.ones(len(events)) * 0.013497770657061326 * 0.82877651,
        np.ones(len(events)),
      ],
      default=0.0
    )
  return np.ones(len(events))

wl_func = WeightLambda.wrap_func(
    name="TTToHadronicRun2Stitching",
    function=stitching_func,
    has_variations=False
)

TTToHadronicRun2Stitching = [wl_func]
