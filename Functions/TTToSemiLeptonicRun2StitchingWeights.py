from pocket_coffea.lib.weights import WeightLambda
import numpy as np

def stitching_func(params, metadata, events, size, shape_variations):
  if not (((events.count_l==1) & (events.count_nu==1))):
    return np.ones(len(events))
  if metadata["year"] == "2016_PreVFP" and metadata["sample"] == "TTToSemiLeptonic":
    return np.select(
      condlist=[
        (events.GenTT.mass>=0.0) & (events.GenTT.mass<700.0),
        (events.GenTT.mass>=700.0) & (events.GenTT.mass<1000.0),
        (events.GenTT.mass>=1000.0) & (events.GenTT.mass<5000.0),
      ],
      choicelist=[
        np.ones(len(events)) * 1.0,
        np.ones(len(events)) * 0.5480761435829583,
        np.ones(len(events)) * 0.24555780002558994,
      ],
      default=0.0
    )
  if metadata["year"] == "2016_PostVFP" and metadata["sample"] == "TTToSemiLeptonic":
    return np.select(
      condlist=[
        (events.GenTT.mass>=0.0) & (events.GenTT.mass<700.0),
        (events.GenTT.mass>=700.0) & (events.GenTT.mass<1000.0),
        (events.GenTT.mass>=1000.0) & (events.GenTT.mass<5000.0),
      ],
      choicelist=[
        np.ones(len(events)) * 1.0,
        np.ones(len(events)) * 0.4846784917928235,
        np.ones(len(events)) * 0.2590964293686057,
      ],
      default=0.0
    )
  if metadata["year"] == "2017" and metadata["sample"] == "TTToSemiLeptonic":
    return np.select(
      condlist=[
        (events.GenTT.mass>=0.0) & (events.GenTT.mass<700.0),
        (events.GenTT.mass>=700.0) & (events.GenTT.mass<1000.0),
        (events.GenTT.mass>=1000.0) & (events.GenTT.mass<5000.0),
      ],
      choicelist=[
        np.ones(len(events)) * 1.0,
        np.ones(len(events)) * 0.6775574530070404,
        np.ones(len(events)) * 0.46572456838555654,
      ],
      default=0.0
    )
  if metadata["year"] == "2018" and metadata["sample"] == "TTToSemiLeptonic":
    return np.select(
      condlist=[
        (events.GenTT.mass>=0.0) & (events.GenTT.mass<700.0),
        (events.GenTT.mass>=700.0) & (events.GenTT.mass<1000.0),
        (events.GenTT.mass>=1000.0) & (events.GenTT.mass<5000.0),
      ],
      choicelist=[
        np.ones(len(events)) * 1.0,
        np.ones(len(events)) * 0.7718484122486218,
        np.ones(len(events)) * 0.5358678883895625,
      ],
      default=0.0
    )
  elif metadata["year"] == "2016_PreVFP" and metadata["sample"] == "TTMtt700To1000":
    return np.ones(len(events)) * 0.042636771369697526
  elif metadata["year"] == "2016_PreVFP" and metadata["sample"] == "TTMtt1000":
    return np.ones(len(events)) * 0.018067002915342696
  elif metadata["year"] == "2016_PostVFP" and metadata["sample"] == "TTMtt700To1000":
    return np.ones(len(events)) * 0.048620505496083224
  elif metadata["year"] == "2016_PostVFP" and metadata["sample"] == "TTMtt1000":
    return np.ones(len(events)) * 0.01774615739245455
  elif metadata["year"] == "2017" and metadata["sample"] == "TTMtt700To1000":
    return np.ones(len(events)) * 0.0304254756294062
  elif metadata["year"] == "2017" and metadata["sample"] == "TTMtt1000":
    return np.ones(len(events)) * 0.012803281812712203
  elif metadata["year"] == "2018" and metadata["sample"] == "TTMtt700To1000":
    return np.ones(len(events)) * 0.021529729442973513
  elif metadata["year"] == "2018" and metadata["sample"] == "TTMtt1000":
    return np.ones(len(events)) * 0.01112417721867686
  return np.ones(len(events))

wl_func = WeightLambda.wrap_func(
    name="TTToSemiLeptonicRun2Stitching",
    function=stitching_func,
    has_variations=False
)

TTToSemiLeptonicRun2Stitching = [wl_func]