from pocket_coffea.lib.weights import WeightLambda
import numpy as np

def stitching_func(params, metadata, events, size, shape_variations):
  if not (((events.count_l==2) & (events.count_nu==2))):
    return np.ones(len(events))
  if metadata["year"] == "2016_PreVFP" and metadata["sample"] == "TTTo2L2Nu":
    return np.select(
      condlist=[
        (events.GenTT.mass>=0.0) & (events.GenTT.mass<700.0),
        (events.GenTT.mass>=700.0) & (events.GenTT.mass<1000.0),
        (events.GenTT.mass>=1000.0) & (events.GenTT.mass<5000.0),
      ],
      choicelist=[
        np.ones(len(events)) * 1.0,
        np.ones(len(events)) * 0.589622499755721,
        np.ones(len(events)) * 0.27789225981122434,
      ],
      default=0.0
    )
  if metadata["year"] == "2016_PostVFP" and metadata["sample"] == "TTTo2L2Nu":
    return np.select(
      condlist=[
        (events.GenTT.mass>=0.0) & (events.GenTT.mass<700.0),
        (events.GenTT.mass>=700.0) & (events.GenTT.mass<1000.0),
        (events.GenTT.mass>=1000.0) & (events.GenTT.mass<5000.0),
      ],
      choicelist=[
        np.ones(len(events)) * 1.0,
        np.ones(len(events)) * 0.5412369503368883,
        np.ones(len(events)) * 0.30544925962324343,
      ],
      default=0.0
    )
  if metadata["year"] == "2017" and metadata["sample"] == "TTTo2L2Nu":
    return np.select(
      condlist=[
        (events.GenTT.mass>=0.0) & (events.GenTT.mass<700.0),
        (events.GenTT.mass>=700.0) & (events.GenTT.mass<1000.0),
        (events.GenTT.mass>=1000.0) & (events.GenTT.mass<5000.0),
      ],
      choicelist=[
        np.ones(len(events)) * 1.0,
        np.ones(len(events)) * 0.7297795419063815,
        np.ones(len(events)) * 0.5283741468295263,
      ],
      default=0.0
    )
  if metadata["year"] == "2018" and metadata["sample"] == "TTTo2L2Nu":
    return np.select(
      condlist=[
        (events.GenTT.mass>=0.0) & (events.GenTT.mass<700.0),
        (events.GenTT.mass>=700.0) & (events.GenTT.mass<1000.0),
        (events.GenTT.mass>=1000.0) & (events.GenTT.mass<5000.0),
      ],
      choicelist=[
        np.ones(len(events)) * 1.0,
        np.ones(len(events)) * 0.811198628052345,
        np.ones(len(events)) * 0.5945988761136708,
      ],
      default=0.0
    )
  elif metadata["year"] == "2016_PreVFP" and metadata["sample"] == "TTMtt700To1000":
    return np.ones(len(events)) * 0.03872025341993595
  elif metadata["year"] == "2016_PreVFP" and metadata["sample"] == "TTMtt1000":
    return np.ones(len(events)) * 0.017299262740808147
  elif metadata["year"] == "2016_PostVFP" and metadata["sample"] == "TTMtt700To1000":
    return np.ones(len(events)) * 0.0432686438244733
  elif metadata["year"] == "2016_PostVFP" and metadata["sample"] == "TTMtt1000":
    return np.ones(len(events)) * 0.016646685167727385
  elif metadata["year"] == "2017" and metadata["sample"] == "TTMtt700To1000":
    return np.ones(len(events)) * 0.025489565255557466
  elif metadata["year"] == "2017" and metadata["sample"] == "TTMtt1000":
    return np.ones(len(events)) * 0.011300227151267012
  elif metadata["year"] == "2018" and metadata["sample"] == "TTMtt700To1000":
    return np.ones(len(events)) * 0.017814369033896563
  elif metadata["year"] == "2018" and metadata["sample"] == "TTMtt1000":
    return np.ones(len(events)) * 0.009711899106991379
  return np.ones(len(events))

wl_func = WeightLambda.wrap_func(
    name="TTRun2Stitching",
    function=stitching_func,
    has_variations=False
)

TTRun2Stitching = [wl_func]
