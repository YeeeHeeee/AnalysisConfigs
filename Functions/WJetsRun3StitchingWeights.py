from pocket_coffea.lib.weights import WeightLambda
import numpy as np

def stitching_func(params, metadata, events, size, shape_variations):
  if metadata["year"] in ["2022_preEE","2022_postEE","2023_preBPix","2023_postBPix"] and metadata["sample"] == "WJetsToLNu":
    return np.select(
      condlist=[
        (events.LHE.HT>=0.0) & (events.LHE.HT<40.0) & ((events.LNu.mass>=0) & (events.LNu.mass<120)),
        (events.LHE.HT>=40.0) & (events.LHE.HT<100.0) & ((events.LNu.mass>=0) & (events.LNu.mass<120)),
        (events.LHE.HT>=100.0) & (events.LHE.HT<400.0) & ((events.LNu.mass>=0) & (events.LNu.mass<120)),
        (events.LHE.HT>=400.0) & (events.LHE.HT<800.0) & ((events.LNu.mass>=0) & (events.LNu.mass<120)),
        (events.LHE.HT>=800.0) & (events.LHE.HT<1500.0) & ((events.LNu.mass>=0) & (events.LNu.mass<120)),
        (events.LHE.HT>=1500.0) & (events.LHE.HT<2500.0) & ((events.LNu.mass>=0) & (events.LNu.mass<120)),
        (events.LHE.HT>=2500.0) & (events.LHE.HT<5000.0) & ((events.LNu.mass>=0) & (events.LNu.mass<120)),
        (events.LHE.HT>=0.0) & (events.LHE.HT<40.0) & (events.LNu.mass>=120),
        (events.LHE.HT>=40.0) & (events.LHE.HT<100.0) & (events.LNu.mass>=120),
        (events.LHE.HT>=100.0) & (events.LHE.HT<400.0) & (events.LNu.mass>=120),
        (events.LHE.HT>=400.0) & (events.LHE.HT<800.0) & (events.LNu.mass>=120),
        (events.LHE.HT>=800.0) & (events.LHE.HT<1500.0) & (events.LNu.mass>=120),
        (events.LHE.HT>=1500.0) & (events.LHE.HT<2500.0) & (events.LNu.mass>=120),
        (events.LHE.HT>=2500.0) & (events.LHE.HT<5000.0) & (events.LNu.mass>=120),
      ],
      choicelist=[
        np.ones(len(events)) * 1.0,
        np.ones(len(events)) * 0.0,
        np.ones(len(events)) * 0.0,
        np.ones(len(events)) * 0.0,
        np.ones(len(events)) * 0.0,
        np.ones(len(events)) * 0.0,
        np.ones(len(events)) * 0.0,
        np.ones(len(events)) * 1.0,
        np.ones(len(events)) * 0.0,
        np.ones(len(events)) * 0.0,
        np.ones(len(events)) * 0.0,
        np.ones(len(events)) * 0.0,
        np.ones(len(events)) * 0.0,
        np.ones(len(events)) * 0.0,
      ],
      default=0.0
    )
  elif metadata["year"] in ["2022_preEE","2022_postEE","2023_preBPix","2023_postBPix"] and metadata["sample"] == "WJetsToLNuHT40To100MLNu0To120":
    return np.ones(len(events)) * 0.076800867
  elif metadata["year"] in ["2022_preEE","2022_postEE","2023_preBPix","2023_postBPix"] and metadata["sample"] == "WJetsToLNuHT100To400MLNu0To120":
    return np.ones(len(events)) * 0.029355479
  elif metadata["year"] in ["2022_preEE","2022_postEE","2023_preBPix","2023_postBPix"] and metadata["sample"] == "WJetsToLNuHT400To800MLNu0To120":
    return np.ones(len(events)) * 0.001083047
  elif metadata["year"] in ["2022_preEE","2022_postEE","2023_preBPix","2023_postBPix"] and metadata["sample"] == "WJetsToLNuHT800To1500MLNu0To120":
    return np.ones(len(events)) * 0.000112475
  elif metadata["year"] in ["2022_preEE","2022_postEE","2023_preBPix","2023_postBPix"] and metadata["sample"] == "WJetsToLNuHT1500To2500MLNu0To120":
    return np.ones(len(events)) * 8.08269e-06
  elif metadata["year"] in ["2022_preEE","2022_postEE","2023_preBPix","2023_postBPix"] and metadata["sample"] == "WJetsToLNuHT2500MLNu0To120":
    return np.ones(len(events)) * 5.55154e-07
  elif metadata["year"] in ["2022_preEE","2022_postEE","2023_preBPix","2023_postBPix"] and metadata["sample"] == "WJetsToLNuHT40To100MLNu120":
    return np.ones(len(events)) * 0.000371186
  elif metadata["year"] in ["2022_preEE","2022_postEE","2023_preBPix","2023_postBPix"] and metadata["sample"] == "WJetsToLNuHT100To400MLNu120":
    return np.ones(len(events)) * 0.000183968
  elif metadata["year"] in ["2022_preEE","2022_postEE","2023_preBPix","2023_postBPix"] and metadata["sample"] == "WJetsToLNuHT400To800MLNu120":
    return np.ones(len(events)) * 9.45839e-06
  elif metadata["year"] in ["2022_preEE","2022_postEE","2023_preBPix","2023_postBPix"] and metadata["sample"] == "WJetsToLNuHT800To1500MLNu120":
    return np.ones(len(events)) * 1.12927e-06
  elif metadata["year"] in ["2022_preEE","2022_postEE","2023_preBPix","2023_postBPix"] and metadata["sample"] == "WJetsToLNuHT1500To2500MLNu120":
    return np.ones(len(events)) * 9.14606e-08
  elif metadata["year"] in ["2022_preEE","2022_postEE","2023_preBPix","2023_postBPix"] and metadata["sample"] == "WJetsToLNuHT2500MLNu120":
    return np.ones(len(events)) * 6.83878e-09
  return np.ones(len(events))

wl_func = WeightLambda.wrap_func(
    name="WJetsRun3Stitching",
    function=stitching_func,
    has_variations=False
)

WJetsRun3Stitching = [wl_func]