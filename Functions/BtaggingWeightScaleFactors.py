from pocket_coffea.lib.weights import WeightLambda
import numpy as np
import awkward as ak
import correctionlib

def btag_weight_central(params, metadata, events, size, shape_variations):
  return reweighting_func(params, metadata, events, variation="central")

def btag_weight_up(params, metadata, events, size, shape_variations):
  return reweighting_func(params, metadata, events, variation="up")

def btag_weight_down(params, metadata, events, size, shape_variations):
  return reweighting_func(params, metadata, events, variation="down")

def btag_weight_up_correlated(params, metadata, events, size, shape_variations):
  if metadata["year"] == "2024":
    return np.ones(len(events["BJetGood"]))
  return reweighting_func(params, metadata, events, variation="up_correlated")

def btag_weight_down_correlated(params, metadata, events, size, shape_variations):
  if metadata["year"] == "2024":
    return np.ones(len(events["BJetGood"]))
  return reweighting_func(params, metadata, events, variation="down_correlated")

def reweighting_func(params, metadata, events, variation="central"):
  
  json_file = params["btagjsonFiles"][metadata["year"]]["AK4"]
  function = params["btagFunc"][metadata["year"]]["AK4"]
  BTagfile = correctionlib.CorrectionSet.from_file(json_file)
  corr = BTagfile[function]

  bjets = events["BJetGood"]
  bjets['abseta'] = np.abs(bjets['eta'])
  bj, nbj = ak.flatten(bjets), ak.num(bjets)

  flat_hflav = np.array(bj["hadronFlavour"])

  if metadata["year"] == "2024":
    apply_mask = (flat_hflav == 5)
  else:
    apply_mask = (flat_hflav == 4) | (flat_hflav == 5)

  flatweightcorr = np.ones(len(flat_hflav))

  # Run the correction only on the valid subset
  valid_eta = np.array(bj["eta"])[apply_mask]
  valid_pt = np.array(bj["pt"])[apply_mask]
  valid_flav = flat_hflav[apply_mask]

  flatweightcorr[apply_mask] = corr.evaluate(
      variation,
      params.object_preselection.Jet.btag.wp,
      valid_flav,
      np.abs(valid_eta),
      valid_pt,
  )
  weightcorr = ak.unflatten(flatweightcorr, nbj)
  per_event_weight = ak.prod(weightcorr, axis=1)

  return per_event_weight


btagging_weight_func = WeightLambda.wrap_func(
    name="BTagWeightCorrection",
    function=btag_weight_central,
    has_variations=False
)

btagging_weight_func_up = WeightLambda.wrap_func(
    name="BTagWeightCorrection_up",
    function=btag_weight_up,
    has_variations=False
)

btagging_weight_func_down = WeightLambda.wrap_func(
    name="BTagWeightCorrection_down",
    function=btag_weight_down,
    has_variations=False
)

btagging_weight_func_up_correlated = WeightLambda.wrap_func(
    name="BTagWeightCorrection_up_correlated",
    function=btag_weight_up_correlated,
    has_variations=False
)

btagging_weight_func_down_correlated = WeightLambda.wrap_func(
    name="BTagWeightCorrection_down_correlated",
    function=btag_weight_down_correlated,
    has_variations=False
)

BTagWeightCorrection = [
  btagging_weight_func,
  btagging_weight_func_up,
  btagging_weight_func_down,
  btagging_weight_func_up_correlated,
  btagging_weight_func_down_correlated
]