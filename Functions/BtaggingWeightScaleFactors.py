from pocket_coffea.lib.weights import WeightLambda
import numpy as np
import awkward as ak
import correctionlib

def reweighting_func(params, metadata, events, size, shape_variations):

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
      "central",
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
    function=reweighting_func,
    has_variations=False
)

BTagWeightCorrection = [btagging_weight_func]