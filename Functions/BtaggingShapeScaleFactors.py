from pocket_coffea.lib.weights import WeightLambda
import numpy as np
import awkward as ak
import correctionlib

from pocket_coffea.lib.weights import WeightLambda
import numpy as np
import awkward as ak
import correctionlib

def compute_sf(jets, corr):
    jets["abseta"] = np.abs(jets["eta"])
    flat_jets = ak.flatten(jets)
    n_jets = ak.num(jets)

    sf = np.ones(len(flat_jets))

    # Apply mask if out of range
    mask = (
        (flat_jets["abseta"] < 2.5) &
        (flat_jets["pt"] > 20) &
        (flat_jets["btagDeepB"] >= 0) &
        (flat_jets["btagDeepB"] <= 1)
    )

    sf[mask] = corr.evaluate(
        "central",
        np.array(flat_jets["hadronFlavour"][mask]),
        np.array(flat_jets["abseta"][mask]),
        np.array(flat_jets["pt"][mask]),
        np.array(flat_jets["btagDeepB"][mask]),
    )

    return ak.prod(ak.unflatten(sf, n_jets), axis=1)

def reweighting_func(params, metadata, events, size, shape_variations):

    json_file = params["btagjsonFiles"][metadata["year"]]["AK4"]
    function = params["btagShapeFunc"][metadata["year"]]["AK4"]
    BTagfile = correctionlib.CorrectionSet.from_file(json_file)
    corr = BTagfile[function]

    # Compute per-event weights for each jet collection
    w_jetgood = compute_sf(events["JetGood"], corr)
    w_subjet1 = compute_sf(events["SubJetGood1"], corr)
    w_subjet2 = compute_sf(events["SubJetGood2"], corr)

    # Combine all three into a single per-event weight
    per_event_weight = w_jetgood * w_subjet1 * w_subjet2

    return per_event_weight


def reweighting_func_subjets(params, metadata, events, size, shape_variations):

    json_file = params["btagjsonFiles"][metadata["year"]]["AK4"]
    function = params["btagShapeFunc"][metadata["year"]]["AK4"]
    BTagfile = correctionlib.CorrectionSet.from_file(json_file)
    corr = BTagfile[function]

    # Compute per-event weights for subjets
    w_subjet1 = compute_sf(events["SubJet1"][:,None], corr)
    w_subjet2 = compute_sf(events["SubJet2"][:,None], corr)

    # Combine the two subjet weights into a single per-event weight
    per_event_weight = w_subjet1 * w_subjet2

    return per_event_weight

btagging_shape_func = WeightLambda.wrap_func(
    name="BTagShapeCorrection",
    function=reweighting_func,
    has_variations=False
)

btagging_shape_subjets_func = WeightLambda.wrap_func(
    name="BTagShapeCorrectionSubjets",
    function=reweighting_func_subjets,
    has_variations=False
)


BTagShapeCorrection = [btagging_shape_func, btagging_shape_subjets_func]