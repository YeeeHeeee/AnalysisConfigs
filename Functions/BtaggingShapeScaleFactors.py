from pocket_coffea.lib.weights import WeightLambda
import numpy as np
import awkward as ak
import correctionlib

from pocket_coffea.lib.weights import WeightLambda
import numpy as np
import awkward as ak
import correctionlib


def compute_sf(jets, corr, variation="central", hadronFlavor_mask=None):
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
    if hadronFlavor_mask is not None:
        first=True
        for flavour in hadronFlavor_mask:
            if first:
                new_mask = (flat_jets["hadronFlavour"] == flavour)
                first=False
            else:
                new_mask = new_mask | (flat_jets["hadronFlavour"] == flavour)
        mask = mask & (new_mask)


    sf[mask] = corr.evaluate(
        variation,
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


def reweighting_func_subjets(params, metadata, events, variation, hadronFlavor_mask=None):

    if metadata["year"] in ["2022_preEE", "2022_postEE", "2023_preBPix", "2023_postBPix", "2024"]:
        year = "2018"
    else:
        year = metadata["year"]

    json_file = params["btagjsonFiles"][year]["AK4"]
    function = params["btagShapeFunc"][year]["AK4"]
    BTagfile = correctionlib.CorrectionSet.from_file(json_file)
    corr = BTagfile[function]

    # Compute per-event weights for subjets
    w_subjet1 = compute_sf(events["SubJet1"][:,None], corr, variation=variation, hadronFlavor_mask=hadronFlavor_mask)
    w_subjet2 = compute_sf(events["SubJet2"][:,None], corr, variation=variation, hadronFlavor_mask=hadronFlavor_mask)

    # Combine the two subjet weights into a single per-event weight
    per_event_weight = w_subjet1 * w_subjet2

    return per_event_weight



def reweight_func_subjets_central(params, metadata, events, size, shape_variations):
    return reweighting_func_subjets(params, metadata, events, variation="central")

def reweight_func_subjets_down_hf(params, metadata, events, size, shape_variations):
    return reweighting_func_subjets(params, metadata, events, variation="down_hf", hadronFlavor_mask=[0,5])

def reweight_func_subjets_up_hf(params, metadata, events, size, shape_variations):
    return reweighting_func_subjets(params, metadata, events, variation="up_hf", hadronFlavor_mask=[0,5])

def reweight_func_subjets_down_lf(params, metadata, events, size, shape_variations):
    return reweighting_func_subjets(params, metadata, events, variation="down_lf", hadronFlavor_mask=[0,5])

def reweight_func_subjets_up_lf(params, metadata, events, size, shape_variations):
    return reweighting_func_subjets(params, metadata, events, variation="up_lf", hadronFlavor_mask=[0,5])

def reweight_func_subjets_down_hfstats1(params, metadata, events, size, shape_variations):
    return reweighting_func_subjets(params, metadata, events, variation="down_hfstats1", hadronFlavor_mask=[0,5])

def reweight_func_subjets_up_hfstats1(params, metadata, events, size, shape_variations):
    return reweighting_func_subjets(params, metadata, events, variation="up_hfstats1", hadronFlavor_mask=[0,5])

def reweight_func_subjets_down_lfstats1(params, metadata, events, size, shape_variations):
    return reweighting_func_subjets(params, metadata, events, variation="down_lfstats1", hadronFlavor_mask=[0,5])

def reweight_func_subjets_up_lfstats1(params, metadata, events, size, shape_variations):
    return reweighting_func_subjets(params, metadata, events, variation="up_lfstats1", hadronFlavor_mask=[0,5])

def reweight_func_subjets_down_hfstats2(params, metadata, events, size, shape_variations):
    return reweighting_func_subjets(params, metadata, events, variation="down_hfstats2", hadronFlavor_mask=[0,5])

def reweight_func_subjets_up_hfstats2(params, metadata, events, size, shape_variations):
    return reweighting_func_subjets(params, metadata, events, variation="up_hfstats2", hadronFlavor_mask=[0,5])

def reweight_func_subjets_down_lfstats2(params, metadata, events, size, shape_variations):
    return reweighting_func_subjets(params, metadata, events, variation="down_lfstats2", hadronFlavor_mask=[0,5])

def reweight_func_subjets_up_lfstats2(params, metadata, events, size, shape_variations):
    return reweighting_func_subjets(params, metadata, events, variation="up_lfstats2", hadronFlavor_mask=[0,5])

def reweight_func_subjets_down_cferr1(params, metadata, events, size, shape_variations):
    return reweighting_func_subjets(params, metadata, events, variation="down_cferr1", hadronFlavor_mask=[4])

def reweight_func_subjets_up_cferr1(params, metadata, events, size, shape_variations):
    return reweighting_func_subjets(params, metadata, events, variation="up_cferr1", hadronFlavor_mask=[4])

def reweight_func_subjets_down_cferr2(params, metadata, events, size, shape_variations):
    return reweighting_func_subjets(params, metadata, events, variation="down_cferr2", hadronFlavor_mask=[4])

def reweight_func_subjets_up_cferr2(params, metadata, events, size, shape_variations):
    return reweighting_func_subjets(params, metadata, events, variation="up_cferr2", hadronFlavor_mask=[4])

btagging_shape_func = WeightLambda.wrap_func(
    name="BTagShapeCorrection",
    function=reweighting_func,
    has_variations=False
)

btagging_shape_subjets_func = WeightLambda.wrap_func(
    name="BTagShapeCorrectionSubjets",
    function=reweight_func_subjets_central,
    has_variations=False
)
btagging_shape_subjets_func_down_hf = WeightLambda.wrap_func(
    name="BTagShapeCorrectionSubjets_down_hf",
    function=reweight_func_subjets_down_hf,
    has_variations=False
)
btagging_shape_subjets_func_up_hf = WeightLambda.wrap_func(
    name="BTagShapeCorrectionSubjets_up_hf",
    function=reweight_func_subjets_up_hf,
    has_variations=False
)
btagging_shape_subjets_func_down_lf = WeightLambda.wrap_func(
    name="BTagShapeCorrectionSubjets_down_lf",
    function=reweight_func_subjets_down_lf,
    has_variations=False
)
btagging_shape_subjets_func_up_lf = WeightLambda.wrap_func(
    name="BTagShapeCorrectionSubjets_up_lf",
    function=reweight_func_subjets_up_lf,
    has_variations=False
)
btagging_shape_subjets_func_down_hfstats1 = WeightLambda.wrap_func(
    name="BTagShapeCorrectionSubjets_down_hfstats1",
    function=reweight_func_subjets_down_hfstats1,
    has_variations=False
)
btagging_shape_subjets_func_up_hfstats1 = WeightLambda.wrap_func(
    name="BTagShapeCorrectionSubjets_up_hfstats1",
    function=reweight_func_subjets_up_hfstats1,
    has_variations=False
)
btagging_shape_subjets_func_down_lfstats1 = WeightLambda.wrap_func(
    name="BTagShapeCorrectionSubjets_down_lfstats1",
    function=reweight_func_subjets_down_lfstats1,
    has_variations=False
)
btagging_shape_subjets_func_up_lfstats1 = WeightLambda.wrap_func(
    name="BTagShapeCorrectionSubjets_up_lfstats1",
    function=reweight_func_subjets_up_lfstats1,
    has_variations=False
)
btagging_shape_subjets_func_down_hfstats2 = WeightLambda.wrap_func(
    name="BTagShapeCorrectionSubjets_down_hfstats2",
    function=reweight_func_subjets_down_hfstats2,
    has_variations=False
)
btagging_shape_subjets_func_up_hfstats2 = WeightLambda.wrap_func(
    name="BTagShapeCorrectionSubjets_up_hfstats2",
    function=reweight_func_subjets_up_hfstats2,
    has_variations=False
)
btagging_shape_subjets_func_down_lfstats2 = WeightLambda.wrap_func(
    name="BTagShapeCorrectionSubjets_down_lfstats2",
    function=reweight_func_subjets_down_lfstats2,
    has_variations=False
)
btagging_shape_subjets_func_up_lfstats2 = WeightLambda.wrap_func(
    name="BTagShapeCorrectionSubjets_up_lfstats2",
    function=reweight_func_subjets_up_lfstats2,
    has_variations=False
)
btagging_shape_subjets_func_down_cferr1 = WeightLambda.wrap_func(
    name="BTagShapeCorrectionSubjets_down_cferr1",
    function=reweight_func_subjets_down_cferr1,
    has_variations=False
)
btagging_shape_subjets_func_up_cferr1 = WeightLambda.wrap_func(
    name="BTagShapeCorrectionSubjets_up_cferr1",
    function=reweight_func_subjets_up_cferr1,
    has_variations=False
)
btagging_shape_subjets_func_down_cferr2 = WeightLambda.wrap_func(
    name="BTagShapeCorrectionSubjets_down_cferr2",
    function=reweight_func_subjets_down_cferr2,
    has_variations=False
)
btagging_shape_subjets_func_up_cferr2 = WeightLambda.wrap_func(
    name="BTagShapeCorrectionSubjets_up_cferr2",
    function=reweight_func_subjets_up_cferr2,
    has_variations=False
)



BTagShapeCorrection = [
    btagging_shape_func, 
    btagging_shape_subjets_func,
    btagging_shape_subjets_func_down_hf,
    btagging_shape_subjets_func_up_hf,
    btagging_shape_subjets_func_down_lf,
    btagging_shape_subjets_func_up_lf,
    btagging_shape_subjets_func_down_hfstats1,
    btagging_shape_subjets_func_up_hfstats1,
    btagging_shape_subjets_func_down_lfstats1,
    btagging_shape_subjets_func_up_lfstats1,
    btagging_shape_subjets_func_down_hfstats2,
    btagging_shape_subjets_func_up_hfstats2,
    btagging_shape_subjets_func_down_lfstats2,
    btagging_shape_subjets_func_up_lfstats2,
    btagging_shape_subjets_func_down_cferr1,
    btagging_shape_subjets_func_up_cferr1,
    btagging_shape_subjets_func_down_cferr2,
    btagging_shape_subjets_func_up_cferr2
]