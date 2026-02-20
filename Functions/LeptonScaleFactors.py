from pocket_coffea.lib.weights import WeightLambda
import numpy as np
import awkward as ak
import correctionlib
from pocket_coffea.lib.scale_factors import sf_ele_id, get_ele_sf

from pocket_coffea.lib.weights.common.weights_run2_UL import sf_ele_trigger as sf_ele_trigger_run2
from pocket_coffea.lib.scale_factors import sf_ele_trigger as sf_ele_trigger_run3


def sf_ele_reco(params, events, year):
    '''
    This function computes the per-electron reco SF and returns the corresponding per-event SF, obtained by multiplying the per-electron SF in each event.
    Additionally, also the up and down variations of the SF are returned.
    Electrons are split into two categories based on a pt cut depending on the Run preiod, so that the proper SF is applied.
    '''
    coll = params.lepton_scale_factors.electron_sf.collection
    ele_pt = events[coll].pt
    ele_eta = events[coll].etaSC # This is added on top of NanoAOD
    ele_phi = events[coll].phi

    pt_ranges = []
    if year in ['2016_PreVFP', '2016_PostVFP','2017','2018']:
        pt_ranges += [("pt_lt_20", (ele_pt < 20)), 
                      ("pt_gt_20", (ele_pt >= 20))]
    elif year in ["2022_preEE", "2022_postEE", "2023_preBPix", "2023_postBPix","2024Reco"]:
        pt_ranges += [("pt_lt_20", (ele_pt < 20)), 
                      ("pt_gt_20_lt_75", (ele_pt >= 20) & (ele_pt < 75)), 
                      ("pt_gt_75", (ele_pt >= 75))]
    else:
        raise Exception("For chosen year "+year+" sf_ele_reco are not implemented yet")
    
    sf_reco, sfup_reco, sfdown_reco = [], [], []

    for pt_range_key, pt_range in pt_ranges:
        ele_pt_inPtRange = ak.flatten(ele_pt[pt_range])
        ele_eta_inPtRange = ak.flatten(ele_eta[pt_range])
        ele_phi_inPtRange = ak.flatten(ele_phi[pt_range])
        ele_counts_inPtRange = ak.num(ele_pt[pt_range])

        sf_reco_inPtRange, sfup_reco_inPtRange, sfdown_reco_inPtRange = get_ele_sf(
            params,
            year,
            ele_pt_inPtRange,
            ele_eta_inPtRange,
            ele_phi_inPtRange,
            ele_counts_inPtRange,
            'reco',
            pt_range_key,
        )
        
        sf_reco.append(sf_reco_inPtRange)
        sfup_reco.append(sfup_reco_inPtRange)
        sfdown_reco.append(sfdown_reco_inPtRange)

    sf_reco = ak.prod(
        ak.concatenate(sf_reco, axis=1), axis=1
    )
    sfup_reco = ak.prod(
        ak.concatenate(sfup_reco, axis=1), axis=1
    )
    sfdown_reco = ak.prod(
        ak.concatenate(sfdown_reco, axis=1), axis=1
    )

    return sf_reco, sfup_reco, sfdown_reco


def sf_ele_reco_func(params, metadata, events, size, shape_variations):
    if metadata["year"] in ["2024"]:
      sf = sf_ele_reco(params, events, "2024Reco")
    else:
      sf = sf_ele_reco(params, events, metadata["year"])
    return sf


def sf_ele_id_func(params, metadata, events, size, shape_variations):
    """
    Function to apply electron ID scale factors.
    
    Parameters:
    - params: Configuration parameters containing scale factor definitions.
    - events: Events data containing electron information.
    - year: Year of the data (e.g., 2016, 2017, 2018).
    
    Returns:
    - Scale factors for electron ID.
    """
    # Get the scale factor for electron ID based on the year
    if metadata["year"] in ["2016_PreVFP", "2016_PostVFP", "2017", "2018"]:
        params["object_preselection"]["Electron"]["id"] = params["object_preselection"]["Electron"]["id1"]
    elif metadata["year"] in ["2022_preEE", "2022_postEE", "2023_preBPix", "2023_postBPix", "2024"]:
        params["object_preselection"]["Electron"]["id"] = params["object_preselection"]["Electron"]["id2"]
    
    # Apply the scale factor to the electrons in the events
    sf = sf_ele_id(params, events, metadata["year"])

    return sf


def sf_ele_trigger_func(params, metadata, events, size, shape_variations):
    """
    Function to apply electron trigger scale factors.
    
    Parameters:
    - params: Configuration parameters containing scale factor definitions.
    - events: Events data containing electron information.
    
    Returns:
    - Scale factors for electron trigger.
    """
    # Get the scale factor for electron trigger based on the year
    if metadata["year"] in ["2016_PreVFP", "2016_PostVFP", "2017", "2018"]:
      #sf = sf_ele_trigger_run2(params, events, metadata["year"])
      sf = np.ones(len(events)) # Needs to be fixed, as the run2 scale factors are not available in the new format
    elif metadata["year"] in ["2022_preEE", "2022_postEE", "2023_preBPix", "2023_postBPix", "2024"]:
      sf = sf_ele_trigger_run3(params, events, metadata["year"])
    else:
      sf = np.ones(len(events))

    return sf


sf_ele_id_custom = WeightLambda.wrap_func(
    name="sf_ele_id_custom",
    function=sf_ele_id_func,
    has_variations=True
    )

sf_ele_trigger_custom = WeightLambda.wrap_func(
    name="sf_ele_trigger_custom",
    function=sf_ele_trigger_func,
    has_variations=True
)

sf_ele_reco_custom = WeightLambda.wrap_func(
    name="sf_ele_reco_custom",
    function=sf_ele_reco_func,
    has_variations=True
)

SF_ele_custom = [
  sf_ele_id_custom,
  sf_ele_trigger_custom,
  sf_ele_reco_custom,
]