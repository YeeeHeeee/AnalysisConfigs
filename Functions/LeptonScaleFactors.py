from pocket_coffea.lib.weights import WeightLambda
import numpy as np
import awkward as ak
import correctionlib

from pocket_coffea.lib.scale_factors import get_ele_sf

def sf_ele_id(params, events, year):
    '''
    This function computes the per-electron id SF and returns the corresponding per-event SF, obtained by multiplying the per-electron SF in each event.
    Additionally, also the up and down variations of the SF are returned.
    '''
    coll = params.lepton_scale_factors.electron_sf.collection
    ele_pt = events[coll].pt
    ele_eta = events[coll].etaSC
    ele_phi = events[coll].phi

    ele_pt = ak.where(ele_pt > 1000, 999.9, ele_pt)
    ele_pt = ak.where(ele_pt < 20, 20.1, ele_pt)
    ele_eta = ak.where(ele_eta > 2.5, 2.499, ele_eta)
    ele_eta = ak.where(ele_eta < -2.5, -2.499, ele_eta)

    ele_pt_flat, ele_eta_flat, ele_phi_flat, ele_counts = (
        ak.flatten(ele_pt),
        ak.flatten(ele_eta),
        ak.flatten(ele_phi),
        ak.num(ele_pt),
    )

    sf_id, sfup_id, sfdown_id = get_ele_sf(
        params, year, ele_pt_flat, ele_eta_flat, ele_phi_flat, ele_counts, 'id'
    )
        

    # The SF arrays corresponding to the electrons are multiplied along the electron axis in order to obtain a per-event scale factor.
    return ak.prod(sf_id, axis=1), ak.prod(sfup_id, axis=1), ak.prod(sfdown_id, axis=1)

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
    elif year in ["2022_preEE", "2022_postEE", "2023_preBPix", "2023_postBPix"]:
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


def sf_ele_trigger_run2(params, events, year):

    coll = "ElectronGood"
    ele_pt = events[coll].pt
    ele_eta = events[coll].etaSC

    ele_pt_flat, ele_eta_flat, ele_counts = (
        ak.flatten(ele_pt).to_numpy(),
        ak.flatten(ele_eta).to_numpy(),
        ak.num(ele_pt),
    )

    sf_flat = np.ones_like(ele_pt_flat, dtype=float)
    uncert_flat = np.zeros_like(ele_pt_flat, dtype=float)

    if year in ["2016_PreVFP", "2016_PostVFP"]:
        year_name = "2016"
    else:
        year_name = year

    pt_bins  = params["electron_trigger_run2"][year_name]["boundaries"]["pt"]
    eta_bins = params["electron_trigger_run2"][year_name]["boundaries"]["eta"]
    sf_table = params["electron_trigger_run2"][year_name]["sf"]
    uncert_table = params["electron_trigger_run2"][year_name]["uncert"]

    for pt_ind, pt_bin in enumerate(pt_bins[::-1]):
        pt_min, pt_max = pt_bin[0], pt_bin[1]
        for eta_ind, eta_bin in enumerate(eta_bins):
            eta_min, eta_max = eta_bin[0], eta_bin[1]
            sf_val = sf_table[pt_ind][eta_ind]
            mask = (
                (ele_pt_flat >= pt_min) & (ele_pt_flat < pt_max) &
                (ele_eta_flat >= eta_min) & (ele_eta_flat < eta_max)
            )
            sf_flat[mask] = sf_val
            uncert_flat[mask] = uncert_table[pt_ind][eta_ind]

    sfup_flat = sf_flat + uncert_flat
    sfdown_flat = sf_flat - uncert_flat

    sf = ak.unflatten(sf_flat, ele_counts)
    sfup = ak.unflatten(sfup_flat, ele_counts)
    sfdown = ak.unflatten(sfdown_flat, ele_counts)

    return (
        ak.prod(sf, axis=1),
        ak.prod(sfup, axis=1),
        ak.prod(sfdown, axis=1),
    )


def sf_ele_trigger(params, events, year):
    """Compute electron trigger scale factors using the EGM JSON files with correctionlib.
    Returns the per-event scale factor for the trigger.

    Returns:
    --------
    tuple: (sf, sfup, sfdown) per-event scale factor
    """
    electronSF = params.lepton_scale_factors.electron_sf
    year_pog = electronSF.era_mapping[year]
    map_name = electronSF.trigger_sf[year].name
    trigger_path = electronSF.trigger_sf[year].path

    coll = electronSF.collection
    ele_pt = events[coll].pt
    ele_eta = events[coll].etaSC

    ele_pt_flat, ele_eta_flat, ele_counts = (
        ak.flatten(ele_pt).to_numpy(),
        ak.flatten(ele_eta).to_numpy(),
        ak.num(ele_pt),
    )

    electron_correctionset = correctionlib.CorrectionSet.from_file(
        electronSF.trigger_sf[year].file
    )
    corr_eval = electron_correctionset[map_name].evaluate

    # get sf, sfup, sfdown per electron
    scale_factors = [
        ak.unflatten(
            corr_eval(year_pog, variation, trigger_path, ele_eta_flat, ele_pt_flat),
            ele_counts,
        )
        for variation in ("sf", "sfup", "sfdown")
    ]

    # return a per-event scale factor by multiplying all electron scale factors
    return tuple(ak.prod(sf, axis=1) for sf in scale_factors)


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
    if metadata["year"] in ["2016_PreVFP", "2016_PostVFP", "2017", "2018"]: # Need to be custom as no jsonpog format
        sf = sf_ele_trigger_run2(params, events, metadata["year"])
    elif metadata["year"] in ["2022_preEE", "2022_postEE", "2023_preBPix", "2023_postBPix"]: # 2024 not available yet
        sf = sf_ele_trigger(params, events, metadata["year"])
    else:
      sf = np.ones(len(events))

    return sf


def get_mu_sf(params, year, pt, eta, counts, key=''):
    '''
    This function computes the per-muon id or iso SF.
    '''
    muonSF = params["lepton_scale_factors"]["muon_sf"]

    muon_correctionset = correctionlib.CorrectionSet.from_file(
        muonSF.JSONfiles[year]['file']
    )

    if key not in ["id","iso","trigger"]:
        raise Exception(f"Muon SF key {key} not recognized")
    
    sfName = muonSF.sf_name[year][key]
    
    sf = muon_correctionset[sfName].evaluate(
        np.abs(eta.to_numpy()), pt.to_numpy(), "nominal"
    )
    sfup = muon_correctionset[sfName].evaluate(
        np.abs(eta.to_numpy()), pt.to_numpy(), "systup"
    )
    sfdown = muon_correctionset[sfName].evaluate(
        np.abs(eta.to_numpy()), pt.to_numpy(), "systdown"
    )
    
    # The unflattened arrays are returned in order to have one row per event.
    return (
        ak.unflatten(sf, counts),
        ak.unflatten(sfup, counts),
        ak.unflatten(sfdown, counts),
    )


def sf_mu(params, events, year, key=''):
    '''
    This function computes the per-muon id SF and returns the corresponding per-event SF, obtained by multiplying the per-muon SF in each event.
    Additionally, also the up and down variations of the SF are returned.
    '''
    coll = params.lepton_scale_factors.muon_sf.collection
    mu_pt = events[coll].pt
    mu_eta = events[coll].eta

    # Since `correctionlib` does not support jagged arrays as an input, the pt and eta arrays are flattened.
    mu_pt_flat, mu_eta_flat, mu_counts = (
        ak.flatten(mu_pt),
        ak.flatten(mu_eta),
        ak.num(mu_pt),
    )
    sf, sfup, sfdown = get_mu_sf(params, year, mu_pt_flat, mu_eta_flat, mu_counts, key)

    # The SF arrays corresponding to all the muons are multiplied along the
    # muon axis in order to obtain a per-event scale factor.
    return ak.prod(sf, axis=1), ak.prod(sfup, axis=1), ak.prod(sfdown, axis=1)


def sf_mu_id_func(params, metadata, events, size, shape_variations):
    return sf_mu(params, events, metadata["year"], key='id')

def sf_mu_iso_func(params, metadata, events, size, shape_variations):
    return sf_mu(params, events, metadata["year"], key='iso')

def sf_mu_trigger_func(params, metadata, events, size, shape_variations):
    return sf_mu(params, events, metadata["year"], key='trigger')

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

sf_mu_id_custom = WeightLambda.wrap_func(
    name="sf_mu_id_custom",
    function=sf_mu_id_func,
    has_variations=True
)

sf_mu_iso_custom = WeightLambda.wrap_func(
    name="sf_mu_iso_custom",
    function=sf_mu_iso_func,
    has_variations=True
)

sf_mu_trigger_custom = WeightLambda.wrap_func(
    name="sf_mu_trigger_custom",
    function=sf_mu_trigger_func,
    has_variations=True
)

SF_ele_custom = [
    sf_ele_id_custom,
    sf_ele_trigger_custom,
    sf_ele_reco_custom,
]

SF_mu_custom = [
    sf_mu_id_custom,
    sf_mu_iso_custom,
    sf_mu_trigger_custom,
]
