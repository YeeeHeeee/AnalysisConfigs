import copy
import importlib
import gzip
import cloudpickle

import awkward as ak
import numpy as np
import correctionlib
from coffea.jetmet_tools import  CorrectedMETFactory

def get_dijet(jets, taggerVars=True):
    '''
    Constructs a dijet candidate from the pair of non b-tagged jets in each event as W,
    that are closest in ΔR (delta-R).

    Parameters
    ----------
    jets : ak.Array
        An array of jets (either reco or gen), expected to have pt, eta, phi, mass, etc.
    
    taggerVars : bool
        If True and jets have b-tagging scores, include those in the output.
        If a string is passed (deprecated), raise an error.

    Returns
    -------
    ak.Array
        A record array (PtEtaPhiMCandidate) of dijet variables constructed 
        from the closest pair of non b-tagged jets in ΔR per event.
    '''

    # Reject deprecated usage where taggerVars is a string (likely a tagger name)
    if isinstance(taggerVars, str):
        raise NotImplementedError(f"Using the tagger name while calling `get_dijet` is deprecated. Please use `jet_tagger={taggerVars}` as an argument to `jet_selection`.")

    # Initialize default values for output fields
    fields = {
        "pt": 0.,
        "eta": 0.,
        "phi": 0.,
        "mass": 0.,
    }

    # Create all unique jet pairs in each event
    pairs = ak.combinations(jets, 2, fields=["j1", "j2"])

    # Compute ΔR between jet pairs
    deltaR = pairs.j1.delta_r(pairs.j2)

    # Find the index of the pair with minimum ΔR in each event
    min_idx = ak.argmin(deltaR, axis=1)

    # Select the closest pair using the minimum ΔR indices
    closest_pairs = pairs[min_idx]

    # Compute 4-vector sum of the closest pair
    dijet = closest_pairs.j1 + closest_pairs.j2

    # Number of jets per event, used to guard against events with <2 jets
    njet = ak.num(jets)

    # Fill kinematic fields if ≥2 jets, else default to zeros
    for var in fields.keys():
        fields[var] = ak.where(
            (njet >= 2),
            getattr(dijet, var),
            fields[var]
        )

    # Add ΔR, Δη, ΔΦ between jet1 and jet2 in the closest pair
    fields["deltaR"] = ak.where((njet >= 2), closest_pairs.j1.delta_r(closest_pairs.j2), -1)
    fields["deltaPhi"] = ak.where((njet >= 2), abs(closest_pairs.j1.delta_phi(closest_pairs.j2)), -1)
    fields["deltaEta"] = ak.where((njet >= 2), abs(closest_pairs.j1.eta - closest_pairs.j2.eta), -1)

    # Add jet-by-jet info
    fields["j1Phi"] = ak.where((njet >= 2), closest_pairs.j1.phi, -1)
    fields["j2Phi"] = ak.where((njet >= 2), closest_pairs.j2.phi, -1)
    fields["j1pt"] = ak.where((njet >= 2), closest_pairs.j1.pt, -1)
    fields["j2pt"] = ak.where((njet >= 2), closest_pairs.j2.pt, -1)

    # If b-tagging info exists (e.g. reco jets), add them
    if "jetId" in jets.fields and taggerVars:
        fields["j1CvsL"] = ak.where((njet >= 2), closest_pairs.j1["btagCvL"], -1)
        fields["j2CvsL"] = ak.where((njet >= 2), closest_pairs.j2["btagCvL"], -1)
        fields["j1CvsB"] = ak.where((njet >= 2), closest_pairs.j1["btagCvB"], -1)
        fields["j2CvsB"] = ak.where((njet >= 2), closest_pairs.j2["btagCvB"], -1)

    # Bundle the outputs into an awkward record array
    dijet = ak.zip(fields, with_name="PtEtaPhiMCandidate")
    
    return dijet

# def combine_jets(*jets):
#     fields = {
#         "pt": 0.,
#         "eta": 0.,
#         "phi": 0.,
#         "mass": 0.,
#     }

#     # Ensure jets are valid, non-empty, and have the same depth
#     jets = [jet for jet in jets if ak.num(jet) > 0]


#     # Extract first jet per event while keeping structure
#     jets = [ak.pad_none(jet[:, 0:1], 1) if ak.num(jet) > 0 else jet for jet in jets]

#     # Debugging: print structure before concatenation
#     print(ak.type(jets[0]) if jets else "No valid jets")

#     # Concatenate along axis=1
#     combined = ak.pad_none(ak.concatenate(jets, axis=1), len(jets), clip=True)

#     # Ensure num has the right shape
#     num = ak.num(combined, axis=1)

#     # Update fields safely
#     for var in fields.keys():
#         if hasattr(combined, var):
#             fields[var] = ak.where((num >= 2), getattr(combined, var), fields[var])

#     return ak.zip(fields, with_name="PtEtaPhiMCandidate")

def bjj_deltaR(bjets, dijet):
    '''
    Reconstructs a top quark candidate by combining a dijet system (W candidate)
    with the closest b-tagged jet in ΔR.

    Parameters
    ----------
    bjets : ak.Array
        Array of b-tagged jets (assumed to be at least 2 per event).
    
    dijet : ak.Array
        Dijet candidate, typically the W boson candidate (from get_dijet).

    Returns
    -------
    ak.Array
        Four-vector sum of the dijet and the closest b-jet, interpreted as the
        reconstructed top quark candidate.
    '''
    
    # Form all unique unordered pairs of bjets per event
    paris = ak.argcombinations(bjets, 2, axis=1)
    b1 = bjets[paris.slot0]
    b2 = bjets[paris.slot1]

    # Compute ΔR between dijet and each b-jet in the pair
    deltaR_b_to_jj_1 = b1.delta_r(dijet)
    deltaR_b_to_jj_2 = b2.delta_r(dijet)

    # Select the b-jet that is closest to the dijet in ΔR
    min_b_to_jj = ak.where(deltaR_b_to_jj_1 < deltaR_b_to_jj_2, b1, b2)

    # Combine the closest b-jet with the dijet to form a top candidate
    bjj = combine_jets(min_b_to_jj, dijet)

    return bjj

def bjj_deltaM(bjets, dijet):
    """
    Reconstructs a top quark candidate by combining the dijet (W boson candidate)
    with one of two b-jets. Selects the combination whose mass is:
      - Closest to the top mass window [170, 180] GeV
      - In case of tie, closest to the nominal top mass (172.52 GeV)

    Parameters
    ----------
    bjets : ak.Array
        Array of b-tagged jets (at least two per event).

    dijet : ak.Array
        Dijet (W candidate) to be combined with each b-jet.

    Returns
    -------
    ak.Array
        Top candidate (bjj) four-vector with best mass match.
    """
    # Form all unique unordered pairs of bjets per event
    paris = ak.argcombinations(bjets, 2, axis=1)
    b1 = bjets[paris.slot0]
    b2 = bjets[paris.slot1]

    # Combine dijet with each b-jet
    bjj_1 = combine_jets(b1, dijet)
    bjj_2 = combine_jets(b2, dijet)

    # Define mass window
    target_low, target_high = 170, 180
    top_nominal = 172.52

    # Distance to [170, 180] window
    def dist_to_range(mass):
        return ak.where(
            mass < target_low, target_low - mass,
            ak.where(mass > target_high, mass - target_high, 0)
        )

    dist_1 = dist_to_range(bjj_1.mass)
    dist_2 = dist_to_range(bjj_2.mass)

    # Tie-breaker: distance to nominal top mass
    tie_1 = abs(bjj_1.mass - top_nominal)
    tie_2 = abs(bjj_2.mass - top_nominal)

    # First compare range distance, then use tie-breaker if equal
    use_1 = (dist_1 < dist_2) | ((dist_1 == dist_2) & (tie_1 < tie_2))

    bjj = ak.where(use_1, bjj_1, bjj_2)
    return bjj





