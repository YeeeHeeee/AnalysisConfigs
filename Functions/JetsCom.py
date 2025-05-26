import copy
import importlib
import gzip
import cloudpickle

import awkward as ak
import numpy as np
import correctionlib
from coffea.jetmet_tools import  CorrectedMETFactory

def get_dijet(jets, taggerVars=True):
    if isinstance(taggerVars, str):
        raise NotImplementedError(
            f"Using the tagger name while calling `get_dijet` is deprecated. "
            f"Please use `jet_tagger={taggerVars}` as an argument to `jet_selection`."
        )

    fields = {
        "pt": 0.,
        "eta": 0.,
        "phi": 0.,
        "mass": 0.,
    }

    # Pad to at least 2 jets per event
    jets = ak.pad_none(jets, 2)

    njet = ak.num(jets[~ak.is_none(jets, axis=1)])

    # Form dijet 4-vector sum from leading two jets
    dijet = jets[:, 0] + jets[:, 1]

    for var in fields.keys():
        fields[var] = ak.where(
            (njet >= 2),
            getattr(dijet, var),
            fields[var]
        )

    # # Angular differences and per-jet variables
    # fields["deltaR"] = ak.where((njet >= 2), jets[:, 0].delta_r(jets[:, 1]), -1)
    # fields["deltaPhi"] = ak.where((njet >= 2), abs(jets[:, 0].delta_phi(jets[:, 1])), -1)
    # fields["deltaEta"] = ak.where((njet >= 2), abs(jets[:, 0].eta - jets[:, 1].eta), -1)
    # fields["j1Phi"] = ak.where((njet >= 2), jets[:, 0].phi, -1)
    # fields["j2Phi"] = ak.where((njet >= 2), jets[:, 1].phi, -1)
    # fields["j1pt"] = ak.where((njet >= 2), jets[:, 0].pt, -1)
    # fields["j2pt"] = ak.where((njet >= 2), jets[:, 1].pt, -1)

    # if "jetId" in jets.fields and taggerVars:
    #     fields["j1CvsL"] = ak.where((njet >= 2), jets[:, 0]["btagCvL"], -1)
    #     fields["j2CvsL"] = ak.where((njet >= 2), jets[:, 1]["btagCvL"], -1)
    #     fields["j1CvsB"] = ak.where((njet >= 2), jets[:, 0]["btagCvB"], -1)
    #     fields["j2CvsB"] = ak.where((njet >= 2), jets[:, 1]["btagCvB"], -1)

    # Main output
    dijet = ak.zip(fields, with_name="PtEtaPhiMCandidate")
    # # Return the pair of leading jets as second output
    # leading_pair = ak.concatenate([jets[:, 0:1], jets[:, 1:2]], axis=1)
    return dijet

def combine_jets(jet1, jet2):
    fields = {
        "pt": 0.,
        "eta": 0.,
        "phi": 0.,
        "mass": 0.,
    }

    # Combine 1 bjet + 2 non-bjets:
    combined = ak.pad_none(ak.concatenate([jet1[:,0:1], jet2[:,0:1]], axis=1), 2)

    # Require all 2 jets to exist
    num = ak.num(combined[~ak.is_none(combined, axis=1)])

    # Sum of jets:
    comb = combined[:,0] + combined[:,1]
    # + combined[:,2]

    for var in fields.keys():
        fields[var] = ak.where(
            (num>=2),
            getattr(comb, var),
            fields[var]
        )
    comb = ak.zip(fields, with_name="PtEtaPhiMCandidate")

    return comb

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

def to_singleton_jet(jets):
    """
    Converts a flat Awkward Array of jet records into a nested structure of singleton Jet objects.
    """
    transformed_jets = ak.zip(
        {
            "pt": jets["pt"],
            "eta": jets["eta"],
            "phi": jets["phi"],
            "mass": jets["mass"]
        },
        with_name="Jet"     
    )
    return transformed_jets[:, None]
