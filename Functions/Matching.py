from collections.abc import Iterable
import numpy as np
import awkward as ak
import numba

@numba.njit
def get_matching_pairs_indices(idx_1, idx_2, builder, builder2):
    for ev_q, ev_j in zip(idx_1, idx_2):
        builder.begin_list()
        builder2.begin_list()
        q_done = []
        j_done = []
        for i, (q, j) in enumerate(zip(ev_q, ev_j)):
            if q not in q_done:
                if j not in j_done:
                    builder.append(i)
                    q_done.append(q)
                    j_done.append(j)
                else:
                    builder2.append(i)
        builder.end_list()
        builder2.end_list()
    return builder, builder2


# This function takes as arguments the indices of two collections of objects that have been
# previously matched. The idx_matched_obj2 indices are supposed to be ordered but they can have missing elements.
# The idx_matched_obj indices are matched to the obj2 ones and no order is required on them.
# The function return an array of the dimension of the maxdim_obj2 (akward dimensions) with the indices of idx_matched_obk
# matched to the elements in idx_matched_obj2. None values are included where
# no match has been found.
@numba.njit
def get_matching_objects_indices_padnone(
    idx_matched_obj, idx_matched_obj2, maxdim_obj2, deltaR, builder, builder2, builder3
):
    for ev1_match, ev2_match, nobj2, dr in zip(
        idx_matched_obj, idx_matched_obj2, maxdim_obj2, deltaR
    ):
        # print(ev1_match, ev2_match)
        builder.begin_list()
        builder2.begin_list()
        builder3.begin_list()
        row1_length = len(ev1_match)
        missed = 0
        for i in range(nobj2):
            # looping on the max dimension of collection 2 and checking if the current index i
            # is matched, e.g is part of ev2_match vector.
            if i in ev2_match:
                # if this index is matched, then take the ev1_match and deltaR results
                # print(i, row1_length)
                builder2.append(i)
                if i - missed < row1_length:
                    builder.append(ev1_match[i - missed])
                    builder3.append(dr[i - missed])
            else:
                # If it is missing a None is added and the missed  is incremented
                # so that the next matched one will get the correct element assigned.
                builder.append(None)
                builder2.append(None)
                builder3.append(None)
                missed += 1
        builder.end_list()
        builder2.end_list()
        builder3.end_list()
    return builder, builder2, builder3


def metric_pt(obj, obj2):
    return abs(obj.pt - obj2.pt)

def metric_eta(obj, obj2):
    return abs(obj.eta - obj2.eta)

def delta_phi(a, b):
    """Compute difference in between two phi values, modulo 2pi
    Returns a value within [-pi, pi)
    """
    return (a - b + np.pi) % (2 * np.pi) - np.pi

def metric_phi(obj, obj2):
    return abs(delta_phi(obj.phi,obj2.phi))


def object_matching1(obj, obj2, dr_min, dpt_max=None, return_indices=False):
    """
    Match objects in `obj` (e.g., RecoJets) to objects in `obj2` (e.g., GenJets) using deltaR and optional dPt matching.
    Ensures one-to-one matching: each obj2 (GenJet) is matched to at most one obj (RecoJet).
    
    Parameters:
    -----------
    obj : awkward.Array
        The reconstructed objects (e.g., RecoJets) to be matched.
    obj2 : awkward.Array
        The generator-level objects (e.g., GenJets) to match to.
    dr_min : float
        Maximum allowed deltaR for a valid match.
    dpt_max : float or iterable, optional
        Maximum allowed relative pT difference for a valid match.
    return_indices : bool, optional
        If True, return indices and deltaR info as well.

    Returns:
    --------
    matched_obj : awkward.Array
        RecoJets matched to GenJets.
    matched_obj2 : awkward.Array
        GenJets matched to RecoJets.
    deltaR_sorted : awkward.Array
        deltaR between matched pairs.
    idx_matched_obj (optional) : indices of matched obj
    idx_matched_obj2 (optional) : indices of matched obj2
    """
    # Compute deltaR between all pairs
    deltaR = ak.flatten(obj.metric_table(obj2), axis=2)
    maskDR = deltaR < dr_min  # Apply deltaR cut

    # Optional: apply relative pT cut
    if dpt_max is not None:
        deltaPt_table = obj.metric_table(obj2, metric=metric_pt)
        if isinstance(dpt_max, Iterable):
            dpt_max_broadcast = ak.broadcast_arrays(
                dpt_max[:, np.newaxis], deltaPt_table
            )[0]
            dpt_max = ak.flatten(dpt_max_broadcast, axis=2)
        deltaPt = ak.flatten(deltaPt_table, axis=2)
        maskPt = deltaPt < dpt_max
        maskDR = maskDR & maskPt  # Combine deltaR and dPt masks

    # Sort all deltaR values for closest-first matching
    idx_pairs_sorted = ak.argsort(deltaR, axis=1)
    pairs = ak.argcartesian([obj, obj2])
    pairs_sorted = pairs[idx_pairs_sorted]
    deltaR_sorted = deltaR[idx_pairs_sorted]
    maskDR_sorted = maskDR[idx_pairs_sorted]
    idx_obj, idx_obj2 = ak.unzip(pairs_sorted)

    # Track which GenJets (obj2) have been matched
    matched_genjets = ak.Array([False] * ak.num(obj2))

    matched_pairs = []
    for i in range(ak.num(deltaR_sorted)):
        if maskDR_sorted[i]:
            if not matched_genjets[idx_obj2[i]]:
                matched_pairs.append((idx_obj[i], idx_obj2[i]))
                matched_genjets[idx_obj2[i]] = True  # Mark GenJet as matched

    # Separate indices of matched pairs
    if len(matched_pairs) == 0:
        # Handle empty match case to avoid ValueError in zip
        idx_matched_obj, idx_matched_obj2 = ak.Array([]), ak.Array([])
    else:
        idx_matched_obj, idx_matched_obj2 = zip(*matched_pairs)
        idx_matched_obj = ak.Array(idx_matched_obj)
        idx_matched_obj2 = ak.Array(idx_matched_obj2)

    # Get matched objects
    matched_obj = obj[idx_matched_obj]
    matched_obj2 = obj2[idx_matched_obj2]

    if return_indices:
        return matched_obj, matched_obj2, deltaR_sorted, idx_matched_obj, idx_matched_obj2
    else:
        return matched_obj, matched_obj2, deltaR_sorted

        
##################################################################3
# Not unique deltaR matching

def deltaR_matching_nonunique(obj1, obj2, radius=0.4):  # NxM , NxG arrays
    '''
    Doing this you can keep the assignment on the obj2 collection unique,
    but you are not checking the uniqueness of the matching to the first collection.
    '''
    _, obj2 = ak.unzip(ak.cartesian([obj1, obj2], nested=True))  # Obj2 is now NxMxG
    obj2['dR'] = obj1.delta_r(obj2)  # Calculating delta R
    t_index = ak.argmin(obj2.dR, axis=-2)  # Finding the smallest dR (NxG array)
    s_index = ak.local_index(obj1.eta, axis=-1)  #  NxM array
    _, t_index = ak.unzip(ak.cartesian([s_index, t_index], nested=True))
    obj2 = obj2[s_index == t_index]  # Pairwise comparison to keep smallest delta R
    # Cutting on delta R
    obj2 = obj2[obj2.dR < radius]  # Additional cut on delta R, now a NxMxG' array
    return obj2
