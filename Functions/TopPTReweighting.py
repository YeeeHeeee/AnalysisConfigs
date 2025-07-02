from pocket_coffea.lib.weights import WeightLambda
import numpy as np
import awkward as ak

def top_pt_reweight(pt):
  return np.where(pt > 0, np.exp(0.0615 - 0.0005 * pt), 1.0)

def reweighting_func(params, metadata, events, size, shape_variations):
  if metadata["sample"].startswith("TT") or metadata["sample"].startswith("ST"):
    # find all parton level top quarks
    is_top = abs(events["GenPart"].pdgId) == 6
    is_last = (events["GenPart"].statusFlags & (1 << 13)) != 0  # isLastCopy
    top_quarks = events["GenPart"][is_top & is_last]
    per_event_weight = ak.prod(top_pt_reweight(top_quarks.pt), axis=1)
  else:
    per_event_weight = np.ones(len(events))
  return per_event_weight

top_pt_reweighting_func = WeightLambda.wrap_func(
    name="TopPTReweighting",
    function=reweighting_func,
    has_variations=False
)

TopPTReweighting = [top_pt_reweighting_func]