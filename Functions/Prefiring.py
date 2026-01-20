from pocket_coffea.lib.weights import WeightLambda
import numpy as np
import awkward as ak
from pocket_coffea.lib.scale_factors import sf_L1prefiring


def prefiring_func(params, metadata, events, size, shape_variations):
    """
    Function to apply L1 prefiring scale factors.
    
    Parameters:
    - params: Configuration parameters containing scale factor definitions.
    - events: Events data containing L1 prefiring information.
    
    Returns:
    - Scale factors for L1 prefiring.
    """
    # Get the scale factor for L1 prefiring based on the year
    if metadata["year"] in ["2016_PreVFP", "2016_PostVFP", "2017"]:
      sf = sf_L1prefiring(events)
    else:
      sf = np.ones(len(events))  # No prefiring correction for these years
    return sf

prefiring = WeightLambda.wrap_func(
    name="prefiring",
    function=prefiring_func,
    has_variations=True
)

Prefiring = [prefiring]