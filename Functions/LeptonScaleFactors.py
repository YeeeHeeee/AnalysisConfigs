from pocket_coffea.lib.weights import WeightLambda
import numpy as np
import awkward as ak
from pocket_coffea.lib.scale_factors import sf_ele_id

from pocket_coffea.lib.weights.common.weights_run2_UL import sf_ele_trigger as sf_ele_trigger_run2
from pocket_coffea.lib.scale_factors import sf_ele_trigger as sf_ele_trigger_run3


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

SF_ele_custom = [
  sf_ele_id_custom,
  sf_ele_trigger_custom
]