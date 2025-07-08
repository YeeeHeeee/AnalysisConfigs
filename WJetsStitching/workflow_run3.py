import awkward as ak
import numpy as np

from pocket_coffea.workflows.base import BaseProcessorABC
from pocket_coffea.utils.configurator import Configurator

class WJetsStitchingWorkflow(BaseProcessorABC):
    def __init__(self, cfg: Configurator):
        super().__init__(cfg)

    def _get_pairs(self, arr1, arr2):

        # Cartesian product: all pairwise combinations per event
        pairs = ak.cartesian([arr1, arr2], axis=1, nested=False)
        
        # Unpack the pairs
        left, right = ak.unzip(pairs)        
        di_arr = left + right

        # Keep only the specified fields
        fields = {
            "mass": di_arr.mass,
        }

        # Zip together the fields
        out = ak.zip(fields, with_name="PtEtaPhiMCandidate")
        
        return out

    def apply_object_preselection(self, variation):
      

      l_mask = (self.events["LHEPart"].pdgId == 11) | (self.events["LHEPart"].pdgId == -11) | \
             (self.events["LHEPart"].pdgId == 13) | (self.events["LHEPart"].pdgId == -13) | \
             (self.events["LHEPart"].pdgId == 15) | (self.events["LHEPart"].pdgId == -15)
      nu_mask = (self.events["LHEPart"].pdgId == 12) | (self.events["LHEPart"].pdgId == -12) | \
             (self.events["LHEPart"].pdgId == 14) | (self.events["LHEPart"].pdgId == -14) | \
             (self.events["LHEPart"].pdgId == 16) | (self.events["LHEPart"].pdgId == -16)

      l = self.events["LHEPart"][l_mask]
      nu = self.events["LHEPart"][nu_mask]
      self.events["LNu"] = ak.firsts(self._get_pairs(l, nu))


    def define_common_variables_after_presel(self, variation):
      pass


    def count_objects(self, variation):
      pass
   

    def define_common_variables_before_presel(self, variation):
      pass


