import awkward as ak
import numpy as np

from pocket_coffea.workflows.base import BaseProcessorABC
from pocket_coffea.utils.configurator import Configurator

class TTStitchingWorkflow(BaseProcessorABC):
    def __init__(self, cfg: Configurator):
        super().__init__(cfg)


    def apply_object_preselection(self, variation):

      GenTopFirstCopy = self.events["GenPart"][((np.abs(self.events["GenPart"].pdgId) == 6) & ((self.events["GenPart"].statusFlags & (1 << 12)) > 0))]
      top_pairs = ak.combinations(GenTopFirstCopy, 2, fields=["left", "right"])
      self.events["GenTT"] = ak.firsts(ak.zip({"mass": (top_pairs.left + top_pairs.right).mass}, with_name="PtEtaPhiMCandidate"))

    def define_common_variables_after_presel(self, variation):
      pass


    def count_objects(self, variation):
      pass
   

    def define_common_variables_before_presel(self, variation):
      pass


