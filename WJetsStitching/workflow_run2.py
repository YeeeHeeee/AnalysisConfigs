import awkward as ak
import numpy as np

from pocket_coffea.workflows.base import BaseProcessorABC
from pocket_coffea.utils.configurator import Configurator

class WJetsStitchingWorkflow(BaseProcessorABC):
    def __init__(self, cfg: Configurator):
        super().__init__(cfg)


    def apply_object_preselection(self, variation):
      pass


    def define_common_variables_after_presel(self, variation):
      pass


    def count_objects(self, variation):
      pass
   

    def define_common_variables_before_presel(self, variation):
      pass


