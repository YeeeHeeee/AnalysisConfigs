import awkward as ak
import numpy as np

#export PYTHONPATH=..:$PYTHONPATH
from pocket_coffea.workflows.base import BaseProcessorABC
from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.hist_manager import Axis
from pocket_coffea.lib.deltaR_matching import metric_eta, metric_phi
from pocket_coffea.lib.deltaR_matching import object_matching
from pocket_coffea.lib.gen_objects import getGenJets, getGenLeptons

from pocket_coffea.lib.objects import (
    jet_correction,
    jet_selection,
    btagging,
    get_dilepton,
    get_dijet,
    met_xy_correction,
)

from Functions.JetsCom import to_singleton_jet, combine_jets
from Functions.Leptons import lepton_selection

class ttBaseProcessor_merge(BaseProcessorABC):
    def __init__(self, cfg: Configurator):
        super().__init__(cfg)


    def apply_object_preselection(self, variation):
        # Avoid code duplicate
        super().apply_object_preselection(variation=variation)
        
###########################################################################
        # # MET:
        # met_pt_corr, met_phi_corr = met_xy_correction(self.params, self.events, self._year, self._era)
        # self.events["MET"] = ak.with_field(
        #     self.events.MET, met_pt_corr, "pt"
        # )
        # self.events["MET"] = ak.with_field(
        #     self.events.MET, met_phi_corr, "phi"
        # )

###########################################################################        
        # Leptons:
        # Include the supercluster pseudorapidity variable
        electron_etaSC = self.events.Electron.eta + self.events.Electron.deltaEtaSC
        self.events["Electron"] = ak.with_field(
            self.events.Electron, electron_etaSC, "etaSC"
        )
        # Build masks for selection of muons, electrons, jets, fatjets
        self.events["MuonGood"] = lepton_selection(
            self.events, "Muon", self.params, self._year
        )
        self.events["ElectronGood"] = lepton_selection(
            self.events, "Electron", self.params, self._year
        )
        leptons = ak.with_name(
            ak.concatenate((self.events.MuonGood, self.events.ElectronGood), axis=1),
            name='PtEtaPhiMCandidate',
        )
        self.events["LeptonGood"] = leptons[ak.argsort(leptons.pt, ascending=False)]
        self.events["LeptonSave"] = ak.firsts(self.events["LeptonGood"])

        self.events["ll"] = get_dilepton(
            self.events.ElectronGood, self.events.MuonGood
        )

###########################################################################
        # AK4 Jets:
        self.events["JetGood"], self.jetGoodMask = jet_selection(
            self.events, "Jet", self.params, 
            year=self._year, 
            leptons_collection="LeptonGood"
        )
        self.events["BJetGood"] = btagging(
            self.events["JetGood"], self.params.btagging.working_point[self._year], wp=self.params.object_preselection.Jet.btag.wp)
        
        self.events["BJetBad"] = btagging(
            self.events["JetGood"], self.params.btagging.working_point[self._year], wp=self.params.object_preselection.Jet.btag.wp, veto=True)
###########################################################################
        # AK8 Jets:
        self.events["FatJetGood"], self.fatjetGoodMask = jet_selection(
            self.events, "FatJet", self.params,
            year=self._year,
            leptons_collection="LeptonGood" # used for cleaning jets by removing thoes that overlap with leptons in an events.
        )
        self.events["FatJet"] = ak.firsts(self.events["FatJetGood"])
        
        self.events["SubJetGood1"] = self.events.SubJet[self.events["FatJetGood"].subJetIdx1]
        self.events["SubJetGood2"] = self.events.SubJet[self.events["FatJetGood"].subJetIdx2]

        # combine two subjet for validation
        self.events["CombinedSubJets"] = combine_jets(
            self.events["SubJetGood1"], self.events["SubJetGood2"]
        )

        self.events["SubJet1"] = ak.firsts(self.events["SubJetGood1"])
        self.events["SubJet2"] = ak.firsts(self.events["SubJetGood2"])
    
###########################################################################
        # Select GenJetAK8 jets matched to top or anti-top quarks:
        if self._isMC:
            self.events["GenTop_AK8"] = ak.firsts(self.events["GenJetAK8"])

    def define_common_variables_after_presel(self, variation):

 ###########################################################################
        # Change data type:
       fatjet = to_singleton_jet(self.events["FatJet"])
       GenTop_AK8 = to_singleton_jet(self.events["GenTop_AK8"])

 ########################################################################### 
        # Match the AK8 to the GenJetAK8 jets and GenPart objects:
       self.events["MatchedTop_AK81"], self.events["MatchedGenTop_AK8"], deltaR_padnon = object_matching(
            fatjet, GenTop_AK8, dr_min = 0.8)  
       self.events["MatchedTop_AK8"] = ak.firsts(self.events["MatchedTop_AK81"])
    #    self.events["MatchedTop_Part"] = ak.firsts(self.events["MatchedTop_Part1"])

    def count_objects(self, variation):
        self.events["nMuonGood"] = ak.num(self.events.MuonGood)
        self.events["nElectronGood"] = ak.num(self.events.ElectronGood)
        self.events["nLeptonGood"] = (
            self.events["nMuonGood"] + self.events["nElectronGood"]
        )
        self.events["nJetGood"] = ak.num(self.events.JetGood)
        self.events["nBJetGood"] = ak.num(self.events.BJetGood)
        self.events["nBJetBad"] = ak.num(self.events.BJetBad)  
        self.events["nFatJet"] = ak.num(self.events["FatJetGood"])
   
    # Function that defines common variables employed in analyses and save them as attributes of `events`
    def define_common_variables_before_presel(self, variation):
        self.events["JetGood_Ht"] = ak.sum(abs(self.events.JetGood.pt), axis=1)



