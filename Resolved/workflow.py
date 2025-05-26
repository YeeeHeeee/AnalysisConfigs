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
    lepton_selection,
    jet_selection,
    btagging,
    get_dilepton,
    get_dijet,
    met_xy_correction,
)

from Functions.JetsCom import bjj_deltaR, bjj_deltaM, to_singleton_jet
# from Functions.Matching import object_matching

class ttBaseProcessor_res(BaseProcessorABC):
    def __init__(self, cfg: Configurator):
        super().__init__(cfg)


    def apply_object_preselection(self, variation):
        # Avoid code duplicate
        super().apply_object_preselection(variation=variation)
        
###########################################################################
        # MET:
        met_pt_corr, met_phi_corr = met_xy_correction(self.params, self.events, self._year, self._era)
        self.events["MET"] = ak.with_field(
            self.events.MET, met_pt_corr, "pt"
        )
        self.events["MET"] = ak.with_field(
            self.events.MET, met_phi_corr, "phi"
        )

###########################################################################        
        # Leptons:
        # Include the supercluster pseudorapidity variable
        electron_etaSC = self.events.Electron.eta + self.events.Electron.deltaEtaSC
        self.events["Electron"] = ak.with_field(
            self.events.Electron, electron_etaSC, "etaSC"
        )
        # Build masks for selection of muons, electrons, jets, fatjets
        self.events["MuonGood"] = lepton_selection(
            self.events, "Muon", self.params
        )
        self.events["ElectronGood"] = lepton_selection(
            self.events, "Electron", self.params
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
        # Get GenJets by flavours:    
        if self._isMC:
            # GenJets acceptance cuts
            mask_pt = self.events["GenJet"].pt > 20
            mask_eta = abs(self.events["GenJet"].eta) < 2.4
            mask_genjet = mask_pt & mask_eta
            # Ghost-hadron matching
            mask_b = self.events["GenJet"].hadronFlavour == 5 # b jets
            mask_l = self.events["GenJet"].hadronFlavour < 5 # light-flavour jets
            # Ghost-parton matching 
            mask_b_parton = abs(self.events["GenJet"].partonFlavour) == 5 # b jet
            mask_l_parton = abs(self.events["GenJet"].partonFlavour) < 5

            self.events["GenJetSave"] = ak.firsts(self.events["GenJet"])

            # New GenJet collections split by flavours
            self.events["GenJetGood"] = self.events.GenJet[mask_genjet]
            # self.events["BGenJetGood"] = self.events.GenJet[mask_genjet & mask_b]
            # self.events["LGenJetGood"] = self.events.GenJet[mask_genjet & mask_l] # non b-jets
            self.events["GenBJetGood"] = self.events.GenJet[mask_genjet & mask_b & mask_b_parton]
            self.events["GenBJetBad"] = self.events.GenJet[mask_genjet & mask_l & mask_l_parton] # non b-jets
            
            self.events["GenJetGoodSave"] = ak.firsts(self.events["GenJetGood"])
            self.events["GenBJetGoodSave"] = ak.firsts(self.events["GenBJetGood"])
            self.events["GenBJetBadSave"] = ak.firsts(self.events["GenBJetBad"])
            
    def define_common_variables_after_presel(self, variation):

###########################################################################
        # combine two AK4 jets to be W
        dijet = get_dijet(
            self.events["BJetBad"], taggerVars=False
        )
        self.events["jj"] = to_singleton_jet(dijet)  # transform the format to singleton
        
        top_deltaR = bjj_deltaR(self.events["BJetGood"], self.events["jj"])
        self.events["bjj_deltaR"] = to_singleton_jet(top_deltaR)
        top_deltaM = bjj_deltaM(self.events["BJetGood"], self.events["jj"])
        self.events["bjj_deltaM"] = to_singleton_jet(top_deltaM)

###########################################################################
        Genjj = get_dijet(
                self.events["GenBJetBad"], taggerVars=False)
        self.events["Genjj"] = to_singleton_jet(Genjj)

        # Reconstuct the top with Gen-level data:
        Gentop_deltaR = bjj_deltaR(self.events["GenBJetGood"], self.events["Genjj"])
        self.events["Genbjj_deltaR"] = to_singleton_jet(Gentop_deltaR)
        Gentop_deltaM = bjj_deltaM(self.events["GenBJetGood"], self.events["Genjj"])
        self.events["Genbjj_deltaM"] = to_singleton_jet(Gentop_deltaM)

###########################################################################
        # # Match the Reco w, top to he Gen Reco w, top:
        self.events["Matchedbjj_deltaR"], self.events["MatchedGenbjj_deltaR"], deltaR_padnone = object_matching(
            self.events["bjj_deltaR"], self.events["Genbjj_deltaR"], dr_min = 0.4
        )
        self.events["Matchedbjj_deltaM"], self.events["MatchedGenbjj_delta"], deltaR_padnone = object_matching(
            self.events["bjj_deltaM"], self.events["Genbjj_deltaM"], dr_min = 0.4
        )
        self.events["Matchedjj"], self.events["MatchedGenjj"], deltaR_padnone = object_matching(
            self.events["jj"], self.events["Genjj"], dr_min = 0.4
        )

        # self.events["MatchedGenjj"], self.events["Matchedjj"], deltaR_padnone = object_matching(
        #     self.events["Genjj"], self.events["jj"], dr_min = 0.4
        # )
        # self.events["MatchedGenbjj_deltaR"], self.events["Matchedbjj_deltaR"], deltaR_padnone = object_matching(
        #     self.events["Genbjj_deltaR"], self.events["bjj_deltaR"], dr_min = 0.4
        # )
        # self.events["MatchedGenbjj_deltaM"], self.events["Matchedbjj_deltaM"], deltaR_padnone = object_matching(
        #     self.events["Genbjj_deltaM"], self.events["bjj_deltaM"], dr_min = 0.4
        # )

 ###########################################################################
        # Flatten all the vars with more than one dim for saving:
        # Gen:
        self.events["GenW"] =  ak.firsts(self.events["Genjj"])
        self.events["GenTop_deltaR"] = ak.firsts(self.events["Genbjj_deltaR"])
        self.events["GenTop_deltaM"] = ak.firsts(self.events["Genbjj_deltaM"])
        # Reco:
        self.events["W"] = ak.firsts(self.events["jj"])
        self.events["Top_deltaR"] = ak.firsts(self.events["bjj_deltaR"])
        self.events["Top_deltaM"] = ak.firsts(self.events["bjj_deltaM"])
        # Matched: 
        self.events["MatchedW"] = ak.firsts(self.events["Matchedjj"])
        self.events["MatchedTop_deltaR"] = ak.firsts(self.events["Matchedbjj_deltaR"])
        self.events["MatchedTop_deltaM"] = ak.firsts(self.events["Matchedbjj_deltaM"])

    def count_objects(self, variation):
        self.events["nMuonGood"] = ak.num(self.events.MuonGood)
        self.events["nElectronGood"] = ak.num(self.events.ElectronGood)
        self.events["nLeptonGood"] = (
            self.events["nMuonGood"] + self.events["nElectronGood"]
        )
        self.events["nJetGood"] = ak.num(self.events.JetGood)
        self.events["nBJetGood"] = ak.num(self.events.BJetGood)
        self.events["nBJetBad"] = ak.num(self.events.BJetBad)  
   
    # Function that defines common variables employed in analyses and save them as attributes of `events`
    def define_common_variables_before_presel(self, variation):
        self.events["JetGood_Ht"] = ak.sum(abs(self.events.JetGood.pt), axis=1)



