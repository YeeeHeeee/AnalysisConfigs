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
    # get_dijet,
    met_xy_correction,
)

from Functions.JetsCom import get_dijet, bjj_deltaR, bjj_deltaM
from Functions.Matching import object_matching1

class ttBaseProcessor_res(BaseProcessorABC):
    def __init__(self, cfg: Configurator):
        super().__init__(cfg)


    def apply_object_preselection(self, variation):
        # Avoid code duplicate
        super().apply_object_preselection(variation=variation)
        
        self.events["N_before_cuts"] = len(self.events)
        # print("total number of events", len(self.events))

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
        # combine two AK4 jets
        self.events["jj"] = get_dijet(
            self.events["BJetBad"], taggerVars=False
        )

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
        # Reconstruct the top by combining the W with 1 b-jets based on deltaR and deltaM with recon data
        self.events["bjj_deltaR"] = bjj_deltaR(self.events["BJetGood"], self.events["jj"])
        self.events["bjj_deltaM"] = bjj_deltaM(self.events["BJetGood"], self.events["jj"])

###########################################################################
        # Reconstuct the top with Gen-level data:
        self.events["Genbjj_deltaR"] = bjj_deltaR(self.events["GenBJetGood"], self.events["Genjj"])
        self.events["Genbjj_deltaM"] = bjj_deltaR(self.events["GenBJetGood"], self.events["Genjj"])
        self.events["Genjj"] = get_dijet(self.events["GenBJetBad"], taggerVars=False)

###########################################################################
        # Match the Reco w, top to he Gen Reco w, top:
        self.events["Matchedjj"], self.events["MatchedGenjj"], deltaR_padnone = object_matching1(
            self.events["jj"], self.events["Genjj"], dr_min = 0.4
        )
        self.events["Matchedbjj_deltaR"], self.events["MatchedGenbjj_deltaR"], deltaR_padnone = object_matching1(
            self.events["bjj_deltaR"], self.events["Genbjj_deltaR"], dr_min = 0.4
        )
        self.events["Matchedbjj_deltaM"], self.events["MatchedGenbjj_deltaM"], deltaR_padnone = object_matching1(
            self.events["bjj_deltaM"], self.events["Genbjj_deltaM"], dr_min = 0.4
        )
        
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



