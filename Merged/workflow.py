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
from Functions.jec_config import JECversions, JERversions, JECjsonFiles
from Functions.corrections import jet_correction_correctionlib

class ttBaseProcessor_merge(BaseProcessorABC):
    def __init__(self, cfg: Configurator):
        super().__init__(cfg)


    def apply_object_preselection(self, variation):
        # Avoid code duplicate
        super().apply_object_preselection(variation=variation)
        
        # MET
        if self._year in ["2016_PreVFP", "2016_PostVFP", "2017", "2018"]:
            met_pt_corr, met_phi_corr = met_xy_correction(self.params, self.events, "MET", self._year, self._era)
            self.events["MET"] = ak.with_field(
                self.events.MET, met_pt_corr, "pt"
            )
            self.events["MET"] = ak.with_field(
                self.events.MET, met_phi_corr, "phi"
            )

        # Leptons
        electron_etaSC = self.events.Electron.eta + self.events.Electron.deltaEtaSC
        self.events["Electron"] = ak.with_field(
            self.events.Electron, electron_etaSC, "etaSC"
        )
        self.events["MuonGood"] = lepton_selection(
            self.events, "Muon", self.params, self._year
        )
        self.events["ElectronGood"] = lepton_selection(
            self.events, "Electron", self.params, self._year
        )
        # Add variable to MuonGood, ElectronGood for whether it is electon (0) or muon (1)
        self.events["ElectronGood"]= ak.with_field(
            self.events["ElectronGood"], 0, "leptonType"
        )
        self.events["MuonGood"] = ak.with_field(
            self.events["MuonGood"], 1, "leptonType"
        )   

        leptons = ak.with_name(
            ak.concatenate((self.events.MuonGood, self.events.ElectronGood), axis=1),
            name='PtEtaPhiMCandidate',
        )
        self.events["LeptonGood"] = leptons[ak.argsort(leptons.pt, ascending=False)]
        self.events["LeptonSave"] = ak.firsts(self.events["LeptonGood"])

        # JEC and JER corrections
        self.events["JetUncorrected"] = self.events.Jet
        self.events["FatJetUncorrected"] = self.events.FatJet
        AK4_name = "AK4PFchs" if self._year in ["2016_PreVFP", "2016_PostVFP", "2017", "2018"] else "AK4PFPuppi"
        if self._isMC:
            self.events["Jet"], _ = jet_correction_correctionlib(self.events, "Jet", AK4_name, JECversions[self._year]["MC"], JERversions[self._year]["MC"], JECjsonFiles[self._year], self._year, True)
            self.events["FatJet"], _ = jet_correction_correctionlib(self.events, "FatJet", "AK8PFPuppi", JECversions[self._year]["MC"], JERversions[self._year]["MC"], JECjsonFiles[self._year], self._year, True)
        else:
            self.events["Jet"] = jet_correction_correctionlib(self.events, "Jet", AK4_name, JECversions[self._year]["Data"][self._era], None, JECjsonFiles[self._year], self._year, False)
            self.events["FatJet"] = jet_correction_correctionlib(self.events, "FatJet", "AK8PFPuppi", JECversions[self._year]["Data"][self._era], None, JECjsonFiles[self._year], self._year, False)

        # Recalculate MET after JEC/JER
        px = (self.events["MET"].pt * np.cos(self.events["MET"].phi)) - ak.sum((self.events["Jet"].pt * np.cos(self.events["Jet"].phi)) - (self.events["JetUncorrected"].pt * np.cos(self.events["JetUncorrected"].phi)), axis=1)
        py = (self.events["MET"].pt * np.sin(self.events["MET"].phi)) - ak.sum((self.events["Jet"].pt * np.sin(self.events["Jet"].phi)) - (self.events["JetUncorrected"].pt * np.sin(self.events["JetUncorrected"].phi)), axis=1)
        self.events["METUncorrected"] = ak.zip({
            "pt": self.events["MET"].pt,
            "phi": self.events["MET"].phi
        })
        self.events["MET"] = ak.zip({
            "pt": np.hypot(px, py),
            "phi": np.arctan2(py, px)
        })

        # AK8 Jets
        self.events["FatJetGood"], self.fatjetGoodMask = jet_selection(
            self.events, "FatJet", self.params,
            year=self._year,
            leptons_collection="LeptonGood" # used for cleaning jets by removing those that overlap with leptons in an events.
        )
        self.events["FatJet"] = ak.firsts(self.events["FatJetGood"])

        # AK4 Jets
        self.events["JetGood"], self.jetGoodMask = jet_selection(
            self.events, "Jet", self.params, 
            year=self._year, 
            leptons_collection="LeptonGood"
        )

        # Get b tagged and non-b tagged jets
        self.events["BJetGood"] = btagging(
            self.events["JetGood"], self.params.btagging.working_point[self._year], wp=self.params.object_preselection.Jet.btag.wp)
        self.events["BJetBad"] = btagging(
            self.events["JetGood"], self.params.btagging.working_point[self._year], wp=self.params.object_preselection.Jet.btag.wp, veto=True)

        # Remove b jets that overlap with fat jets in deltaR
        self.events["BJetGood"] = ak.where(
            ak.is_none(self.events["FatJet"]),
            ak.Array([[]] * len(self.events)),
            self.events["BJetGood"][(self.events["BJetGood"].delta_r(self.events["FatJet"]) > 0.8)],
        )

        # Get subjets from the fat jets
        self.events["SubJetGood1"] = self.events.SubJet[self.events["FatJetGood"].subJetIdx1]
        self.events["SubJetGood2"] = self.events.SubJet[self.events["FatJetGood"].subJetIdx2]

        # Combine two subjet for validation
        self.events["CombinedSubJets"] = combine_jets(
            self.events["SubJetGood1"], self.events["SubJetGood2"]
        )

        self.events["SubJet1"] = ak.firsts(self.events["SubJetGood1"])
        self.events["SubJet2"] = ak.firsts(self.events["SubJetGood2"])
    

    def define_common_variables_after_presel(self, variation):

        # Change data type
        fatjet = to_singleton_jet(self.events["FatJet"])

        dummy_candidate = ak.zip({"pt":-999.0*np.ones(len(self.events)), "eta":-999.0*np.ones(len(self.events)), "phi":-999.0*np.ones(len(self.events)), "mass":-999.0*np.ones(len(self.events))}, with_name="PtEtaPhiMCandidate")
        if self._isMC:
            # Add GenTop information
            self.events["GenTop_AK8"] = ak.firsts(self.events["GenJetAK8"])
            GenTop_AK8 = to_singleton_jet(self.events["GenTop_AK8"])
            self.events["GenTop_AK8"] = ak.where(
                ak.is_none(GenTop_AK8),
                dummy_candidate,
                GenTop_AK8,
            )
            self.events["GenTop"] = self.events["GenPart"][((np.abs(self.events["GenPart"].pdgId) == 6) & ((self.events["GenPart"].statusFlags & (1 << 13)) > 0))]
            self.events["GenTop1"] = ak.pad_none(self.events["GenTop"], 2, axis=1)[:, 0]
            self.events["GenTop2"] = ak.pad_none(self.events["GenTop"], 2, axis=1)[:, 1]
            self.events["GenTop_AK8"] = ak.firsts(self.events["GenJetAK8"])
            self.events["MatchedTop_AK81"], self.events["MatchedGenTop_AK8"], deltaR_padnon = object_matching(fatjet, GenTop_AK8, dr_min = 0.8)  
            self.events["MatchedTop_AK8"] = ak.firsts(self.events["MatchedTop_AK81"])
            # Get the LNu for W + jets samples
            if self.events.metadata["sample"].startswith("WJetsToLNu"):
                l_mask = (self.events["LHEPart"].pdgId == 11) | (self.events["LHEPart"].pdgId == -11) | \
                        (self.events["LHEPart"].pdgId == 13) | (self.events["LHEPart"].pdgId == -13) | \
                        (self.events["LHEPart"].pdgId == 15) | (self.events["LHEPart"].pdgId == -15)
                nu_mask = (self.events["LHEPart"].pdgId == 12) | (self.events["LHEPart"].pdgId == -12) | \
                        (self.events["LHEPart"].pdgId == 14) | (self.events["LHEPart"].pdgId == -14) | \
                        (self.events["LHEPart"].pdgId == 16) | (self.events["LHEPart"].pdgId == -16)
                l = self.events["LHEPart"][l_mask]
                nu = self.events["LHEPart"][nu_mask]
                lnu_pairs = ak.cartesian([l, nu], axis=1, nested=False)
                left, right = ak.unzip(lnu_pairs)        
                di_arr = left + right
                fields = {
                    "mass": di_arr.mass,
                }
                self.events["LNu"] = ak.firsts(ak.zip(fields, with_name="PtEtaPhiMCandidate"))   
            else:
                self.events["LNu"] = dummy_candidate
        else:
            self.events["GenTop_AK8"] = dummy_candidate
            self.events["GenTop1"] = dummy_candidate
            self.events["GenTop2"] = dummy_candidate
            self.events["MatchedTop_AK81"] = dummy_candidate
            self.events["MatchedTop_AK8"] = dummy_candidate
            self.events["LNu"] = dummy_candidate
            
        if "LHE" not in self.events:
            self.events["LHE"] = ak.zip({"HT":-999.0*np.ones(len(self.events))})


        # Highest pT b jet
        self.events["BJet_HighestPt"] = ak.firsts(self.events["BJetGood"])

        # Closest b jet to the leading lepton
        self.events["BJet_ClosestToLepton"] = ak.firsts(self.events["BJetGood"][ak.argsort(self.events["BJetGood"].delta_r(self.events["LeptonSave"]), ascending=False)])


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



