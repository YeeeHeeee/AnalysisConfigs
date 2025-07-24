import awkward as ak
import numpy as np

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
from Functions.jec_config import JECversions, JERversions, JECjsonFiles, JECvariations, nom_jec_variations
from Functions.corrections import jet_correction_correctionlib
from Functions.BtaggingShapeScaleFactors import BTagShapeCorrection
from Functions.WJetsRun2StitchingWeights import WJetsRun2Stitching
from Functions.WJetsRun3StitchingWeights import WJetsRun3Stitching
from Functions.TTTo2L2NuRun2StitchingWeights import TTTo2L2NuRun2Stitching
from Functions.TTToSemiLeptonicRun2StitchingWeights import TTToSemiLeptonicRun2Stitching
from Functions.TTToHadronicRun2StitchingWeights import TTToHadronicRun2Stitching
from Functions.TopPTReweighting import TopPTReweighting

class ttBaseProcessor_merge(BaseProcessorABC):
    def __init__(self, cfg: Configurator):
        super().__init__(cfg)

    def _get_ptrel(self, p1, p2):
        p1_px = p1.pt * np.cos(p1.phi)
        p1_py = p1.pt * np.sin(p1.phi)
        p1_pz = p1.pt * np.sinh(p1.eta)
        p2_px = p2.pt * np.cos(p2.phi)
        p2_py = p2.pt * np.sin(p2.phi)
        p2_pz = p2.pt * np.sinh(p2.eta)
        cross_x = p1_py * p2_pz - p1_pz * p2_py
        cross_y = p1_pz * p2_px - p1_px * p2_pz
        cross_z = p1_px * p2_py - p1_py * p2_px
        cross_mag = np.sqrt(cross_x**2 + cross_y**2 + cross_z**2)
        p2_mag = np.sqrt(p2_px**2 + p2_py**2 + p2_pz**2)
        pt_rel = cross_mag / p2_mag
        return pt_rel

    def _get_pairs(self, arr1, arr2):

        # Cartesian product: all pairwise combinations per event
        pairs = ak.cartesian([arr1, arr2], axis=1, nested=False)
        
        # Unpack the pairs
        left, right = ak.unzip(pairs)        
        di_arr = left + right

        # Keep only the specified fields
        fields = {
            "pt": di_arr.pt,
            "eta": di_arr.eta,
            "phi": di_arr.phi,
            "mass": di_arr.mass,
            "deltaR": left.delta_r(right),
            "deltaPhi": abs(left.delta_phi(right)),
            "deltaEta": abs(left.eta - right.eta),
            "arr1" : left,
            "arr2" : right,
        }

        # Zip together the fields
        out = ak.zip(fields, with_name="PtEtaPhiMCandidate")
        
        return out


    def _get_extra_weights(self):

        weights_inputs = BTagShapeCorrection + WJetsRun2Stitching + WJetsRun3Stitching + TTTo2L2NuRun2Stitching + TTToSemiLeptonicRun2Stitching + TTToHadronicRun2Stitching + TopPTReweighting
        weight_names = [
            "BTagShapeCorrectionSubjets",
            "WJetsRun2Stitching", "WJetsRun3Stitching",
            "TTTo2L2NuRun2Stitching", "TTToSemiLeptonicRun2Stitching", "TTToHadronicRun2Stitching",
            "TopPTReweighting",
        ]

        self.events["ExtraWeights"] = ak.zip({k:np.ones(len(self.events)) for k in weight_names})
        if self._isMC:
            weight_input_names = [i.name for i in weights_inputs]
            for weight_name in weight_names:
                if weight_name in weight_input_names:
                    weight_index = weight_input_names.index(weight_name)
                    weight_func = weights_inputs[weight_index]
                    per_event_weight = weight_func._function(self.params, self.events.metadata, self.events, len(self.events), "nominal")
                    self.events["ExtraWeights"] = ak.with_field(
                        self.events["ExtraWeights"], per_event_weight, weight_name
                    ) 


    def _remove_object_4_vector(self, collection, obj, dr=0.4):

        deltaR = collection.delta_r(obj)
        close_mask = deltaR < dr
        zero_obj = 0 * obj
        obj_to_subtract = ak.where(close_mask, obj, zero_obj)
        cleaned_vectors = collection - obj_to_subtract
        collection["pt"] = cleaned_vectors.pt
        collection["eta"] = cleaned_vectors.eta
        collection["phi"] = cleaned_vectors.phi
        collection["mass"] = cleaned_vectors.mass

        return collection


    def apply_object_preselection(self, variation):
        # Avoid code duplicate
        super().apply_object_preselection(variation=variation)
        
        # MET
        #if self._year in ["2016_PreVFP", "2016_PostVFP", "2017", "2018"]:
        #    met_pt_corr, met_phi_corr = met_xy_correction(self.params, self.events, "MET", self._year, self._era)
        #    self.events["MET"] = ak.with_field(
        #        self.events.MET, met_pt_corr, "pt"
        #    )
        #    self.events["MET"] = ak.with_field(
        #        self.events.MET, met_phi_corr, "phi"
        #    )

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
        # Add pf isolation to MuonGood, ElectronGood
        self.events["MuonGood"] = ak.with_field(
            self.events["MuonGood"], self.events["MuonGood"].pfRelIso04_all, "RelIso"
        )
        self.events["ElectronGood"] = ak.with_field(
            self.events["ElectronGood"], self.events["ElectronGood"].pfRelIso03_all, "RelIso"
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

        # Apply JEC and JER corrections
        AK4_name = "AK4PFchs" if self._year in ["2016_PreVFP", "2016_PostVFP", "2017", "2018"] else "AK4PFPuppi"
        if self._isMC:
            self.events["Jet"], _ = jet_correction_correctionlib(self.events, "Jet", AK4_name, JECversions[self._year]["MC"], JERversions[self._year]["MC"], JECjsonFiles[self._year], self._year, True, add_uncertainty=JECvariations[self._year])
            self.events["FatJet"], _ = jet_correction_correctionlib(self.events, "FatJet", "AK8PFPuppi", JECversions[self._year]["MC"], JERversions[self._year]["MC"], JECjsonFiles[self._year], self._year, True, add_uncertainty=JECvariations[self._year])
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
            #leptons_collection="LeptonGood" # used for cleaning jets by removing those that overlap with leptons in an events.
        )

        # Select fat jet as furthest away from the lepton
        self.events["FatJet"] = ak.firsts(
            self.events["FatJetGood"][ak.argsort(self.events["FatJetGood"].delta_r(self.events["LeptonSave"]), ascending=False)]
        ) 

        # Remove lepton 4 vectors from overlapping AK4 jets
        self.events["Jet"] = self._remove_object_4_vector(
            self.events["Jet"], self.events["LeptonSave"], dr=0.4
        )

        # AK4 Jets
        self.events["JetGood"], self.jetGoodMask = jet_selection(
            self.events, "Jet", self.params, 
            year=self._year, 
            #leptons_collection="LeptonGood"
        )

        # Get subjets from the fat jets
        self.events["SubJetGood1"] = self.events.SubJet[self.events["FatJetGood"].subJetIdx1]
        self.events["SubJetGood2"] = self.events.SubJet[self.events["FatJetGood"].subJetIdx2]

        # Apply JEC and JER corrections to subjets
        if self._isMC:
            self.events["SubJetGood1"], _ = jet_correction_correctionlib(self.events, "SubJetGood1", AK4_name, JECversions[self._year]["MC"], JERversions[self._year]["MC"], JECjsonFiles[self._year], self._year, True, area=0.5, add_uncertainty=JECvariations[self._year])
            self.events["SubJetGood2"], _ = jet_correction_correctionlib(self.events, "SubJetGood2", AK4_name, JECversions[self._year]["MC"], JERversions[self._year]["MC"], JECjsonFiles[self._year], self._year, True, area=0.5, add_uncertainty=JECvariations[self._year])
        else:
            self.events["SubJetGood1"] = jet_correction_correctionlib(self.events, "SubJetGood1", AK4_name, JECversions[self._year]["Data"][self._era], None, JECjsonFiles[self._year], self._year, False, area=0.5)
            self.events["SubJetGood2"] = jet_correction_correctionlib(self.events, "SubJetGood2", AK4_name, JECversions[self._year]["Data"][self._era], None, JECjsonFiles[self._year], self._year, False, area=0.5)

        # Get variables between the lepton and the closest jet
        self.events["ClosestJetToLepton"] = ak.firsts(
            self.events["JetGood"][ak.argsort(self.events["JetGood"].delta_r(self.events["LeptonSave"]), ascending=True)]
        )
        self.events["JetLepton"] = ak.firsts(self._get_pairs(self.events["LeptonSave"][:,None], self.events["ClosestJetToLepton"][:,None]))
        self.events["JetLepton"] = ak.with_field(
            self.events["JetLepton"],
            self._get_ptrel(self.events["LeptonSave"], self.events["ClosestJetToLepton"]),
            "ptrel"
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

        # Remove Jets that overlap with the fat jets in deltaR
        self.events["JetGood"] = ak.where(
            ak.is_none(self.events["FatJet"]),
            ak.Array([[]] * len(self.events)),
            self.events["JetGood"][(self.events["JetGood"].delta_r(self.events["FatJet"]) > 0.8)],
        )

        # Remove all FatJetGood that overlap with any BJetLep in deltaR 
        self.events["BJetLep"] = ak.firsts(self.events["BJetGood"])
        self.events["FatJetGood"] = ak.where(
            ak.is_none(self.events["BJetLep"]),
            ak.Array([[]] * len(self.events)),
            self.events["FatJetGood"][(self.events["FatJetGood"].delta_r(self.events["BJetLep"]) > 0.8)],
        )

        # Combine two subjet for validation
        self.events["CombinedSubJets"] = combine_jets(
            self.events["SubJetGood1"], self.events["SubJetGood2"]
        )

        self.events["SubJet1"] = ak.firsts(self.events["SubJetGood1"])
        self.events["SubJet2"] = ak.firsts(self.events["SubJetGood2"])
    
        # Get the transverse mass of the lepton and the MET
        self.events["LeptonMET"] = ak.zip({
            "mt" : np.sqrt(2 * self.events["LeptonSave"].pt * self.events["MET"].pt * (1 - np.cos(self.events["LeptonSave"].delta_phi(self.events["MET"]))))
        })


    def define_common_variables_after_presel(self, variation):

        # Change data type
        fatjet = to_singleton_jet(self.events["FatJet"])

        dummy_candidate = ak.zip({"pt":-999.0*np.ones(len(self.events)), "eta":-999.0*np.ones(len(self.events)), "phi":-999.0*np.ones(len(self.events)), "mass":-999.0*np.ones(len(self.events))}, with_name="PtEtaPhiMCandidate")
        if self._isMC:
            # Get the gen top
            self.events["GenTop"] = self.events["GenPart"][((np.abs(self.events["GenPart"].pdgId) == 6) & ((self.events["GenPart"].statusFlags & (1 << 13)) > 0))]
            self.events["GenTop1"] = ak.pad_none(self.events["GenTop"], 2, axis=1)[:, 0]
            self.events["GenTop2"] = ak.pad_none(self.events["GenTop"], 2, axis=1)[:, 1]

            # Get the gen top AK8
            if self.events.metadata["sample"].startswith("TT"):
                self.events["GenTop_AK8"] = ak.firsts(self.events["GenJetAK8"])
                GenTop_AK8 = to_singleton_jet(self.events["GenTop_AK8"])
                self.events["GenTop_AK8"] = ak.where(
                    ak.is_none(GenTop_AK8),
                    dummy_candidate,
                    GenTop_AK8,
                )
                self.events["GenTop_AK8"] = ak.firsts(self.events["GenJetAK8"])
                self.events["MatchedTop_AK81"], self.events["MatchedGenTop_AK8"], deltaR_padnon = object_matching(fatjet, GenTop_AK8, dr_min = 0.8)  
                self.events["MatchedTop_AK8"] = ak.firsts(self.events["MatchedTop_AK81"])

            # Get the GenTop pairs - first copy
            if self.events.metadata["sample"].startswith("TT"):
                GenTopFirstCopy = self.events["GenPart"][((np.abs(self.events["GenPart"].pdgId) == 6) & ((self.events["GenPart"].statusFlags & (1 << 12)) > 0))]
                top_pairs = ak.combinations(GenTopFirstCopy, 2, fields=["left", "right"])
                self.events["GenTT"] = ak.firsts(ak.zip({"mass": (top_pairs.left + top_pairs.right).mass}, with_name="PtEtaPhiMCandidate"))
                l_mask = (self.events["LHEPart"].pdgId == 11) | (self.events["LHEPart"].pdgId == -11) | \
                        (self.events["LHEPart"].pdgId == 13) | (self.events["LHEPart"].pdgId == -13) | \
                        (self.events["LHEPart"].pdgId == 15) | (self.events["LHEPart"].pdgId == -15)
                nu_mask = (self.events["LHEPart"].pdgId == 12) | (self.events["LHEPart"].pdgId == -12) | \
                        (self.events["LHEPart"].pdgId == 14) | (self.events["LHEPart"].pdgId == -14) | \
                        (self.events["LHEPart"].pdgId == 16) | (self.events["LHEPart"].pdgId == -16)
                self.events["count_l"] = ak.num(self.events["LHEPart"][l_mask])
                self.events["count_nu"] = ak.num(self.events["LHEPart"][nu_mask])
                # add count_l field to GenTT
                self.events["GenTT"] = ak.with_field(
                    self.events["GenTT"], self.events["count_l"], "count_l"
                )

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

        if not hasattr(self.events, "GenTop1"): self.events["GenTop1"] = dummy_candidate
        if not hasattr(self.events, "GenTop2"): self.events["GenTop2"] = dummy_candidate
        if not hasattr(self.events, "LNu"): self.events["LNu"] = dummy_candidate
        if not hasattr(self.events, "GenTop_AK8"): self.events["GenTop_AK8"] = dummy_candidate
        if not hasattr(self.events, "MatchedTop_AK8"): self.events["MatchedTop_AK8"] = dummy_candidate
        if not hasattr(self.events, "LHE"): self.events["LHE"] = ak.zip({"HT":-999.0*np.ones(len(self.events))})
        if not hasattr(self.events, "GenTT"): self.events["GenTT"] = ak.zip({"count_l":-999.0*np.ones(len(self.events))})
        for collection in ["BJetLep", "FatJet", "SubJet1", "SubJet2"]:
            fields = [f"corrFactor_{i}" for i in nom_jec_variations]+["pt_raw","mass_raw","corrFactor","smearFactor"]
            if collection == "FatJet": fields.append("msoftdrop_raw")
            for field in fields:
                if field not in self.events[collection].fields:
                    self.events[collection] = ak.with_field(self.events[collection], -999.0 * np.ones(len(self.events)), field)
        if not hasattr(self.events, "PSWeight"): self.events["PSWeight"] = ak.Array(np.ones((len(self.events),4)))
        self.events["PSWeight"] = ak.fill_none(ak.pad_none(array, 4, clip=True, axis=1), 1)
        if not hasattr(self.events, "LHEScaleWeight"): self.events["LHEScaleWeight"] = ak.Array(np.ones((len(self.events),8)))
        self.events["LHEScaleWeight"] = ak.fill_none(ak.pad_none(self.events.LHEScaleWeight, 8, clip=True, axis=1), 1)

        self.events["GenWeights"] = ak.zip({
            "isr2fsr1": self.events.PSWeight[:, 0],
            "isr1fsr2": self.events.PSWeight[:, 1],
            "isr0p5fsr1": self.events.PSWeight[:, 2],
            "isr1fsr0p5": self.events.PSWeight[:, 3],
            "muF0p5muR0p5": self.events.LHEScaleWeight[:, 0],
            "muF1muR0p5": self.events.LHEScaleWeight[:, 1],
            "muF2muR0p5": self.events.LHEScaleWeight[:, 2],
            "muF0p5muR1": self.events.LHEScaleWeight[:, 3],
            "muF2muR1": self.events.LHEScaleWeight[:, 4],
            "muF0p5muR2": self.events.LHEScaleWeight[:, 5],
            "muF1muR2": self.events.LHEScaleWeight[:, 6],
            "muF2muR2": self.events.LHEScaleWeight[:, 7],
        })

        # Get extra weights
        self._get_extra_weights()


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



