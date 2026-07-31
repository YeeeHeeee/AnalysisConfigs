import copy
import gc
import json
import os
import uproot
import awkward as ak
import numpy as np

from pocket_coffea.workflows.base import BaseProcessorABC
from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.deltaR_matching import object_matching
from pocket_coffea.lib.objects import (
    jet_selection,
    btagging,
)

from Functions.JetsCom import to_singleton_jet, combine_jets
from Functions.Leptons import lepton_selection
from Functions.jec_config import JECversions, JERversions, JECjsonFiles, JECvariations, nom_jec_variations
from Functions.corrections import jet_correction_correctionlib
from Functions.BtaggingShapeScaleFactors import BTagShapeCorrection
from Functions.BtaggingWeightScaleFactors import BTagWeightCorrection
from Functions.met_xy_correction import met_xy_correction_run2, met_xy_correction_run3
from Functions.jet_veto_maps import apply_jet_veto_maps

class ttBaseProcessor_merge(BaseProcessorABC):
    def __init__(self, cfg: Configurator):
        super().__init__(cfg)
        bfrag_weights_file = "/eos/user/g/guttley/bfrag/bfragweights_vs_pt.root"
        self.bfrag_weights_file = uproot.open(bfrag_weights_file)

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
    

    def _get_first_copy(self, particle, max_iter=20):

        # Early exit if empty
        if len(ak.flatten(particle)) == 0:
            return particle
            
        first_copy_flag = 1 << 12
        result = particle
        
        for _ in range(max_iter):

            # Check which particles need updating
            needs_update = (result.statusFlags & first_copy_flag) == 0
            
            # Exit if nothing needs updating
            if not ak.any(needs_update):
                break
                
            result = ak.where(
                needs_update,
                self.events["GenPart"][result.genPartIdxMother],
                result
            )

            gc.collect()

        return result


    def _get_top_parent(self, particle, max_iter=20):

        # Early exit if empty
        if len(ak.flatten(particle)) == 0:
            return particle
            
        result = particle
        
        for _ in range(max_iter):

            # Check which particles need updating
            needs_update = (result.pdgId != 6) & (result.pdgId != -6) & (result.genPartIdxMother >= 0)
            
            # Exit if nothing needs updating
            if not ak.any(needs_update):
                break
                
            result = ak.where(
                needs_update,
                self.events["GenPart"][result.genPartIdxMother],
                result
            )

            gc.collect()

        return result

    def _get_gen_semi_leptonic_ttbar(self):

        # Add index to GenPart
        self.events["GenPart"] = ak.with_field(self.events["GenPart"], ak.local_index(self.events["GenPart"], axis=1), "index")

        # Get last copy of tops - gen level
        self.events["GenTop"] = self.events["GenPart"][((np.abs(self.events["GenPart"].pdgId) == 6) & ((self.events["GenPart"].statusFlags & (1 << 13)) > 0))]

        # Get final state particles of semi leptonic ttbar decays - gen level
        gen_light_quarks_last_copy = self.events["GenPart"][((np.abs(self.events["GenPart"].pdgId) < 6) & ((self.events["GenPart"].statusFlags & (1 << 13)) > 0))]
        gen_leptons_last_copy = self.events["GenPart"][((np.abs(self.events["GenPart"].pdgId) == 11) | (np.abs(self.events["GenPart"].pdgId) == 13)) & ((self.events["GenPart"].statusFlags & (1 << 13)) > 0)] 
        gen_b_quarks_last_copy = self.events["GenPart"][((np.abs(self.events["GenPart"].pdgId) == 5) & ((self.events["GenPart"].statusFlags & (1 << 13)) > 0))]

        # Get last copy of W bosons - gen level
        gen_W_had_last_copy = self.events["GenPart"][self._get_first_copy(gen_light_quarks_last_copy).genPartIdxMother]
        gen_W_lep_last_copy = self.events["GenPart"][self._get_first_copy(gen_leptons_last_copy).genPartIdxMother]

        # Update the indices - gen level
        gen_light_quarks_last_copy = gen_light_quarks_last_copy[(np.abs(gen_W_had_last_copy.pdgId) == 24)]
        gen_W_had_last_copy = gen_W_had_last_copy[(np.abs(gen_W_had_last_copy.pdgId) == 24)]
        gen_leptons_last_copy = gen_leptons_last_copy[(np.abs(gen_W_lep_last_copy.pdgId) == 24)]
        gen_W_lep_last_copy = gen_W_lep_last_copy[(np.abs(gen_W_lep_last_copy.pdgId) == 24)]

        # Get last copy of top quarks - gen level
        gen_top_had_last_copy = self.events["GenPart"][self._get_first_copy(gen_W_had_last_copy).genPartIdxMother]
        gen_top_lep_last_copy = self.events["GenPart"][self._get_first_copy(gen_W_lep_last_copy).genPartIdxMother]
        gen_top_b_last_copy = self.events["GenPart"][self._get_first_copy(gen_b_quarks_last_copy).genPartIdxMother]

        # Update the indices - gen level
        gen_light_quarks_last_copy = gen_light_quarks_last_copy[(np.abs(gen_top_had_last_copy.pdgId) == 6)]
        gen_W_had_last_copy = gen_W_had_last_copy[(np.abs(gen_top_had_last_copy.pdgId) == 6)]
        gen_top_had_last_copy = gen_top_had_last_copy[(np.abs(gen_top_had_last_copy.pdgId) == 6)]
        gen_leptons_last_copy = gen_leptons_last_copy[(np.abs(gen_top_lep_last_copy.pdgId) == 6)]
        gen_W_lep_last_copy = gen_W_lep_last_copy[(np.abs(gen_top_lep_last_copy.pdgId) == 6)]
        gen_top_lep_last_copy = gen_top_lep_last_copy[(np.abs(gen_top_lep_last_copy.pdgId) == 6)]
        gen_b_quarks_last_copy = gen_b_quarks_last_copy[(np.abs(gen_top_b_last_copy.pdgId) == 6)]
        gen_top_b_last_copy = gen_top_b_last_copy[(np.abs(gen_top_b_last_copy.pdgId) == 6)]

        # Make GenTopHadronic and GenTopLeptonic
        self.events["GenTopHadronic"] = ak.firsts(gen_top_had_last_copy)
        self.events["GenTopLeptonic"] = ak.firsts(gen_top_lep_last_copy)

        # Make GenWHadronic and GenWLeptonic
        self.events["GenWHadronic"] = ak.firsts(gen_W_had_last_copy)
        self.events["GenWLeptonic"] = ak.firsts(gen_W_lep_last_copy)

        # Make GenBQuarkHadronic and GenBQuarkLeptonic
        self.events["GenBQuarkHadronic"] = ak.firsts(gen_b_quarks_last_copy[gen_top_b_last_copy.index == self.events["GenTopHadronic"].index])
        self.events["GenBQuarkLeptonic"] = ak.firsts(gen_b_quarks_last_copy[gen_top_b_last_copy.index == self.events["GenTopLeptonic"].index])

        # Make GenLepton
        self.events["GenLepton"] = ak.firsts(gen_leptons_last_copy)

        # Make GenLightQuark
        self.events["GenLightQuark"] = ak.pad_none(gen_light_quarks_last_copy, 2, axis=1)[:, :2]
        self.events["GenLightQuark1"] = self.events["GenLightQuark"][:, 0]
        self.events["GenLightQuark2"] = self.events["GenLightQuark"][:, 1]


    def _get_extra_weights(self):

        weight_names = {
            "WJetsRun2Stitching" : ["nominal"], 
            "WJetsRun3Stitching" : ["nominal"],
            "TTTo2L2NuRun2Stitching" : ["nominal"], 
            "TTToSemiLeptonicRun2Stitching" : ["nominal"], 
            "TTToHadronicRun2Stitching" : ["nominal"],
            "TopPTReweighting" : ["nominal"],
            "sf_ele_id_custom" : ["nominal", "up", "down"],
            "sf_ele_reco_custom" : ["nominal", "up", "down"],
            "sf_ele_trigger_custom" : ["nominal", "up", "down"],
            "sf_mu_id_custom" : ["nominal", "up", "down"],
            "sf_mu_iso_custom" : ["nominal", "up", "down"],
            "sf_mu_trigger_custom" : ["nominal", "up", "down"],
            "prefiring" : ["nominal", "up", "down"],
            "pileup" : ["nominal", "up", "down"],
            "BTagWeightCorrection" : ["nominal"],
        }

        all_inputs = []

        weights_inputs = self.weights_manager._weightsObj

        btag_shape_correction_names = ["BTagShapeCorrectionSubjets","BTagShapeCorrectionSubjets_down_hf","BTagShapeCorrectionSubjets_up_hf","BTagShapeCorrectionSubjets_down_lf","BTagShapeCorrectionSubjets_up_lf","BTagShapeCorrectionSubjets_down_hfstats1","BTagShapeCorrectionSubjets_up_hfstats1","BTagShapeCorrectionSubjets_down_hfstats2","BTagShapeCorrectionSubjets_up_hfstats2","BTagShapeCorrectionSubjets_down_lfstats1","BTagShapeCorrectionSubjets_up_lfstats1","BTagShapeCorrectionSubjets_down_lfstats2","BTagShapeCorrectionSubjets_up_lfstats2","BTagShapeCorrectionSubjets_down_cferr1","BTagShapeCorrectionSubjets_up_cferr1","BTagShapeCorrectionSubjets_down_cferr2","BTagShapeCorrectionSubjets_up_cferr2"]
        for name in btag_shape_correction_names:
            weights_inputs[name] = BTagShapeCorrection[[i.name for i in BTagShapeCorrection].index(name)]
            all_inputs.append(name)
            weight_names[name] = ["nominal"]

        btag_weight_correction_names = ["BTagWeightCorrection_up", "BTagWeightCorrection_down", "BTagWeightCorrection_up_correlated", "BTagWeightCorrection_down_correlated"]
        for name in btag_weight_correction_names:
            weights_inputs[name] = BTagWeightCorrection[[i.name for i in BTagWeightCorrection].index(name)]
            all_inputs.append(name)
            weight_names[name] = ["nominal"]

        initial_dict = {}
        for k in weight_names:
            for var in weight_names[k]:
                if var == "nominal":
                    initial_dict[f"{k}"] = np.ones(len(self.events))
                else:
                    initial_dict[f"{k}_{var}"] = np.ones(len(self.events))
        self.events["Extra"] = ak.zip(initial_dict)
        if self._isMC:
            for weight_name, variations in weight_names.items():
                if weight_name in weights_inputs.keys():
                    weight_func = weights_inputs[weight_name]
                    if weight_name in all_inputs:
                        per_event_weight = weight_func._function(self.params, self.events.metadata, self.events, len(self.events), "nominal")
                    else:
                        per_event_weight = weight_func.compute(self.events, len(self.events), "nominal")
                    if weight_func.has_variations:
                        if weight_name in all_inputs:
                            nominal_weight = per_event_weight[0]
                            up_weight = per_event_weight[1]
                            down_weight = per_event_weight[2]
                        else:
                            nominal_weight = per_event_weight.nominal
                            up_weight = per_event_weight.up
                            down_weight = per_event_weight.down
                        for variation in variations:
                            if variation == "nominal":
                                self.events["Extra"] = ak.with_field(
                                    self.events["Extra"], nominal_weight, weight_name
                                ) 
                            elif variation == "up":
                                self.events["Extra"] = ak.with_field(
                                    self.events["Extra"], up_weight, weight_name+"_up"
                                )
                            elif variation == "down":
                                self.events["Extra"] = ak.with_field(
                                    self.events["Extra"], down_weight, weight_name+"_down"
                                )                                     
                    else:
                        if weight_name in all_inputs:
                            nominal_weight = per_event_weight
                        else:
                            nominal_weight = per_event_weight.nominal
                        self.events["Extra"] = ak.with_field(
                            self.events["Extra"], nominal_weight, weight_name
                        ) 
                else:
                    print(f"Warning: Weight {weight_name} not found in inputs. Skipping.")  


    def _remove_object_4_vector(self, collection, obj, dr=0.4):

        deltaR = collection.delta_r(obj)
        if dr is None:
            close_mask = ak.zeros_like(deltaR, dtype=bool)
        else:
            close_mask = deltaR < dr
        zero_obj = 0 * obj
        obj_to_subtract = ak.where(close_mask, obj, zero_obj)
        cleaned_vectors = collection - obj_to_subtract
        collection["pt"] = cleaned_vectors.pt
        collection["eta"] = cleaned_vectors.eta
        collection["phi"] = cleaned_vectors.phi
        collection["mass"] = cleaned_vectors.mass

        return collection


    def _get_value_from_bfrag_histogram(self, hist_name, xb, pt):

        hist = self.bfrag_weights_file[hist_name]
        values, x_edges, y_edges = hist.to_numpy()
        x_np = ak.to_numpy(xb)
        y_np = ak.to_numpy(pt)
        x_bin = np.searchsorted(x_edges, x_np, side="right") - 1
        y_bin = np.searchsorted(y_edges, y_np, side="right") - 1
        valid = (
            np.isfinite(x_np)
            & np.isfinite(y_np)
            & (x_bin >= 0)
            & (x_bin < values.shape[0])
            & (y_bin >= 0)
            & (y_bin < values.shape[1])
        )                
        out = np.full(len(x_np), 1.0, dtype=float)
        out[valid] = values[x_bin[valid], y_bin[valid]]
        return out


    def apply_object_preselection(self, variation):
        # Avoid code duplicate
        super().apply_object_preselection(variation=variation)
        
        ## Use the puppi MET
        self.events["MET"] = self.events.PuppiMET

        # MET
        self.events["METUncorrected"] = ak.zip({
            "pt": self.events["MET"].pt,
            "phi": self.events["MET"].phi
        })
        if self._year in ["2016_PreVFP", "2016_PostVFP", "2017", "2018"]:
            met_pt_corr, met_phi_corr = met_xy_correction_run2(self.params, self.events, "MET", self._year, self._era, self._isMC)
        elif self._year in ["2022_preEE", "2022_postEE", "2023_preBPix", "2023_postBPix"]:
            met_pt_corr, met_phi_corr = met_xy_correction_run3(self.params, self.events, "MET", self._year, self._era, self._isMC)
        if self._year != "2024":
            self.events["MET"] = ak.with_field(
                self.events.MET, met_pt_corr, "pt"
            )
            self.events["MET"] = ak.with_field(
                self.events.MET, met_phi_corr, "phi"
            )

        # Leptons
        electron_etaSC = self.events.Electron.eta + self.events.Electron.deltaEtaSC
        self.events["Muon"] = ak.with_field(
            self.events.Muon, ak.local_index(self.events.Muon, axis=1), "index"
        )
        self.events["Electron"] = ak.with_field(
            self.events.Electron, ak.local_index(self.events.Electron, axis=1), "index"
        )
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
        self.events["JetUncorrected"] = ak.copy(self.events.Jet)
        self.events["FatJetUncorrected"] = ak.copy(self.events.FatJet)

        # Apply JEC and JER corrections
        AK4_name = "AK4PFchs" if self._year in ["2016_PreVFP", "2016_PostVFP", "2017", "2018"] else "AK4PFPuppi"
        if self._isMC:
            self.events["Jet"], _ = jet_correction_correctionlib(self.events, "Jet", AK4_name, JECversions[self._year]["MC"], JERversions[self._year]["MC"], JECjsonFiles[self._year], self._year, True, add_uncertainty=JECvariations[self._year])
            if self._year != "2024":
                self.events["FatJet"], _ = jet_correction_correctionlib(self.events, "FatJet", "AK8PFPuppi", JECversions[self._year]["MC"], JERversions[self._year]["MC"], JECjsonFiles[self._year], self._year, True, add_uncertainty=JECvariations[self._year])
        else:
            self.events["Jet"] = jet_correction_correctionlib(self.events, "Jet", AK4_name, JECversions[self._year]["Data"][self._era], None, JECjsonFiles[self._year], self._year, False)
            if self._year != "2024":
                self.events["FatJet"] = jet_correction_correctionlib(self.events, "FatJet", "AK8PFPuppi", JECversions[self._year]["Data"][self._era], None, JECjsonFiles[self._year], self._year, False)


        # Recalculate MET after JEC/JER
        px = (self.events["MET"].pt * np.cos(self.events["MET"].phi)) - ak.sum(((self.events["Jet"].pt * np.cos(self.events["Jet"].phi)) - (self.events["JetUncorrected"].pt * np.cos(self.events["JetUncorrected"].phi))), axis=1)
        py = (self.events["MET"].pt * np.sin(self.events["MET"].phi)) - ak.sum(((self.events["Jet"].pt * np.sin(self.events["Jet"].phi)) - (self.events["JetUncorrected"].pt * np.sin(self.events["JetUncorrected"].phi))), axis=1)
        self.events["MET"] = ak.zip({
            "pt": np.hypot(px, py),
            "phi": np.arctan2(py, px)
        })

        # Apply Jet veto maps
        self.events["Jet"] = apply_jet_veto_maps(self.params, self.events, "Jet", self._year)

        # Patch missing jetID for 2024 NanoAODv15
        if self._year == "2024":
            self.events["Jet"] = ak.with_field(
                self.events["Jet"], 2*np.ones(len(self.events["Jet"]), dtype=bool), "jetId"
            )
            self.events["FatJet"] = ak.with_field(
                self.events["FatJet"], 2*np.ones(len(self.events["FatJet"]), dtype=bool), "jetId"
            )

        # AK8 Jets
        self.events["FatJetGood"], self.fatjetGoodMask = jet_selection(
            self.events, "FatJet", self.params,
            year=self._year,
        )

        # Select fat jet as furthest away from the lepton
        self.events["FatJet"] = ak.firsts(
            self.events["FatJetGood"][ak.argsort(self.events["FatJetGood"].delta_r(self.events["LeptonSave"]), ascending=False)]
        ) 

        # Add unselected jet
        self.events["JetUnselected"] = ak.copy(self.events["Jet"])

        # AK4 Jets
        self.events["JetGood"], self.jetGoodMask = jet_selection(
            self.events, "Jet", self.params, 
            year=self._year, 
        )

        # Remove lepton 4 vectors from overlapping AK4 jets
        self.events["JetDRSubtracted"] = self._remove_object_4_vector(
            self.events["JetGood"], self.events["LeptonSave"], dr=0.4
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
            self.events["JetDRSubtracted"][ak.argsort(self.events["JetDRSubtracted"].delta_r(self.events["LeptonSave"]), ascending=True)]
        )
        self.events["JetLepton"] = ak.firsts(self._get_pairs(self.events["LeptonSave"][:,None], self.events["ClosestJetToLepton"][:,None]))
        self.events["JetLepton"] = ak.with_field(
            self.events["JetLepton"],
            self._get_ptrel(self.events["LeptonSave"], self.events["ClosestJetToLepton"]),
            "ptrel"
        )

        # Corrected version of this
        jets_before = ak.copy(self.events["Jet"])
        jets = ak.copy(self.events["JetUncorrected"])
        jets['pt_raw'] = (1 - jets['rawFactor']) * jets['pt']
        jets['mass_raw'] = (1 - jets['rawFactor']) * jets['mass']
        lep = self.events["LeptonSave"]
        is_mu = (lep.leptonType == 1)
        is_el = (lep.leptonType == 0)
        jet_object_mask = (
            (is_mu & ((jets.muonIdx1 == lep.index) | (jets.muonIdx2 == lep.index))) |
            (is_el & ((jets.electronIdx1 == lep.index) | (jets.electronIdx2 == lep.index)))
        )
        jets_4vec = ak.zip(
            {
                "pt": jets.pt_raw,
                "eta": jets.eta,
                "phi": jets.phi,
                "mass": jets.mass_raw,
            },
            with_name="PtEtaPhiMCandidate",
        )
        lep_4vec = ak.zip(
            {
                "pt": lep.pt,
                "eta": lep.eta,
                "phi": lep.phi,
                "mass": lep.mass,
            },
            with_name="PtEtaPhiMCandidate",
        )
        sub_all = jets_4vec.subtract(lep_4vec)
        new_pt   = ak.where(jet_object_mask, sub_all.pt,   jets.pt)
        new_eta  = ak.where(jet_object_mask, sub_all.eta,  jets.eta)
        new_phi  = ak.where(jet_object_mask, sub_all.phi,  jets.phi)
        new_mass = ak.where(jet_object_mask, sub_all.mass, jets.mass)
        jets = ak.with_field(jets, new_pt, "pt_raw")
        jets = ak.with_field(jets, new_eta, "eta")
        jets = ak.with_field(jets, new_phi, "phi")
        jets = ak.with_field(jets, new_mass, "mass_raw")

        self.events["JetSubtracted"] = jets
        if self._isMC:
            self.events["JetSubtracted"], _ = jet_correction_correctionlib(self.events, "JetSubtracted", AK4_name, JECversions[self._year]["MC"], JERversions[self._year]["MC"], JECjsonFiles[self._year], self._year, True, input_raw=True)
        else:
            self.events["JetSubtracted"] = jet_correction_correctionlib(self.events, "JetSubtracted", AK4_name, JECversions[self._year]["Data"][self._era], None, JECjsonFiles[self._year], self._year, False, input_raw=True)
        self.events["Jet"] = ak.copy(self.events["JetSubtracted"])
        params = copy.deepcopy(self.params)
        params["object_preselection"]["Jet"]["pt"] = 15
        params["object_preselection"]["Jet"]["eta"] = 3.0
        jets_sub_4vec, _ = jet_selection(
            self.events, "Jet", params, 
            year=self._year, 
        )

        self.events["JetNotSubtracted"] = ak.copy(self.events["JetUncorrected"])
        if self._isMC:
            self.events["JetNotSubtracted"], _ = jet_correction_correctionlib(self.events, "JetNotSubtracted", AK4_name, JECversions[self._year]["MC"], JERversions[self._year]["MC"], JECjsonFiles[self._year], self._year, True, input_raw=False)
        else:
            self.events["JetNotSubtracted"] = jet_correction_correctionlib(self.events, "JetNotSubtracted", AK4_name, JECversions[self._year]["Data"][self._era], None, JECjsonFiles[self._year], self._year, False, input_raw=False)
        self.events["Jet"] = ak.copy(self.events["JetNotSubtracted"])
        params = copy.deepcopy(self.params)
        params["object_preselection"]["Jet"]["pt"] = 15
        params["object_preselection"]["Jet"]["eta"] = 3.0
        jets_4vec, _ = jet_selection(
            self.events, "Jet", params, 
            year=self._year, 
        )        

        deltaR_nom = jets_4vec.delta_r(lep_4vec)
        deltaR_sub = jets_sub_4vec.delta_r(lep_4vec)
        ptrel_nom = self._get_ptrel(lep_4vec, jets_4vec)
        ptrel_sub = self._get_ptrel(lep_4vec, jets_sub_4vec)
        jets_sub_4vec = ak.with_field(jets_sub_4vec, deltaR_sub, "deltaR")
        jets_sub_4vec = ak.with_field(jets_sub_4vec, ptrel_sub, "ptrel")
        jets_4vec = ak.with_field(jets_4vec, deltaR_nom, "deltaR")
        jets_4vec = ak.with_field(jets_4vec, ptrel_nom, "ptrel")
        closest_sub = ak.firsts(jets_sub_4vec[ak.argsort(deltaR_sub, axis=1, ascending=True)])
        closest_nom = ak.firsts(jets_4vec[ak.argsort(deltaR_nom, axis=1, ascending=True)])
        self.events["ClosestJetWithLeptonRemoved"] = closest_sub
        self.events["ClosestJetWithoutLeptonRemoved"] = closest_nom
        self.events["Jet"] = ak.copy(jets_before)

        # Get b tagged and non-b tagged jets
        self.events["BJetGood"] = btagging(
            self.events["JetGood"], self.params.btagging.working_point[self._year], wp=self.params.object_preselection.Jet.btag.wp)
        self.events["BJetBad"] = btagging(
            self.events["JetGood"], self.params.btagging.working_point[self._year], wp=self.params.object_preselection.Jet.btag.wp, veto=True)

        # Make a consistent b tagging name
        if self._year == "2024":
            self.events["FatJet"] = ak.with_field(
                self.events["FatJet"], -1*np.ones(len(self.events["FatJet"]), dtype=float), "btagDeepB") # No b tagging for AK8 in 2024 NanoAODv15, so set to -1
            self.events["SubJetGood1"] = ak.with_field(
                self.events["SubJetGood1"], self.events["SubJetGood1"].btagUParTAK4B, "btagDeepB")
            self.events["SubJetGood2"] = ak.with_field(
                self.events["SubJetGood2"], self.events["SubJetGood2"].btagUParTAK4B, "btagDeepB")

        # Remove b jets that overlap with fat jets in deltaR
        self.events["BJetGood"] = ak.where(
            ak.is_none(self.events["FatJet"]),
            ak.Array([[]] * len(self.events)),
            self.events["BJetGood"][(self.events["BJetGood"].delta_r(self.events["FatJet"]) > 0.8)],
        )

        # Remove b jets that overlap with the subjets in deltaR
        self.events["BJetGood"] = ak.where(
            ak.is_none(self.events["SubJetGood1"]),
            ak.Array([[]] * len(self.events)),
            self.events["BJetGood"][
                (self.events["BJetGood"].delta_r(ak.firsts(self.events["SubJetGood1"])) > 0.4)
                & (self.events["BJetGood"].delta_r(ak.firsts(self.events["SubJetGood2"])) > 0.4)
            ],
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

        # Add b tagging score to BJetLep
        if self._year == "2024":
            self.events["BJetLep"] = ak.with_field(
                self.events["BJetLep"], self.events["BJetLep"].btagUParTAK4B, "btagScore"
            )
        else:
            self.events["BJetLep"] = ak.with_field(
                self.events["BJetLep"], self.events["BJetLep"].btagDeepFlavB, "btagScore"
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

        # Get the index of the file in the list of files
        fname = self.events.metadata["filename"]
        dataset = self.events.metadata["dataset"]
        file_index = -1
        if "files" in self.events.metadata.keys():
            if fname in self.events.metadata["files"]:
                file_index = self.events.metadata["files"].index(fname)

        # Add event info (event, luminosityBlok run)
        self.events["EventInfo"] = ak.zip({
            "event": self.events.event,
            "luminosityBlock": self.events.luminosityBlock,
            "run": self.events.run,
            "file_index" : file_index,
        })

        # Change data type
        fatjet = to_singleton_jet(self.events["FatJet"])

        dummy_candidate = ak.zip({"pt":-999.0*np.ones(len(self.events)), "eta":-999.0*np.ones(len(self.events)), "phi":-999.0*np.ones(len(self.events)), "mass":-999.0*np.ones(len(self.events))}, with_name="PtEtaPhiMCandidate")
        if self._isMC:

            # Need to recalculate MET after JEC/JER for the variations and get factors
            for uncert in JECvariations[self._year]:
                corr_factor = 1 + (self.events["Jet"][f"corrFactor_{uncert}"] / self.events["Jet"]["corrFactor"])
                px_var = (self.events["MET"].pt * np.cos(self.events["MET"].phi)) - ak.sum((self.events["Jet"].pt * corr_factor * np.cos(self.events["Jet"].phi)) - (self.events["Jet"].pt * np.cos(self.events["Jet"].phi)), axis=1)
                py_var = (self.events["MET"].pt * np.sin(self.events["MET"].phi)) - ak.sum((self.events["Jet"].pt * corr_factor * np.sin(self.events["Jet"].phi)) - (self.events["Jet"].pt * np.sin(self.events["Jet"].phi)), axis=1)
                met_factor = np.hypot(px_var, py_var) / self.events["MET"].pt
                self.events["MET"] = ak.with_field(
                    self.events["MET"], met_factor, f"corrFactor_{uncert}"
                )
            # Recorrect the MET for JER variations
            smear_factor = 1 + ((self.events["Jet"]["smearFactor_up"] - self.events["Jet"]["smearFactor"]) / self.events["Jet"]["smearFactor"])
            px_var = (self.events["MET"].pt * np.cos(self.events["MET"].phi)) - ak.sum((self.events["Jet"].pt * smear_factor * np.cos(self.events["Jet"].phi)) - (self.events["Jet"].pt * np.cos(self.events["Jet"].phi)), axis=1)
            py_var = (self.events["MET"].pt * np.sin(self.events["MET"].phi)) - ak.sum((self.events["Jet"].pt * smear_factor * np.sin(self.events["Jet"].phi)) - (self.events["Jet"].pt * np.sin(self.events["Jet"].phi)), axis=1)
            met_factor = np.hypot(px_var, py_var) / self.events["MET"].pt
            self.events["MET"] = ak.with_field(
                self.events["MET"], met_factor, f"smearFactor_up"
            )
            smear_factor = 1 + ((self.events["Jet"]["smearFactor_down"] - self.events["Jet"]["smearFactor"]) / self.events["Jet"]["smearFactor"])
            px_var = (self.events["MET"].pt * np.cos(self.events["MET"].phi)) - ak.sum((self.events["Jet"].pt * smear_factor * np.cos(self.events["Jet"].phi)) - (self.events["Jet"].pt * np.cos(self.events["Jet"].phi)), axis=1)
            py_var = (self.events["MET"].pt * np.sin(self.events["MET"].phi)) - ak.sum((self.events["Jet"].pt * smear_factor * np.sin(self.events["Jet"].phi)) - (self.events["Jet"].pt * np.sin(self.events["Jet"].phi)), axis=1)
            met_factor = np.hypot(px_var, py_var) / self.events["MET"].pt
            self.events["MET"] = ak.with_field(
                self.events["MET"], met_factor, f"smearFactor_down"
            )


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
                self._get_gen_semi_leptonic_ttbar()

            
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

            # Add the gen info for what in the gen top is merged and what is not
            if self.events.metadata["sample"].startswith("TTToSemiLeptonic") or self.events.metadata["sample"].startswith("TTMtt"):
                pass
        
            # Add the GenJet information for jets matched to the subjets
            self.events["MatchedGenJet_SubJet1"] = ak.firsts(
                self.events["GenJet"][ak.argsort(self.events["GenJet"].delta_r(self.events["SubJet1"]), ascending=True)]
            ) 
            self.events["MatchedGenJet_SubJet2"] = ak.firsts(
                self.events["GenJet"][ak.argsort(self.events["GenJet"].delta_r(self.events["SubJet2"]), ascending=True)]
            ) 

            # Add the GenJet information for BJetLep
            self.events["MatchedGenJet_BJetLep"] = ak.firsts(
                self.events["GenJet"][ak.argsort(self.events["GenJet"].delta_r(self.events["BJetLep"]), ascending=True)]
            )

            if self.events.metadata["sample"].startswith("TT"):
                self.events["deltaR_Jet_Gen"] = ak.zip({
                    "FatJet_GenTopHadronic": self.events["FatJet"].delta_r(self.events["GenTopHadronic"]),
                    "FatJet_GenWHadronic": self.events["FatJet"].delta_r(self.events["GenWHadronic"]),
                    "FatJet_GenBQuarkHadronic": self.events["FatJet"].delta_r(self.events["GenBQuarkHadronic"]),
                    "FatJet_GenLightQuark1": self.events["FatJet"].delta_r(self.events["GenLightQuark1"]),
                    "FatJet_GenLightQuark2": self.events["FatJet"].delta_r(self.events["GenLightQuark2"]),
                    "SubJet1_GenLightQuark1": self.events["SubJet1"].delta_r(self.events["GenLightQuark1"]),
                    "SubJet1_GenLightQuark2": self.events["SubJet1"].delta_r(self.events["GenLightQuark2"]),
                    "SubJet1_GenBQuarkHadronic": self.events["SubJet1"].delta_r(self.events["GenBQuarkHadronic"]),
                    "SubJet2_GenLightQuark1": self.events["SubJet2"].delta_r(self.events["GenLightQuark1"]),
                    "SubJet2_GenLightQuark2": self.events["SubJet2"].delta_r(self.events["GenLightQuark2"]),
                    "SubJet2_GenBQuarkHadronic": self.events["SubJet2"].delta_r(self.events["GenBQuarkHadronic"]),
                })
                self.events["MergingInfo"] = ak.zip({
                    "FatJet_TopDecaysMerged": ak.where(
                        ((self.events["deltaR_Jet_Gen"].FatJet_GenBQuarkHadronic < 0.8) & (self.events["deltaR_Jet_Gen"].FatJet_GenLightQuark1 < 0.8) & (self.events["deltaR_Jet_Gen"].FatJet_GenLightQuark2 < 0.8)),
                        1,
                        0
                    ),
                    "FatJet_WDecaysMerged": ak.where(
                        ((self.events["deltaR_Jet_Gen"].FatJet_GenLightQuark1 < 0.8) & (self.events["deltaR_Jet_Gen"].FatJet_GenLightQuark2 < 0.8)),
                        1,
                        0
                    ),
                    "SubJet1_WDecaysMerged": ak.where(
                        ((self.events["deltaR_Jet_Gen"].SubJet1_GenLightQuark1 < 0.4) & (self.events["deltaR_Jet_Gen"].SubJet1_GenLightQuark2 < 0.4)),
                        1,
                        0
                    ),
                    "SubJet2_WDecaysMerged": ak.where(
                        ((self.events["deltaR_Jet_Gen"].SubJet2_GenLightQuark1 < 0.4) & (self.events["deltaR_Jet_Gen"].SubJet2_GenLightQuark2 < 0.4)),
                        1,
                        0
                    ),
                    "SubJet1_BMerged": ak.where(
                        (self.events["deltaR_Jet_Gen"].SubJet1_GenBQuarkHadronic < 0.4),
                        1,
                        0
                    ),
                    "SubJet2_BMerged": ak.where(
                        (self.events["deltaR_Jet_Gen"].SubJet2_GenBQuarkHadronic < 0.4),
                        1,
                        0
                    ),
                })

            # Add B fragmentation code
            if self.events.metadata["sample"].startswith("TT"):
                b_hadron_mask = ((np.abs(self.events["GenPart"].pdgId) == 511) | (np.abs(self.events["GenPart"].pdgId) == 521) | (np.abs(self.events["GenPart"].pdgId) == 531) | (np.abs(self.events["GenPart"].pdgId) == 5122))
                first_copy_mask = ((self.events["GenPart"].statusFlags & (1 << 12)) > 0)
                b_hadrons = self.events["GenPart"][b_hadron_mask & first_copy_mask]
                top_parents = self._get_top_parent(b_hadrons)
                top_parent_mask = (top_parents.pdgId == 6)
                anti_top_parent_mask = (top_parents.pdgId == -6)
                b_hadrons_top = b_hadrons[top_parent_mask]
                b_hadrons_anti_top = b_hadrons[anti_top_parent_mask]
                self.events["GenBHadronHadronic"] = ak.firsts(b_hadrons_top)
                self.events["GenBHadronLeptonic"] = ak.firsts(b_hadrons_anti_top)
                # Need to match the b hadrons to the gen jets
                genjet_match_mask = (self.events["GenJet"].delta_r(self.events["GenBHadronHadronic"]) < 0.4)
                self.events["MatchedGenJet_BHadronHadronic"] = ak.firsts(self.events["GenJet"][genjet_match_mask])
                genjet_match_mask = (self.events["GenJet"].delta_r(self.events["GenBHadronLeptonic"]) < 0.4)
                self.events["MatchedGenJet_BHadronLeptonic"] = ak.firsts(self.events["GenJet"][genjet_match_mask])
                # Add all neutrino 4 vectors to the match gen jets if within the radius
                neutrino_mask = ((np.abs(self.events["GenPart"].pdgId) == 12) | ((np.abs(self.events["GenPart"].pdgId) == 14)) | ((np.abs(self.events["GenPart"].pdgId) == 16)))
                neutrinos = self.events["GenPart"][neutrino_mask]
                neutrino_match_mask = (self.events["MatchedGenJet_BHadronHadronic"].delta_r(neutrinos) < 0.4)
                neutrinos_hadronic = neutrinos[neutrino_match_mask]
                neutrinos_hadronic_p4 = ak.zip(
                    {
                        "pt": neutrinos_hadronic.pt,
                        "eta": neutrinos_hadronic.eta,
                        "phi": neutrinos_hadronic.phi,
                        "mass": ak.zeros_like(neutrinos_hadronic.pt),
                    },
                    with_name="Momentum4D",
                )
                total_neutrinos_hadronic = ak.sum(neutrinos_hadronic_p4, axis=-1)
                neutrino_match_mask = (self.events["MatchedGenJet_BHadronLeptonic"].delta_r(neutrinos) < 0.4)
                neutrinos_leptonic = neutrinos[neutrino_match_mask]
                neutrinos_leptonic_p4 = ak.zip(
                    {
                        "pt": neutrinos_leptonic.pt,
                        "eta": neutrinos_leptonic.eta,
                        "phi": neutrinos_leptonic.phi,
                        "mass": ak.zeros_like(neutrinos_leptonic.pt),
                    },
                    with_name="Momentum4D",
                )
                total_neutrinos_leptonic = ak.sum(neutrinos_leptonic_p4, axis=-1)
                total_px = total_neutrinos_hadronic.pt * np.cos(total_neutrinos_hadronic.phi) + self.events["MatchedGenJet_BHadronHadronic"].pt * np.cos(self.events["MatchedGenJet_BHadronHadronic"].phi)
                total_py = total_neutrinos_hadronic.pt * np.sin(total_neutrinos_hadronic.phi) + self.events["MatchedGenJet_BHadronHadronic"].pt * np.sin(self.events["MatchedGenJet_BHadronHadronic"].phi)
                total_pt_hadronic = np.sqrt(total_px**2 + total_py**2)
                total_px = total_neutrinos_leptonic.pt * np.cos(total_neutrinos_leptonic.phi) + self.events["MatchedGenJet_BHadronLeptonic"].pt * np.cos(self.events["MatchedGenJet_BHadronLeptonic"].phi)
                total_py = total_neutrinos_leptonic.pt * np.sin(total_neutrinos_leptonic.phi) + self.events["MatchedGenJet_BHadronLeptonic"].pt * np.sin(self.events["MatchedGenJet_BHadronLeptonic"].phi)
                total_pt_leptonic = np.sqrt(total_px**2 + total_py**2)

                # define x_b
                xb_hadronic = ak.where(
                    ak.is_none(self.events["MatchedGenJet_BHadronHadronic"]),
                    -999.0,
                    self.events["GenBHadronHadronic"].pt / total_pt_hadronic
                )
                xb_leptonic = ak.where(
                    ak.is_none(self.events["MatchedGenJet_BHadronLeptonic"]),
                    -999.0,
                    self.events["GenBHadronLeptonic"].pt / total_pt_leptonic
                )

                # Get the hisogram fragCP5BL from the root file self.bfrag_weights_file
                bfrag_weight = self._get_value_from_bfrag_histogram("fragCP5BL_smooth", xb_hadronic, total_pt_hadronic) * self._get_value_from_bfrag_histogram("fragCP5BL_smooth", xb_leptonic, total_pt_leptonic)
                bfrag_weight_down = self._get_value_from_bfrag_histogram("fragCP5BLdown_smooth", xb_hadronic, total_pt_hadronic) * self._get_value_from_bfrag_histogram("fragCP5BLdown_smooth", xb_leptonic, total_pt_leptonic)
                bfrag_weight_up = self._get_value_from_bfrag_histogram("fragCP5BLup_smooth", xb_hadronic, total_pt_hadronic) * self._get_value_from_bfrag_histogram("fragCP5BLup_smooth", xb_leptonic, total_pt_leptonic)
                self.events["bfrag_weight"] = ak.zip({
                    "nominal": bfrag_weight,
                    "up": bfrag_weight_up,
                    "down": bfrag_weight_down,
                    "xb_hadronic": xb_hadronic,
                    "xb_leptonic": xb_leptonic,
                    "pt_hadronic": total_pt_hadronic,
                    "pt_leptonic": total_pt_leptonic,
                })


        if not hasattr(self.events, "bfrag_weight"): self.events["bfrag_weight"] = ak.zip({"nominal":np.ones(len(self.events)),"up":np.ones(len(self.events)),"down":np.ones(len(self.events)), "xb_hadronic":-999.0*np.ones(len(self.events)),"xb_leptonic":-999.0*np.ones(len(self.events)),"pt_hadronic":-999.0*np.ones(len(self.events)),"pt_leptonic":-999.0*np.ones(len(self.events))})
        if not hasattr(self.events, "GenTop1"): self.events["GenTop1"] = dummy_candidate
        if not hasattr(self.events, "GenTop2"): self.events["GenTop2"] = dummy_candidate
        if not hasattr(self.events, "LNu"): self.events["LNu"] = dummy_candidate
        if not hasattr(self.events, "GenTop_AK8"): self.events["GenTop_AK8"] = dummy_candidate
        if not hasattr(self.events, "MatchedTop_AK8"): self.events["MatchedTop_AK8"] = dummy_candidate
        if not hasattr(self.events, "LHE"): self.events["LHE"] = ak.zip({"HT":-999.0*np.ones(len(self.events))})
        if not hasattr(self.events, "GenTT"): self.events["GenTT"] = ak.zip({"count_l":-999.0*np.ones(len(self.events)),"mass":-999.0*np.ones(len(self.events))})
        if not hasattr(self.events, "MatchedGenJet_SubJet1"): self.events["MatchedGenJet_SubJet1"] = ak.zip({"partonFlavour":-999.0*np.ones(len(self.events))})
        if not hasattr(self.events, "MatchedGenJet_SubJet2"): self.events["MatchedGenJet_SubJet2"] = ak.zip({"partonFlavour":-999.0*np.ones(len(self.events))})
        if not hasattr(self.events, "MatchedGenJet_BJetLep"): self.events["MatchedGenJet_BJetLep"] = ak.zip({"partonFlavour":-999.0*np.ones(len(self.events))})
        if not hasattr(self.events, "GenTopHadronic"): self.events["GenTopHadronic"] = dummy_candidate
        if not hasattr(self.events, "GenTopLeptonic"): self.events["GenTopLeptonic"] = dummy_candidate
        if not hasattr(self.events, "GenWHadronic"): self.events["GenWHadronic"] = dummy_candidate
        if not hasattr(self.events, "GenWLeptonic"): self.events["GenWLeptonic"] = dummy_candidate
        if not hasattr(self.events, "GenBQuarkHadronic"): self.events["GenBQuarkHadronic"] = dummy_candidate
        if not hasattr(self.events, "GenBQuarkLeptonic"): self.events["GenBQuarkLeptonic"] = dummy_candidate
        if not hasattr(self.events, "GenLepton"): self.events["GenLepton"] = dummy_candidate
        if not hasattr(self.events, "GenLightQuark1"): self.events["GenLightQuark1"] = dummy_candidate
        if not hasattr(self.events, "GenLightQuark2"): self.events["GenLightQuark2"] = dummy_candidate
        if not hasattr(self.events, "deltaR_Jet_Gen"): 
            self.events["deltaR_Jet_Gen"] = ak.zip({
                "FatJet_GenTopHadronic": -999.0 * np.ones(len(self.events)),
                "FatJet_GenWHadronic": -999.0 * np.ones(len(self.events)),
                "FatJet_GenBQuarkHadronic": -999.0 * np.ones(len(self.events)),
                "FatJet_GenLightQuark1": -999.0 * np.ones(len(self.events)),
                "FatJet_GenLightQuark2": -999.0 * np.ones(len(self.events)),
                "SubJet1_GenLightQuark1": -999.0 * np.ones(len(self.events)),
                "SubJet1_GenLightQuark2": -999.0 * np.ones(len(self.events)),
                "SubJet1_GenBQuarkHadronic": -999.0 * np.ones(len(self.events)),
                "SubJet2_GenLightQuark1": -999.0 * np.ones(len(self.events)),
                "SubJet2_GenLightQuark2": -999.0 * np.ones(len(self.events)),
                "SubJet2_GenBQuarkHadronic": -999.0 * np.ones(len(self.events)),
            })
        if not hasattr(self.events, "MergingInfo"): 
            self.events["MergingInfo"] = ak.zip({
                "FatJet_TopDecaysMerged": -999.0 * np.ones(len(self.events)),
                "FatJet_WDecaysMerged": -999.0 * np.ones(len(self.events)),
                "SubJet1_WDecaysMerged": -999.0 * np.ones(len(self.events)),
                "SubJet2_WDecaysMerged": -999.0 * np.ones(len(self.events)),
                "SubJet1_BMerged": -999.0 * np.ones(len(self.events)),
                "SubJet2_BMerged": -999.0 * np.ones(len(self.events)),
            })

        for collection in ["BJetLep", "FatJet", "SubJet1", "SubJet2", "ClosestJetToLepton", "MET"]:
            fields = [f"corrFactor_{i}" for i in nom_jec_variations]+["pt_raw","mass_raw","corrFactor","smearFactor","smearFactor_up","smearFactor_down"]
            if collection != "MET":
                fields += ["pt_raw","mass_raw","corrFactor","smearFactor"]
            if collection == "FatJet": fields.append("msoftdrop_raw")
            for field in fields:
                if field not in self.events[collection].fields:
                    self.events[collection] = ak.with_field(self.events[collection], -999.0 * np.ones(len(self.events)), field)
                    
        if not hasattr(self.events, "PSWeight"): self.events["PSWeight"] = ak.Array(np.ones((len(self.events),4)))
        self.events["PSWeight"] = ak.fill_none(ak.pad_none(self.events.PSWeight, 4, clip=True, axis=1), 1)
        if not hasattr(self.events, "LHEScaleWeight"): self.events["LHEScaleWeight"] = ak.Array(np.ones((len(self.events),9)))
        self.events["LHEScaleWeight"] = ak.fill_none(ak.pad_none(self.events.LHEScaleWeight, 9, clip=True, axis=1), 1)

        # Get RMSE of PDF weights
        if not hasattr(self.events, "LHEPdfWeight"): 
            self.events["LHEPdfWeight"] = ak.Array(np.ones((len(self.events),100)))
        pdf_weights = ak.fill_none(self.events.LHEPdfWeight, 100)
        pdf_mean = ak.mean(pdf_weights, axis=1)
        pdf_rmse = np.sqrt(ak.mean((pdf_weights - pdf_mean[:, None])**2, axis=1))

        self.events["GenWeights"] = ak.zip({
            "isr2fsr1": self.events.PSWeight[:, 0],
            "isr1fsr2": self.events.PSWeight[:, 1],
            "isr0p5fsr1": self.events.PSWeight[:, 2],
            "isr1fsr0p5": self.events.PSWeight[:, 3],
            "muF0p5muR0p5": self.events.LHEScaleWeight[:, 0],
            "muF1muR0p5": self.events.LHEScaleWeight[:, 1],
            "muF2muR0p5": self.events.LHEScaleWeight[:, 2],
            "muF0p5muR1": self.events.LHEScaleWeight[:, 3],
            "muF1muR1": self.events.LHEScaleWeight[:, 4],
            "muF2muR1": self.events.LHEScaleWeight[:, 5],
            "muF0p5muR2": self.events.LHEScaleWeight[:, 6],
            "muF1muR2": self.events.LHEScaleWeight[:, 7],
            "muF2muR2": self.events.LHEScaleWeight[:, 8],
            "pdf_max" : ak.max(self.events.LHEPdfWeight, axis=1),
            "pdf_min" : ak.min(self.events.LHEPdfWeight, axis=1),
            "pdf_rmse": pdf_rmse,
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



