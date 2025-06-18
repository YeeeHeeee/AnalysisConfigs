import gc

import awkward as ak
import numpy as np

from pocket_coffea.workflows.base import BaseProcessorABC
from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.objects import jet_selection

from Functions.Leptons import lepton_selection


class ttBaseProcessor_GenInfo(BaseProcessorABC):


    def __init__(self, cfg: Configurator):
        super().__init__(cfg)


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

    
    def _define_objects(self):

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

        # AK4 Jets
        self.events["JetGood"], self.jetGoodMask = jet_selection(
            self.events, "Jet", self.params, 
            year=self._year, 
            leptons_collection="LeptonGood"
        )

        # Give JetsGood an index for each jet
        self.events["JetGood"] = ak.with_field(self.events["JetGood"], ak.local_index(self.events["JetGood"], axis=1), "index")

        # Adding b tagging information to the jets
        self.events["JetGood"] = ak.with_field(
            self.events["JetGood"],
            self.events["JetGood"][self.params.btagging.working_point[self._year]["btagging_algorithm"]] > self.params.btagging.working_point[self._year]["btagging_WP"][self.params.object_preselection.Jet.btag.wp],
            "btag"
        )

        # Get the btagging mask for the good jets
        self.events["BJetGood"] = self.events["JetGood"][(self.events["JetGood"].btag == 1)]

        # AK8 Jets
        self.events["FatJetGood"], self.fatjetGoodMask = jet_selection(
            self.events, "FatJet", self.params,
            year=self._year,
            leptons_collection="LeptonGood" # used for cleaning jets by removing thoes that overlap with leptons in an events.
        )

        # Get FatJet SubJets
        self.events["SubJetGood1"] = self.events.SubJet[self.events["FatJetGood"].subJetIdx1]
        self.events["SubJetGood2"] = self.events.SubJet[self.events["FatJetGood"].subJetIdx2]


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


    def _match_to_all_jets(self):

        # Pad the arrays
        self.events["LeptonPadded"] = ak.pad_none(self.events["LeptonGood"], 2, axis=1)[:, :2]
        self.events["JetPadded"] = ak.pad_none(self.events["JetGood"], 5, axis=1)[:, :5]
        self.events["BJetPadded"] = ak.pad_none(self.events["BJetGood"], 3, axis=1)[:, :3]
        self.events["FatJetPadded"] = ak.pad_none(self.events["FatJetGood"], 3, axis=1)[:, :3]
        self.events["SubJet1Padded"] = ak.pad_none(self.events["SubJetGood1"], 3, axis=1)[:, :3]
        self.events["SubJet2Padded"] = ak.pad_none(self.events["SubJetGood2"], 3, axis=1)[:, :3]

        # deltaR matching parameter
        matches = [
            ["LeptonPadded","GenLepton", True, 0.1], # Rec collection, Gen collection, is it flat? dR matching radius
            ["JetPadded","GenLightQuark", False, 0.4],
            ["FatJetPadded","GenLightQuark", False, 0.8],
            ["SubJet1Padded","GenLightQuark", False, 0.4],
            ["SubJet2Padded","GenLightQuark", False, 0.4],
            ["JetPadded","GenBQuarkHadronic", True, 0.4],
            ["JetPadded","GenBQuarkLeptonic", True, 0.4],
            ["FatJetPadded","GenBQuarkHadronic", True, 0.8],
            ["FatJetPadded","GenBQuarkLeptonic", True, 0.8],
            ["SubJet1Padded","GenBQuarkHadronic", True, 0.4],
            ["SubJet1Padded","GenBQuarkLeptonic", True, 0.4],
            ["SubJet2Padded","GenBQuarkHadronic", True, 0.4],
            ["SubJet2Padded","GenBQuarkLeptonic", True, 0.4],
        ]
        for match in matches:
            if match[2]:
                max_shape = 1
            else:
                max_shape = max(ak.num(self.events[match[1]]))
            if max_shape > 1:
                for i in range(max_shape):
                    name = f"{match[1]}{i+1}Matched_{match[0]}"
                    fields = {
                        "matched": (self.events[match[0]].delta_r(self.events[match[1]][:,i]) < match[3])
                    }
                    self.events[name] = ak.zip(fields)
                    self.events[name] = ak.where(
                        ak.is_none(self.events[name].matched),
                        ak.Array([[None] * max(ak.num(self.events[match[0]], axis=1))] * len(self.events[name])),
                        self.events[name]
                    )
            else:
                name = f"{match[1]}Matched_{match[0]}"
                fields = {
                    "matched": (self.events[match[0]].delta_r(self.events[match[1]]) < match[3])
                }
                self.events[name] = ak.zip(fields)
                self.events[name] = ak.where(
                    ak.is_none(self.events[name].matched),
                    ak.Array([[None] * max(ak.num(self.events[match[0]], axis=1))] * len(self.events[name])),
                    self.events[name]
                )


    def _select_pair_by_highest_pt(self, collection1, collection2):
        
        # Get all combinations of non btagged AK4 jest
        jet_combos = self._get_pairs(collection1, collection2)

        # Remove items with dR = 0
        jet_combos = jet_combos[jet_combos.deltaR > 0]

        # Sort by pT
        jet_combos = ak.firsts(jet_combos[ak.argsort(jet_combos.pt, ascending=False)])

        return jet_combos.arr1, jet_combos.arr2, jet_combos
    

    def _select_pair_by_dR(self, collection1, collection2):
        
        # Get all combinations of non btagged AK4 jest
        jet_combos = self._get_pairs(collection1, collection2)

        # Remove items with dR = 0
        jet_combos = jet_combos[jet_combos.deltaR > 0]

        # Sort by dR
        jet_combos = ak.firsts(jet_combos[ak.argsort(jet_combos.deltaR, ascending=True)])

        return jet_combos.arr1, jet_combos.arr2, jet_combos


    def _select_pair_by_mass(self, collection1, collection2, mass):
        
        # Get all combinations of non btagged AK4 jest
        jet_combos = self._get_pairs(collection1, collection2)

        # Remove items with dR = 0
        jet_combos = jet_combos[jet_combos.deltaR > 0]

        # Sort by dR
        jet_combos = ak.firsts(jet_combos[ak.argsort(abs(jet_combos.mass - mass), ascending=True)])

        return jet_combos.arr1, jet_combos.arr2, jet_combos


    def apply_object_preselection(self, variation):

        # Avoid code duplicate
        super().apply_object_preselection(variation=variation)
        
        # Get objects
        self._define_objects()

        # Gen level information only to be applied to MC
        if self._isMC:

            # Get the gen level objects
            self._get_gen_semi_leptonic_ttbar()

            # Match to all jet object
            self._match_to_all_jets()


        ### Select from hadronic top decay ###

        # 1) Highest pT tagged objects
        valid_mask = (~ak.is_none(self.events["JetGood"])) & (self.events["JetGood"].btag == 0)
        q1, q2, W = self._select_pair_by_highest_pt(ak.mask(self.events["JetGood"],valid_mask), ak.mask(self.events["JetGood"],valid_mask))
        valid_mask = (~ak.is_none(self.events["BJetGood"])) & (self.events["BJetGood"].btag == 1)
        b, W, top = self._select_pair_by_highest_pt(ak.mask(self.events["BJetGood"],valid_mask), W)
        self.events["HighestPt_q1"] = q1
        self.events["HighestPt_q2"] = q2
        self.events["HighestPt_W"] = W
        self.events["HighestPt_b"] = b
        self.events["HighestPt_top"] = top

        if self._isMC:
            fields = {
                "q1" : ((self.events["HighestPt_q1"].delta_r(self.events["GenLightQuark"][:,0]) < 0.4) | (self.events["HighestPt_q1"].delta_r(self.events["GenLightQuark"][:,1]) < 0.4)),
                "q2" : ((self.events["HighestPt_q2"].delta_r(self.events["GenLightQuark"][:,0]) < 0.4) | (self.events["HighestPt_q2"].delta_r(self.events["GenLightQuark"][:,1]) < 0.4)),
                "b" : (self.events["HighestPt_b"].delta_r(self.events["GenBQuarkHadronic"]) < 0.4),
            }
            self.events["HighestPt_gen_matched"] = ak.zip(fields)

        # 2) Closest tagged jets in dR
        valid_mask = (~ak.is_none(self.events["JetGood"])) & (self.events["JetGood"].btag == 0)
        q1, q2, W = self._select_pair_by_dR(ak.mask(self.events["JetGood"],valid_mask), ak.mask(self.events["JetGood"],valid_mask))
        valid_mask = (~ak.is_none(self.events["BJetGood"])) & (self.events["BJetGood"].btag == 1)
        b, W, top = self._select_pair_by_dR(ak.mask(self.events["BJetGood"],valid_mask), W)
        self.events["ClosestDR_q1"] = q1
        self.events["ClosestDR_q2"] = q2
        self.events["ClosestDR_W"] = W
        self.events["ClosestDR_b"] = b
        self.events["ClosestDR_top"] = top

        if self._isMC:
            fields = {
                "q1" : ((self.events["ClosestDR_q1"].delta_r(self.events["GenLightQuark"][:,0]) < 0.4) | (self.events["ClosestDR_q1"].delta_r(self.events["GenLightQuark"][:,1]) < 0.4)),
                "q2" : ((self.events["ClosestDR_q2"].delta_r(self.events["GenLightQuark"][:,0]) < 0.4) | (self.events["ClosestDR_q2"].delta_r(self.events["GenLightQuark"][:,1]) < 0.4)),
                "b" : (self.events["ClosestDR_b"].delta_r(self.events["GenBQuarkHadronic"]) < 0.4),
            }
            self.events["ClosestDR_gen_matched"] = ak.zip(fields)

        # 3) Closest jets to the masses
        valid_mask = (~ak.is_none(self.events["JetGood"])) & (self.events["JetGood"].btag == 0)
        q1, q2, W = self._select_pair_by_mass(ak.mask(self.events["JetGood"],valid_mask), ak.mask(self.events["JetGood"],valid_mask), 80.4)
        valid_mask = (~ak.is_none(self.events["BJetGood"])) & (self.events["BJetGood"].btag == 1)
        b, W, top = self._select_pair_by_mass(ak.mask(self.events["BJetGood"],valid_mask), W, 172.5)
        self.events["ClosestMass_q1"] = q1
        self.events["ClosestMass_q2"] = q2
        self.events["ClosestMass_W"] = W
        self.events["ClosestMass_b"] = b
        self.events["ClosestMass_top"] = top

        if self._isMC:
            fields = {
                "q1" : ((self.events["ClosestMass_q1"].delta_r(self.events["GenLightQuark"][:,0]) < 0.4) | (self.events["ClosestMass_q1"].delta_r(self.events["GenLightQuark"][:,1]) < 0.4)),
                "q2" : ((self.events["ClosestMass_q2"].delta_r(self.events["GenLightQuark"][:,0]) < 0.4) | (self.events["ClosestMass_q2"].delta_r(self.events["GenLightQuark"][:,1]) < 0.4)),
                "b" : (self.events["ClosestMass_b"].delta_r(self.events["GenBQuarkHadronic"]) < 0.4),
            }
            self.events["ClosestMass_gen_matched"] = ak.zip(fields)


    def count_objects(self, variation):

        self.events["nMuonGood"] = ak.num(self.events["MuonGood"])
        self.events["nElectronGood"] = ak.num(self.events["ElectronGood"])
        self.events["nLeptonGood"] = ak.num(self.events["LeptonGood"])
        self.events["nJetGood"] = ak.num(self.events["JetGood"])
        self.events["nBJetGood"] = ak.num(self.events["BJetGood"])
        self.events["nFatJetGood"] = ak.num(self.events["FatJetGood"])


    def define_common_variables_after_presel(self, variation):
        pass


    def define_common_variables_before_presel(self, variation):
        pass
