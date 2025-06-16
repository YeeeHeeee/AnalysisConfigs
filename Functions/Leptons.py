import awkward as ak
import numpy as np
import correctionlib

def lepton_selection(events, lepton_flavour, params, year):

    leptons = events[lepton_flavour]
    cuts = params.object_preselection[lepton_flavour]
    # Requirements on pT and eta
    passes_eta = abs(leptons.eta) < cuts["eta"]
    passes_pt = leptons.pt > cuts["pt"]

    if lepton_flavour == "Electron":
        # Requirements on SuperCluster eta, isolation and id
        etaSC = abs(leptons.deltaEtaSC + leptons.eta)
        passes_SC = np.invert((etaSC >= 1.4442) & (etaSC <= 1.5660))
        passes_iso = True
        if "iso" in cuts.keys():
            passes_iso = leptons.pfRelIso03_all < cuts["iso"]
        
        Run2 = ['2018','2016_PreVFP', '2016_PostVFP', '2017']
        Run3 = ['2022_preEE', '2022_postEE', '2023_preBPix', '2023_postBPix']
        if year in Run2:
            # Run 2 electron ID
            passes_id = leptons[cuts['id1']] == True
        elif year in Run3:
            # Run 3 electron ID
            passes_id = leptons[cuts['id2']] == True

        good_leptons = passes_eta & passes_pt & passes_SC & passes_iso & passes_id

    elif lepton_flavour == "Muon":
        # Requirements on isolation and id
        passes_iso = leptons.pfRelIso04_all < cuts["iso"]
        passes_id = leptons[cuts['id']] == True

        good_leptons = passes_eta & passes_pt & passes_iso & passes_id

    return leptons[good_leptons]

    