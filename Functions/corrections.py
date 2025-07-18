import copy

import awkward as ak
import numpy as np
import correctionlib

from pocket_coffea.lib.deltaR_matching import object_matching

def jet_correction_correctionlib(
    events, Jet, typeJet, JECversion, JERversion, JECjsonFile, year, MC,verbose=False, area=None
):
    '''
    This function implements the Jet Energy corrections and Jet energy smearning
    using factors from correctionlib common-POG json file
    example here: https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/blob/master/examples/jercExample.py

    '''
    jsonfile = JECjsonFile[
        [t for t in ['AK4', 'AK8'] if typeJet.startswith(t)][0]
    ]
    JECfile = correctionlib.CorrectionSet.from_file(jsonfile)
    #print(list(JECfile.keys()))
    #print(f'{JECversion}_L1L2L3Res_{typeJet}')
    corr = JECfile.compound[f'{JECversion}_L1L2L3Res_{typeJet}']

    # until correctionlib handles jagged data natively we have to flatten and unflatten
    jets = events[Jet]
    jets['pt_raw'] = (1 - jets['rawFactor']) * jets['pt']
    jets['mass_raw'] = (1 - jets['rawFactor']) * jets['mass']
    if "fixedGridRhoFastjetAll" in events.fields:
        jets['rho'] = ak.broadcast_arrays(events.fixedGridRhoFastjetAll, jets.pt)[0]
    else:
        jets['rho'] = ak.broadcast_arrays(events.Rho.fixedGridRhoFastjetAll, jets.pt)[0]
    jets['run'] = ak.broadcast_arrays(events.run, jets.pt)[0]

    # set are to 0.5 if not present
    if "area" not in jets.fields:
        jets['area'] = ak.broadcast_arrays(
            np.full_like(jets.pt, 0.5, dtype=np.float32), jets.pt
        )[0]

    j, nj = ak.flatten(jets), ak.num(jets)

    if MC:
        if year in ['2016_PreVFP', '2016_PostVFP', '2017', '2018','2022_preEE','2022_postEE', '2023_preBPix']:
            flatCorrFactor = corr.evaluate(
                np.array(j['area']),
                np.array(j['eta']),
                np.array(j['pt_raw']),
                np.array(j['rho']),
            )
        elif year in ['2023_postBPix']:
            flatCorrFactor = corr.evaluate(
                np.array(j['area']),
                np.array(j['eta']),
                np.array(j['pt_raw']),
                np.array(j['rho']),
                np.array(j['phi']),
            )
    else:
        if year in ['2016_PreVFP', '2016_PostVFP', '2017', '2018', '2022_preEE', '2022_postEE']:
            flatCorrFactor = corr.evaluate(
                np.array(j['area']),
                np.array(j['eta']),
                np.array(j['pt_raw']),
                np.array(j['rho']),
            )
        elif year in ['2023_preBPix']:
            flatCorrFactor = corr.evaluate(
                np.array(j['area']),
                np.array(j['eta']),
                np.array(j['pt_raw']),
                np.array(j['rho']),
                np.array(j['run']),
            )
        elif year in ['2023_postBPix']:
            flatCorrFactor = corr.evaluate(
                np.array(j['area']),
                np.array(j['eta']),
                np.array(j['pt_raw']),
                np.array(j['rho']),
                np.array(j['phi']),
                np.array(j['run'])
            )
      
    corrFactor = ak.unflatten(flatCorrFactor, nj)

    jets_corrected = copy.copy(jets)
    jets_corrected['pt'] = jets['pt_raw'] * corrFactor
    jets_corrected['mass'] = jets['mass_raw'] * corrFactor
    if hasattr(jets, 'msoftdrop'):
        jets_corrected['msoftdrop'] = jets['msoftdrop'] * (jets_corrected['pt'] / jets['pt_raw'])
    jets_corrected['rho'] = jets['rho']

    seed = events.event[0]

    if verbose:
        print()
        print(seed, 'JEC: starting columns:', ak.fields(jets), end='\n\n')

        print(seed, 'JEC: untransformed pt ratios', jets.pt / jets.pt_raw)
        print(seed, 'JEC: untransformed mass ratios', jets.mass / jets.mass_raw)

        print(
            seed, 'JEC: corrected pt ratios', jets_corrected.pt / jets_corrected.pt_raw
        )
        print(
            seed,
            'JEC: corrected mass ratios',
            jets_corrected.mass / jets_corrected.mass_raw,
        )

        print()
        print(seed, 'JEC: corrected columns:', ak.fields(jets_corrected), end='\n\n')

        # print('JES UP pt ratio',jets_corrected.JES_jes.up.pt/jets_corrected.pt_raw)
        # print('JES DOWN pt ratio',jets_corrected.JES_jes.down.pt/jets_corrected.pt_raw, end='\n\n')

    # Apply JER pt smearing (https://twiki.cern.ch/twiki/bin/viewauth/CMS/JetResolution)
    # The hybrid scaling method is implemented: if a jet is matched to a gen-jet, the scaling method is applied;
    # if a jet is not gen-matched, the stochastic smearing is applied.
    if JERversion:

        sf = JECfile[f'{JERversion}_ScaleFactor_{typeJet}']
        res = JECfile[f'{JERversion}_PtResolution_{typeJet}']
        j, nj = ak.flatten(jets_corrected), ak.num(jets_corrected)
        if len(sf.inputs) == 2:
            scaleFactor_flat = sf.evaluate(j['eta'].to_numpy(), 'nom')
        elif len(sf.inputs) == 3:
            scaleFactor_flat = sf.evaluate(
                j['eta'].to_numpy(), j['pt'].to_numpy(), "nom"
            )
        ptResolution_flat = res.evaluate(
            j['eta'].to_numpy(), j['pt'].to_numpy(), j['rho'].to_numpy()
        )
        scaleFactor = ak.unflatten(scaleFactor_flat, nj)
        ptResolution = ak.unflatten(ptResolution_flat, nj)
        # Match jets with gen-level jets, with DeltaR and DeltaPt requirements
        dr_min = {'AK4PFchs': 0.2, 'AK4PFPuppi': 0.2, 'AK8PFPuppi': 0.4}[
            typeJet
        ]  # Match jets within a cone with half the jet radius
        pt_min = (
            3 * ptResolution * jets_corrected['pt']
        )  # Match jets whose pt does not differ more than 3 sigmas from the gen-level pt
        genJet = {'AK4PFchs': 'GenJet', 'AK4PFPuppi': 'GenJet', 'AK8PFPuppi': 'GenJetAK8'}[typeJet]
        genJetIdx = {'AK4PFchs': 'genJetIdx', 'AK4PFPuppi': 'genJetIdx', 'AK8PFPuppi': 'genJetAK8Idx'}[typeJet]


        genjets = events[genJet]
        Ngenjet = ak.num(genjets)
        # They can be matched manually
        if not hasattr(jets_corrected, genJetIdx):
            matched_genjets, matched_jets, deltaR_matched = object_matching(genjets, jets_corrected, dr_min, pt_min)
        # Or the association in NanoAOD it can be used, removing the indices that are not found. That happens because
        # not all the genJet are saved in the NanoAODs.
        else:
            matched_genjets_idx = ak.mask(
                jets_corrected[genJetIdx],
                (jets_corrected[genJetIdx] < Ngenjet) & (jets_corrected[genJetIdx] != -1),
            )
            # this array of indices has already the dimension of the Jet collection
            # in NanoAOD nomatch == -1 --> convert to None with a mask
            matched_objs_mask = ~ak.is_none(matched_genjets_idx, axis=1)
            matched_genjets = genjets[matched_genjets_idx]
            matched_jets = ak.mask(jets_corrected, matched_objs_mask)

        deltaPt = ak.unflatten(
            np.abs(ak.flatten(matched_jets.pt) - ak.flatten(matched_genjets.pt)),
            ak.num(matched_genjets),
        )
        matched_genjets = ak.mask(matched_genjets, deltaPt < pt_min)
        matched_jets = ak.mask(matched_jets, deltaPt < pt_min)

        # Compute energy correction factor with the scaling method
        detSmear = (
            1
            + (scaleFactor - 1)
            * (matched_jets['pt'] - matched_genjets['pt'])
            / matched_jets['pt']
        )
        # Compute energy correction factor with the stochastic method
        np.random.seed(seed)
        seed_dict = {}
        filename = events.metadata['filename']
        entrystart = events.metadata['entrystart']
        entrystop = events.metadata['entrystop']
        seed_dict[f'chunk_{filename}_{entrystart}-{entrystop}'] = seed
        rand_gaus = np.random.normal(
            np.zeros_like(ptResolution_flat), ptResolution_flat
        )
        jersmear = ak.unflatten(rand_gaus, nj)
        sqrt_arg_flat = scaleFactor_flat**2 - 1
        sqrt_arg_flat = ak.where(
            sqrt_arg_flat > 0, sqrt_arg_flat, ak.zeros_like(sqrt_arg_flat)
        )
        sqrt_arg = ak.unflatten(sqrt_arg_flat, nj)
        stochSmear = 1 + jersmear * np.sqrt(sqrt_arg)
        isMatched = ~ak.is_none(matched_jets.pt, axis=1)
        smearFactor = ak.where(isMatched, detSmear, stochSmear)

        jets_smeared = copy.copy(jets_corrected)
        jets_smeared['pt'] = jets_corrected['pt'] * smearFactor
        jets_smeared['mass'] = jets_corrected['mass'] * smearFactor
        if hasattr(jets_smeared, 'msoftdrop'):
            jets_smeared['msoftdrop'] = jets_corrected['msoftdrop'] * smearFactor

        if verbose:
            print()
            print(seed, "JER: isMatched", isMatched)
            print(seed, "JER: matched_jets.pt", matched_jets.pt)
            print(seed, "JER: smearFactor", smearFactor, end='\n\n')

            print(
                seed,
                'JER: corrected pt ratios',
                jets_corrected.pt / jets_corrected.pt_raw,
            )
            print(
                seed,
                'JER: corrected mass ratios',
                jets_corrected.mass / jets_corrected.mass_raw,
            )

            print(seed, 'JER: smeared pt ratios', jets_smeared.pt / jets_corrected.pt)
            print(
                seed,
                'JER: smeared mass ratios',
                jets_smeared.mass / jets_corrected.mass,
            )

            print()
            print(seed, 'JER: corrected columns:', ak.fields(jets_smeared), end='\n\n')

        return jets_smeared, seed_dict
    else:
        return jets_corrected