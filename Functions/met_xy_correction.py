import numpy as np
import correctionlib
import awkward as ak
import copy

def met_xy_correction_run2(params, events, METcol,  year, era, isMC):
    '''Apply MET xy corrections to MET collection'''
    metx = events[METcol].pt * np.cos(events[METcol].phi)
    mety = events[METcol].pt * np.sin(events[METcol].phi)
    nPV = events.PV.npvs

    if isMC:
        params_ = params["MET_xy"]["MC"][year]
    else:
        params_ = params["MET_xy"]["Data"][year][era]

    metx = metx - (params_[0][0] * nPV + params_[0][1])
    mety = mety - (params_[1][0] * nPV + params_[1][1])
    pt_corr = np.hypot(metx, mety)
    phi_corr = np.arctan2(mety, metx)
    
    return pt_corr, phi_corr

def met_xy_correction_run3(params, events, METcol,  year, era, isMC):
    '''Apply MET xy corrections to MET collection'''

    json_file = params["METjsonFiles"][year]
    function = params["METFunc"][year]

    file = correctionlib.CorrectionSet.from_file(json_file)
    corr = file[function]

    inputs = []
    for ind, var in enumerate(corr.inputs):
        if var.name == "pt_phi":
            inputs.append(None)
            pt_phi_index = ind
        elif var.name == "met_type":
            inputs.append(METcol)
        elif var.name == "epoch":
            if year == "2022_preEE":
                inputs.append("2022")
            elif year == "2022_postEE":
                inputs.append("2022EE")
            elif year == "2023_preBPix":
                inputs.append("2023")
            elif year == "2023_postBPix":
                inputs.append("2023BPix")
        elif var.name == "dtmc":
            inputs.append("MC" if isMC else "DATA")
        elif var.name == "variation":
            inputs.append("nom")
        elif var.name == "met_pt":
            inputs.append(np.array(events[METcol].pt))
        elif var.name == "met_phi":
            inputs.append(np.array(events[METcol].phi))
        elif var.name == "npvGood":
            inputs.append(np.array(events.PV.npvs))

    pt_inputs = copy.deepcopy(inputs)
    phi_inputs = copy.deepcopy(inputs)
    pt_inputs[pt_phi_index] = "pt"
    phi_inputs[pt_phi_index] = "phi"

    pt_corr = corr.evaluate(*pt_inputs)
    phi_corr = corr.evaluate(*phi_inputs)

    return pt_corr, phi_corr