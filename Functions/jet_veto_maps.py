import correctionlib
import awkward as ak

def apply_jet_veto_maps(params, events, jetcol, year):
  '''Apply jet veto maps to jets collection'''
  json_file = params["JetVetojsonFiles"][year]
  function = params["JetVetoFunc"][year]

  file = correctionlib.CorrectionSet.from_file(json_file)
  func = file[function]

  j, nj = ak.flatten(events[jetcol]), ak.num(events[jetcol])

  inputs = []
  for ind, var in enumerate(func.inputs):
    if var.name == "type":
      inputs.append("jetvetomap")
    elif var.name == "eta":
      inputs.append(ak.to_numpy(j["eta"]))
    elif var.name == "phi":
      inputs.append(ak.to_numpy(j["phi"]))

  flatvetomaps = func.evaluate(*inputs)      
  vetomaps = ak.unflatten(flatvetomaps, nj)

  # Apply non zero veto maps to veto those jets
  keepmap = (vetomaps == 0)
  return events[jetcol][keepmap]


