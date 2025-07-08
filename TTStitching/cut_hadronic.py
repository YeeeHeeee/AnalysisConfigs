import awkward as ak
from pocket_coffea.lib.cut_definition import Cut

# tt to semileptonic decay
def hadronic(events, params, year, sample, **kwargs):

    l_mask = (events["LHEPart"].pdgId == 11) | (events["LHEPart"].pdgId == -11) | \
            (events["LHEPart"].pdgId == 13) | (events["LHEPart"].pdgId == -13) | \
            (events["LHEPart"].pdgId == 15) | (events["LHEPart"].pdgId == -15)
    nu_mask = (events["LHEPart"].pdgId == 12) | (events["LHEPart"].pdgId == -12) | \
            (events["LHEPart"].pdgId == 14) | (events["LHEPart"].pdgId == -14) | \
            (events["LHEPart"].pdgId == 16) | (events["LHEPart"].pdgId == -16)

    count_l = ak.num(events["LHEPart"][l_mask])
    count_nu = ak.num(events["LHEPart"][nu_mask])

    mask = (
        ## cut on AK4 ##
        (count_l == 0)
        & (count_nu == 0)
    )
    return ak.where(ak.is_none(mask), False, mask)

hadronic_presel = Cut(
    name="hadronic",
    params={},
    function=hadronic,
)
