import awkward as ak
from pocket_coffea.lib.cut_definition import Cut

############## tt to semileptonic decay ##############
def semileptonic(events, params, year, sample, **kwargs):
    has_one_electron = events.nElectronGood == 1
    has_one_muon = events.nMuonGood == 1

    mask = (
        # Ensure only has one lepton:
        (events.nLeptonGood == 1)
        & 
        # Distinguish between leading muon and leading electron:
        (
            (
                has_one_electron
                &
                (
                    ak.firsts(events.LeptonGood.pt) > params["pt_leading_electron"][year]
                )
            )
            |
            (
                has_one_muon
                &
                (
                    ak.firsts(events.LeptonGood.pt) > params["pt_leading_muon"][year]
                )
            )
        )
        # Events have >= 2 AK4 jets
        & (events.nJetGood >= params["njet"])
        # Events have >= 2 non-b jets
        & (events.nBJetBad >= params["nbjet"])
        # Events have == 1 b jets
        & (events.nBJetGood == params["nbjet"])

    )
    
    return ak.where(ak.is_none(mask), False, mask)

semileptonic_presel = Cut(
    name = "semileptonic",
    params = {
        "njet": 2,
        "nbjet": 2,
        "pt_leading_electron": {
            '2016_PreVFP': 29,
            '2016_PostVFP': 29,
            '2017': 30,
            '2018': 30,
        },
        "pt_leading_muon": {
            '2016_PreVFP': 26,
            '2016_PostVFP': 26,
            '2017': 29,
            '2018': 26,
        }
    },
    function=semileptonic
)