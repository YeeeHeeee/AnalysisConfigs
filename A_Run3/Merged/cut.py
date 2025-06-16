import awkward as ak
from pocket_coffea.lib.cut_definition import Cut

# tt to semileptonic decay
def semileptonic_merge(events, params, year, sample, **kwargs):

    has_one_electron = events.nElectronGood == 1
    has_one_muon = events.nMuonGood == 1

    mask = (
        ## cut on AK4 ##
        (events.nBJetGood == params["nbjet"])
        ## cut on AK8 ##
        & (events.nFatJet == params["nfatjet"])  
        & (events.FatJet.pt >= params["pt"])
        ## cut on lepton ## 
        & (events.nLeptonGood == 1)
        &
        (
            (
                has_one_electron
                & (
                    ak.firsts(events.LeptonGood.pt)
                    > params["pt_leading_electron"]
                )
            )
            | (
                has_one_muon
                & (
                    ak.firsts(events.LeptonGood.pt) > params["pt_leading_muon"]
                )
            )
        )   
    )
    return ak.where(ak.is_none(mask), False, mask)

semileptonic_presel_merge = Cut(
    name="semileptonic_merge",
    params={
        "pt": 500,
        "eta": 2.4,
        "nfatjet": 1,
        "jetpt": 30,
        "nbjet": 1,
        "pt_leading_electron": 60,
        "pt_leading_muon": 60
    },
    function=semileptonic_merge,
)
