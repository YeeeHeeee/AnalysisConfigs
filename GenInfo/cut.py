import awkward as ak
from pocket_coffea.lib.cut_definition import Cut

# tt to semileptonic decay
def LeptonAndB(events, params, year, sample, **kwargs):

    mask = (
        events.nLeptonGood >= 0
    )
    return ak.where(ak.is_none(mask), False, mask)

gen_info_presel = Cut(
    name="LeptonAndB",
    function=LeptonAndB,
    params={}
)