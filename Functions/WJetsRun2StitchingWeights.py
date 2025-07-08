from pocket_coffea.lib.weights import WeightLambda
import numpy as np

def stitching_func(params, metadata, events, size, shape_variations):
  if metadata["year"] == "2016_PreVFP" and metadata["sample"] == "WJetsToLNu":
    return np.select(
      condlist=[
        (events.LHE.HT>=0.0) & (events.LHE.HT<70.0),
        (events.LHE.HT>=70.0) & (events.LHE.HT<100.0),
        (events.LHE.HT>=100.0) & (events.LHE.HT<200.0),
        (events.LHE.HT>=200.0) & (events.LHE.HT<400.0),
        (events.LHE.HT>=400.0) & (events.LHE.HT<600.0),
        (events.LHE.HT>=600.0) & (events.LHE.HT<800.0),
        (events.LHE.HT>=800.0) & (events.LHE.HT<1200.0),
        (events.LHE.HT>=1200.0) & (events.LHE.HT<2500.0),
        (events.LHE.HT>=2500.0) & (events.LHE.HT<5000.0),
      ],
      choicelist=[
        np.ones(len(events)) * 1.0,
        np.ones(len(events)) * 6.053537454182782e-05,
        np.ones(len(events)) * 5.0857307877842474e-05,
        np.ones(len(events)) * 3.8694943207027956e-05,
        np.ones(len(events)) * 9.06949122549848e-06,
        np.ones(len(events)) * 0.00614717955771354,
        np.ones(len(events)) * 0.00244411940646801,
        np.ones(len(events)) * 0.0006024704898207631,
        np.ones(len(events)) * 1.0,
      ],
      default=0.0
    )
  if metadata["year"] == "2016_PostVFP" and metadata["sample"] == "WJetsToLNu":
    return np.select(
      condlist=[
        (events.LHE.HT>=0.0) & (events.LHE.HT<70.0),
        (events.LHE.HT>=70.0) & (events.LHE.HT<100.0),
        (events.LHE.HT>=100.0) & (events.LHE.HT<200.0),
        (events.LHE.HT>=200.0) & (events.LHE.HT<400.0),
        (events.LHE.HT>=400.0) & (events.LHE.HT<600.0),
        (events.LHE.HT>=600.0) & (events.LHE.HT<800.0),
        (events.LHE.HT>=800.0) & (events.LHE.HT<1200.0),
        (events.LHE.HT>=1200.0) & (events.LHE.HT<2500.0),
        (events.LHE.HT>=2500.0) & (events.LHE.HT<5000.0),
      ],
      choicelist=[
        np.ones(len(events)) * 1.0,
        np.ones(len(events)) * 0.08667155673812035,
        np.ones(len(events)) * 0.08480219236686763,
        np.ones(len(events)) * 0.03181076512422451,
        np.ones(len(events)) * 0.03044316546737202,
        np.ones(len(events)) * 0.00694382702290774,
        np.ones(len(events)) * 0.003238095072910146,
        np.ones(len(events)) * 0.0006663419239524197,
        np.ones(len(events)) * 1.0,
      ],
      default=0.0
    )
  if metadata["year"] == "2017" and metadata["sample"] == "WJetsToLNu":
    return np.select(
      condlist=[
        (events.LHE.HT>=0.0) & (events.LHE.HT<70.0),
        (events.LHE.HT>=70.0) & (events.LHE.HT<100.0),
        (events.LHE.HT>=100.0) & (events.LHE.HT<200.0),
        (events.LHE.HT>=200.0) & (events.LHE.HT<400.0),
        (events.LHE.HT>=400.0) & (events.LHE.HT<600.0),
        (events.LHE.HT>=600.0) & (events.LHE.HT<800.0),
        (events.LHE.HT>=800.0) & (events.LHE.HT<1200.0),
        (events.LHE.HT>=1200.0) & (events.LHE.HT<2500.0),
        (events.LHE.HT>=2500.0) & (events.LHE.HT<5000.0),
      ],
      choicelist=[
        np.ones(len(events)) * 1.0,
        np.ones(len(events)) * 1.8137469853300753e-05,
        np.ones(len(events)) * 1.964988433566148e-05,
        np.ones(len(events)) * 4.546324132433333e-06,
        np.ones(len(events)) * 4.129190060073219e-06,
        np.ones(len(events)) * 9.1688302712788e-07,
        np.ones(len(events)) * 0.0013061387452087628,
        np.ones(len(events)) * 0.00027685062437891075,
        np.ones(len(events)) * 1.0,
      ],
      default=0.0
    )
  if metadata["year"] == "2018" and metadata["sample"] == "WJetsToLNu":
    return np.select(
      condlist=[
        (events.LHE.HT>=0.0) & (events.LHE.HT<70.0),
        (events.LHE.HT>=70.0) & (events.LHE.HT<100.0),
        (events.LHE.HT>=100.0) & (events.LHE.HT<200.0),
        (events.LHE.HT>=200.0) & (events.LHE.HT<400.0),
        (events.LHE.HT>=400.0) & (events.LHE.HT<600.0),
        (events.LHE.HT>=600.0) & (events.LHE.HT<800.0),
        (events.LHE.HT>=800.0) & (events.LHE.HT<1200.0),
        (events.LHE.HT>=1200.0) & (events.LHE.HT<2500.0),
        (events.LHE.HT>=2500.0) & (events.LHE.HT<5000.0),
      ],
      choicelist=[
        np.ones(len(events)) * 1.0,
        np.ones(len(events)) * 1.0374764194843457e-05,
        np.ones(len(events)) * 1.3436852651333747e-05,
        np.ones(len(events)) * 3.465124513006681e-06,
        np.ones(len(events)) * 2.9662277899420162e-06,
        np.ones(len(events)) * 6.652299925337401e-07,
        np.ones(len(events)) * 3.3199485070633147e-07,
        np.ones(len(events)) * 0.00023075538558485755,
        np.ones(len(events)) * 1.0,
      ],
      default=0.0
    )
  elif metadata["year"] == "2016_PreVFP" and metadata["sample"] == "WJetsToLNuHT7OTo100":
    return np.ones(len(events)) * 0.023964488457510476
  elif metadata["year"] == "2016_PreVFP" and metadata["sample"] == "WJetsToLNuHT100To200":
    return np.ones(len(events)) * 0.023279963142934873
  elif metadata["year"] == "2016_PreVFP" and metadata["sample"] == "WJetsToLNuHT200To400":
    return np.ones(len(events)) * 0.005684939531164952
  elif metadata["year"] == "2016_PreVFP" and metadata["sample"] == "WJetsToLNuHT400To600":
    return np.ones(len(events)) * 0.0010221156491274051
  elif metadata["year"] == "2016_PreVFP" and metadata["sample"] == "WJetsToLNuHT600To800":
    return np.ones(len(events)) * 0.00016782136288964064
  elif metadata["year"] == "2016_PreVFP" and metadata["sample"] == "WJetsToLNuHT800To1200":
    return np.ones(len(events)) * 7.146649247998813e-05
  elif metadata["year"] == "2016_PreVFP" and metadata["sample"] == "WJetsToLNuHT1200To2500":
    return np.ones(len(events)) * 1.4875908504517582e-05
  elif metadata["year"] == "2016_PostVFP" and metadata["sample"] == "WJetsToLNuHT7OTo100":
    return np.ones(len(events)) * 0.020950225732266358
  elif metadata["year"] == "2016_PostVFP" and metadata["sample"] == "WJetsToLNuHT100To200":
    return np.ones(len(events)) * 0.020830613869394186
  elif metadata["year"] == "2016_PostVFP" and metadata["sample"] == "WJetsToLNuHT200To400":
    return np.ones(len(events)) * 0.00596058903877212
  elif metadata["year"] == "2016_PostVFP" and metadata["sample"] == "WJetsToLNuHT400To600":
    return np.ones(len(events)) * 0.0008009118969865824
  elif metadata["year"] == "2016_PostVFP" and metadata["sample"] == "WJetsToLNuHT600To800":
    return np.ones(len(events)) * 0.00019444793573229017
  elif metadata["year"] == "2016_PostVFP" and metadata["sample"] == "WJetsToLNuHT800To1200":
    return np.ones(len(events)) * 8.586411815639534e-05
  elif metadata["year"] == "2016_PostVFP" and metadata["sample"] == "WJetsToLNuHT1200To2500":
    return np.ones(len(events)) * 1.7322427319652646e-05
  elif metadata["year"] == "2017" and metadata["sample"] == "WJetsToLNuHT7OTo100":
    return np.ones(len(events)) * 0.023072249154104395
  elif metadata["year"] == "2017" and metadata["sample"] == "WJetsToLNuHT100To200":
    return np.ones(len(events)) * 0.021452469579043287
  elif metadata["year"] == "2017" and metadata["sample"] == "WJetsToLNuHT200To400":
    return np.ones(len(events)) * 0.006495125500106488
  elif metadata["year"] == "2017" and metadata["sample"] == "WJetsToLNuHT400To600":
    return np.ones(len(events)) * 0.0009639027594407762
  elif metadata["year"] == "2017" and metadata["sample"] == "WJetsToLNuHT600To800":
    return np.ones(len(events)) * 0.0002640697625211807
  elif metadata["year"] == "2017" and metadata["sample"] == "WJetsToLNuHT800To1200":
    return np.ones(len(events)) * 6.277839033296108e-05
  elif metadata["year"] == "2017" and metadata["sample"] == "WJetsToLNuHT1200To2500":
    return np.ones(len(events)) * 1.2426901451743393e-05
  elif metadata["year"] == "2018" and metadata["sample"] == "WJetsToLNuHT7OTo100":
    return np.ones(len(events)) * 0.023714689559324327
  elif metadata["year"] == "2018" and metadata["sample"] == "WJetsToLNuHT100To200":
    return np.ones(len(events)) * 0.023014662330726313
  elif metadata["year"] == "2018" and metadata["sample"] == "WJetsToLNuHT200To400":
    return np.ones(len(events)) * 0.005783728116131987
  elif metadata["year"] == "2018" and metadata["sample"] == "WJetsToLNuHT400To600":
    return np.ones(len(events)) * 0.00095663610733996
  elif metadata["year"] == "2018" and metadata["sample"] == "WJetsToLNuHT600To800":
    return np.ones(len(events)) * 0.0002306584573850086
  elif metadata["year"] == "2018" and metadata["sample"] == "WJetsToLNuHT800To1200":
    return np.ones(len(events)) * 0.0001585288340009812
  elif metadata["year"] == "2018" and metadata["sample"] == "WJetsToLNuHT1200To2500":
    return np.ones(len(events)) * 1.2248903260215276e-05
  return np.ones(len(events))

wl_func = WeightLambda.wrap_func(
    name="WJetsRun2Stitching",
    function=stitching_func,
    has_variations=False
)

WJetsRun2Stitching = [wl_func]
