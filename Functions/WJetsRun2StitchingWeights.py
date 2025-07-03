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
        np.ones(len(events)) * 0.0005740361984679119,
        np.ones(len(events)) * 0.0005417179401811718,
        np.ones(len(events)) * 3.231998251286542e-05,
        np.ones(len(events)) * 1.0447377666692847e-06,
        np.ones(len(events)) * 2.8513473180218254e-08,
        np.ones(len(events)) * 5.132515414485858e-09,
        np.ones(len(events)) * 2.2155943509658546e-10,
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
        np.ones(len(events)) * 0.0005258901520662323,
        np.ones(len(events)) * 0.0005177846541382414,
        np.ones(len(events)) * 3.790019090442677e-05,
        np.ones(len(events)) * 6.823742754422377e-07,
        np.ones(len(events)) * 3.834061047754377e-08,
        np.ones(len(events)) * 7.420626261123658e-09,
        np.ones(len(events)) * 3.0046677991623227e-10,
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
        np.ones(len(events)) * 0.0005320648788878696,
        np.ones(len(events)) * 0.00046001489841421554,
        np.ones(len(events)) * 4.218526776322013e-05,
        np.ones(len(events)) * 9.291154844380106e-07,
        np.ones(len(events)) * 6.973297337622792e-08,
        np.ones(len(events)) * 3.9514424154533895e-09,
        np.ones(len(events)) * 1.545134462227199e-10,
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
        np.ones(len(events)) * 0.0005620821246735322,
        np.ones(len(events)) * 0.0005294085798746587,
        np.ones(len(events)) * 3.3450627832751235e-05,
        np.ones(len(events)) * 9.151573446835511e-07,
        np.ones(len(events)) * 5.3203398382531896e-08,
        np.ones(len(events)) * 2.5131410318924687e-08,
        np.ones(len(events)) * 1.5010491632986337e-10,
        np.ones(len(events)) * 1.0,
      ],
      default=0.0
    )
  elif metadata["year"] == "2016_PreVFP" and metadata["sample"] == "WJetsToLNuHT7OTo100":
    return np.ones(len(events)) * 0.023952174868073262
  elif metadata["year"] == "2016_PreVFP" and metadata["sample"] == "WJetsToLNuHT100To200":
    return np.ones(len(events)) * 0.023268529860187938
  elif metadata["year"] == "2016_PreVFP" and metadata["sample"] == "WJetsToLNuHT200To400":
    return np.ones(len(events)) * 0.005684974752063177
  elif metadata["year"] == "2016_PreVFP" and metadata["sample"] == "WJetsToLNuHT400To600":
    return np.ones(len(events)) * 0.001022123610524815
  elif metadata["year"] == "2016_PreVFP" and metadata["sample"] == "WJetsToLNuHT600To800":
    return np.ones(len(events)) * 0.0001688593271548839
  elif metadata["year"] == "2016_PreVFP" and metadata["sample"] == "WJetsToLNuHT800To1200":
    return np.ones(len(events)) * 7.164157583514717e-05
  elif metadata["year"] == "2016_PreVFP" and metadata["sample"] == "WJetsToLNuHT1200To2500":
    return np.ones(len(events)) * 1.4884872691679189e-05
  elif metadata["year"] == "2016_PostVFP" and metadata["sample"] == "WJetsToLNuHT7OTo100":
    return np.ones(len(events)) * 0.022926264231535674
  elif metadata["year"] == "2016_PostVFP" and metadata["sample"] == "WJetsToLNuHT100To200":
    return np.ones(len(events)) * 0.022748990157591183
  elif metadata["year"] == "2016_PostVFP" and metadata["sample"] == "WJetsToLNuHT200To400":
    return np.ones(len(events)) * 0.006156196429611078
  elif metadata["year"] == "2016_PostVFP" and metadata["sample"] == "WJetsToLNuHT400To600":
    return np.ones(len(events)) * 0.0008260592047835228
  elif metadata["year"] == "2016_PostVFP" and metadata["sample"] == "WJetsToLNuHT600To800":
    return np.ones(len(events)) * 0.0001958075815885109
  elif metadata["year"] == "2016_PostVFP" and metadata["sample"] == "WJetsToLNuHT800To1200":
    return np.ones(len(events)) * 8.614305663289388e-05
  elif metadata["year"] == "2016_PostVFP" and metadata["sample"] == "WJetsToLNuHT1200To2500":
    return np.ones(len(events)) * 1.7333977611210644e-05
  elif metadata["year"] == "2017" and metadata["sample"] == "WJetsToLNuHT7OTo100":
    return np.ones(len(events)) * 0.023060394312598462
  elif metadata["year"] == "2017" and metadata["sample"] == "WJetsToLNuHT100To200":
    return np.ones(len(events)) * 0.02144302415023246
  elif metadata["year"] == "2017" and metadata["sample"] == "WJetsToLNuHT200To400":
    return np.ones(len(events)) * 0.006494881689946622
  elif metadata["year"] == "2017" and metadata["sample"] == "WJetsToLNuHT400To600":
    return np.ones(len(events)) * 0.0009639059192589427
  elif metadata["year"] == "2017" and metadata["sample"] == "WJetsToLNuHT600To800":
    return np.ones(len(events)) * 0.0002640700068420122
  elif metadata["year"] == "2017" and metadata["sample"] == "WJetsToLNuHT800To1200":
    return np.ones(len(events)) * 6.286049951948753e-05
  elif metadata["year"] == "2017" and metadata["sample"] == "WJetsToLNuHT1200To2500":
    return np.ones(len(events)) * 1.2430343768329396e-05
  elif metadata["year"] == "2018" and metadata["sample"] == "WJetsToLNuHT7OTo100":
    return np.ones(len(events)) * 0.023701607294836665
  elif metadata["year"] == "2018" and metadata["sample"] == "WJetsToLNuHT100To200":
    return np.ones(len(events)) * 0.023002789101111497
  elif metadata["year"] == "2018" and metadata["sample"] == "WJetsToLNuHT200To400":
    return np.ones(len(events)) * 0.005783555038922758
  elif metadata["year"] == "2018" and metadata["sample"] == "WJetsToLNuHT400To600":
    return np.ones(len(events)) * 0.000956638127596107
  elif metadata["year"] == "2018" and metadata["sample"] == "WJetsToLNuHT600To800":
    return np.ones(len(events)) * 0.00023065861256829387
  elif metadata["year"] == "2018" and metadata["sample"] == "WJetsToLNuHT800To1200":
    return np.ones(len(events)) * 0.0001585288922794104
  elif metadata["year"] == "2018" and metadata["sample"] == "WJetsToLNuHT1200To2500":
    return np.ones(len(events)) * 1.2251731155527854e-05
  return np.ones(len(events))

wl_func = WeightLambda.wrap_func(
    name="WJetsRun2Stitching",
    function=stitching_func,
    has_variations=False
)

WJetsRun2Stitching = [wl_func]
