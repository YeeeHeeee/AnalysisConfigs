from pocket_coffea.lib.weights import WeightLambda
import numpy as np

def stitching_func(params, metadata, events, size, shape_variations):
  if metadata["year"] == "2022_preEE" and metadata["sample"] == "WJetsToLNu":
    return np.select(
      condlist=[
        (events.LHE.HT>=0.0) & (events.LHE.HT<40.0) & ((events.LNu.mass>=0) & (events.LNu.mass<120)),
        (events.LHE.HT>=40.0) & (events.LHE.HT<100.0) & ((events.LNu.mass>=0) & (events.LNu.mass<120)),
        (events.LHE.HT>=100.0) & (events.LHE.HT<400.0) & ((events.LNu.mass>=0) & (events.LNu.mass<120)),
        (events.LHE.HT>=400.0) & (events.LHE.HT<800.0) & ((events.LNu.mass>=0) & (events.LNu.mass<120)),
        (events.LHE.HT>=800.0) & (events.LHE.HT<1500.0) & ((events.LNu.mass>=0) & (events.LNu.mass<120)),
        (events.LHE.HT>=1500.0) & (events.LHE.HT<2500.0) & ((events.LNu.mass>=0) & (events.LNu.mass<120)),
        (events.LHE.HT>=2500.0) & (events.LHE.HT<5000.0) & ((events.LNu.mass>=0) & (events.LNu.mass<120)),
        (events.LHE.HT>=0.0) & (events.LHE.HT<40.0) & (events.LNu.mass>=120),
        (events.LHE.HT>=40.0) & (events.LHE.HT<100.0) & (events.LNu.mass>=120),
        (events.LHE.HT>=100.0) & (events.LHE.HT<400.0) & (events.LNu.mass>=120),
        (events.LHE.HT>=400.0) & (events.LHE.HT<800.0) & (events.LNu.mass>=120),
        (events.LHE.HT>=800.0) & (events.LHE.HT<1500.0) & (events.LNu.mass>=120),
        (events.LHE.HT>=1500.0) & (events.LHE.HT<2500.0) & (events.LNu.mass>=120),
        (events.LHE.HT>=2500.0) & (events.LHE.HT<5000.0) & (events.LNu.mass>=120),
      ],
      choicelist=[
        np.ones(len(events)) * 1.0,
        np.ones(len(events)) * 0.005755922103451193,
        np.ones(len(events)) * 0.0008583431479992099,
        np.ones(len(events)) * 1.1472612592703246e-06,
        np.ones(len(events)) * 1.0790475662086518e-08,
        np.ones(len(events)) * 4.0274747845928086e-11,
        np.ones(len(events)) * 8.19585749712385e-14,
        np.ones(len(events)) * 1.0,
        np.ones(len(events)) * 1.4599911686530576e-07,
        np.ones(len(events)) * 3.51631615409894e-08,
        np.ones(len(events)) * 9.100264149131543e-11,
        np.ones(len(events)) * 8.770174374345613e-13,
        np.ones(len(events)) * 9.009292524893863e-15,
        np.ones(len(events)) * 0.0,
      ],
      default=0.0
    )
  if metadata["year"] == "2022_postEE" and metadata["sample"] == "WJetsToLNu":
    return np.select(
      condlist=[
        (events.LHE.HT>=0.0) & (events.LHE.HT<40.0) & ((events.LNu.mass>=0) & (events.LNu.mass<120)),
        (events.LHE.HT>=40.0) & (events.LHE.HT<100.0) & ((events.LNu.mass>=0) & (events.LNu.mass<120)),
        (events.LHE.HT>=100.0) & (events.LHE.HT<400.0) & ((events.LNu.mass>=0) & (events.LNu.mass<120)),
        (events.LHE.HT>=400.0) & (events.LHE.HT<800.0) & ((events.LNu.mass>=0) & (events.LNu.mass<120)),
        (events.LHE.HT>=800.0) & (events.LHE.HT<1500.0) & ((events.LNu.mass>=0) & (events.LNu.mass<120)),
        (events.LHE.HT>=1500.0) & (events.LHE.HT<2500.0) & ((events.LNu.mass>=0) & (events.LNu.mass<120)),
        (events.LHE.HT>=2500.0) & (events.LHE.HT<5000.0) & ((events.LNu.mass>=0) & (events.LNu.mass<120)),
        (events.LHE.HT>=0.0) & (events.LHE.HT<40.0) & (events.LNu.mass>=120),
        (events.LHE.HT>=40.0) & (events.LHE.HT<100.0) & (events.LNu.mass>=120),
        (events.LHE.HT>=100.0) & (events.LHE.HT<400.0) & (events.LNu.mass>=120),
        (events.LHE.HT>=400.0) & (events.LHE.HT<800.0) & (events.LNu.mass>=120),
        (events.LHE.HT>=800.0) & (events.LHE.HT<1500.0) & (events.LNu.mass>=120),
        (events.LHE.HT>=1500.0) & (events.LHE.HT<2500.0) & (events.LNu.mass>=120),
        (events.LHE.HT>=2500.0) & (events.LHE.HT<5000.0) & (events.LNu.mass>=120),
      ],
      choicelist=[
        np.ones(len(events)) * 1.0,
        np.ones(len(events)) * 0.005848017156001407,
        np.ones(len(events)) * 0.0008687784725008341,
        np.ones(len(events)) * 1.1477058629506203e-06,
        np.ones(len(events)) * 1.0902523735651541e-08,
        np.ones(len(events)) * 3.916485038280941e-11,
        np.ones(len(events)) * 1.1105294492496935e-13,
        np.ones(len(events)) * 1.0,
        np.ones(len(events)) * 1.4312671574157142e-07,
        np.ones(len(events)) * 3.498431408297812e-08,
        np.ones(len(events)) * 8.34327409928116e-11,
        np.ones(len(events)) * 1.0127082186657795e-12,
        np.ones(len(events)) * 4.963321650779388e-15,
        np.ones(len(events)) * 8.645084226452983e-18,
      ],
      default=0.0
    )
  if metadata["year"] == "2023_preBPix" and metadata["sample"] == "WJetsToLNu":
    return np.select(
      condlist=[
        (events.LHE.HT>=0.0) & (events.LHE.HT<40.0) & ((events.LNu.mass>=0) & (events.LNu.mass<120)),
        (events.LHE.HT>=40.0) & (events.LHE.HT<100.0) & ((events.LNu.mass>=0) & (events.LNu.mass<120)),
        (events.LHE.HT>=100.0) & (events.LHE.HT<400.0) & ((events.LNu.mass>=0) & (events.LNu.mass<120)),
        (events.LHE.HT>=400.0) & (events.LHE.HT<800.0) & ((events.LNu.mass>=0) & (events.LNu.mass<120)),
        (events.LHE.HT>=800.0) & (events.LHE.HT<1500.0) & ((events.LNu.mass>=0) & (events.LNu.mass<120)),
        (events.LHE.HT>=1500.0) & (events.LHE.HT<2500.0) & ((events.LNu.mass>=0) & (events.LNu.mass<120)),
        (events.LHE.HT>=2500.0) & (events.LHE.HT<5000.0) & ((events.LNu.mass>=0) & (events.LNu.mass<120)),
        (events.LHE.HT>=0.0) & (events.LHE.HT<40.0) & (events.LNu.mass>=120),
        (events.LHE.HT>=40.0) & (events.LHE.HT<100.0) & (events.LNu.mass>=120),
        (events.LHE.HT>=100.0) & (events.LHE.HT<400.0) & (events.LNu.mass>=120),
        (events.LHE.HT>=400.0) & (events.LHE.HT<800.0) & (events.LNu.mass>=120),
        (events.LHE.HT>=800.0) & (events.LHE.HT<1500.0) & (events.LNu.mass>=120),
        (events.LHE.HT>=1500.0) & (events.LHE.HT<2500.0) & (events.LNu.mass>=120),
        (events.LHE.HT>=2500.0) & (events.LHE.HT<5000.0) & (events.LNu.mass>=120),
      ],
      choicelist=[
        np.ones(len(events)) * 1.0,
        np.ones(len(events)) * 0.006083108528589771,
        np.ones(len(events)) * 0.0008983629440460499,
        np.ones(len(events)) * 1.1436702729339062e-06,
        np.ones(len(events)) * 1.0714994613921108e-08,
        np.ones(len(events)) * 3.637758777185127e-11,
        np.ones(len(events)) * 1.4870557581027056e-13,
        np.ones(len(events)) * 1.0,
        np.ones(len(events)) * 1.4167728628179853e-07,
        np.ones(len(events)) * 3.4560383773730696e-08,
        np.ones(len(events)) * 8.798357794773811e-11,
        np.ones(len(events)) * 9.169612870500124e-13,
        np.ones(len(events)) * 9.950170878162927e-16,
        np.ones(len(events)) * 0.0,
      ],
      default=0.0
    )
  if metadata["year"] == "2023_postBPix" and metadata["sample"] == "WJetsToLNu":
    return np.select(
      condlist=[
        (events.LHE.HT>=0.0) & (events.LHE.HT<40.0) & ((events.LNu.mass>=0) & (events.LNu.mass<120)),
        (events.LHE.HT>=40.0) & (events.LHE.HT<100.0) & ((events.LNu.mass>=0) & (events.LNu.mass<120)),
        (events.LHE.HT>=100.0) & (events.LHE.HT<400.0) & ((events.LNu.mass>=0) & (events.LNu.mass<120)),
        (events.LHE.HT>=400.0) & (events.LHE.HT<800.0) & ((events.LNu.mass>=0) & (events.LNu.mass<120)),
        (events.LHE.HT>=800.0) & (events.LHE.HT<1500.0) & ((events.LNu.mass>=0) & (events.LNu.mass<120)),
        (events.LHE.HT>=1500.0) & (events.LHE.HT<2500.0) & ((events.LNu.mass>=0) & (events.LNu.mass<120)),
        (events.LHE.HT>=2500.0) & (events.LHE.HT<5000.0) & ((events.LNu.mass>=0) & (events.LNu.mass<120)),
        (events.LHE.HT>=0.0) & (events.LHE.HT<40.0) & (events.LNu.mass>=120),
        (events.LHE.HT>=40.0) & (events.LHE.HT<100.0) & (events.LNu.mass>=120),
        (events.LHE.HT>=100.0) & (events.LHE.HT<400.0) & (events.LNu.mass>=120),
        (events.LHE.HT>=400.0) & (events.LHE.HT<800.0) & (events.LNu.mass>=120),
        (events.LHE.HT>=800.0) & (events.LHE.HT<1500.0) & (events.LNu.mass>=120),
        (events.LHE.HT>=1500.0) & (events.LHE.HT<2500.0) & (events.LNu.mass>=120),
        (events.LHE.HT>=2500.0) & (events.LHE.HT<5000.0) & (events.LNu.mass>=120),
      ],
      choicelist=[
        np.ones(len(events)) * 1.0,
        np.ones(len(events)) * 0.006367015011974064,
        np.ones(len(events)) * 0.0009273538013485699,
        np.ones(len(events)) * 1.169457061780796e-06,
        np.ones(len(events)) * 1.0347837564110143e-08,
        np.ones(len(events)) * 3.5114490903700883e-11,
        np.ones(len(events)) * 1.1601656794663466e-13,
        np.ones(len(events)) * 1.0,
        np.ones(len(events)) * 1.4782068784753769e-07,
        np.ones(len(events)) * 3.5481757318780544e-08,
        np.ones(len(events)) * 8.331033448485315e-11,
        np.ones(len(events)) * 6.319183462438486e-13,
        np.ones(len(events)) * 5.5001164510652775e-15,
        np.ones(len(events)) * 1.1402498174009709e-16,
      ],
      default=0.0
    )
  elif metadata["year"] == "2022_preEE" and metadata["sample"] == "WJetsToLNuHT40To100MLNu0To120":
    return np.ones(len(events)) * 0.07564913392888378
  elif metadata["year"] == "2022_preEE" and metadata["sample"] == "WJetsToLNuHT100To400MLNu0To120":
    return np.ones(len(events)) * 0.029284917535132186
  elif metadata["year"] == "2022_preEE" and metadata["sample"] == "WJetsToLNuHT400To800MLNu0To120":
    return np.ones(len(events)) * 0.0010711022094375157
  elif metadata["year"] == "2022_preEE" and metadata["sample"] == "WJetsToLNuHT800To1500MLNu0To120":
    return np.ones(len(events)) * 0.00010387721379422993
  elif metadata["year"] == "2022_preEE" and metadata["sample"] == "WJetsToLNuHT1500To2500MLNu0To120":
    return np.ones(len(events)) * 6.346238873876876e-06
  elif metadata["year"] == "2022_preEE" and metadata["sample"] == "WJetsToLNuHT2500MLNu0To120":
    return np.ones(len(events)) * 2.862840808903488e-07
  elif metadata["year"] == "2022_preEE" and metadata["sample"] == "WJetsToLNuHT40To100MLNu120":
    return np.ones(len(events)) * 0.00038209827996153506
  elif metadata["year"] == "2022_preEE" and metadata["sample"] == "WJetsToLNuHT100To400MLNu120":
    return np.ones(len(events)) * 0.00018751842657334096
  elif metadata["year"] == "2022_preEE" and metadata["sample"] == "WJetsToLNuHT400To800MLNu120":
    return np.ones(len(events)) * 9.5395304644953e-06
  elif metadata["year"] == "2022_preEE" and metadata["sample"] == "WJetsToLNuHT800To1500MLNu120":
    return np.ones(len(events)) * 9.364920914955941e-07
  elif metadata["year"] == "2022_preEE" and metadata["sample"] == "WJetsToLNuHT1500To2500MLNu120":
    return np.ones(len(events)) * 9.491729307609747e-08
  elif metadata["year"] == "2022_preEE" and metadata["sample"] == "WJetsToLNuHT2500MLNu120":
    return np.ones(len(events)) * 0.0
  elif metadata["year"] == "2022_postEE" and metadata["sample"] == "WJetsToLNuHT40To100MLNu0To120":
    return np.ones(len(events)) * 0.07624839572964483
  elif metadata["year"] == "2022_postEE" and metadata["sample"] == "WJetsToLNuHT100To400MLNu0To120":
    return np.ones(len(events)) * 0.029462241877809523
  elif metadata["year"] == "2022_postEE" and metadata["sample"] == "WJetsToLNuHT400To800MLNu0To120":
    return np.ones(len(events)) * 0.0010713097337940471
  elif metadata["year"] == "2022_postEE" and metadata["sample"] == "WJetsToLNuHT800To1500MLNu0To120":
    return np.ones(len(events)) * 0.00010441515032209893
  elif metadata["year"] == "2022_postEE" and metadata["sample"] == "WJetsToLNuHT1500To2500MLNu0To120":
    return np.ones(len(events)) * 6.2581826740097274e-06
  elif metadata["year"] == "2022_postEE" and metadata["sample"] == "WJetsToLNuHT2500MLNu0To120":
    return np.ones(len(events)) * 3.3324607263245736e-07
  elif metadata["year"] == "2022_postEE" and metadata["sample"] == "WJetsToLNuHT40To100MLNu120":
    return np.ones(len(events)) * 0.00037832088926771494
  elif metadata["year"] == "2022_postEE" and metadata["sample"] == "WJetsToLNuHT100To400MLNu120":
    return np.ones(len(events)) * 0.00018704093899217863
  elif metadata["year"] == "2022_postEE" and metadata["sample"] == "WJetsToLNuHT400To800MLNu120":
    return np.ones(len(events)) * 9.134152450328961e-06
  elif metadata["year"] == "2022_postEE" and metadata["sample"] == "WJetsToLNuHT800To1500MLNu120":
    return np.ones(len(events)) * 1.006334049242474e-06
  elif metadata["year"] == "2022_postEE" and metadata["sample"] == "WJetsToLNuHT1500To2500MLNu120":
    return np.ones(len(events)) * 7.045084563565837e-08
  elif metadata["year"] == "2022_postEE" and metadata["sample"] == "WJetsToLNuHT2500MLNu120":
    return np.ones(len(events)) * 2.9402524086297396e-09
  elif metadata["year"] == "2023_preBPix" and metadata["sample"] == "WJetsToLNuHT40To100MLNu0To120":
    return np.ones(len(events)) * 0.0777566995134128
  elif metadata["year"] == "2023_preBPix" and metadata["sample"] == "WJetsToLNuHT100To400MLNu0To120":
    return np.ones(len(events)) * 0.029959237107556907
  elif metadata["year"] == "2023_preBPix" and metadata["sample"] == "WJetsToLNuHT400To800MLNu0To120":
    return np.ones(len(events)) * 0.0010694245952624303
  elif metadata["year"] == "2023_preBPix" and metadata["sample"] == "WJetsToLNuHT800To1500MLNu0To120":
    return np.ones(len(events)) * 0.00010351325760070543
  elif metadata["year"] == "2023_preBPix" and metadata["sample"] == "WJetsToLNuHT1500To2500MLNu0To120":
    return np.ones(len(events)) * 6.031383570170938e-06
  elif metadata["year"] == "2023_preBPix" and metadata["sample"] == "WJetsToLNuHT2500MLNu0To120":
    return np.ones(len(events)) * 3.8562361936251844e-07
  elif metadata["year"] == "2023_preBPix" and metadata["sample"] == "WJetsToLNuHT40To100MLNu120":
    return np.ones(len(events)) * 0.0003764004067603343
  elif metadata["year"] == "2023_preBPix" and metadata["sample"] == "WJetsToLNuHT100To400MLNu120":
    return np.ones(len(events)) * 0.00018590422958962116
  elif metadata["year"] == "2023_preBPix" and metadata["sample"] == "WJetsToLNuHT400To800MLNu120":
    return np.ones(len(events)) * 9.37995618006806e-06
  elif metadata["year"] == "2023_preBPix" and metadata["sample"] == "WJetsToLNuHT800To1500MLNu120":
    return np.ones(len(events)) * 9.57580955872229e-07
  elif metadata["year"] == "2023_preBPix" and metadata["sample"] == "WJetsToLNuHT1500To2500MLNu120":
    return np.ones(len(events)) * 3.1543891450109506e-08
  elif metadata["year"] == "2023_preBPix" and metadata["sample"] == "WJetsToLNuHT2500MLNu120":
    return np.ones(len(events)) * 0.0
  elif metadata["year"] == "2023_postBPix" and metadata["sample"] == "WJetsToLNuHT40To100MLNu0To120":
    return np.ones(len(events)) * 0.07953914842271925
  elif metadata["year"] == "2023_postBPix" and metadata["sample"] == "WJetsToLNuHT100To400MLNu0To120":
    return np.ones(len(events)) * 0.030438360932804748
  elif metadata["year"] == "2023_postBPix" and metadata["sample"] == "WJetsToLNuHT400To800MLNu0To120":
    return np.ones(len(events)) * 0.001081413747901781
  elif metadata["year"] == "2023_postBPix" and metadata["sample"] == "WJetsToLNuHT800To1500MLNu0To120":
    return np.ones(len(events)) * 0.00010172432087280015
  elif metadata["year"] == "2023_postBPix" and metadata["sample"] == "WJetsToLNuHT1500To2500MLNu0To120":
    return np.ones(len(events)) * 5.925748130191484e-06
  elif metadata["year"] == "2023_postBPix" and metadata["sample"] == "WJetsToLNuHT2500MLNu0To120":
    return np.ones(len(events)) * 3.406120490332384e-07
  elif metadata["year"] == "2023_postBPix" and metadata["sample"] == "WJetsToLNuHT40To100MLNu120":
    return np.ones(len(events)) * 0.000384474532312066
  elif metadata["year"] == "2023_postBPix" and metadata["sample"] == "WJetsToLNuHT100To400MLNu120":
    return np.ones(len(events)) * 0.00018836601620203535
  elif metadata["year"] == "2023_postBPix" and metadata["sample"] == "WJetsToLNuHT400To800MLNu120":
    return np.ones(len(events)) * 9.127449505634777e-06
  elif metadata["year"] == "2023_postBPix" and metadata["sample"] == "WJetsToLNuHT800To1500MLNu120":
    return np.ones(len(events)) * 7.949329193356187e-07
  elif metadata["year"] == "2023_postBPix" and metadata["sample"] == "WJetsToLNuHT1500To2500MLNu120":
    return np.ones(len(events)) * 7.416276997972263e-08
  elif metadata["year"] == "2023_postBPix" and metadata["sample"] == "WJetsToLNuHT2500MLNu120":
    return np.ones(len(events)) * 1.0678248065113353e-08
  return np.ones(len(events))

wl_func = WeightLambda.wrap_func(
    name="WJetsRun3Stitching",
    function=stitching_func,
    has_variations=False
)

WJetsRun3Stitching = [wl_func]
