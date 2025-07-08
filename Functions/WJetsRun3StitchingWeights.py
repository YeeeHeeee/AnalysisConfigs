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
        np.ones(len(events)) * 0.042211298608064295,
        np.ones(len(events)) * 0.024125704034901805,
        np.ones(len(events)) * 0.00666287211389584,
        np.ones(len(events)) * 0.006103685168835849,
        np.ones(len(events)) * 0.001123950705423108,
        np.ones(len(events)) * 4.676623913434717e-05,
        np.ones(len(events)) * 1.0,
        np.ones(len(events)) * 0.00710745973484424,
        np.ones(len(events)) * 0.007404075336522949,
        np.ones(len(events)) * 0.001594344719712577,
        np.ones(len(events)) * 0.00016313967751262104,
        np.ones(len(events)) * 1.6435893106193558e-05,
        np.ones(len(events)) * nan,
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
        np.ones(len(events)) * 0.1018570379309504,
        np.ones(len(events)) * 0.051251411651421525,
        np.ones(len(events)) * 0.014858284464528249,
        np.ones(len(events)) * 0.01294488055684442,
        np.ones(len(events)) * 0.002486788430434093,
        np.ones(len(events)) * 0.00013926324504083467,
        np.ones(len(events)) * 1.0,
        np.ones(len(events)) * 0.01690950299353732,
        np.ones(len(events)) * 0.017247891927396966,
        np.ones(len(events)) * 0.003703747825627973,
        np.ones(len(events)) * 0.0004027631603583225,
        np.ones(len(events)) * 2.7954573263400488e-05,
        np.ones(len(events)) * 1.1348746158936156e-06,
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
        np.ones(len(events)) * 0.051369997535209655,
        np.ones(len(events)) * 0.029784077460869987,
        np.ones(len(events)) * 0.007559463544539573,
        np.ones(len(events)) * 0.006794620204157047,
        np.ones(len(events)) * 0.0011092751536458795,
        np.ones(len(events)) * 7.964926853494944e-05,
        np.ones(len(events)) * 1.0,
        np.ones(len(events)) * 0.007789797168599686,
        np.ones(len(events)) * 0.00838803580621803,
        np.ones(len(events)) * 0.0017065938713679847,
        np.ones(len(events)) * 0.0001988642542682738,
        np.ones(len(events)) * 6.654329972499241e-06,
        np.ones(len(events)) * nan,
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
        np.ones(len(events)) * 0.05291568085410299,
        np.ones(len(events)) * 0.029448147271713167,
        np.ones(len(events)) * 0.007707324676469757,
        np.ones(len(events)) * 0.006657397627195943,
        np.ones(len(events)) * 0.001069576471664212,
        np.ones(len(events)) * 6.402585464199861e-05,
        np.ones(len(events)) * 1.0,
        np.ones(len(events)) * 0.008295568811644989,
        np.ones(len(events)) * 0.008438893445915715,
        np.ones(len(events)) * 0.0016826106030981287,
        np.ones(len(events)) * 0.00015158870124510997,
        np.ones(len(events)) * 1.403307486588504e-05,
        np.ones(len(events)) * 2.0776897152655225e-06,
      ],
      default=0.0
    )
  elif metadata["year"] == "2022_preEE" and metadata["sample"] == "WJetsToLNuHT40To100MLNu0To120":
    return np.ones(len(events)) * 0.07287534563153654
  elif metadata["year"] == "2022_preEE" and metadata["sample"] == "WJetsToLNuHT100To400MLNu0To120":
    return np.ones(len(events)) * 0.02860294720865638
  elif metadata["year"] == "2022_preEE" and metadata["sample"] == "WJetsToLNuHT400To800MLNu0To120":
    return np.ones(len(events)) * 0.0010639666974138564
  elif metadata["year"] == "2022_preEE" and metadata["sample"] == "WJetsToLNuHT800To1500MLNu0To120":
    return np.ones(len(events)) * 0.00010324315702672822
  elif metadata["year"] == "2022_preEE" and metadata["sample"] == "WJetsToLNuHT1500To2500MLNu0To120":
    return np.ones(len(events)) * 6.339106538325922e-06
  elif metadata["year"] == "2022_preEE" and metadata["sample"] == "WJetsToLNuHT2500MLNu0To120":
    return np.ones(len(events)) * 2.850972211689156e-07
  elif metadata["year"] == "2022_preEE" and metadata["sample"] == "WJetsToLNuHT40To100MLNu120":
    return np.ones(len(events)) * 0.0003793826101965761
  elif metadata["year"] == "2022_preEE" and metadata["sample"] == "WJetsToLNuHT100To400MLNu120":
    return np.ones(len(events)) * 0.0001861300597543236
  elif metadata["year"] == "2022_preEE" and metadata["sample"] == "WJetsToLNuHT400To800MLNu120":
    return np.ones(len(events)) * 9.524320162858577e-06
  elif metadata["year"] == "2022_preEE" and metadata["sample"] == "WJetsToLNuHT800To1500MLNu120":
    return np.ones(len(events)) * 9.363392939680019e-07
  elif metadata["year"] == "2022_preEE" and metadata["sample"] == "WJetsToLNuHT1500To2500MLNu120":
    return np.ones(len(events)) * 9.49157309163664e-08
  elif metadata["year"] == "2022_preEE" and metadata["sample"] == "WJetsToLNuHT2500MLNu120":
    return np.ones(len(events)) * nan
  elif metadata["year"] == "2022_postEE" and metadata["sample"] == "WJetsToLNuHT40To100MLNu0To120":
    return np.ones(len(events)) * 0.06888479255390825
  elif metadata["year"] == "2022_postEE" and metadata["sample"] == "WJetsToLNuHT100To400MLNu0To120":
    return np.ones(len(events)) * 0.027976561805195168
  elif metadata["year"] == "2022_postEE" and metadata["sample"] == "WJetsToLNuHT400To800MLNu0To120":
    return np.ones(len(events)) * 0.0010553930540337759
  elif metadata["year"] == "2022_postEE" and metadata["sample"] == "WJetsToLNuHT800To1500MLNu0To120":
    return np.ones(len(events)) * 0.00010306349869691278
  elif metadata["year"] == "2022_postEE" and metadata["sample"] == "WJetsToLNuHT1500To2500MLNu0To120":
    return np.ones(len(events)) * 6.242619768533506e-06
  elif metadata["year"] == "2022_postEE" and metadata["sample"] == "WJetsToLNuHT2500MLNu0To120":
    return np.ones(len(events)) * 3.3183492870964305e-07
  elif metadata["year"] == "2022_postEE" and metadata["sample"] == "WJetsToLNuHT40To100MLNu120":
    return np.ones(len(events)) * 0.000371923734882804
  elif metadata["year"] == "2022_postEE" and metadata["sample"] == "WJetsToLNuHT100To400MLNu120":
    return np.ones(len(events)) * 0.0001838148495599703
  elif metadata["year"] == "2022_postEE" and metadata["sample"] == "WJetsToLNuHT400To800MLNu120":
    return np.ones(len(events)) * 9.100321106738956e-06
  elif metadata["year"] == "2022_postEE" and metadata["sample"] == "WJetsToLNuHT800To1500MLNu120":
    return np.ones(len(events)) * 1.0059284671465263e-06
  elif metadata["year"] == "2022_postEE" and metadata["sample"] == "WJetsToLNuHT1500To2500MLNu120":
    return np.ones(len(events)) * 7.04488856339666e-08
  elif metadata["year"] == "2022_postEE" and metadata["sample"] == "WJetsToLNuHT2500MLNu120":
    return np.ones(len(events)) * 2.9268020353761206e-09
  elif metadata["year"] == "2023_preBPix" and metadata["sample"] == "WJetsToLNuHT40To100MLNu0To120":
    return np.ones(len(events)) * 0.07421378147081381
  elif metadata["year"] == "2023_preBPix" and metadata["sample"] == "WJetsToLNuHT100To400MLNu0To120":
    return np.ones(len(events)) * 0.029093067647390652
  elif metadata["year"] == "2023_preBPix" and metadata["sample"] == "WJetsToLNuHT400To800MLNu0To120":
    return np.ones(len(events)) * 0.0010613415712030683
  elif metadata["year"] == "2023_preBPix" and metadata["sample"] == "WJetsToLNuHT800To1500MLNu0To120":
    return np.ones(len(events)) * 0.000102809913857952
  elif metadata["year"] == "2023_preBPix" and metadata["sample"] == "WJetsToLNuHT1500To2500MLNu0To120":
    return np.ones(len(events)) * 6.02469340630475e-06
  elif metadata["year"] == "2023_preBPix" and metadata["sample"] == "WJetsToLNuHT2500MLNu0To120":
    return np.ones(len(events)) * 3.8403900056866347e-07
  elif metadata["year"] == "2023_preBPix" and metadata["sample"] == "WJetsToLNuHT40To100MLNu120":
    return np.ones(len(events)) * 0.00037346834947733185
  elif metadata["year"] == "2023_preBPix" and metadata["sample"] == "WJetsToLNuHT100To400MLNu120":
    return np.ones(len(events)) * 0.0001843448513214469
  elif metadata["year"] == "2023_preBPix" and metadata["sample"] == "WJetsToLNuHT400To800MLNu120":
    return np.ones(len(events)) * 9.363948724547041e-06
  elif metadata["year"] == "2023_preBPix" and metadata["sample"] == "WJetsToLNuHT800To1500MLNu120":
    return np.ones(len(events)) * 9.573905501386952e-07
  elif metadata["year"] == "2023_preBPix" and metadata["sample"] == "WJetsToLNuHT1500To2500MLNu120":
    return np.ones(len(events)) * 3.154368332216403e-08
  elif metadata["year"] == "2023_preBPix" and metadata["sample"] == "WJetsToLNuHT2500MLNu120":
    return np.ones(len(events)) * nan
  elif metadata["year"] == "2023_postBPix" and metadata["sample"] == "WJetsToLNuHT40To100MLNu0To120":
    return np.ones(len(events)) * 0.07581297922343738
  elif metadata["year"] == "2023_postBPix" and metadata["sample"] == "WJetsToLNuHT100To400MLNu0To120":
    return np.ones(len(events)) * 0.029569431619275995
  elif metadata["year"] == "2023_postBPix" and metadata["sample"] == "WJetsToLNuHT400To800MLNu0To120":
    return np.ones(len(events)) * 0.0010730802406485045
  elif metadata["year"] == "2023_postBPix" and metadata["sample"] == "WJetsToLNuHT800To1500MLNu0To120":
    return np.ones(len(events)) * 0.00010104710771078462
  elif metadata["year"] == "2023_postBPix" and metadata["sample"] == "WJetsToLNuHT1500To2500MLNu0To120":
    return np.ones(len(events)) * 5.919410734493913e-06
  elif metadata["year"] == "2023_postBPix" and metadata["sample"] == "WJetsToLNuHT2500MLNu0To120":
    return np.ones(len(events)) * 3.39146144207902e-07
  elif metadata["year"] == "2023_postBPix" and metadata["sample"] == "WJetsToLNuHT40To100MLNu120":
    return np.ones(len(events)) * 0.0003812851685466557
  elif metadata["year"] == "2023_postBPix" and metadata["sample"] == "WJetsToLNuHT100To400MLNu120":
    return np.ones(len(events)) * 0.0001867764099941625
  elif metadata["year"] == "2023_postBPix" and metadata["sample"] == "WJetsToLNuHT400To800MLNu120":
    return np.ones(len(events)) * 9.112092272042358e-06
  elif metadata["year"] == "2023_postBPix" and metadata["sample"] == "WJetsToLNuHT800To1500MLNu120":
    return np.ones(len(events)) * 7.948125639805503e-07
  elif metadata["year"] == "2023_postBPix" and metadata["sample"] == "WJetsToLNuHT1500To2500MLNu120":
    return np.ones(len(events)) * 7.416173522535769e-08
  elif metadata["year"] == "2023_postBPix" and metadata["sample"] == "WJetsToLNuHT2500MLNu120":
    return np.ones(len(events)) * 1.0629814021943305e-08
  return np.ones(len(events))

wl_func = WeightLambda.wrap_func(
    name="WJetsRun3Stitching",
    function=stitching_func,
    has_variations=False
)

WJetsRun3Stitching = [wl_func]
