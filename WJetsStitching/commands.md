You will need to run the follow to configs (make sure the output directories are unchanged). You can submit them or run them locally. These will make histogram for the important variables using both the gen weight and the gen weight squared. Remember to collect your histograms to output_all.coffea if runnning on a batch.

```bash
for e in run2 run3; do for t in w w2; do pocket-coffea run --cfg config_${e}_${t}.py -o output_${e}_${t} --executor=condor@ic --scaleout=100 --jobs-dir="jobs_${e}_${t}"; done; done
```

```bash
for e in run2 run3; do for t in w w2; do pocket-coffea merge-outputs - -o output_${e}_${t}/output_all.coffea -jc "jobs_${e}_${t}/job/jobs_config.yaml"; done; done
```

To get the fractions and the relevant weight function, you can run use the stitch_samples script with the following options.

```bash
python3 ../scripts/stitch_samples.py --input="output_run2_w/output_all.coffea" --input-w2="output_run2_w2/output_all.coffea" --output-name=WJetsRun2Stitching --output-file="../Functions/WJetsRun2StitchingWeights.py"
```

```bash
python3 ../scripts/stitch_samples.py --input="output_run3_w/output_all.coffea" --input-w2="output_run3_w2/output_all.coffea" --output-name=WJetsRun3Stitching --output-file="../Functions/WJetsRun3StitchingWeights.py" --category-conversion="MLNu0To120:(events.LNu.mass>=0) & (events.LNu.mass<120),MLNu120:events.LNu.mass>=120"
```


You can also do it by saving the datasets and not just the histograms. This is only set up for run 2 at the moment.

```bash
e=run2; pocket-coffea run --cfg config_${e}.py -o output_${e} --executor=condor@ic --scaleout=100 --jobs-dir="jobs_${e}"
```

```bash
e=run2; pocket-coffea merge-outputs -o output_${e}/output_all.coffea -jc "jobs_${e}/job/jobs_config.yaml";
```

To get the fractions run this.

```bash
python3 ../scripts/stitch_samples.py --input="output_run2/output_all.coffea" --output-name=WJetsRun2Stitching --output-file="../Functions/WJetsRun2StitchingWeights.py" --dataset --dataset-bins="((events.LHE.HT>=0) & (events.LHE.HT<70)),((events.LHE.HT>=70) & (events.LHE.HT<100)),((events.LHE.HT>=100) & (events.LHE.HT<200)),((events.LHE.HT>=200) & (events.LHE.HT<400)),((events.LHE.HT>=400) & (events.LHE.HT<600)),((events.LHE.HT>=600) & (events.LHE.HT<800)),((events.LHE.HT>=800) & (events.LHE.HT<1200)),((events.LHE.HT>=1200) & (events.LHE.HT<2500)),(events.LHE.HT>=2500)"
```
