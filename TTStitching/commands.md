You will need to run the follow to configs (make sure the output directories are unchanged). You can submit them or run them locally. These will make histogram for the important variables using both the gen weight and the gen weight squared. Remember to collect your histograms to output_all.coffea if runnning on a batch.

```bash
for c in hadronic semileptonic fullyleptonic; do for t in w w2; do pocket-coffea run --cfg config_${c}_run2_${t}.py -o output_${c}_run2_${t} --executor=condor@ic --scaleout=100 --jobs-dir="jobs_${c}_run2_${t}"; done; done
```

```bash
for c in hadronic semileptonic fullyleptonic; do for t in w w2; do pocket-coffea merge-outputs -o output_${c}_run2_${t}/output_all.coffea  -jc "jobs_${c}_run2_${t}/job/jobs_config.yaml"; done; done
```

To get the fractions and the relevant weight function, you can run use the stitch_samples script with the following options.

```bash
for c in hadronic semileptonic fullyleptonic; do python3 ../scripts/stitch_samples.py --input="output_${c}_run2_w/output_all.coffea" --input-w2="output_${c}_run2_w2/output_all.coffea" --output-name=TTRun2Stitching --output-file="../Functions/TT${c}Run2StitchingWeights.py"; done
```

```bash
python3 ../scripts/stitch_samples.py --input="output_hadronic_run2_w/output_all.coffea" --input-w2="output_hadronic_run2_w2/output_all.coffea" --output-name=TTToHadronicRun2Stitching --output-file="../Functions/TTToHadronicRun2StitchingWeights.py" --extra-sel="((events.count_l==0) & (events.count_nu==0))"
```
```bash
python3 ../scripts/stitch_samples.py --input="output_semileptonic_run2_w/output_all.coffea" --input-w2="output_semileptonic_run2_w2/output_all.coffea" --output-name=TTToSemiLeptonicRun2Stitching --output-file="../Functions/TTToSemiLeptonicRun2StitchingWeights.py" --extra-sel="((events.count_l==1) & (events.count_nu==1))"
```
```bash
python3 ../scripts/stitch_samples.py --input="output_fullyleptonic_run2_w/output_all.coffea" --input-w2="output_fullyleptonic_run2_w2/output_all.coffea" --output-name=TTTo2L2NuRun2Stitching --output-file="../Functions/TTTo2L2NuRun2StitchingWeights.py" --extra-sel="((events.count_l==2) & (events.count_nu==2))"
```