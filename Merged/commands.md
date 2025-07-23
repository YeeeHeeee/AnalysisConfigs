# Running on lxplus

## Setup environment

```bash
voms-proxy-init -voms cms --valid 96:0
```

```bash
apptainer shell \
  -B /afs \
  -B /cvmfs/cms.cern.ch \
  -B /tmp \
  -B /eos/cms/ \
  -B /eos/user/$(whoami | cut -c1)/$(whoami) \
  -B /etc/sysconfig/ngbauth-submit \
  -B ${XDG_RUNTIME_DIR} \
  --env KRB5CCNAME="FILE:${XDG_RUNTIME_DIR}/krb5cc" \
  /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/pocketcoffea:lxplus-el9-stable
```

```bash
export PYTHONPATH="$(realpath $(pwd | grep -o '.*/AnalysisConfigs')):$PYTHONPATH"
```

## Setup directory names
```bash
job_name="220725"
eos_folder="/eos/user/g/guttley/pc_output"
```

Make sure you are in the `Merged` directory.

## Run a test
```bash
pocket-coffea run --cfg config.py -o "output/${job_name}_test" --skip-bad=files --test 
```

## Run jobs
```bash
pocket-coffea run --cfg config.py -o "${eos_folder}/${job_name}" --jobs-dir="jobs/${job_name}" --skip-bad-files --executor=condor@lxplus --scaleout=5000 -ro "../params/lxplus_run_options.yaml"
```

Check jobs can use `--resubmit`, `--set-to-fail`, `--skip-bad-files`, and `--sub-replace` options.
```bash
python3 ../scripts/check_jobs.py --jobs-folder="jobs/${job_name}/job"
```

Sometimes it is easier to split per year.
```bash
years=("2016_PreVFP" "2016_PostVFP" "2017" "2018" "2022_preEE" "2022_postEE" "2023_preBPix" "2023_postBPix")
```

```bash
for yr in "${years[@]}"; do pocket-coffea run --cfg config.py -o "${eos_folder}/${job_name}_${yr}" --jobs-dir="jobs/${job_name}_${yr}" --skip-bad-files --executor=condor@lxplus --scaleout=1000 -ro "../params/lxplus_run_options.yaml" --filter-years="${yr}"; done
```

## Convert to parquet

Convert to parquet nominal.
```bash
python3 ../scripts/convert_coffea_to_parquet.py --input="${eos_folder}/${job_name}/*.coffea" --output="${eos_folder}/${job_name}_parquet"
```

Convert to parquet and rescale with b tagging shape corrections.
```bash
python3 ../scripts/convert_coffea_to_parquet.py --input="${eos_folder}/${job_name}/*.coffea" --output="${eos_folder}/${job_name}_parquet" --weight="weight*ExtraWeights_BTagShapeCorrectionSubjets" --norm-weight="weight" --norm-files="TTToSemiLeptonic_*,TTToHadronic_*,TTTo2L2Nu_*,TTMtt*,WJetsToLNu_*,WJetsToLNuHT*,ST_*,QCD_*,DY_*,WW*,WZ*,ZZ*"
```

## Run plotting

Plots
```bash
python3 ../scripts/plot_merged.py --input="${eos_folder}/${job_name}_parquet" --output="../plots/${job_name}" --extra-args="--include-fraction"
```