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
  /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/pocketcoffea:lxplus-el9-576bd3cd
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
pocket-coffea run --cfg config.py -o "output/${job_name}_test" --skip-bad-files --test 
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
```bash
python3 ../scripts/convert_coffea_to_parquet.py --input="${eos_folder}/${job_name}/*.coffea" --output="${eos_folder}/${job_name}_parquet"
```

If you split by year you can combine all into a single parquet folder replaceing the input with this:

```bash
file_string=""
for year in "${years[@]}"; do file_string+="${eos_folder}/${job_name}_${year}/*.coffea,"; done
file_string="${file_string%,}"
```

## Make BW reweighted samples

```bash
for yr in "${years[@]}"; do for i in 166 169 170 171 172 173 174 175 178; do python3 ../scripts/make_bw_samples.py --input="${eos_folder}/${job_name}_parquet/TTTo*${yr}.parquet,${eos_folder}/${job_name}_parquet/TTMtt*${yr}.parquet" --output="${eos_folder}/${job_name}_parquet_bw/TT_${i}p5_${yr}.parquet" --yield-input="${eos_folder}/${job_name}_parquet/TTToSemiLeptonic_${yr}.parquet,${eos_folder}/${job_name}_parquet/TTToHadronic_${yr}.parquet,${eos_folder}/${job_name}_parquet/TTTo2L2Nu_${yr}.parquet,${eos_folder}/${job_name}_parquet/TTMtt*_${yr}.parquet" --mass-to=${i}.5; done; done
```

```bash
cp ${eos_folder}/${job_name}_parquet_bw/TT_*.parquet ${eos_folder}/${job_name}_parquet/
```

## Run plotting

Plots
```bash
python3 ../scripts/plot_from_parquet.py --input="${eos_folder}/${job_name}_parquet" --output="../plots/${job_name}" --include-fraction --cfg="../params/plotting_extra_mass.py" --year=all
```

If you want to apply extra selection on the event you can add '--pre-sel="(JetLepton_deltaR>0.25) & (JetLepton_ptrel>30) & (SubJet2_btagDeepB > 0.2783) & (MET_pt > 50) & ((FatJet_tau3/FatJet_tau2) < 0.7) & ((SubJet1_tau2/SubJet1_tau1) < 0.7)"'

## Creating datacards

```bash
for yr in "${years[@]}"; do python3 ../scripts/plot_from_parquet.py --input="${eos_folder}/${job_name}_parquet" --output="../plots/${job_name}_${yr}_datacards" --include-fraction --cfg="../params/plotting_extra_mass.py" --year=${yr} --var="CombinedSubJets_mass" --write --syst --plot-syst-variation; done
```