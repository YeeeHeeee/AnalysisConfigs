# AnalysisConfigs

Repository containing analysis configurations for [`PocketCoffea`](https://github.com/gputtley/PocketCoffea.git).  
These configurations are intended for the semileptonic top-antitop (`tt̄`) analysis in the resolved regime using AK4 jets (incomplete) and the merged regime using AK8. 

---

## Setup Instructions

### Clone the `AnalysisConfigs` Repository

This repository is set up to work with a particular version of `PocketCoffea` which can be run using the apptainer lxplus and using the correct commit on cvmfs. Therefore, the setup of `AnalysisConfigs` must be done on lxplus.

The command to clone the `AnalysisConfigs` repository is shown below.

```bash
git clone git@github.com:YeeeHeeee/AnalysisConfigs.git
```

Now you can move into the repository.
```bash
cd AnalysisConfigs
```

### Using the `PocketCoffea` Apptainer

It is best to set your proxy before opening the container.
```bash
voms-proxy-init -voms cms -rfc --valid 168:0
```

To use the apptainer you can run the following command.

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

With this you will also need to export `AnalysisConfigs` as the PYTHONPATH. This can be run from any directory of the `AnalysisConfigs` repository.

```bash
export PYTHONPATH="$(realpath $(pwd | grep -o '.*/AnalysisConfigs')):$PYTHONPATH"
```

---

## Preparing the Dataset Definitions

Make sure you proxy is set. Then you can run this for data.

```bash
pocket-coffea build-datasets --cfg Datasets/data_definitions.json -rs 'T[123]_(FR|IT|DE|BE|CH|UK)_\w+' -o -ir
```

And run this for MC.

```bash
pocket-coffea build-datasets --cfg Datasets/mc_definitions.json -rs 'T[123]_(FR|IT|DE|BE|CH|UK)_\w+' -o -ir
```

To check the dataset definitions you have created you can run this.

```bash
ls -lrt Datasets/
```


## Running the Merged Analysis

Make sure you move into the `Merged` directory.

```bash
cd Merged
```

### Setup Directory Names

In the following commands we use global variables defined the configure the output directories. Here is an example of how to do this.

```bash
job_name="260126_v2"
eos_folder="/eos/user/g/guttley/pc_output"
```

### Doing a Local Test Run of `PocketCoffea`

To perform a test of the config and workflow, you can run the following command.

```bash
pocket-coffea run --cfg config.py -o "output/${job_name}_test" --skip-bad-files --test 
```

You also have the following options you can use when runnning `PocketCoffea`:
* `--test`: Run interactively with a file limit of 1 (useful for quick debugging).
* `--limit-chunks`: Limit the number of chunks processed (splits of files).
* `--limit-files`: Limit the total number of files to process.
* `--chunksize`: Overrides the number of events processed per task, allowing you to control memory usage and performance without editing the config file.
* `--process-separately`: Process each dataset independently instead of merging everything in one job.
* `--filter-years`: Comma-separated list to select specific data-taking years to process.

When running this step you may want to comment out the processes and years you do not want to run in the `config.py` file.


### Running All Eras Simultaneously on the Batch

To run all the datasets on the CERN condor cluster, you can run the following command. The `--scaleout` option is the approximate number of jobs you wish scaleout the processing to. 

```bash
pocket-coffea run --cfg config.py -o "${eos_folder}/${job_name}" --jobs-dir="jobs/${job_name}" --skip-bad-files --executor=condor@lxplus --scaleout=5000 -ro "../params/lxplus_run_options.yaml"
```

You can check the jobs with the follow command. Check jobs can use `--resubmit`, `--set-to-fail`, `--skip-bad-files`, and `--sub-replace` options, using `--sub-replace='+JobFlavour: "tomorrow"'` for example if you want to request more time.

```bash
python3 ../scripts/check_jobs.py --jobs-folder="jobs/${job_name}/job"
```


### Running All Eras Separately on the Batch

It is often useful to run the eras of data-taking separately. This is useful for debugging, and easier resubmission. You can define the years list with this.

```bash
years=("2016_PreVFP" "2016_PostVFP" "2017" "2018" "2022_preEE" "2022_postEE" "2023_preBPix" "2023_postBPix")
```

You can then submit each year to a separate folder with the following command.

```bash
for yr in "${years[@]}"; do pocket-coffea run --cfg config.py -o "${eos_folder}/${job_name}_${yr}" --jobs-dir="jobs/${job_name}_${yr}" --skip-bad-files --executor=condor@lxplus --scaleout=1000 -ro "../params/lxplus_run_options.yaml" --filter-years="${yr}"; done
```

Checking jobs can then be done with the same command but with the era extension, an example is shown below.

```bash
python3 ../scripts/check_jobs.py --jobs-folder="jobs/${job_name}_2016_PreVFP/job"
```


### Collecting and Converting the Output Coffea Files to Parquet for Test

`PocketCoffea` creates an output `.coffea` file for every job submitted. Here we used a script to both collect and convert these `.coffea` files into `.parquet` files. This will create a `.parquet` file for every process and era.

To do this for the local test, you can simply run this.

```bash
python3 ../scripts/convert_coffea_to_parquet.py --input="output/${job_name}_test/*.coffea" --output="output/${job_name}_test_parquet"
```


### Collecting and Converting the Output Coffea Files to Parquet for All Eras Simultaneously on the Batch

You can use the following command for this.

```bash
python3 ../scripts/convert_coffea_to_parquet.py --input="${eos_folder}/${job_name}/*.coffea" --output="${eos_folder}/${job_name}_parquet"
```

The `convert_coffea_to_parquet.py` also offers a functionality to apply an additional weight but renomalise back to the yield without the weight. This is needed for the b tagging shape corrections. This example is run like this.

```bash
python3 ../scripts/convert_coffea_to_parquet.py --input="${eos_folder}/${job_name}/*.coffea" --output="${eos_folder}/${job_name}_btag_parquet" --weight="weight*ExtraWeights_BTagShapeCorrectionSubjets" --norm-weight="weight" --norm-files="TTToSemiLeptonic_*,TTToHadronic_*,TTTo2L2Nu_*,TTMtt*,WJetsToLNu_*,WJetsToLNuHT*,ST_*,QCD_*,DY_*,WW*,WZ*,ZZ*"
```


### Collecting and Converting the Output Coffea Files to Parquet for All Eras Separately on the Batch

To do this, first define the input string (assuming `years` is already defined).

```bash
years=("2016_PreVFP" "2016_PostVFP" "2017" "2018" "2022_preEE" "2022_postEE" "2023_preBPix" "2023_postBPix")
file_string=""
for year in "${years[@]}"; do file_string+="${eos_folder}/${job_name}_${year}/*.coffea,"; done
file_string="${file_string%,}"
```

Then you can run the nominal command with this.

```bash
python3 ../scripts/convert_coffea_to_parquet.py --input="${file_string}" --output="${eos_folder}/${job_name}_parquet"
```

And the b taggin shape correction command like this.

```bash
python3 ../scripts/convert_coffea_to_parquet.py --input="${file_string}" --output="${eos_folder}/${job_name}_btag_parquet" --weight="weight*ExtraWeights_BTagShapeCorrectionSubjets" --norm-weight="weight" --norm-files="TTToSemiLeptonic_*,TTToHadronic_*,TTTo2L2Nu_*,TTMtt*,WJetsToLNu_*,WJetsToLNuHT*,ST_*,QCD_*,DY_*,WW*,WZ*,ZZ*"
```


### Making Optimally Reweighted BW ttbar Files

To make a combined ttbar file using all available samples, you can run the following command.

```bash
for yr in "${years[@]}"; do python3 ../scripts/make_bw_samples.py --input="${eos_folder}/${job_name}_parquet/TTTo*${yr}.parquet,${eos_folder}/${job_name}_parquet/TTMtt*${yr}.parquet" --output="${eos_folder}/${job_name}_parquet_bw" --file-ext="_${yr}" --yield-input="${eos_folder}/${job_name}_parquet/TTToSemiLeptonic_${yr}.parquet,${eos_folder}/${job_name}_parquet/TTToHadronic_${yr}.parquet,${eos_folder}/${job_name}_parquet/TTTo2L2Nu_${yr}.parquet,${eos_folder}/${job_name}_parquet/TTMtt*_${yr}.parquet" --mass-to="166.5,169.5,170.5,171.0,171.5,172.0,172.5,173.0,173.5,174.0,174.5,175.5,178.5"; done
```


### Run Plotting

To run the plotting we first define a few globale variables.

```bash
plots_name="310126"
pre_selection="((JetLepton_deltaR>0.25) & (JetLepton_ptrel>30))"
post_selection="((CombinedSubJets_pt>400) & (LeptonicTop_mass<CombinedSubJets_mass) & (MET_pt>50))"
years=("run2" "run3")
```

To run the plotting you can use the `plot_from_parquet` file.

```bash
for yr in "${years[@]}"; do python3 ../scripts/plot_from_parquet.py --input="${eos_folder}/${job_name}_parquet" --output="../plots/${plots_name}" --include-fraction --cfg="../params/plotting_extra_mass.py" --year=${yr} --pre-sel="${pre_selection}" --sel="${post_selection}" --norm-groups-to-data="TT Merged (172.5 GeV),TT Unmerged (172.5 GeV)"; done
```

There are some booleans defined in `plotting_extra_mass.py` which can alter what is run. The variables plotted are also defined here.


### Creating ROOT Datacards

To create ROOT datacards locally you can run the following command.

```bash
for yr in "${years[@]}"; do python3 ../scripts/plot_from_parquet.py --input="${eos_folder}/${job_name}_parquet" --output="../plots/${plots_name}_datacards" --include-fraction --cfg="../params/plotting_extra_mass.py" --year=${yr} --pre-sel="${pre_selection}" --sel="${post_selection}" --write --syst --plot-syst-variation --rebin --norm-to-bin-width; done
```

This may be slow so you may want to run it on the batch. Here we define the command first.

```bash
yr=run2
cmd='python3 ../scripts/plot_from_parquet.py --input="${eos_folder}/${job_name}_parquet" --output="../plots/${plots_name}_jobs" --include-fraction --cfg="../params/plotting_extra_mass.py" --year=${yr} --var="CombinedSubJets_mass" --write --syst --pre-sel="${pre_selection}" --sel="${post_selection}" --points-per-job=20'
```

Then you can run these commands to submit, then to hadd (once finished) and then to plot.

```bash
eval ${cmd} --submit
eval ${cmd} --hadd
eval ${cmd} --output="../plots/${plots_name}_datacards" --load-from-root="../plots/${plots_name}_jobs/datacard_CombinedSubJets_mass.root" --rebin --norm-to-bin-width --write-after-load --syst
```