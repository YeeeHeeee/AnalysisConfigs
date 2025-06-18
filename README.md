# Configs

Repository containing analysis configurations for [`PocketCoffea`](https://github.com/YeeeHeeee/PocketCoffea.git).  
These configurations are intended for the semileptonic top-antitop (`tt̄`) analysis in the resolved regime using AK4 jets and merged regime using AK8, also combined topology. 

---

## Setup Instructions

### 1. Clone the `PocketCoffea` Repository

```bash
git clone git@github.com:gputtley/PocketCoffea.git
```

### 2. Install Micromamba and Create the Environment
   ```bash
   "${SHELL}" <(curl -L micro.mamba.pm/install.sh)
   micromamba create -n pocket-coffea python=3.11 -c conda-forge
   micromamba activate pocket-coffea
   micromamba install XRootD
   ```

### 3. Install `PocketCoffea` Locally
   ```bash
   cd PocketCoffea
   pip install -e .
   ```

After set up the environment, you can leave the directory of the PocketCoffea

**Note:** detail set up tutorial can be found here [`Installation`](https://pocketcoffea.readthedocs.io/en/stable/installation.html)

---

## Setup the Workflow

### 1. Clone the `AnalysisConfigs` Repository

```bash
git clone git@github.com:YeeeHeeee/AnalysisConfigs.git
```

### 2. Activate the Environment
```bash
micromamba activate pocket-coffea
```

### 3. Install `AnalysisConfigs` Locally
   ```bash
   cd AnalysisConfigs
   pip install -e .
   ```

---

## Running the Analysis
The main steps that need to be performed are following:

### 1. Build the json Datasets
Authorised first:
```bash
source /vols/grid/cms/setup.sh
voms-proxy-init -voms cms -rfc --valid 168:0
```
Then run:
```bash
pocket-coffea build-datasets --cfg Datasets/datasets_definitions.json -o
```
Check:
```bash
ls -lrt Datasets/
```

### 2. Process the Datasets
PocketCoffea provides a flexible command-line interface to configure. The basic usage is:
```bash
pocket-coffea run --cfg config.py -o output_test --skip-bad-files
```

**Note:** Ensure you've pointed to the desired directory before executing this command.

Additional Options:
* `--test`: Run interactively with a file limit of 1 (useful for quick debugging).
* `--limit-chunks`: Limit the number of chunks processed (splits of files).
* `--limit-files`: Limit the total number of files to process.
* `--chunksize`: Overrides the number of events processed per task, allowing you to control memory usage and performance without editing the config file.
* `--process-separately`: Process each dataset independently instead of merging everything in one job.
* `--filter-years`: Comma-separated list to select specific data-taking years to process.
To run with predefined executor using `--executor` with 100 workers:
```bash
pocket-coffea run --cfg config.py  --executor condor@ic  -o output_condor --scaleout=100 --skip-bad-files --jobs-dir="jobs/"
```

**Note:** Rename the `--jobs-dir` if you wanna submit multiple jobs at the same time.


The Executors available are:

| Site                        | Supported Executors       | Executor String(s)         |
|----------------------------|---------------------------|-----------------------------|
| CERN lxplus                | Dask                      | `dask@lxplus`               |
| CERN SWAN                  | Dask                      | `dask@swan`                 |
| T3_CH_PSI                  | Dask                      | `dask@T3_CH_PSI`            |
| DESY NAF                   | Dask                      | `dask@DESY_NAF`             |
| RWTH Aachen LX-Cluster     | Parsl, Dask               | `parsl@RWTH`, `dask@RWTH`   |
| RWTH CLAIX                 | Dask                      | `dask@CLAIX`                |
| Purdue Analysis Facility   | Dask                      | `dask@purdue-af`            |
| INFN Analysis Facility     | Dask                      | `dask@infn-af`              |
| Brown brux20 cluster       | Dask                      | `dask@brux`                 |
| Brown CCV Oscar            | Dask                      | `dask@oscar`                |
| Maryland rubin cluster     | Dask, Condor              | `dask@rubin`, `condor@rubin`|
| Imperial College (lx06, lx05, lx04)| Condor                    | `condor@ic`                 |

After submitting, to merge the files:
```bash
pocket-coffea merge-outputs -o output_condor/output_all.coffea -jc jobs-dir/job/jobs_config.yaml output_condor/output_job_*.coffea
```

## Others
1. Removee files:
```bash
rm -rf ./jobs-dir/job
```

```bash
rm jobs-dir/job/jobs_config.yaml
```
2. Check the queues (if you are using the condor_ic):
```bash
/vols/cms/tr1123/condor_tools/condor_stat.py
```
3. Removes submitting jobs:
```bash
condor_rm job_id
```
