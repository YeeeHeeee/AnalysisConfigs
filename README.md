# Resolved Configs
Repository containing analysis configurations for PocketCoffea_top.
Analysis for semileptonic ttbar in resolved case where use the AK4 jets.

## Set up
Install the `PocketCoffea_top` package in your python environment.
1. Clone locally the PocketCoffea_top repo:
   ```bash
  git@github.com:YeeeHeeee/PocketCoffea_top.git
   ```
2. Install micromamba:
  ```bash
  "${SHELL}" <(curl -L micro.mamba.pm/install.sh)
  micromamba create -n pocket-coffea python=3.11 -c conda-forge
  micromamba activate pocket-coffea
```
3. Install the PocketCoffea_top package locally:
   ```bash
   cd PocketCoffea_top
   pip install -e .
   ```


