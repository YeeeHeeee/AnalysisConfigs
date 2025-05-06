# Resolved Configs

Repository containing analysis configurations for [`PocketCoffea_top`](https://github.com/YeeeHeeee/PocketCoffea_top).  
These configurations are intended for the semileptonic top-antitop (`tt̄`) analysis in the resolved regime using AK4 jets.

---

## Setup Instructions

### 1. Clone the `PocketCoffea_top` Repository

```bash
git clone git@github.com:YeeeHeeee/PocketCoffea_top.git
```

### 2. Install Micromamba and Create the Environment
   ```bash
   "${SHELL}" <(curl -L micro.mamba.pm/install.sh)
   micromamba create -n pocket-coffea python=3.11 -c conda-forge
   micromamba activate pocket-coffea
   ```

### 1. Install `PocketCoffea_top` Locally
   ```bash
   cd PocketCoffea_top
   pip install -e .
   ```
