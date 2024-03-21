<!--
SPDX-FileCopyrightText: 2023 Iegor Riepin, Tom Brown

SPDX-License-Identifier: CC0-1.0
-->

# Code for the paper "Spatio-temporal load shifting for truly clean computing"

This repository contains the code to reproduce the complete workflow behind the manuscript.

### Abstract

Increasing energy demand for cloud computing raises concerns about its carbon footprint. Companies within the datacenter sector procure significant amounts of renewable energy to reduce their environmental impact. There is increasing interest in achieving 24/7 Carbon-Free Energy (CFE) matching in electricity usage, aiming to eliminate all carbon footprints associated with electricity consumption. However, the variability of renewable energy resources poses significant challenges for achieving this goal. In this work, we explore the role of spatio-temporal load-shifting flexibility provided by hyperscale datacenters in achieving a net zero carbon footprint. We develop an optimization model to simulate a network of geographically distributed datacenters managed by a company leveraging spatio-temporal load flexibility to achieve 24/7 carbon-free energy matching. We isolate three signals relevant fo informed use of load flexiblity: varying quality of renewable energy resources, low correlation between wind power generation
over long distances due to different weather conditions, and lags in solar radiation peak due to Earth rotation. Further, we illustrate how these signals can be used for effective load-shaping strategies depending on load locations and time of year. Finally, we show that optimal energy procurement and load-shifting decisions based on these signals facilitate resource-efficiency and cost-effectiveness of clean computing. The costs of 24/7 CFE matching are reduced by 1.29±0.07 EUR/MWh for every additional percentage of flexible load.

### How to reproduce results from the paper?

1. Clone the repository:

```
git clone git@github.com:Irieo/space-time-optimization.git
```

2. Install the necessary dependencies using `environment.yml` file. The following commands will do the job:

```
conda env create -f envs/environment.yaml
conda activate 247-env
```
3. The results of the paper can be conveniently reproduced by running the [snakemake](https://snakemake.readthedocs.io/en/stable/) workflows.  The following terminal commands will run the workflows for sections 1-4 of the paper:

```
snakemake -call --configfile custom_config_s1.yaml
snakemake -call --configfile custom_config_s2.yaml
snakemake -call --configfile custom_config_s3.yaml
snakemake -call --configfile custom_config_s4.yaml
```

NB Size of a mathematical problem for this paper is optimized so that it is possible to reproduce the results on a private laptop with 8GB RAM.

Model results will be stored in the `results` directory. For each workflow, the directory will contain:
- solved networks (.nc) for individual optimization problems
- summary (.yaml) for individual optimization problems
- summary (.csv) for aggregated results
- log files (memory, python, solver)
- detailed plots (.pdf) of the results

4. At this point, a curious reader can even reproduce the dashboards from the paper by running the jupyter notebooks in the `scripts/` directory and compile the LaTeX project `/manuscript/manuscript.tex` to reproduce the paper .pdf file.

### Data requirements

The workflow is based on PyPSA networks exported from [PyPSA-Eur](https://github.com/PyPSA/pypsa-eur) built with `myopic` setting to get brownfield networks for 2025/2030. For convenience, the workflow uses already networks provided in the `input` folder by default.

Technology data assumptions are automatically retrieved from [technology-data](https://github.com/PyPSA/technology-data) repository for `<year>` and `<version>`, as specified in `config.yaml`.

Several plots from the paper require high-resolution geographical data that is not included in this repository.
To reproduce those plots, three files (`elec_s_256_ec.nc`, `profile_solar.nc`, `regions_onshore_elec_s_256.geojson`) from [PyPSA-Eur Zenodo repository](https://zenodo.org/records/7646728) have to be retrieved and placed in the `input/` directory. The following command automates this task:

```
snakemake -call retrieve_data
```

### Software requirements

The code is known to work with PyPSA 0.26.0, pandas 2.0.3, numpy 1.26.2, vresutils 0.3.1 and gurobi 10.0.1. The complete list of dependencies is in the [envs/environment.yml](envs/environment.yml) file.


### License

There are different open licenses for different types of files in the repository. See [specifications here](.reuse/dep5).
