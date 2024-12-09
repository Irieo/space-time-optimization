<!--
SPDX-FileCopyrightText: 2023 Iegor Riepin, Tom Brown

SPDX-License-Identifier: CC0-1.0
-->

# Code for the paper "Spatio-temporal load shifting for truly clean computing"

This repository contains the code to reproduce the complete workflow behind the manuscript: "Spatio-temporal load shifting for truly clean computing" by Iegor Riepin, Victor Zavala and Tom Brown. The paper is available on [arXiv](https://arxiv.org/abs/2405.00036).

### Abstract

Companies operating datacenters are increasingly committed to procuring renewable energy to reduce their carbon footprint, with a growing emphasis on achieving 24/7 Carbon-Free Energy (CFE) matching—eliminating carbon emissions from electricity use on an hourly basis.
However, variability in renewable energy resources poses significant challenges to achieving this goal.
This study investigates how shifting computing workloads and associated power loads across time and location supports 24/7 CFE matching.
We develop an optimization model to simulate a network of geographically distributed datacenters managed by a company leveraging spatio-temporal load flexibility to achieve 24/7 CFE matching.
We isolate three signals relevant for informed use of load flexibility: (1) varying average quality of renewable energy resources, (2) low correlation between wind power generation over long distances due to different weather conditions, and (3) lags in solar radiation peak due to Earth's rotation.
Our analysis reveals that datacenter location and time of year influence which signal drives an effective load-shaping strategy.
By leveraging these signals for coordinated energy procurement and load-shifting decisions, clean computing becomes both more resource-efficient and cost-effective—the costs of 24/7 CFE are reduced by 1.29±0.07 EUR/MWh for every additional percentage of flexible load.
This study provides practical guidelines for datacenter companies to harness spatio-temporal load flexibility for clean computing.
Our results and the open-source optimization model offer insights applicable to a broader range of industries aiming to eliminate their carbon footprints.


### How to reproduce results from the paper?

1. Clone the repository:

```
git clone git@github.com:Irieo/space-time-optimization.git
```

2. Install the necessary dependencies using `environment.yaml` file. The following commands will do the job:

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

The code is known to work with PyPSA 0.26.0, pandas 2.0.3, numpy 1.26.2, and gurobi 10.0.1. The complete list of dependencies is in the [envs/environment.yaml](envs/environment.yaml) file.


### License

There are different open licenses for different types of files in the repository. See [specifications here](.reuse/dep5).
