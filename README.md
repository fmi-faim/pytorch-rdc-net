# Pytorch Rdc-Net

## Repository Overview
The repository uses [pixi](https://pixi.sh) to manage the project dependencies.
Please follow the installation instructions in [docs/installation.md](docs/installation.md) to get started.
Once pixi is installed you can run `pixi run build_docs` in the root of the project to build the documentation.
The documentation is then available offline in `site/index.html`.

* `docs`: Contains the source files for the documentation.
* `infrastructure`: Contains installed tools e.g. pixi.
* `processed_data`: Contains all processed data which is used for the final results.
* `raw_data`: Contains the raw data from your experiments.
* `results`: Contains final results like figures and plots.
* `runs`: Contains all config files which were used to produce processed data or results.
* `sandbox`: Contains all experimental code and scripts which are not used for final results.
* `source`: Contains all scripts which are used to produce processed data and final results.

---
This project was generated with the [faim-ipa-project](https://fmi-faim.github.io/ipa-project-template/) copier template.
