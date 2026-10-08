# FedSPL

**Structure Prototype Learning for Federated Molecular Property Prediction under
Scaffold Heterogeneity**

Tingkai Sun, Xiucai Ye, and Tetsuya Sakurai, University of Tsukuba.

FedSPL groups local molecules using fixed structural descriptors, averages GNN
embeddings within those groups, and exchanges structure prototypes alongside
federated model updates. The prototypes guide local representation learning
without requiring class labels for prototype assignment.

This repository expands the manuscript's reproducibility archive into browsable
source, configurations, data inputs, and result files. It contains no manuscript,
reviewer correspondence, trained checkpoints, or server credentials.

## Quick start: reproduce result tables

```sh
git clone https://github.com/TINGKAISUN5619/FedSPL.git
cd FedSPL/reproducibility
python3 -B reproduce_tables.py --output ../../fedspl_numeric_tables
```

Table reproduction uses Python 3.8+ standard library only. It does not install
dependencies, access a server, or start training. Choose a new output directory.

For the optional CPU tests and figure reproduction, first prepare the scientific
environment described in [ENVIRONMENT.md](reproducibility/ENVIRONMENT.md):

```sh
python -B test_cpu.py
python -B reproduce_figures.py --output ../../fedspl_figures
```

## Train a new run

From `reproducibility/`, after installing the documented dependencies:

```sh
python -B train.py --config classification_example.json --workspace ../../fedspl_import --import-smoke
python -B train.py --config classification_example.json --workspace ../../fedspl_training --execute
```

Only `--execute` starts training. The wrapper prepares a separate workspace and
leaves the archived code and results unchanged. Use `regression_example.json`
for regression. An import check is not a full training test; see the environment
notes for the distinction between the tested CPU environment and historical CUDA
runs. Training can be computationally expensive.

## Repository map

| Location | Contents |
| --- | --- |
| [Detailed reproducibility guide](reproducibility/README.md) | Protocols, commands, provenance and known limitations |
| [Corrected source](reproducibility/corrected/source) | Frozen main, regression and encoder implementations |
| [Corrected records](reproducibility/corrected/records) | Per-round curves, split records and summaries |
| [Official FedChem references](reproducibility/official_audit) | Separately identified historical implementation references |
| [Historical task analysis](reproducibility/historical_task) | Single-seed Tox21 per-task predictions and figure inputs |
| [Datasets](reproducibility/datasets) | Public benchmark inputs, scaffold labels and saved splits |
| [Coverage](reproducibility/COVERAGE.md) | Supported reproduction tasks and their limits |

The corrected classification cohort contains 360 runs / 120 three-seed cells;
the regression cohort contains 162 runs / 54 three-seed cells. Other diagnostics
have separately documented protocols and seed counts.

**Do not pool the three result namespaces.** Official FedChem references differ
from the corrected protocol in supervision and splitting. Historical per-task
figures are single-seed analyses, not estimates of multi-seed variability.
The detailed guide describes these distinctions and known molecular-identity
overlaps in BBBP. Trained model checkpoints are not included. Keep generated
outputs outside `reproducibility/` so that the supplied results remain unchanged.

## Attribution and licensing

Original author-owned contributions are offered under [MIT](LICENSE), subject
to the scope and third-party exclusions in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Bundled upstream code and public datasets are **not** relicensed by this release.
Some pinned upstream projects do not provide an explicit top-level license;
their inclusion is not a claim of an unrestricted reuse grant.

Please acknowledge [FedChem](https://github.com/ur-whitelab/fedchem),
[MMGNN](https://github.com/MathIntelligence/MMGNN), the benchmark data sources,
and the software dependencies when using their components. Citation metadata
for this research software is provided in [CITATION.cff](CITATION.cff).
