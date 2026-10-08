# Environments and Public Data

## Local Reproduction

`reproduce_tables.py`: Python 3.8+ standard library, no network, GPU or dataset download.

CPU tests and plot reexports were run with Python 3.8.20 on macOS, PyTorch 2.4.1,
DGL 1.1.2, DGLLife 0.3.2, RDKit 2024.3.5, NumPy 1.24.4, pandas 2.0.3,
scikit-learn 1.3.2, SciPy 1.10.1, matplotlib 3.7.5 and easydict 1.13.
Install into a separate compatible environment; this is a tested package-version
inventory, not a solver-tested cross-platform lockfile. No new environment was
installed during packaging. Plotting uses the noninteractive Agg backend.

Historical CUDA run metadata record Python 3.8.20, PyTorch 1.8.1, CUDA 11.1,
DGL 0.6.1, RDKit 2022.09.5 and NVIDIA A100-PCIE-40GB in relevant cohorts.
Consult each run spec for its own environment; the CPU environment above is not
declared equivalent. A complete original conda/container lock is not available.
MMGNN upstream source is pinned in the SI to
`78dcbff3a3d576506434241e4f26c70416453d9e`, from
https://github.com/MathIntelligence/MMGNN . Upstream attribution/license files are retained when available;
no new blanket software or dataset license is asserted.

## Data Acquisition and Preprocessing

Public CSV inputs are included for all eight datasets under `datasets/`.
Classification inputs and 15 split JSONs come from the frozen central bundle;
regression public CSVs come from local public-download copies. These inputs are
not model weights. Scaffold-label tensors are small original partition inputs,
not checkpoints. Do not unpickle arbitrary untrusted `.pt` files.

The frozen dataset loaders specify DGL's public dataset distribution endpoints
using `_get_dgl_url`: `dataset/bbbp.zip`, `dataset/sider.zip`,
`dataset/tox21.csv.gz`, `dataset/hiv.zip`, `dataset/toxcast.zip`,
`dataset/ESOL.zip`, `dataset/FreeSolv.zip`, `dataset/lipophilicity.zip`.
These are source-declared acquisition locations, not newly tested network links.
The included CSVs permit offline inspection even if upstream hosting changes.
Dataset-specific filenames are BBBP.csv, sider.csv, tox21.csv.gz, HIV.csv,
toxcast_data.csv, delaney-processed.csv, SAMPL.csv, and Lipophilicity.csv.

For a fresh execution workspace, copy the chosen frozen implementation to a new
directory. Copy the appropriate scaffold tensor to
`data/scaffold_result/scffoldLabel_DATASET.pt`. Populate DGL's public download
cache with the corresponding CSV/archive layout named by that dataset's loader,
or use its documented public download. Do not reuse graph caches from another
RDKit/DGL version. The frozen `data_loader.py` creates molecular graphs with
canonical atom/bond features, self loops, and the dataset-specific constructors.
`data/csv_dataset.py` removes graph-construction failures, records `valid_ids`,
and creates float labels with observation masks. Dataset-specific `__getitem__`
overrides supply stable IDs. Do not replace missing targets by observed negatives.

Before training, verify valid counts against the archived dataset statistics
(2039,1427,7831,41127,8577 for the five classification sets), label/task dimensions,
CSV row order and valid_ids, and the stored scaffold tensor length. Any RDKit
version-induced change requires an explicit new protocol, not silent repair.
Graph caches/valid-id tensors are not included; the old exact graph pipeline must
be reproduced and validated for full training replay. No automatic new filtering,
deduplication or scaffold recomputation is licensed by this package.

A source-entry example, after those preparations in a disposable workspace:

```sh
python -B main.py -dataset Tox21 -fedmid fedavg_proto -part_alpha 0.5 -seed 0 \
  --split_seed 0 --partition_seed 0 --partition_method hetero --encoder mpnn \
  -comm_round 50 -numClient 4 --clients_per_round 4 --local_steps_per_round 200 \
  --topoproto_clusters 16 --topoproto_global_clusters 16 --lambda_proto 0.1 \
  --proto_descriptor_space fingerprint --global_only_eval --num_workers 0 \
  --results_dir fresh_results
```

This is a user-controlled training entry example, not an executed/tested replay
command or an exact historical hardware/worker configuration. Consult frozen
configuration CSVs and run specs; do not change worker counts/RNG-consuming
evaluation paths and claim exact trajectory identity. No queue or server access
is required for table reproduction or CPU tests.
