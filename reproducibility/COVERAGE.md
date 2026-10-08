# Main and Unified SI Coverage

Paths below are original relative identifiers located by the portable entry
points. All referenced result CSVs are physically included, not merely
listed. Main and SI TeX source files are deliberately excluded.

| Object | Included numeric/source evidence | Reproduction scope |
|---|---|---|
| Main Table 1 dataset statistics | public CSVs; source_dataset_landscape_stats.csv; dataset loaders; partition/identity audits | Rounded landscape statistics preserved; graph filtering is not rerun |
| Main classification table and paired analyses | main_classification_source_runs/summary/paired CSVs; 360 original curves; corrected snapshot; manifests | Means and sample SDs aggregated from stored endpoints |
| Central reference | 15 specs/completion records/split IDs/block curves/endpoints; central_source_runs/summary; central_mpnn.py | Final block 49 and 800 updates/block checked; no checkpoint inference |
| Main regression table | regression_table_source_runs/summary/contrasts; 162 original curves; frozen regression code and specs | 54 means/SDs aggregated from stored endpoints |
| Main convergence figure | all corrected source curves and plot CSVs; original plotting script; current official-reference display PDF and separate official curve CSVs | Corrected convergence reexport supported; historical substitution is audit-only |
| Main Figure 5 | current two-panel PDF; restore_historical_task_panel_20260925.py; held-out predictions; 36 task AUC/CI rows; run summaries and training metadata | Reexport recomputes 36 task AUCs and three macro summaries; stored bootstrap bounds retained, not resampled |
| Prototype count/support figure | k_summary, run/round/support audits; occupancy and original result CSVs; plotting/audit code | Source records included; plot reexport, not new clustering |
| Main schematics and TOC | exact currently referenced raster assets | Illustrative assets, no empirical CSV implied |
| SI independent-client LR/NB/RF | 135 source conditions, 45 summary cells, raw per-client-task CSVs, full-coverage fallback records, implementations | Means/SDs; no estimator refitting |
| SI mechanism endpoints/sizes/contrasts | 30 raw endpoint curves, task curves, split/batch trace metadata, 120 endpoints, 96 differences and 32 summaries, control implementations | Final endpoints recomputed; archived pairing certificates not new trajectory replay |
| SI encoder/descriptor robustness | sources.csv, summary/contrast CSVs; raw selected curves; MPNN/AttentiveFP implementations | Final endpoints; no retraining |
| SI adaptation disclosure | unchanged corrected trainer/client/server source; README adaptation differences | Code-level definition, not upstream benchmark reproduction |
| SI official historical references | all original 18 cases and 9 Tox21 replay cases; specs, final rescores, per-round/per-task replay data; separate official source | Audit-only; missing weights/probabilities preclude new inference for all runs |
| SI independent MPNN | 15 seed-0 case curves/specs/completion/selection/split records, worker, summary | Final client means; no three-seed uncertainty claim |
| SI MMGNN server/seeds/client means | 102 endpoint rows, 12 federated and 6 independent cases, raw-logit curves, selection/development records, adapter/upstream source/workers | Final 50-round endpoints; client mean derived from four client AUCs |
| Split/identity/allocation claims | 15 split JSONs, 45 client membership JSONs, duplicate/scaffold summaries and all 45000 permutation-null observations | Index coverage/disjointness checked; chemical-identity overlap disclosed |

## Honest Gaps

Not a full checkpoint/inference archive. Graph caches and checkpoints were
deliberately excluded. Some historical scripts import excluded queue
utilities, assume an old layout, or emit TeX; they are reference implementations,
not the recommended portable commands. Not all old code versions have an
exact run-to-source correspondence. Historical source namespaces must not be
presented as corrected execution paths.

No 50-round training, fresh CUDA execution, long-run RNG equivalence, public URL
download, or clean-environment installation was performed. CPU tests use finite
missing-label fill values; they do not assert NaN/Inf fill safety. The restored
historical task panel checks literal SMILES separation in saved metadata, not
canonical identity or a fresh training replay. BBBP split tests deliberately do
not claim molecule-identity disjointness. No zero-critique guarantee is made.
