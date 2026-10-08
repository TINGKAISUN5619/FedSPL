# JCIM FedSPL Reviewer Reproducibility Archive

Archive: `JCIM_FedSPL_reproducibility_20260926.zip` (26 September 2026).
This is a standalone source/result audit and figure-reproduction package, not a
claim of bitwise training replay. No training was performed while packaging.
No manuscript TeX, checkpoints, graph caches, credentials, server login commands,
or executable queue orchestrators are distributed.

## Start Here

Run from the extracted archive, with bytecode generation disabled:

```sh
python3 -B verify.py
python3 -B reproduce_tables.py --output ../numeric_tables_20260926
python -B test_cpu.py
python -B reproduce_figures.py --output ../reexport_20260926
```

`verify.py` needs only Python 3.8+ standard library. It checks every SHA256SUMS
entry, exact inventory coverage, original numerical curve hashes, final-round
endpoints, means/sample SDs, classification split/allocation membership, and
negative tests rejecting truncated/reordered/duplicate curves and best-test
selection. It neither imports scientific training modules nor starts jobs.
The optional CPU and figure commands need the environment in ENVIRONMENT.md.
Their output directory must be new and outside this immutable archive.

For an explicit single-run execution path, `train.py` prepares a disposable copy
of the frozen source and public inputs, and substitutes only local filesystem
acquisition for network downloads. Defaults do not train:

```sh
python -B train.py --config classification_example.json --workspace ../classification_import --import-smoke
python -B train.py --config regression_example.json --workspace ../regression_import --import-smoke
python -B train.py --config classification_example.json --workspace ../new_classification_run --execute
```

Only the last command starts training, and it was NOT run by the packager.
Use `regression_example.json` for the frozen raw-target regression implementation.
The import-smoke checks load the actual frozen entry and parse its help; they do
not test graph construction, training or checkpoint regeneration. Generated
`PREPARED.json` records exact arguments and copied source hashes. Fresh graph
construction requires compatible RDKit/DGL, valid-count/split validation, and
the resources described in ENVIRONMENT.md. No full-matrix runner is promised.

## Three Distinct Evidence Namespaces

* `corrected/`: reviewer-v3 masked/disjoint-index main classification and raw-scale
  regression, plus separately identified central, mechanism, encoder, traditional,
  independent-client and prototype-support diagnostics. Separate endpoints and
  seed designs are never pooled simply because this directory is shared.
* `official_audit/`: official historical FedChem references, original Tox21/HIV
  results and complete Tox21 masked-curve replay. Different split and supervision;
  never a matched baseline in corrected paired statistics. The mixed-protocol
  published convergence display is preserved here only for traceability.
* `historical_task/`: July single-seed Tox21 Figure 5, current two-panel PDF,
  unchanged `restore_historical_task_panel_20260925.py`, held-out predictions,
  task/bootstrap CSV, run summaries and training metadata. No old three-panel
  hard-coded table graphic is included. Task SEM is not seed uncertainty.

`PROVENANCE.json` maps every repository-relative input to its actual archive
path, original hash, payload hash and transformation. Scientific `.py` files
are byte-identical copies. Metadata containing deployment paths is explicitly
marked `deployment_paths_only`; these copies are not original-byte certificates.
Numerical curve CSVs used for primary endpoint verification retain their original
bytes and hashes. Portable adapters are new packaging code, not historical code.
`PACKAGING_AUDIT.json` records inspected manuscript/SI hashes and referenced labels,
but includes no manuscript text. Source paths retained in manifests are provenance
identifiers, not commands or prerequisites for access to another filesystem.

## Corrected Main Protocol

Classification: BBBP, SIDER, Tox21, HIV, ToxCast; alpha 0.1/0.5/1.0; model,
split and partition seeds jointly 0/1/2; eight methods; 360 runs and 120 cells.
Use the manifest-selected source, including the three declared SIDER alpha 0.5
K16 reuses. Never replace a missing condition by a nearby experiment.
The outer split is NumPy RandomState(split_seed), first floor(0.8*N) for training.
Remaining indices are shuffled with a separate Torch generator seeded
split_seed+104729; first floor(heldout/2) is validation, remainder is test.
Four clients partition training by stored scaffold labels and balanced Dirichlet
allocation. All four participate for 50 rounds, 200 local updates per round,
batch 64, Adam recreated locally, learning rate 1e-4, weight decay 1e-5,
gradient norm clip 1. Incomplete training batches are dropped; complete local
training sets are visited for prototype collection.

Use observed-label BCE-with-logits, averaged over observed batch entries.
Final aggregated server-model test AUC is task-macro sigmoid-probability AUC,
excluding missing entries and tasks without both observed classes. CSV round 49
is the endpoint; best_test/best_val fields are not reported endpoints. Summaries
use arithmetic mean and sample SD (ddof=1), not SE. Paired t intervals use df=2,
are exploratory/unadjusted, and do not establish equivalence or significance
after multiple comparisons. Rounds and dataset-alpha cells are not extra seeds.

FedSPL: Morgan radius 2, 2048 bits; Euclidean k-means, n_init=10; hard cosine
descriptor matching; embedding MSE; lambda=0.1; local/global K16 (ToxCast K32).
Unweighted clustering, support-weighted subsequent prototype averages; local
clustering seed is partition seed plus zero-based client ID. No public proxy
distillation dataset. MPNN embedding dimension 128, node hidden 64, edge hidden
16, three message-passing steps, Set2Set and 128->64->task prediction head.

Regression: ESOL, FreeSolv, Lipophilicity; three alphas and three joint seeds;
FedAvg, FedProx, FLIT+, MOON-MSE, fingerprint SPL and physchem SPL; 162 runs,
54 cells. Final round-49 server RMSE on unstandardized targets. Units:
log10(mol/L), kcal/mol, logD respectively. SPL uses K16 and lambda=0.1.
The 19-feature physchem variant is separate, not a per-cell best-descriptor choice.
Frozen regression code includes the MSE adaptation; it is not substituted for
the classification snapshot. No centralized regression reference is supplied.

## Source and Execution Boundaries

The primary source snapshot is `corrected/source/reviewer_v3_snapshot/`, with
separate frozen regression and encoder snapshots. The imported trainer is
`fedml_api/standalone/fedavg/my_model_trainer_classification.py`, not a similarly
named historical root-level file. The loader exposes labels, masks and stable IDs
as separate fields and disables local-test loader aliases. Five-field molecular
samples have stable IDs; the generic 3/4-field fallback can use batch-local IDs
and must not be advertised as a globally stable-ID path.

Some archived audit inventories explicitly call the September snapshot a later
semantics reference, not the exact original version of every early run. The
archive does not invent per-run code attestations. Available run specs, hashes,
and curve certificates identify narrower exact-source relationships.
Keep frozen code unchanged even where later protocol work identified RNG-path
differences: an added DataLoader/prototype traversal can change subsequent RNG
consumption. A short CPU check is not 50-round CUDA trajectory equivalence.
No optimizer-step parity or full CUDA replay was newly run for this archive.

Original builders are retained for algorithm and selection transparency, but
some expect the old repository layout or excluded orchestration dependencies.
Do not run their write-mode mains against source records. `verify.py` is the
portable read-only result adapter; `reproduce_figures.py` redirects only I/O of
unchanged plot functions and never rewrites their scientific code. The package
does not claim every legacy worker is a turnkey executable on modern platforms.

## Known Scientific Limits

BBBP has 60 canonical-isomeric-SMILES duplicate groups. Test records with a
training identity number 12/12/8 for seeds 0/1/2. Index disjointness is not strict
unseen-molecule or scaffold-disjoint evaluation. No deduplication, salt/tautomer
normalization or retrospective score correction was done. ToxCast stored scaffold
groups match stereochemistry-preserving Murcko groups; other classification sets
match non-stereochemical grouping. Frozen allocations are authoritative.

Historical FedChem supervision sigmoids both predictions and binary targets
(0/1 become 0.5/about 0.731), does not mask unavailable supervised entries, uses
fixed outer split seed 42 and different evaluation RNG behavior. Masked rescoring
does not repair this training difference. Cross-protocol held-out sets overlap
the other protocol's training data; no common-test reevaluation is asserted.
Official repeat trajectories were not numerically identical; whole cohorts and
differences are retained, without selecting favorable seeds or rounds.

FedProto-adapted, FPL-inspired and FedTGP-inspired retain shared-model averaging
and observed task-class prototypes; they are not upstream method reproductions.
FPL omits domain/prototype clustering and hierarchical contrastive loss. FedTGP
uses directly optimized vectors with fixed-temperature contrastive loss/anchor,
not a generator and adaptive margin. Prototype losses differ from SPL's MSE.

Mechanism controls: separate 30-run paired diagnostic cohort, server and
client-mean endpoints separate, not main-table replacements. Traditional ML:
135 client-local conditions; constant AUC 0.5 fallback analytically restores
unfittable tasks, not new fits. Independent MPNN: 15 seed-0 references only,
no three-seed SD. MMGNN: two datasets, fixed split/allocation seed 0, initialization
0/1/2; raw-logit AUC, validation-selected learning rate, 12 federated plus 6
individual runs. None of these is a universal performance superiority claim.

No retained weights means checkpoint inference and original prediction integrity
cannot be newly reverified for every run. Source hashes attest bytes, not the truth
of a scientific claim. Saved metadata attestations are explicitly weaker than
new training or inference. See COVERAGE.md and RELEASE_TESTS.json for actual scope.
