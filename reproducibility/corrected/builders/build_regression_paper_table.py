"""Build a traceable regression table from certified, complete three-seed runs."""

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from deadline_queue_20260920.summarize_regression_complete import collect
import regression_physchem_20260924 as physchem


ROOT = Path(__file__).resolve().parents[2]
REVISION = ROOT / "results/reviewer_revision_20260831"
INPUT = REVISION / "regression_preparation_20260916/verified_existing_runs.csv"
OUT = REVISION / "revision_approval_20260924/regression"
TEX = ROOT / "paper_acs_latex/regression_results_table_generated.tex"
PHYSCHEM = REVISION / "regression_physchem_20260924"
DATASETS = (("esol", "ESOL"), ("freesolv", "FreeSolv"), ("lipo", "Lipophilicity"))
ALPHAS = ("0.1", "0.5", "1.0")
METHODS = ("FedAvg", "FedProx", "FLIT+", "MOON-MSE", "FedSPL (fingerprint)", "FedSPL (physchem)")
BASELINE_CODES = {"FedProx": "fedprox", "FLIT+": "oursvatFLITPLUS", "MOON-MSE": "moon"}


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def emit(path, content, check=False):
    if check:
        if not path.exists() or path.read_text() != content:
            raise ValueError("Generated output is stale: " + str(path))
    else:
        path.write_text(content)


def write_csv(path, rows, check=False, fields=None):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    emit(path, stream.getvalue(), check)


def validate_existing(row):
    source = Path(row["source"])
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != row["source_sha256"]:
        raise ValueError("Source changed since prior verification: " + str(source))
    curve = [r for r in read_csv(source) if r.get("scope") == "global"]
    if len(curve) != 50 or [int(r["round"]) for r in curve] != list(range(50)):
        raise ValueError("Expected all 50 communication rounds")
    for r in curve:
        if r["metric"] != "rmse" or r["encoder"] != "mpnn" or r["partition_method"] != "hetero":
            raise ValueError("Incompatible metric/encoder/partition")
        if r["method"] != row["method"] or r["dataset"] != row["dataset"]:
            raise ValueError("Method/dataset metadata mismatch")
        if any(int(r[k]) != int(row["seed"]) for k in ("model_seed", "split_seed", "partition_seed")):
            raise ValueError("Unexpected randomization metadata")
        if not all(math.isfinite(float(r[k])) and float(r[k]) >= 0 for k in ("val", "test")):
            raise ValueError("Invalid RMSE endpoint")
        if row["method"] == "fedavg_proto":
            if (r["descriptor"] != row["descriptor"] or float(r["lambda_proto"]) != 0.1
                    or int(r["local_clusters"]) != 16 or int(r["global_clusters"]) != 16):
                raise ValueError("Unexpected prototype configuration")
    if any(float(curve[-1][key]) != float(row["final_" + key + "_rmse"]) for key in ("val", "test")):
        raise ValueError("Certified endpoint differs from raw final round")
    if row["method"] == "avg":
        method = "FedAvg"
    elif row["method"] == "fedavg_proto" and row["descriptor"] in ("fingerprint", "physchem"):
        method = "FedSPL (" + row["descriptor"] + ")"
    elif row["method"] in BASELINE_CODES.values():
        method = next(name for name, code in BASELINE_CODES.items() if code == row["method"])
    else:
        raise ValueError("Unexpected existing method")
    return dict(dataset=row["dataset"], alpha=str(float(row["alpha"])), seed=int(row["seed"]),
                method=method, final_test_rmse=float(row["final_test_rmse"]),
                final_val_rmse=float(row["final_val_rmse"]), endpoint="final_server_round49",
                cohort="reviewer_v3_regression_raw_targets", source=str(source), source_sha256=digest)


def load_physchem_records():
    approval = json.loads((PHYSCHEM / 'approval.json').read_text())
    expected = {case['case_id']: case for case in physchem.cases()}
    if approval['cases'] != physchem.cases() or approval['approved'] is not True:
        raise ValueError('Physchem approval does not match frozen 18-condition extension')
    parent = json.loads((REVISION / 'regression_followup_20260918/raw_baselines/queue_lock.json').read_text())
    if (approval['parent_lock_sha256'] != parent['lock_sha256'] or approval['gpu'] != 2
            or approval['allocator_cap_enabled'] is not False):
        raise ValueError('Physchem frozen parent or device approval mismatch')
    if hashlib.sha256(Path(physchem.__file__).read_bytes()).hexdigest() != approval['launcher_sha256']:
        raise ValueError('Physchem launcher changed since approval')
    rows = read_csv(PHYSCHEM / 'completed_runs.csv')
    if len(rows) != 18 or {row['case_id'] for row in rows} != set(expected):
        raise ValueError('Require all 18 distinct completed physchem runs')
    records = []
    approval_sha = hashlib.sha256((PHYSCHEM / 'approval.json').read_bytes()).hexdigest()
    for row in rows:
        case = expected[row['case_id']]
        for key, value in case.items():
            actual = row[key]
            matches = float(actual) == value if isinstance(value, (int, float)) else actual == value
            if not matches:
                raise ValueError('Physchem source condition mismatch: ' + key)
        directory = PHYSCHEM / 'cases' / case['case_id']
        spec = json.loads((directory / 'case_spec.json').read_text())
        done = json.loads((directory / 'completed.json').read_text())
        source = directory / Path(row['source']).name
        log = directory / 'run.log'
        if (done['case'] != case or spec['case'] != case or done['exit_code'] != 0
                or done['formal'] is not True or spec['formal'] is not True):
            raise ValueError('Incomplete physchem run or wrong completion identity')
        if (spec['argv'] != physchem.argv(case, Path(row['source']).parent)
                or spec['approval_sha256'] != approval_sha
                or spec['parent_lock_sha256'] != approval['parent_lock_sha256']
                or spec['labels_standardized'] is not False or spec['source_changed'] is not False
                or spec['torch'] != '1.8.1' or spec['cuda'] != '11.1' or spec['gpu'] != 2
                or spec['allocator_cap_enabled'] is not False):
            raise ValueError('Physchem protocol or runtime mismatch')
        if (hashlib.sha256(log.read_bytes()).hexdigest() != done['log_sha256']
                or done['curve_sha256'] != row['source_sha256']
                or 'PHYSCHEM_WORKER_COMPLETE ' + case['case_id'] not in log.read_text()):
            raise ValueError('Physchem completion hashes or log marker mismatch')
        records.append(validate_existing(dict(row, source=str(source))))
    return records


def load_records():
    raw = [validate_existing(r) for r in read_csv(INPUT)]
    if len(raw) != 63:
        raise ValueError("Expected the audited 54 main and nine descriptor runs")
    # Reuse the completed-cohort audit, including frozen specs and curve hashes.
    additions = [r for r in collect() if r["cohort"] == "completed_81"]
    if len(additions) != 81:
        raise ValueError("Expected all 81 completed baseline runs")
    for row in additions:
        raw.append(validate_existing(dict(
            row, method=BASELINE_CODES[row["method"]], source=row["curve_path"],
            source_sha256=row["curve_sha256"])))
    raw.extend(load_physchem_records())
    keys = {(r["dataset"], r["alpha"], r["method"], r["seed"]) for r in raw}
    expected = {(dataset, alpha, method, seed) for dataset, _ in DATASETS
                for alpha in ALPHAS for method in METHODS for seed in (0, 1, 2)}
    if keys != expected or len(raw) != 162:
        raise ValueError("Duplicate or missing regression source runs")
    return raw


def paired_contrast(dataset, alpha, treatment, control, values, references):
    treated = {r['seed']: r['final_test_rmse'] for r in values}
    baseline = {r['seed']: r['final_test_rmse'] for r in references}
    if len(values) != 3 or len(references) != 3 or set(treated) != set(baseline) or set(treated) != {0, 1, 2}:
        raise ValueError('A contrast requires three matched distinct seeds')
    differences = [baseline[seed] - treated[seed] for seed in (0, 1, 2)]
    mean, sd = statistics.mean(differences), statistics.stdev(differences)
    margin = 4.302652729911275 * sd / math.sqrt(3)
    return dict(dataset=dataset, alpha=alpha, treatment=treatment, control=control, n=3,
                mean_rmse_reduction=mean, sd_paired_reduction=sd,
                ci95_low=mean-margin, ci95_high=mean+margin,
                positive_seeds=sum(x > 0 for x in differences),
                interpretation='Exploratory unadjusted paired t interval, df=2')


def statistics_cell(row, rank=None):
    if row["status"] != "complete":
        return "--"
    mean = format(row["mean_rmse"], ".4f")
    if rank == 1:
        mean = r"\textbf{" + mean + "}"
    elif rank == 2:
        mean = r"\underline{" + mean + "}"
    return (r"\shortstack{" + mean + r"\\[-0.1em]"
            + r"{\scriptsize$\pm " + format(row["sd_rmse"], ".4f") + r"$}}")


def main(check=False):
    raw = load_records()
    groups = defaultdict(list)
    for row in raw:
        groups[row["dataset"], row["alpha"], row["method"]].append(row)
    summaries = []
    for dataset, _ in DATASETS:
        for alpha in ALPHAS:
            for method in METHODS:
                values = groups.get((dataset, alpha, method), [])
                if sorted(r["seed"] for r in values) != [0, 1, 2]:
                    raise ValueError("Each published cell requires exactly seeds 0, 1 and 2")
                scores = [r["final_test_rmse"] for r in values]
                summaries.append(dict(dataset=dataset, alpha=alpha, method=method, n=len(values),
                                      mean_rmse=statistics.mean(scores),
                                      sd_rmse=statistics.stdev(scores), status="complete",
                                      endpoint="final_server_round49", cohort="reviewer_v3_regression_raw_targets"))
    if not check:
        OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "regression_table_source_runs.csv", raw, check)
    write_csv(OUT / "regression_table_summary.csv", summaries, check)
    write_csv(OUT / "regression_missing_cells.csv", [], check, fields=list(summaries[0]))
    lookup = {(r["dataset"], r["alpha"], r["method"]): r for r in summaries}
    paired = []
    for (dataset, alpha, method), values in groups.items():
        if method == "FedAvg":
            continue
        paired.append(paired_contrast(dataset, alpha, method, "FedAvg", values,
                                      groups[dataset, alpha, "FedAvg"]))
    write_csv(OUT / "regression_paired_differences.csv", paired, check)
    descriptor_pairs = [paired_contrast(dataset, alpha, 'FedSPL (physchem)', 'FedSPL (fingerprint)',
        groups[dataset, alpha, 'FedSPL (physchem)'], groups[dataset, alpha, 'FedSPL (fingerprint)'])
        for dataset, _ in DATASETS for alpha in ALPHAS]
    write_csv(OUT / 'regression_descriptor_contrasts.csv', descriptor_pairs, check)
    lines = [
        r"% Generated by scripts/reviewer_revision/build_regression_paper_table.py.",
        r"\begin{table}[!tbp]", r"\caption{Final RMSE on molecular regression benchmarks.}",
        r"\label{tab:regression_results}", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{3pt}", r"\renewcommand{\arraystretch}{1.18}",
        r"\begin{tabular}{lc@{\hspace{8pt}}cccccc}", r"\toprule",
        r"Dataset & $\alpha$ & FedAvg & FedProx & FLIT+ & \shortstack{MOON\\(MSE)} & \shortstack{FedSPL\\fingerprint} & \shortstack{FedSPL\\physchem} \\",
        r"\midrule",
    ]
    for j, (dataset, title) in enumerate(DATASETS):
        if j:
            lines.append(r"\midrule")
        for i, alpha in enumerate(ALPHAS):
            label = r"\multirow{3}{*}{" + title + "}" if i == 0 else ""
            ranked_values = sorted({lookup[dataset, alpha, m]["mean_rmse"] for m in METHODS[:-1]})
            ranks = {value: i + 1 for i, value in enumerate(ranked_values)}
            cells = [statistics_cell(lookup[dataset, alpha, method],
                     ranks[lookup[dataset, alpha, method]["mean_rmse"]] if method in METHODS[:-1] else None)
                     for method in METHODS]
            lines.append(" & ".join([label, alpha] + cells) + r" \\")
    lines.extend([
        r"\bottomrule", r"\end{tabular}", r"\vspace{0.4em}",
        r"\parbox{\linewidth}{\footnotesize\textit{Note:} Mean $\pm$ sample SD over three paired seeds (0, 1, 2), each varying the molecular split, client allocation, and initialization; lower is better. The endpoint is the final aggregated server model after 50 rounds, not the best test round. Targets are unstandardized; units are $\log_{10}(\mathrm{mol/L})$ for ESOL, kcal/mol for FreeSolv, and logD for Lipophilicity. MOON uses MSE supervision and retains its representation-contrastive objective. Both FedSPL variants use 16 local/global clusters and $\lambda_{\mathrm{proto}}=0.1$. Bold and underline identify the lowest and second-lowest unrounded means among the five main methods (FedAvg, FedProx, FLIT+, MOON-MSE, and fingerprint FedSPL), not statistical significance. Physchem denotes the separate 19-feature physicochemical-descriptor variant, evaluated at all three alpha values and excluded from this ranking. Neither FedSPL column selects the better descriptor per condition.}",
        r"\end{table}", "",
    ])
    emit(TEX, "\n".join(lines), check)
    status = dict(verified_runs=len(raw), complete_cells=sum(r["status"]=="complete" for r in summaries),
                  pending_baseline_cells=sum(r["status"]=="pending_baseline" for r in summaries),
                  descriptor_runs=27, new_physchem_sources=18, descriptor_complete_cells=9,
                  baseline_runs_needed=0, table=str(TEX), original_results_modified=False,
                  final_submission_ready=False, missing_methods=[],
                  excluded_by_request=["FedProto"],
                  label_prototype_regression_adaptation_not_assumed=["FPL", "FedTGP"],
                  centralized_regression_reference_available=False)
    emit(OUT / "table_status.json", json.dumps(status, indent=2) + "\n", check)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Verify generated files without writing")
    main(parser.parse_args().check)
