#!/usr/bin/env python3
"""Build only the approved final-round reviewer-v3 classification table."""

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[2]
REVISION = ROOT / "results/reviewer_revision_20260831"
STATUS = REVISION / "robustness_audit_20260910/summaries/expected_run_status.csv"
FORMAL = REVISION / "server_synced/formal_v3"
OUT = REVISION / "revision_approved_20260920"
TEX = ROOT / "paper_acs_latex/main_results_table_generated.tex"
CENTRAL = REVISION / "deadline_queue_20260920/central_mpnn/queue"
CENTRAL_OUT = REVISION / "revision_approval_20260924/central"
CENTRAL_PROTOCOL = "central_mpnn_reviewer_v3_50x800_v1"
DATASETS = ("BBBP", "SIDER", "Tox21", "HIV", "ToxCast")
ALPHAS = ("0.1", "0.5", "1.0")
SEEDS = (0, 1, 2)
METHODS = ("avg", "fedprox", "oursvatFLITPLUS", "fedproto", "moon", "fpl", "fedtgp", "fedavg_proto")
LABELS = dict(zip(METHODS, ("Avg", "Prox", "FLIT+", "Proto", "MOON", "FPL", "TGP", "SPL")))
T_CRITICAL = 4.302652729911275
COHORT = "server_synced/formal_v3:main_classification"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def csv_rows(data):
    return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))


def key(row):
    return row["dataset"], row["alpha"], row["method"], int(row["seed"])


def expected_keys():
    return set(itertools.product(DATASETS, ALPHAS, METHODS, SEEDS))


def validate_manifest(rows):
    require(len(rows) == 360, "Expected exactly 360 main_classification conditions")
    require(len({key(r) for r in rows}) == 360, "Duplicate condition")
    require({key(r) for r in rows} == expected_keys(), "Missing/unapproved dataset, alpha, method or seed")
    require(len({r["case_id"] for r in rows}) == 360, "Duplicate case ID")
    for r in rows:
        for field, value in (("suite", "main_classification"), ("status", "complete"),
                             ("partition", "hetero"), ("encoder", "mpnn"), ("metric", "auc"),
                             ("target_round", "49"), ("max_round", "49")):
            require(r[field] == value, "Invalid manifest " + field + ": " + r["case_id"])
        is_spl = r["method"] == "fedavg_proto"
        require(r["descriptor"] == ("fingerprint" if is_spl else ""), "Unexpected descriptor")
        require(r["k"] == (str(32 if r["dataset"] == "ToxCast" else 16) if is_spl else ""),
                "Unexpected manifest cluster count")
        require(r["lambda_proto"] == ("0.1" if is_spl else ""), "Unexpected lambda")


def resolve_source(row):
    original = Path(row["csv_file"])
    # Preserve the manifest-selected file even if the repository is relocated.
    prefix = "results/reviewer_revision_20260831/server_synced/formal_v3/"
    require(prefix in str(original), "Source is not in the frozen formal-v3 mirror")
    path = (FORMAL / str(original).split(prefix, 1)[1]).resolve()
    relative = path.relative_to(FORMAL.resolve())
    if relative.parts[0] == "priority_suite":
        expected_parent = ("priority_suite/k_occupancy/"
                           "k_occupancy_SIDER_a0p5_hetero_seed%d_fedavg_proto_mpnn_fingerprint_k16" % int(row["seed"]))
        require(row["dataset"] == "SIDER" and row["alpha"] == "0.5" and row["method"] == "fedavg_proto"
                and str(relative.parent) == expected_parent, "Unapproved priority-suite reuse")
        reuse = "manifest_declared_SIDER_alpha0.5_K16_reuse"
    else:
        expected_parent = "main_classification/%s/alpha_%s/hetero/seed_%s/%s" % (
            row["dataset"], row["alpha"], row["seed"], row["method"])
        require(str(relative.parent) == expected_parent, "Source directory/condition mismatch")
        reuse = "main_queue"
    require(path.is_file(), "Missing original curve: " + str(path))
    return path, reuse


def validate_curve(curve, expected):
    require(len(curve) == 50, "Need exactly 50 global CSV rows")
    require([int(r["round"]) for r in curve] == list(range(50)), "Missing, duplicate or reordered rounds")
    for r in curve:
        for field, value in (("scope", "global"), ("metric", "auc"), ("encoder", "mpnn"),
                             ("partition_method", "hetero"), ("dataset", expected["dataset"]),
                             ("method", expected["method"]), ("descriptor", expected["descriptor"])):
            require(r[field] == value, "Curve metadata mismatch: " + field)
        require(all(int(r[k]) == int(expected["seed"]) for k in
                    ("model_seed", "split_seed", "partition_seed")), "Unmatched model/split/partition seed")
        require(all(math.isfinite(float(r[k])) and 0 <= float(r[k]) <= 1 for k in ("val", "test")),
                "Nonfinite/out-of-range AUC")
        if expected["method"] == "fedavg_proto":
            require(float(r["lambda_proto"]) == 0.1 and
                    r["local_clusters"] == r["global_clusters"] == expected["k"], "SPL settings mismatch")
        else:
            require(all(r[k] == "" for k in ("lambda_proto", "local_clusters", "global_clusters")),
                    "Unexpected prototype configuration")
    last = curve[-1]
    require(all(float(last[k]) == float(expected["final_" + k]) for k in ("val", "test")),
            "Manifest final endpoint differs from raw round49")
    return last


def dense_ranks(means):
    ordered = sorted(set(means.values()), reverse=True)
    return {method: ordered.index(value) + 1 for method, value in means.items()}


def paired_values(treatment, control):
    require(set(treatment) == set(control) == set(SEEDS), "Pair requires exactly the same three seeds")
    return [treatment[s] - control[s] for s in SEEDS]


def paired_statistics(values):
    mean, sd = statistics.mean(values), statistics.stdev(values)
    margin = T_CRITICAL * sd / math.sqrt(3)
    return mean, sd, mean - margin, mean + margin


def csv_text(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def table_cell(row):
    mean = format(row["mean_auc"], ".4f")
    if row.get("rank") == 1:
        mean = r"\textbf{" + mean + "}"
    elif row.get("rank") == 2:
        mean = r"\underline{" + mean + "}"
    return (r"\shortstack{" + mean + r"\\[-0.1em]{\footnotesize$\pm " +
            format(row["sample_sd_auc"], ".4f") + r"$}}")


def central_final_endpoint(endpoints, split):
    selected = [r for r in endpoints if r["checkpoint"] == "final" and r["split"] == split]
    require(len(selected) == 1, "Need one final central endpoint per split")
    row = selected[0]
    require(row["block"] == 49 and math.isfinite(row["auc"]) and 0 <= row["auc"] <= 1
            and row["valid_tasks"] > 0, "Invalid final central endpoint")
    return row


def build_central():
    certificate_bytes = (CENTRAL / "verified_cases.json").read_bytes()
    certificates = json.loads(certificate_bytes)
    expected_ids = {"central_%s_seed%d" % (d, s) for d in DATASETS for s in SEEDS}
    require(len(certificates) == 15 and {r["case_id"] for r in certificates} == expected_ids,
            "Central certificate must contain the exact 15 approved runs")
    verified = {r["case_id"]: r for r in certificates}
    sources, summary = [], []
    for dataset in DATASETS:
        values = []
        for seed in SEEDS:
            case_id = "central_%s_seed%d" % (dataset, seed)
            directory = CENTRAL / "cases" / case_id
            done_bytes = (directory / "completed.json").read_bytes()
            done = json.loads(done_bytes)
            for record in (done, verified[case_id]):
                require(record["case_id"] == case_id and record["status"] == "complete"
                        and record["formal"] is True and record["protocol"] == CENTRAL_PROTOCOL
                        and record["blocks"] == 50 and record["successful_updates"] == 40000,
                        "Central completion identity/budget mismatch")
            # Recheck mirrored metadata; checkpoint/prediction validation was performed on the server.
            for filename in ("spec.json", "endpoints.json", "block_curve.csv", "split_ids.json",
                             "selection.json", "events.json"):
                require(sha((directory / filename).read_bytes()) == done["files"][filename],
                        "Central source hash mismatch: " + case_id + "/" + filename)
            spec = json.loads((directory / "spec.json").read_bytes())
            require(spec["protocol"] == CENTRAL_PROTOCOL and spec["formal"] is True
                    and spec["synthetic"] is False and spec["device"] == "cuda:0"
                    and spec["blocks"] == 50 and spec["updates_per_block"] == 800
                    and spec["batch_size"] == 64 and spec["case"]["dataset"] == dataset
                    and spec["case"]["seed"] == seed and spec["case"]["case_id"] == case_id,
                    "Central specification mismatch")
            curve = csv_rows((directory / "block_curve.csv").read_bytes())
            require([int(r["block"]) for r in curve] == list(range(50))
                    and all(int(r["successful_updates"]) == 800
                            and int(r["cumulative_updates"]) == 800 * (i + 1)
                            for i, r in enumerate(curve)), "Central training budget/coverage mismatch")
            endpoints = json.loads((directory / "endpoints.json").read_bytes())
            require(endpoints == verified[case_id]["endpoints"], "Central endpoints differ from server certificate")
            test, val = (central_final_endpoint(endpoints, split) for split in ("test", "val"))
            require(test["checkpoint_sha256"] == val["checkpoint_sha256"] == done["files"]["final.pt"],
                    "Central final checkpoint identity mismatch")
            values.append(test["auc"])
            sources.append(dict(dataset=dataset, seed=seed, case_id=case_id, method="Centralized MPNN",
                endpoint="final_test_block49", final_test_auc=test["auc"], final_val_auc=val["auc"],
                valid_test_tasks=test["valid_tasks"], blocks=50, updates_per_block=800,
                successful_updates=40000, alpha="not_applicable", protocol=CENTRAL_PROTOCOL,
                source=str((directory / "endpoints.json").relative_to(ROOT)),
                source_sha256=done["files"]["endpoints.json"], completion_sha256=sha(done_bytes),
                final_checkpoint_sha256=test["checkpoint_sha256"],
                server_certificate_sha256=sha(certificate_bytes)))
        summary.append(dict(dataset=dataset, method="Centralized MPNN", n=3,
            seed0_auc=values[0], seed1_auc=values[1], seed2_auc=values[2],
            mean_auc=statistics.mean(values), sample_sd_auc=statistics.stdev(values),
            alpha="not_applicable", endpoint="final_test_block49", ranked=False))
    return sources, summary


def latex_table(summary, central_summary):
    lookup = {(r["dataset"], r["alpha"], r["method"]): r for r in summary}
    central = {r["dataset"]: r for r in central_summary}
    require(len(central_summary) == 5 and set(central) == set(DATASETS), "Need five central summary cells")
    lines = ["% Generated by scripts/reviewer_revision/build_main_classification_table_20260920.py.",
             r"\begin{table}[!tbp]", r"\caption{Final classification ROC-AUC under scaffold heterogeneity.}",
             r"\label{tab:main_results}", r"\centering", r"\small",
             r"\setlength{\tabcolsep}{2pt}", r"\renewcommand{\arraystretch}{1.15}",
             r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lcc*{8}{c}@{}}", r"\toprule",
             r"Dataset & Central & $\alpha$ & Avg & Prox & FLIT+ & \shortstack{Proto\\adapted} & MOON & \shortstack{FPL\\inspired} & \shortstack{TGP\\inspired} & SPL \\",
             r"\midrule"]
    for d, dataset in enumerate(DATASETS):
        if d:
            lines.append(r"\midrule")
        for i, alpha in enumerate(ALPHAS):
            cells = [table_cell(lookup[dataset, alpha, m]) for m in METHODS]
            reference = r"\multirow{3}{*}{" + table_cell(central[dataset]) + "}" if i == 0 else ""
            lines.append(" & ".join([dataset if i == 0 else "", reference, alpha] + cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular*}", r"\vspace{0.4em}",
                  r"\parbox{\linewidth}{\footnotesize\textit{Note:} Mean $\pm$ sample SD over seeds 0, 1, and 2; higher is better. Seeds vary molecular splitting and initialization, and client allocation for federated methods. Federated entries are final aggregated server-model test AUCs after 50 rounds, not best-round or client-average scores. Central denotes pooled-data MPNN training for 50 blocks of 800 updates, matching the federated total of $50\times4\times200$ updates. Its final test AUC is shown once per dataset: $\alpha$ does not apply. Central is excluded from rankings and is not a performance upper bound. Avg, Prox, and SPL abbreviate FedAvg, FedProx, and FedSPL. Proto-adapted, FPL-inspired, and TGP-inspired denote the disclosed multi-task shared-model implementations of FedProto, FPL, and FedTGP, not unmodified reference implementations (Supporting Information Table~S8). Bold and underline mark the highest and second-highest unrounded federated means, not statistical significance. All methods use the revised MPNN split and masked-AUC protocol; no CPU or historical-protocol runs are pooled.}",
                  r"\end{table}", ""])
    return "\n".join(lines)


def build():
    central_sources, central_summary = build_central()
    manifest_bytes = STATUS.read_bytes()
    manifest = [r for r in csv_rows(manifest_bytes) if r["suite"] == "main_classification"]
    validate_manifest(manifest)
    by_key = {key(r): r for r in manifest}
    sources, runs = [], {}
    for dataset, alpha, method, seed in itertools.product(DATASETS, ALPHAS, METHODS, SEEDS):
        row = by_key[dataset, alpha, method, seed]
        path, reuse = resolve_source(row)
        raw = path.read_bytes()
        last = validate_curve(csv_rows(raw), row)
        result = dict(dataset=dataset, alpha=alpha, method=method, method_label=LABELS[method], seed=seed,
                      model_seed=int(last["model_seed"]), split_seed=int(last["split_seed"]),
                      partition_seed=int(last["partition_seed"]), case_id=row["case_id"],
                      endpoint="global_test_round49", final_test_auc=float(last["test"]),
                      final_val_auc=float(last["val"]), global_rounds=50, raw_endpoint_csv_line=51,
                      partition="hetero", encoder="mpnn", descriptor=row["descriptor"],
                      k=row["k"], lambda_proto=row["lambda_proto"], cohort=COHORT, source_role=reuse,
                      source=str(path.relative_to(ROOT)), source_sha256=sha(raw),
                      manifest_csv_file=row["csv_file"])
        runs[dataset, alpha, method, seed] = result
        sources.append(result)
    require(len({r["source"] for r in sources}) == 360, "One source reused for different conditions")
    summary, differences, intervals, ranks = [], [], [], []
    for dataset, alpha in itertools.product(DATASETS, ALPHAS):
        values = {m: {s: runs[dataset, alpha, m, s]["final_test_auc"] for s in SEEDS} for m in METHODS}
        means = {m: statistics.mean(values[m].values()) for m in METHODS}
        ranking = dense_ranks(means)
        for method in METHODS:
            summary.append(dict(dataset=dataset, alpha=alpha, method=method, method_label=LABELS[method],
                                n=3, seed0_auc=values[method][0], seed1_auc=values[method][1],
                                seed2_auc=values[method][2], mean_auc=means[method],
                                sample_sd_auc=statistics.stdev(values[method].values()), rank=ranking[method],
                                highlight="best" if ranking[method] == 1 else "second" if ranking[method] == 2 else "",
                                endpoint="global_test_round49", cohort=COHORT))
        for baseline in METHODS[:-1]:
            delta = paired_values(values["fedavg_proto"], values[baseline])
            for seed, value in zip(SEEDS, delta):
                a, b = runs[dataset, alpha, "fedavg_proto", seed], runs[dataset, alpha, baseline, seed]
                differences.append(dict(dataset=dataset, alpha=alpha, seed=seed, treatment="SPL",
                    baseline=LABELS[baseline], spl_auc=a["final_test_auc"], baseline_auc=b["final_test_auc"],
                    difference_auc=value, difference_pp=100*value, same_seed_metadata=True,
                    spl_case_id=a["case_id"], baseline_case_id=b["case_id"],
                    spl_source=a["source"], baseline_source=b["source"],
                    spl_source_sha256=a["source_sha256"], baseline_source_sha256=b["source_sha256"]))
            mean, sd, low, high = paired_statistics(delta)
            intervals.append(dict(dataset=dataset, alpha=alpha, treatment="SPL", baseline=LABELS[baseline],
                n=3, seed0_difference_pp=100*delta[0], seed1_difference_pp=100*delta[1],
                seed2_difference_pp=100*delta[2], mean_difference_pp=100*mean, paired_sd_pp=100*sd,
                ci95_low_pp=100*low, ci95_high_pp=100*high, ci_includes_zero=low <= 0 <= high,
                positive_seeds=sum(v > 0 for v in delta), negative_seeds=sum(v < 0 for v in delta),
                zero_seeds=sum(v == 0 for v in delta), t_critical_df2=T_CRITICAL,
                interpretation="Exploratory unadjusted paired t CI; not equivalence or multiplicity-controlled significance"))
        ordered = sorted(METHODS, key=lambda m: -means[m])
        ranks.append(dict(dataset=dataset, alpha=alpha, spl_rank=ranking["fedavg_proto"],
            methods=8, spl_mean_auc=means["fedavg_proto"], avg_mean_auc=means["avg"],
            spl_minus_avg_pp=100*(means["fedavg_proto"]-means["avg"]),
            descending_unrounded_mean_order=" > ".join(LABELS[m] for m in ordered),
            best_methods=";".join(LABELS[m] for m in METHODS if ranking[m] == 1),
            second_methods=";".join(LABELS[m] for m in METHODS if ranking[m] == 2),
            tie_note="Exact equal means share a dense rank; order within ties follows table order"))
    versus = {}
    for baseline in METHODS[:-1]:
        rr = [r for r in intervals if r["baseline"] == LABELS[baseline]]
        versus[LABELS[baseline]] = dict(positive_cells=sum(r["mean_difference_pp"] > 0 for r in rr),
            negative_cells=sum(r["mean_difference_pp"] < 0 for r in rr),
            tied_cells=sum(r["mean_difference_pp"] == 0 for r in rr),
            unadjusted_ci_above_zero=sum(r["ci95_low_pp"] > 0 for r in rr),
            unadjusted_ci_below_zero=sum(r["ci95_high_pp"] < 0 for r in rr))
    status = dict(verified_runs=360, global_rows=18000, cells=120, seeds=list(SEEDS),
        paired_seed_differences=len(differences), paired_ci_groups=len(intervals),
        manifest_sha256=sha(manifest_bytes), builder_sha256=sha(Path(__file__).read_bytes()),
        main_queue_sources=357, manifest_declared_priority_reuses=3, cpu_or_legacy_sources=0,
        endpoint="global_test_round49", sd="sample SD, ddof=1", rank="descending unrounded mean; dense ties",
        spl_rank_counts={str(i): sum(r["spl_rank"] == i for r in ranks) for i in range(1, 9)},
        spl_versus_baselines=versus,
        central=dict(verified_runs=15, cells=5, endpoint="final_test_block49", alpha="not_applicable",
            protocol=CENTRAL_PROTOCOL, successful_updates_per_run=40000, ranked=False,
            verification="Mirrored metadata hashes, exact budget and endpoints agree with prior server validation certificates"),
        pairing_verified="All 315 differences match dataset/alpha/model_seed/split_seed/partition_seed/encoder/partition",
        pairing_limit="Aggregate CSVs do not independently certify molecule IDs, full RNG trajectories or per-run hardware",
        cohort="Only manifest-listed original server formal-v3 main conditions; no mechanism, CPU or legacy pooling",
        adapted_baselines={"methods": ["FedProto", "FPL", "FedTGP"],
            "label": "Multi-task shared-model task-label prototype adaptations; not unmodified reference implementations",
            "owner_reported_limitations": {"FPL": "No domain clustering", "FedTGP": "No adaptive margin"}},
        manuscript_main_tex_modified=False, other_tables_modified=False, experiments_started=False)
    report = ["# Main Classification Table Audit", "", "2026-09-20 main cohort; 2026-09-24 approved centralized reference addition. No new experiments.", "",
              "- 360 unique approved conditions, 360 distinct original CSVs, 18,000 global rows covering rounds 0..49.",
              "- 120 cells, each with seeds 0/1/2. Sample SD (ddof=1), never best_val/best_test or best-seed selection.",
              "- 357 main-queue sources and the manifest's three SIDER alpha=0.5 SPL K=16 priority-suite reuses.",
              "- 315 seed-matched SPL-minus-baseline differences, 105 unadjusted paired t intervals (df=2).",
              "- CSV metadata pairing is verified; molecule-ID/RNG/hardware equivalence is not newly established by aggregate logs.",
              "- Rank highlighting uses unrounded federated means and is descriptive, not a significance test.",
              "- Central adds 15 completed CUDA MPNN runs as five final-test mean/SD cells, independent of alpha and excluded from ranking.",
              "- Central uses 50 blocks x 800 updates, matching the 50 x 4 x 200 federated update count, not optimization trajectories or wall time.",
              "- Central metadata/curve hashes and endpoints match the prior server validation certificates. This builder does not rerun checkpoint inference.",
              "- Daggered FedProto/FPL/FedTGP are this study's multi-task shared-model task-label prototype adaptations, not unmodified reference implementations. Per the owner's implementation audit, FPL lacks domain clustering and FedTGP lacks adaptive margin; full disclosure is handled in the main text by the main task.",
              "- No CPU, legacy, endpoint-mechanism or regression result is substituted for the federated cohort.", "",
              "## SPL versus Avg and All-Method Rank", "",
              "| Dataset | Alpha | SPL-Avg (pp) | Paired 95% CI (pp) | Positive seeds | SPL rank / 8 | Best |",
              "|---|---:|---:|---|---:|---:|---|"]
    for rank in ranks:
        pair = next(r for r in intervals if r["dataset"] == rank["dataset"] and r["alpha"] == rank["alpha"] and r["baseline"] == "Avg")
        report.append("| %s | %s | %+.4f | [%+.4f, %+.4f] | %d/3 | %d | %s |" % (
            rank["dataset"], rank["alpha"], pair["mean_difference_pp"], pair["ci95_low_pp"],
            pair["ci95_high_pp"], pair["positive_seeds"], rank["spl_rank"], rank["best_methods"]))
    report.extend(["", "## Descriptive Wins", "", "```json", json.dumps(versus, indent=2), "```", "",
                   "The 15 alpha/dataset cells are not 15 independent seed replicates. CI multiplicity is not corrected; retain all negative results.",
                   "", "## Rebuild", "", "```sh", "/opt/anaconda3/envs/MPP/bin/python -B scripts/reviewer_revision/build_main_classification_table_20260920.py", "```", "",
                   "Use --check for a read-only source/integrity/reproducibility check. Source paths in source_runs are repository-relative; manifest_csv_file preserves the original inventory entry.", ""])
    outputs = {TEX: latex_table(summary, central_summary),
        CENTRAL_OUT / "central_source_runs.csv": csv_text(central_sources),
        CENTRAL_OUT / "central_summary.csv": csv_text(central_summary),
        OUT / "main_classification_source_runs.csv": csv_text(sources),
        OUT / "main_classification_summary.csv": csv_text(summary),
        OUT / "main_classification_paired_differences.csv": csv_text(differences),
        OUT / "main_classification_paired_ci.csv": csv_text(intervals),
        OUT / "main_classification_spl_ranking.csv": csv_text(ranks),
        OUT / "MAIN_CLASSIFICATION_TABLE_AUDIT.md": "\n".join(report)}
    status["output_sha256"] = {str(p.relative_to(ROOT)): sha(text.encode()) for p, text in outputs.items()}
    outputs[OUT / "main_classification_build_status.json"] = json.dumps(status, indent=2) + "\n"
    return outputs, status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify existing outputs without writing")
    args = parser.parse_args()
    outputs, status = build()
    if args.check:
        for path, text in outputs.items():
            require(path.is_file() and path.read_bytes() == text.encode(), "Generated output mismatch: " + str(path))
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        for path, text in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(text.encode())
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
