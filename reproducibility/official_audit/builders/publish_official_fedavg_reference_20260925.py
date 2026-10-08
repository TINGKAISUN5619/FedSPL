#!/usr/bin/env python3
"""Render the approved, explicitly unmatched official-source reference cells.

Run after build_main_classification_table_20260920.py to restore this display.
Frozen scientific summaries and raw observations are never rewritten.
"""
import csv
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import statistics

import build_main_classification_table_20260920 as base

ROOT = base.ROOT
CAMPAIGN = base.REVISION / "official_fedavg_20260925"
OUT = base.REVISION / "revision_approval_20260925/official_reference"
PAPER = ROOT / "paper_acs_latex"
REPLAY = base.REVISION / "tox21_masked_curves_20260925/attempt_003"


def load_replay():
    state = json.loads((REPLAY / "queue_state.json").read_text())
    assert state["state"] == "complete" and state["completed"] == state["total"] == 9
    approval = json.loads((REPLAY / "approval.json").read_text())
    values, sources, hashes = {}, [], {}
    for case in approval["cases"]:
        assert case["dataset"] == "Tox21"
        folder = REPLAY / "cases" / case["case_id"]
        done = json.loads((folder / "completed.json").read_text())
        assert done["case"] == case and done["exit_code"] == 0 and done["formal"]
        assert done["approval_sha256"] == digest(REPLAY / "approval.json")
        for name, expected in done["output_sha256"].items():
            if name.endswith((".pt", ".npz")):
                continue
            assert digest(folder / name) == expected
            hashes[str((folder / name).relative_to(ROOT))] = expected
        curve = read_csv(folder / "masked_rounds.csv")
        raw = read_csv(folder / "rounds.csv")
        assert [int(r["round"]) for r in curve] == list(range(50))
        assert [int(r["round"]) for r in raw] == list(range(50))
        assert all(abs(float(x["unmasked_test_auc"]) - float(y["test_auc"])) < 1e-6
                   for x, y in zip(curve, raw))
        scores = json.loads((folder / "final_rescore.json").read_text())
        value = float(curve[-1]["masked_test_auc"])
        assert abs(value - scores["final_masked_test_auc"]) < 1e-12
        key = (case["dataset"], str(case["alpha"]), case["seed"])
        assert key not in values
        values[key] = value
        sources.append(dict(dataset=key[0], alpha=key[1], seed=key[2],
            cohort="official_FedChem_fixed_outer_split42_masked_curve_replay", final_masked_auc=value,
            final_unmasked_auc=scores["final_unmasked_rescore_test_auc"],
            source=str((folder / "final_rescore.json").relative_to(ROOT)),
            source_sha256=digest(folder / "final_rescore.json")))
    assert set(values) == {("Tox21", a, s) for a in base.ALPHAS for s in base.SEEDS}
    summaries = read_csv(REPLAY / "summary.csv")
    assert len(summaries) == 3
    for row in summaries:
        scores = [values["Tox21", row["alpha"], s] for s in base.SEEDS]
        assert abs(statistics.mean(scores) - float(row["masked_mean_auc"])) < 1e-12
        assert abs(statistics.stdev(scores) - float(row["masked_sd_auc"])) < 1e-12
    return values, sources, hashes, summaries


def read_csv(path):
    return list(csv.DictReader(io.StringIO(path.read_text())))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--tox21-replay", action="store_true",
                        help="Use the approved complete masked-curve replay for Tox21 only")
    args = parser.parse_args()
    if args.tox21_replay:
        OUT = base.REVISION / "revision_approval_20260925/official_reference_replay"
    state = json.loads((CAMPAIGN / "queue_state.json").read_text())
    assert state["state"] == "complete" and state["completed"] == state["total"] == 18
    approval = json.loads((CAMPAIGN / "approval.json").read_text())
    summaries = read_csv(CAMPAIGN / "summary.csv")
    assert len(summaries) == 6
    official, source_rows, hashes = {}, [], {}
    for case in approval["cases"]:
        folder = CAMPAIGN / "cases" / case["case_id"]
        done = json.loads((folder / "completed.json").read_text())
        assert done["case"] == case and done["exit_code"] == 0 and done["formal"]
        assert done["approval_sha256"] == digest(CAMPAIGN / "approval.json")
        for name, expected in done["output_sha256"].items():
            if name.endswith((".pt", ".npz")):
                continue  # These large outputs were verified on the server.
            assert digest(folder / name) == expected, (case["case_id"], name)
            hashes[str((folder / name).relative_to(ROOT))] = expected
        curve = read_csv(folder / "rounds.csv")
        assert [int(r["round"]) for r in curve] == list(range(50))
        scores = json.loads((folder / "final_rescore.json").read_text())
        value = scores["final_masked_test_auc"]
        assert math.isfinite(value) and 0 <= value <= 1
        d, a, s = case["dataset"], str(case["alpha"]), case["seed"]
        official[d, a, s] = value
        source_rows.append(dict(dataset=d, alpha=a, seed=s,
            cohort="official_FedChem_fixed_outer_split42", final_masked_auc=value,
            final_unmasked_auc=scores["final_unmasked_rescore_test_auc"],
            source=str((folder / "final_rescore.json").relative_to(ROOT)),
            source_sha256=digest(folder / "final_rescore.json")))
    expected = {(d, a, s) for d in ("Tox21", "HIV") for a in base.ALPHAS for s in base.SEEDS}
    assert set(official) == expected
    for row in summaries:
        vals = [official[row["dataset"], row["alpha"], s] for s in base.SEEDS]
        assert int(row["n"]) == 3
        assert abs(statistics.mean(vals) - float(row["masked_mean_auc"])) < 1e-12
        assert abs(statistics.stdev(vals) - float(row["masked_sd_auc"])) < 1e-12

    if args.tox21_replay:
        replay_values, replay_sources, replay_hashes, replay_summaries = load_replay()
        official.update(replay_values)
        source_rows = replay_sources + [r for r in source_rows if r["dataset"] != "Tox21"]
        hashes.update(replay_hashes)
        summaries = replay_summaries + [r for r in summaries if r["dataset"] != "Tox21"]

    revised = read_csv(base.OUT / "main_classification_summary.csv")
    revised_sources = read_csv(base.OUT / "main_classification_source_runs.csv")
    retained = [r for r in revised_sources if r["dataset"] in ("Tox21", "HIV") and r["method"] == "avg"]
    assert len(retained) == 18
    for row in retained:
        assert digest(ROOT / row["source"]) == row["source_sha256"]
    display = []
    for row in revised:
        row = dict(row)
        for field in ("mean_auc", "sample_sd_auc"):
            row[field] = float(row[field])
        row["rank"] = int(row["rank"])
        if row["dataset"] in ("Tox21", "HIV"):
            row["rank"] = 0
            if row["method"] == "avg":
                vals = [official[row["dataset"], row["alpha"], s] for s in base.SEEDS]
                row["mean_auc"], row["sample_sd_auc"] = statistics.mean(vals), statistics.stdev(vals)
        display.append(row)
    # Rank only the mutually matched revised methods, excluding official Avg.
    for dataset in ("Tox21", "HIV"):
        for alpha in base.ALPHAS:
            eligible = [r for r in display if r["dataset"] == dataset
                        and r["alpha"] == alpha and r["method"] != "avg"]
            ranks = base.dense_ranks({r["method"]: r["mean_auc"] for r in eligible})
            for row in eligible:
                row["rank"] = ranks[row["method"]]
    _, central = base.build_central()
    tex = base.latex_table(display, central)
    lines = tex.splitlines()
    dataset = None
    for i, line in enumerate(lines):
        if " & " not in line:
            continue
        cells = line.split(" & ")
        if cells[0] in base.DATASETS:
            dataset = cells[0]
        if dataset in ("Tox21", "HIV") and len(cells) == 11:
            cells[3] = cells[3].replace(r"\\[-0.1em]", r"$^{\dagger}$\\[-0.1em]", 1)
            lines[i] = " & ".join(cells)
    note = (r"\parbox{\linewidth}{\footnotesize\textit{Note:} Mean $\pm$ sample SD over three seeds; higher is better. "
        r"All entries use final server-model test AUC after 50 rounds, except Central (pooled-data MPNN, final block of 50, 800 updates per block). "
        r"$^{\dagger}$Official-source FedChem reference, with a fixed seed-42 outer split and historical supervision; its test molecules differ from the revised-protocol entries. "
        r"These six cells use observed-label (masked) rescoring, not the original unmasked metric. They are not matched controls for the other columns. "
        r"Supporting Information Tables~S9--S10 retain the revised FedAvg results and disclose both protocols and all seeds. "
        r"Bold and underline mark the highest and second-highest unrounded means among revised-protocol federated methods, not significance; dagger-marked references are excluded. "
        r"Central is excluded from rankings; $\alpha$ does not apply. Avg, Prox, and SPL denote FedAvg, FedProx, and FedSPL. "
        r"Proto-adapted, FPL-inspired, and TGP-inspired are the disclosed shared-model implementations (Table~S8).}")
    lines = [note if line.startswith(r"\parbox{\linewidth}") else line for line in lines]
    lines[0] = "% Generated by publish_official_fedavg_reference_20260925.py; raw summaries unchanged."
    OUT.mkdir(parents=True, exist_ok=True)
    backup = OUT / "before"
    backup.mkdir(exist_ok=True)
    for name in ("main_results_table_generated.tex", "main.tex"):
        if not (backup / name).exists():
            shutil.copy2(PAPER / name, backup / name)
    base.TEX.write_text("\n".join(lines) + "\n")
    (OUT / "official_sources.csv").write_text(base.csv_text(source_rows))
    (OUT / "retained_revised_sources.csv").write_text(base.csv_text(retained))

    lookup = {(r["dataset"], r["alpha"], r["method"]): r for r in revised}
    si = [r"\begin{table}[H]", r"\caption{FedAvg summaries from two distinct evaluation protocols.}",
        r"\label{tab:official_reference}", r"\centering\small",
        r"\begin{tabular}{llccc}\toprule",
        r"Dataset & $\alpha$ & Revised, masked & Official, masked & Official, unmasked \\\midrule"]
    for r in summaries:
        old = lookup[r["dataset"], r["alpha"], "avg"]
        def cell(m, sd):
            return "$%.4f \\pm %.4f$" % (float(m), float(sd))
        si.append(" & ".join([r["dataset"], r["alpha"], cell(old["mean_auc"], old["sample_sd_auc"]),
            cell(r["masked_mean_auc"], r["masked_sd_auc"]), cell(r["original_mean_auc"], r["original_sd_auc"])]) + r" \\")
    si += [r"\bottomrule\end{tabular}", r"\end{table}", r"\begin{table}[H]",
        r"\caption{Final masked FedAvg AUC for every reported seed.}", r"\label{tab:official_seeds}",
        r"\centering\small", r"\begin{tabular}{llrrrrrr}\toprule",
        r" & & \multicolumn{3}{c}{Revised protocol} & \multicolumn{3}{c}{Official-source protocol} \\",
        r"Dataset & $\alpha$ & Seed 0 & Seed 1 & Seed 2 & Seed 0 & Seed 1 & Seed 2 \\\midrule"]
    for r in summaries:
        old = lookup[r["dataset"], r["alpha"], "avg"]
        values = [float(old["seed%d_auc" % s]) for s in base.SEEDS]
        values += [official[r["dataset"], r["alpha"], s] for s in base.SEEDS]
        si.append(" & ".join([r["dataset"], r["alpha"]] + ["%.6f" % v for v in values]) + r" \\")
    si += [r"\bottomrule\end{tabular}", r"\end{table}"]
    (PAPER / "official_reference_tables_generated.tex").write_text("\n".join(si) + "\n")
    (OUT / "VERIFICATION.json").write_text(json.dumps(dict(completed=18, cells=6,
        tox21_masked_curve_replay=args.tox21_replay,
        verified_unique_runs=27 if args.tox21_replay else 18,
        metric="final masked test AUC, mean and sample SD", source_hashes=hashes,
        retained_revised_seeds=len(retained), training_changed=False,
        mixed_protocol_ranking=False, source_data_changed=False,
        table_sha256=digest(base.TEX)), indent=2) + "\n")
    print("Updated official reference display; Tox21 replay=" + str(args.tox21_replay)
          + "; retained all raw observations and revised controls.")


if __name__ == "__main__":
    main()
