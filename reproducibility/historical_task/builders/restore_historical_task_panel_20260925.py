"""Retain the historical single-seed task diagnostic without obsolete table bars."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "paper_acs_latex/figures/mpp_diagnostics"
OUT = ROOT / "results/reviewer_revision_20260831/revision_approval_20260925/per_task_historical"
FIG = ROOT / "paper_acs_latex/figures/revision_20260925/tox21_historical_single_seed_tasks"
METHODS = ["FedAvg", "FedProto", "FedSPL"]
COLORS = {"FedAvg": "#334155", "FedProto": "#2f80ed", "FedSPL": "#d62728"}
MARKERS = {"FedAvg": "o", "FedProto": "^", "FedSPL": "s"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pred_path = SOURCE / "tox21_alpha0p5_heldout_predictions.csv"
    ci_path = SOURCE / "tox21_alpha0p5_task_auc_bootstrap_ci.csv"
    pred = pd.read_csv(pred_path)
    task = pd.read_csv(ci_path)
    shared = None
    summary = []
    cols = ["dataset_index", "smiles"] + [f"{field}_{i}" for i in range(12) for field in ("y", "mask")]
    for method in METHODS:
        data = pred[pred.method == method].sort_values("dataset_index").reset_index(drop=True)
        assert len(data) == data.dataset_index.nunique() == 1567
        if shared is None:
            shared = data[cols]
        else:
            pd.testing.assert_frame_equal(shared, data[cols])
        directory = ROOT / f"TEST/results/embedding_mechanism_tox21_alpha0p5_seed0_{method.lower()}"
        spec_path = directory / "run_summary.json"
        spec = json.loads(spec_path.read_text())
        assert (spec["dataset"], spec["alpha"], spec["seed"], spec["comm_round"],
                spec["clients_per_round"], spec["local_steps_per_round"]) == ("Tox21", .5, 0, 50, 4, 200)
        meta_path = directory / "embedding_metadata.csv"
        meta = pd.read_csv(meta_path, usecols=["split", "smiles"])
        train = set(meta.loc[meta["split"] == "train_client_split", "smiles"])
        assert len(train) == 6263
        assert not train.intersection(set(data.smiles))
        scores = []
        for i in range(12):
            observed = data[f"mask_{i}"] > .5
            value = roc_auc_score(data.loc[observed, f"y_{i}"], data.loc[observed, f"prob_{i}"])
            saved = task[(task.method == method) & (task.task_idx == i)]
            assert len(saved) == 1
            row = saved.iloc[0]
            assert abs(value - row.auc) < 1e-12
            assert 0 <= row.ci_low <= row.auc <= row.ci_high <= 1
            scores.append(value)
        summary.append(dict(method=method, seed=0, alpha=.5, n_heldout=1567,
                            mean_task_auc=float(np.mean(scores)),
                            task_sem=float(np.std(scores, ddof=1) / np.sqrt(12))))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.titlesize": 8.5, "axes.labelsize": 8,
                         "axes.linewidth": .7, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, (ax, bars) = plt.subplots(1, 2, figsize=(7.1, 3.05),
                                 gridspec_kw={"width_ratios": [4.25, 1.15], "wspace": .34})
    x = np.arange(12)
    for offset, method in zip([-.18, 0, .18], METHODS):
        data = task[task.method == method].sort_values("task_idx")
        err = np.vstack([data.auc - data.ci_low, data.ci_high - data.auc])
        ax.errorbar(x + offset, data.auc, yerr=err, fmt=MARKERS[method],
                    markersize=3.4, linewidth=.85, capsize=2, color=COLORS[method], label=method)
    names = task[task.method == "FedAvg"].sort_values("task_idx").task.tolist()
    ax.set_xticks(x, names, rotation=43, ha="right", fontsize=7)
    ax.set_ylim(.45, .95)
    ax.set_ylabel("Task ROC-AUC")
    ax.set_title("a  Task-level AUC (bootstrap 95% CI)", loc="left")
    ax.legend(frameon=False, ncol=3, loc="lower left", fontsize=7, handletextpad=.4, columnspacing=.9)
    rng = np.random.default_rng(20260925)
    for i, record in enumerate(summary):
        method = record["method"]
        bars.bar(i, record["mean_task_auc"], yerr=record["task_sem"], capsize=3,
                 color=COLORS[method], width=.58, alpha=.85)
        values = task[task.method == method].sort_values("task_idx").auc.to_numpy()
        bars.scatter(i + rng.normal(0, .032, len(values)), values, s=12,
                     facecolor="white", edgecolor=COLORS[method], linewidth=.65, zorder=3)
    bars.set_xticks(np.arange(3), METHODS, rotation=40, ha="right", fontsize=7)
    bars.set_ylim(.45, .95)
    bars.set_ylabel("Mean task AUC")
    bars.set_title("b  Mean across tasks", loc="left")
    for axis in (ax, bars):
        axis.grid(True, axis="y", color="#e5e7eb", linewidth=.65)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=.07, right=.99, bottom=.30, top=.79)
    fig.suptitle("Historical single-seed diagnostic: Tox21, alpha = 0.5, seed = 0", y=.98, fontsize=9)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(FIG.with_suffix("." + extension), dpi=450, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(summary).to_csv(OUT / "verified_summary.csv", index=False)
    pivot = task.pivot(index="task_idx", columns="method", values="auc")
    report = dict(
        cohort="historical July single-seed checkpoint diagnostic, not revised main-table cohort",
        figure_contract="Task-dependent differences in one historical run; not multi-seed superiority",
        backend="Python/matplotlib", archetype="quantitative grid",
        summary=summary,
        same_molecules_labels_masks=True, auc_recomputed=True,
        training_smiles_overlap_in_recorded_metadata=0,
        identity_audit_scope="literal SMILES equality in saved metadata; not full training replay",
        positive_task_differences_spl_vs_avg=int((pivot.FedSPL > pivot.FedAvg).sum()),
        task_intervals="Original stored percentile bootstrap intervals retained; 1000 resamples in source builder",
        macro_error="SD of 12 task AUCs divided by sqrt(12), not seed uncertainty or an independent-task inference",
        removed_panel="Historical hard-coded main-table bars, not the task-prediction cohort",
        new_training=False, raw_csv_modified=False,
    )
    (OUT / "task_analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"summary": summary, "spl_positive_tasks": report["positive_task_differences_spl_vs_avg"]}, indent=2))


if __name__ == "__main__":
    main()
