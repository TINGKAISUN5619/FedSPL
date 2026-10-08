"""Plot the approved, protocol-matched classification and support analyses."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REVISION = ROOT / "results/reviewer_revision_20260831"
STATUS = REVISION / "robustness_audit_20260910/summaries/expected_run_status.csv"
SUPPORT = REVISION / "cluster_support_review_20260916/k_summary.csv"
OUTPUT = REVISION / "revision_approved_20260920"
FIGURES = ROOT / "paper_acs_latex/figures/revision_20260920"
DATASETS = ("SIDER", "Tox21", "ToxCast")
ALPHAS = (0.1, 0.5, 1.0)
METHODS = (
    ("avg", "FedAvg", "#263241", "--"),
    ("fedprox", "FedProx", "#c18518", ":"),
    ("oursvatFLITPLUS", "FLIT+", "#8162a9", "-."),
    ("fedproto", "FedProto-adapted", "#327ca7", (0, (5, 2))),
    ("moon", "MOON", "#848b91", (0, (3, 1))),
    ("fpl", "FPL-inspired", "#45846e", (0, (1, 1.4))),
    ("fedtgp", "FedTGP-inspired", "#ac745b", (0, (5, 1.5, 1, 1.5))),
    ("fedavg_proto", "FedSPL", "#c73442", "-"),
)


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_figure(fig, stem):
    for suffix in ("pdf", "svg"):
        fig.savefig(FIGURES / f"{stem}.{suffix}", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def classification_curves():
    records = [row for row in read_rows(STATUS)
               if row["suite"] == "main_classification" and row["dataset"] in DATASETS]
    expected = {(d, a, method, seed) for d in DATASETS for a in ALPHAS
                for method, *_ in METHODS for seed in (0, 1, 2)}
    observed = set()
    curves = defaultdict(list)
    sources, long_rows = [], []
    for record in records:
        key = (record["dataset"], float(record["alpha"]), record["method"], int(record["seed"]))
        assert key not in observed and key in expected and record["status"] == "complete"
        observed.add(key)
        path = Path(record["csv_file"])
        rows = sorted((r for r in read_rows(path) if r.get("scope") == "global"),
                      key=lambda r: int(r["round"]))
        assert [int(row["round"]) for row in rows] == list(range(50)), path
        values = np.asarray([float(row["test"]) for row in rows])
        assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
        assert abs(values[-1] - float(record["final_test"])) < 1e-12
        curves[key[:3]].append(values)
        sources.append({"case_id": record["case_id"], "path": str(path), "sha256": digest(path)})
        for index, value in enumerate(values):
            long_rows.append(dict(dataset=key[0], alpha=key[1], method=key[2], seed=key[3],
                                  communication_round=index + 1, csv_round=index,
                                  server_test_auc=float(value), source=str(path)))
    assert observed == expected
    summary = []
    for key, values in sorted(curves.items()):
        assert len(values) == 3
        array = np.asarray(values)
        for index, (mean, sd) in enumerate(zip(array.mean(0), array.std(0, ddof=1))):
            summary.append(dict(dataset=key[0], alpha=key[1], method=key[2], round=index + 1,
                                n_seeds=3, mean_auc=float(mean), sample_sd=float(sd)))
    write_rows(OUTPUT / "convergence_seed_curves.csv", long_rows)
    write_rows(OUTPUT / "convergence_mean_sd.csv", summary)
    fig, axes = plt.subplots(3, 3, figsize=(9.3, 6.45))
    handles = []
    for r, dataset in enumerate(DATASETS):
        means = [np.asarray(v).mean(0) for key, v in curves.items() if key[0] == dataset]
        low = min(0.5, np.floor((min(v.min() for v in means) - .015) * 20) / 20)
        high = min(1., np.ceil((max(v.max() for v in means) + .015) * 20) / 20)
        for c, alpha in enumerate(ALPHAS):
            ax = axes[r, c]
            for method, label, color, style in METHODS:
                mean = np.asarray(curves[(dataset, alpha, method)]).mean(0)
                line, = ax.plot(range(1, 51), mean, label=label, color=color, linestyle=style,
                                linewidth=1.65 if method == "fedavg_proto" else 1.15)
                if r == c == 0:
                    handles.append(line)
            ax.set_title(f"{dataset}, $\\alpha={alpha:.1f}$", fontsize=10.2, pad=5)
            ax.set(xlim=(1, 50), ylim=(low, high), xticks=(1, 10, 20, 30, 40, 50))
            ax.set_xlabel("Communication round", labelpad=3)
            ax.set_ylabel("ROC-AUC", labelpad=3)
            ax.grid(color="#dedede", linewidth=.55, alpha=.8)
            ax.set_axisbelow(True)
    fig.legend(handles, [item[1] for item in METHODS], loc="lower center", ncol=4,
               bbox_to_anchor=(.52, .005), handlelength=2.5, columnspacing=1.1, fontsize=9.2)
    fig.subplots_adjust(left=.074, right=.985, bottom=.145, top=.947, hspace=.55, wspace=.30)
    save_figure(fig, "classification_convergence_three_seeds")
    return sources


def prototype_support():
    rows = sorted(read_rows(SUPPORT), key=lambda row: int(row["k"]))
    assert [int(row["k"]) for row in rows] == [8, 16, 32, 64, 128]
    assert all(row["seeds"] == "0;1;2" and int(row["n_seeds"]) == 3 for row in rows)
    x = np.arange(5)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7))
    mean = np.array([float(r["final_test_auc_mean"]) for r in rows])
    sd = np.array([float(r["final_test_auc_sd"]) for r in rows])
    axes[0].errorbar(x, mean, yerr=sd, color="#c73442", marker="o", markersize=4,
                     linewidth=1.4, capsize=3, elinewidth=1.05)
    axes[0].set_ylabel("Final ROC-AUC")
    axes[0].set_ylim(.52, .66)
    axes[0].set_title("a  Prediction performance", loc="left", fontweight="bold", fontsize=9)
    for key, label, color, style in (
        ("singleton", "1 molecule", "#577b91", "--"),
        ("lt3", "<3 molecules", "#b57a2b", "-"),
        ("lt5", "<5 molecules", "#587d63", ":"),
    ):
        values = 100 * np.array([float(r[f"local_{key}_fraction_final_mean"]) for r in rows])
        variation = 100 * np.array([float(r[f"local_{key}_fraction_final_sd"]) for r in rows])
        axes[1].errorbar(x, values, yerr=variation, label=label, color=color, linestyle=style,
                         linewidth=1.2, marker="o", markersize=3, capsize=2, elinewidth=.8)
    axes[1].set_ylim(0, 102)
    axes[1].set_ylabel("Local clusters (%)")
    axes[1].set_title("b  Molecular support", loc="left", fontweight="bold", fontsize=9)
    axes[1].legend(loc="upper left", fontsize=7.2, handlelength=2.2)
    for ax in axes:
        ax.set_xticks(x, [str(int(row["k"])) for row in rows])
        ax.set_xlabel("Requested prototypes, $P$")
        ax.grid(axis="y", color="#e2e2e2", linewidth=.55)
        ax.set_axisbelow(True)
    fig.subplots_adjust(left=.105, right=.99, bottom=.22, top=.86, wspace=.38)
    save_figure(fig, "sider_prototype_count_support")
    write_rows(OUTPUT / "prototype_count_plot_source.csv", rows)
    return {"path": str(SUPPORT), "sha256": digest(SUPPORT)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classification-only', action='store_true')
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 9, "axes.linewidth": .75, "legend.frameon": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    if args.classification_only:
        existing = json.loads((OUTPUT / 'figure_provenance.json').read_text())
        support = existing['support_source']
    else:
        support = prototype_support()
    provenance = {"convergence_sources": classification_curves(), "support_source": support,
                  "convergence_statistic": "Arithmetic mean across seeds 0/1/2; no smoothing; no error bands",
                  "support_statistic": "Mean and sample SD across three joint split/init/partition seeds",
                  "uncertainty": "Sample SD, not SE; no significance symbols",
                  "endpoints": "Round49 masked server-global held-out test ROC-AUC",
                  "protocol": "reviewer-v3 only; no legacy/CPU/client-mean fallback"}
    (OUTPUT / "figure_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("Verified 216 classification curves; support plot unchanged." if args.classification_only
          else "Verified 216 classification curves and five three-seed support settings.")
    print(FIGURES)


if __name__ == "__main__":
    main()
