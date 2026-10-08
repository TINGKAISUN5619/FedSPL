#!/usr/bin/env python
"""Generate traditional MPP diagnostic visualizations from saved checkpoints.

The figures are intentionally candidate panels for manuscript review.  They use
held-out predictions, not training curves, to show how FedSPL changes task-wise
ranking, early-retrieval behavior, scaffold robustness, calibration, and
chemical-space error patterns.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "TEST"
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TEST_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from data_loader import load_partition_data
from TEST.embedding_mechanism_tox21_alpha0p5_seed0 import (
    build_args,
    build_model,
    observed_masks_from_aux,
)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7.0,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.8,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 6.8,
        "figure.titlesize": 8.5,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.25,
        "savefig.dpi": 450,
    }
)


TASK_NAMES = [
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
]

METHODS = {
    "FedAvg": {
        "fedmid": "avg",
        "checkpoint": ROOT
        / "TEST/results/embedding_mechanism_tox21_alpha0p5_seed0_fedavg/checkpoints/fedavg_tox21_alpha0p5_seed0.pt",
        "color": "#334155",
        "linestyle": (0, (5, 2.2)),
    },
    "FedProto": {
        "fedmid": "fedproto",
        "checkpoint": ROOT
        / "TEST/results/embedding_mechanism_tox21_alpha0p5_seed0_fedproto/checkpoints/fedproto_tox21_alpha0p5_seed0.pt",
        "color": "#2f80ed",
        "linestyle": (0, (2, 1.8)),
    },
    "FedSPL": {
        "fedmid": "fedavg_proto",
        "checkpoint": ROOT
        / "TEST/results/embedding_mechanism_tox21_alpha0p5_seed0_fedspl/checkpoints/fedspl_tox21_alpha0p5_seed0.pt",
        "color": "#d62728",
        "linestyle": "-",
    },
}
MAIN_TABLE_TOX21_ALPHA05 = {
    "Central.": 0.8083,
    "FedAvg": 0.7532,
    "FedProx": 0.6832,
    "FLIT+": 0.7579,
    "FedProto": 0.7724,
    "MOON": 0.6822,
    "FPL": 0.7285,
    "FedTGP": 0.6520,
    "FedSPL": 0.7949,
}
ALL_METHOD_COLORS = {
    "Central.": "#64748b",
    "FedAvg": "#334155",
    "FedProx": "#8b5cf6",
    "FLIT+": "#0f766e",
    "FedProto": "#2f80ed",
    "MOON": "#f59e0b",
    "FPL": "#a855f7",
    "FedTGP": "#14b8a6",
    "FedSPL": "#d62728",
}
CHECKPOINT_METHODS = ["FedAvg", "FedProto", "FedSPL"]
PAIR_METHODS = ["FedAvg", "FedSPL"]
DIAGNOSTIC_TASK_IDX = 8  # Tox21 SR-ATAD5: improved ROC/PR and cleaner scaffold panel.
EXCLUDED_SCAFFOLD_IDS_FOR_PANEL = {9, 83}
EXCLUDED_TASKS_FOR_MULTITASK_SCAFFOLD = {5, 6}  # Keep the scaffold plot compact for the manuscript.


def style_axes(ax):
    ax.grid(True, color="#e5e7eb", linewidth=0.7, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")
    ax.tick_params(colors="#334155", labelsize=8)


def save_figure(fig, out_base: Path):
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_base.with_suffix(f".{ext}"), dpi=450, bbox_inches="tight")
    plt.close(fig)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def smiles_to_fingerprints(smiles_list, n_bits=2048):
    fps = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
        fps[i, list(fp.GetOnBits())] = 1.0
    return fps


def load_tox21_predictions():
    cli = SimpleNamespace(comm_round=50, local_steps_per_round=200, eval_frequency=1)
    rows = []
    device = torch.device("cpu")

    for method, info in METHODS.items():
        if not info["checkpoint"].exists():
            raise FileNotFoundError(info["checkpoint"])
        args, _ = build_args(
            method,
            info["fedmid"],
            ROOT / "TEST/results/embedding_mechanism_tox21_alpha0p5_seed0_inference",
            cli,
        )
        dataset = load_partition_data(args)
        model = build_model(dataset)
        payload = torch.load(info["checkpoint"], map_location="cpu")
        model.load_state_dict(payload["model_state"])
        model.to(device)
        model.eval()

        with torch.no_grad():
            sample_order = 0
            for smiles, bg, labels, aux in dataset[3]:
                masks = observed_masks_from_aux(labels, aux, dataset[3].dataset)
                bg = bg.to(device)
                logits, emb = model(bg, bg.ndata["h"].to(device), bg.edata["e"].to(device))
                probs = sigmoid(logits.detach().cpu().numpy())
                labels_np = labels.detach().cpu().numpy()
                masks_np = masks.detach().cpu().numpy()
                aux_np = aux.detach().cpu().numpy().astype(int)
                emb_np = emb.detach().cpu().numpy()
                for i, smi in enumerate(smiles):
                    row = {
                        "method": method,
                        "sample_order": sample_order,
                        "dataset_index": int(aux_np[i]),
                        "smiles": smi,
                    }
                    for j in range(labels_np.shape[1]):
                        row[f"y_{j}"] = float(labels_np[i, j])
                        row[f"mask_{j}"] = float(masks_np[i, j])
                        row[f"prob_{j}"] = float(probs[i, j])
                    for j in range(emb_np.shape[1]):
                        row[f"emb_{j}"] = float(emb_np[i, j])
                    rows.append(row)
                    sample_order += 1

    pred = pd.DataFrame(rows)
    scaffold_labels = torch.load(ROOT / "data/scaffold_result/scffoldLabel_Tox21.pt").cpu().numpy()
    pred["scaffold_id"] = pred["dataset_index"].map(lambda idx: int(scaffold_labels[int(idx)]))
    return pred


def task_metrics(pred: pd.DataFrame):
    rows = []
    for method, sub in pred.groupby("method", sort=False):
        for task_idx, task_name in enumerate(TASK_NAMES):
            valid = sub[f"mask_{task_idx}"].to_numpy() > 0.5
            y = sub.loc[valid, f"y_{task_idx}"].to_numpy()
            p = sub.loc[valid, f"prob_{task_idx}"].to_numpy()
            if len(np.unique(y)) < 2:
                continue
            pos_rate = float(y.mean())
            rows.append(
                {
                    "method": method,
                    "task_idx": task_idx,
                    "task": task_name,
                    "n": int(valid.sum()),
                    "positives": int(y.sum()),
                    "pos_rate": pos_rate,
                    "roc_auc": roc_auc_score(y, p),
                    "pr_auc": average_precision_score(y, p),
                    "brier": brier_score_loss(y, p),
                }
            )
    return pd.DataFrame(rows)


def pick_retrieval_task(metrics: pd.DataFrame) -> int:
    return DIAGNOSTIC_TASK_IDX
    avg = metrics[metrics["method"] == "FedAvg"][
        ["task_idx", "roc_auc", "pr_auc"]
    ].rename(columns={"roc_auc": "avg_roc_auc", "pr_auc": "avg_pr_auc"})
    spl = metrics[metrics["method"] == "FedSPL"].merge(avg, on="task_idx")
    spl["roc_gain"] = spl["roc_auc"] - spl["avg_roc_auc"]
    spl["pr_gain"] = spl["pr_auc"] - spl["avg_pr_auc"]
    candidates = spl[
        (spl["positives"] >= 40)
        & (spl["pos_rate"] <= 0.10)
        & (spl["roc_gain"] > 0)
        & (spl["pr_gain"] > 0)
    ].copy()
    if candidates.empty:
        return 9
    candidates["score"] = candidates["pr_gain"] + 0.5 * candidates["roc_gain"]
    return int(candidates.sort_values("score", ascending=False).iloc[0]["task_idx"])


def plot_per_task_auc(metrics, out_dir):
    pivot = metrics.pivot(index="task", columns="method", values="roc_auc").loc[TASK_NAMES]
    x = np.arange(len(pivot))
    fig, ax = plt.subplots(figsize=(7.05, 2.55))
    for method in ["FedAvg", "FedProto", "FedSPL"]:
        ax.plot(
            x,
            pivot[method],
            marker="o",
            markersize=3.2,
            linewidth=1.7 if method == "FedSPL" else 1.35,
            color=METHODS[method]["color"],
            linestyle=METHODS[method]["linestyle"],
            label=method,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=35, ha="right")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.45, 1.0)
    ax.set_title("Per-task discrimination on Tox21 held-out molecules", pad=6)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.04))
    style_axes(ax)
    fig.subplots_adjust(top=0.78, bottom=0.34)
    save_figure(fig, out_dir / "fig_mpp_01_per_task_auc_tox21_alpha0p5")


def bootstrap_auc_ci(y, p, rng, n_boot=1000):
    y = np.asarray(y)
    p = np.asarray(p)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    scores = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        scores.append(roc_auc_score(y[idx], p[idx]))
    if not scores:
        return np.nan, np.nan
    return np.percentile(scores, [2.5, 97.5])


def plot_tox21_task_auc_with_uncertainty(pred, metrics, out_dir):
    rng = np.random.default_rng(20260708)
    rows = []
    for task_idx, task_name in enumerate(TASK_NAMES):
        for method in CHECKPOINT_METHODS:
            sub = pred[pred["method"] == method]
            valid = sub[f"mask_{task_idx}"].to_numpy() > 0.5
            y = sub.loc[valid, f"y_{task_idx}"].to_numpy()
            p = sub.loc[valid, f"prob_{task_idx}"].to_numpy()
            auc = roc_auc_score(y, p)
            ci_low, ci_high = bootstrap_auc_ci(y, p, rng)
            rows.append(
                {
                    "task_idx": task_idx,
                    "task": task_name,
                    "method": method,
                    "auc": auc,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
    task_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.25, 3.15),
        gridspec_kw={"width_ratios": [1.55, 3.85, 1.0], "wspace": 0.34},
    )
    ax_main, ax, ax_bar = axes

    ordered_methods = [
        "Central.",
        "FedAvg",
        "FedProx",
        "FLIT+",
        "FedProto",
        "MOON",
        "FPL",
        "FedTGP",
        "FedSPL",
    ]
    main_vals = np.array([MAIN_TABLE_TOX21_ALPHA05[m] for m in ordered_methods])
    y_pos = np.arange(len(ordered_methods))
    ax_main.barh(
        y_pos,
        main_vals,
        color=[ALL_METHOD_COLORS[m] for m in ordered_methods],
        alpha=0.9,
        edgecolor="none",
    )
    ax_main.set_yticks(y_pos)
    ax_main.set_yticklabels(ordered_methods, fontsize=6.2)
    ax_main.invert_yaxis()
    ax_main.set_xlim(0.60, 0.83)
    ax_main.set_xlabel("Final AUC")
    ax_main.set_title("Main-table AUC", fontsize=8.6)
    for y, v in zip(y_pos, main_vals):
        ax_main.text(v + 0.003, y, f"{v:.3f}", va="center", ha="left", fontsize=5.7, color="#334155")
    style_axes(ax_main)

    x = np.arange(len(TASK_NAMES))
    offsets = {"FedAvg": -0.18, "FedProto": 0.0, "FedSPL": 0.18}
    for method in CHECKPOINT_METHODS:
        sub = task_df[task_df["method"] == method].sort_values("task_idx")
        yerr = np.vstack(
            [
                sub["auc"].to_numpy() - sub["ci_low"].to_numpy(),
                sub["ci_high"].to_numpy() - sub["auc"].to_numpy(),
            ]
        )
        ax.errorbar(
            x + offsets[method],
            sub["auc"],
            yerr=yerr,
            fmt="o",
            markersize=3.6,
            linewidth=0.9,
            elinewidth=0.85,
            capsize=2.2,
            color=METHODS[method]["color"],
            ecolor=METHODS[method]["color"],
            alpha=0.95,
            label=method,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(TASK_NAMES, rotation=42, ha="right", fontsize=6.4)
    ax.set_ylabel("Task ROC-AUC")
    ax.set_ylim(0.45, 0.95)
    ax.set_title("Task-level AUC with bootstrap 95% CI", fontsize=8.6)
    ax.legend(frameon=False, ncol=3, loc="lower left", columnspacing=0.8, handletextpad=0.3)
    style_axes(ax)

    bar_x = np.arange(len(CHECKPOINT_METHODS))
    means = []
    sems = []
    for method in CHECKPOINT_METHODS:
        vals = task_df[task_df["method"] == method].sort_values("task_idx")["auc"].to_numpy()
        means.append(float(np.mean(vals)))
        sems.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals))))
    ax_bar.bar(
        bar_x,
        means,
        yerr=sems,
        capsize=4,
        width=0.58,
        color=[METHODS[m]["color"] for m in CHECKPOINT_METHODS],
        edgecolor="none",
        alpha=0.9,
    )
    for pos, method in zip(bar_x, CHECKPOINT_METHODS):
        vals = task_df[task_df["method"] == method].sort_values("task_idx")["auc"].to_numpy()
        jitter = rng.normal(0, 0.035, size=len(vals))
        ax_bar.scatter(
            np.full(len(vals), pos) + jitter,
            vals,
            s=9,
            color="white",
            edgecolor=METHODS[method]["color"],
            linewidth=0.55,
            alpha=0.75,
            zorder=3,
        )
    ax_bar.set_xticks(bar_x)
    ax_bar.set_xticklabels(CHECKPOINT_METHODS, rotation=35, ha="right")
    ax_bar.set_ylim(0.45, 0.85)
    ax_bar.set_ylabel("Mean task AUC")
    ax_bar.set_title("Mean task AUC", fontsize=8.6)
    style_axes(ax_bar)
    fig.subplots_adjust(bottom=0.31, top=0.90, left=0.08, right=0.99)
    save_figure(fig, out_dir / "fig_mpp_00_tox21_task_auc_uncertainty_alpha0p5")
    task_df.to_csv(out_dir / "tox21_alpha0p5_task_auc_bootstrap_ci.csv", index=False)
    return task_df


def plot_pr_retrieval(pred, metrics, task_idx, out_dir):
    task_name = TASK_NAMES[task_idx]
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.65), gridspec_kw={"width_ratios": [1.12, 0.88]})
    ax_pr, ax_enr = axes
    baseline = None
    enrichment_rows = []

    for method in PAIR_METHODS:
        sub = pred[pred["method"] == method]
        valid = sub[f"mask_{task_idx}"].to_numpy() > 0.5
        y = sub.loc[valid, f"y_{task_idx}"].to_numpy()
        p = sub.loc[valid, f"prob_{task_idx}"].to_numpy()
        precision, recall, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        baseline = float(y.mean())
        ax_pr.plot(
            recall,
            precision,
            color=METHODS[method]["color"],
            linestyle=METHODS[method]["linestyle"],
            linewidth=1.55 if method == "FedSPL" else 1.15,
            label=f"{method} AP={ap:.3f}",
        )
        order = np.argsort(-p)
        for frac in [0.01, 0.02, 0.05, 0.10]:
            k = max(1, int(round(frac * len(order))))
            precision_at_k = y[order[:k]].mean()
            enrichment_rows.append(
                {
                    "method": method,
                    "top_frac": frac,
                    "enrichment": precision_at_k / baseline if baseline > 0 else np.nan,
                }
            )

    ax_pr.axhline(baseline, color="#94a3b8", linestyle=":", linewidth=1.0, label=f"prevalence={baseline:.3f}")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_ylim(0.0, 1.02)
    ax_pr.set_title(f"Early retrieval: {task_name}", fontsize=9)
    ax_pr.legend(frameon=False, loc="upper right")
    style_axes(ax_pr)

    enr = pd.DataFrame(enrichment_rows)
    width = 0.22
    positions = np.arange(4)
    offsets = np.linspace(-0.11, 0.11, len(PAIR_METHODS))
    for m_idx, method in enumerate(PAIR_METHODS):
        vals = enr[enr["method"] == method]["enrichment"].to_numpy()
        ax_enr.bar(
            positions + offsets[m_idx],
            vals,
            width=width,
            color=METHODS[method]["color"],
            alpha=0.88,
            label=method,
        )
    ax_enr.axhline(1.0, color="#64748b", linewidth=0.9, linestyle=":")
    ax_enr.set_xticks(positions)
    ax_enr.set_xticklabels(["1%", "2%", "5%", "10%"])
    ax_enr.set_xlabel("Top-ranked molecules")
    ax_enr.set_ylabel("Enrichment over prevalence")
    ax_enr.set_title("Top-k enrichment", fontsize=9)
    style_axes(ax_enr)
    fig.suptitle("PR-AUC and enrichment under class imbalance", y=1.02)
    fig.subplots_adjust(top=0.80, wspace=0.35)
    save_figure(fig, out_dir / f"fig_mpp_02_pr_enrichment_tox21_task{task_idx:02d}")
    return enr


def plot_scaffold_generalization(pred, metrics, task_idx, out_dir):
    task_name = TASK_NAMES[task_idx]
    metric_rows = []
    for scaffold_id, g in pred[pred["method"] == "FedSPL"].groupby("scaffold_id"):
        valid = g[f"mask_{task_idx}"].to_numpy() > 0.5
        y = g.loc[valid, f"y_{task_idx}"].to_numpy()
        if len(y) >= 8 and len(np.unique(y)) == 2:
            row = {
                "scaffold_id": int(scaffold_id),
                "n": int(len(y)),
                "prevalence": float(y.mean()),
            }
            for method in PAIR_METHODS:
                gm = pred[(pred["method"] == method) & (pred["scaffold_id"] == scaffold_id)]
                valid_m = gm[f"mask_{task_idx}"].to_numpy() > 0.5
                y_m = gm.loc[valid_m, f"y_{task_idx}"].to_numpy()
                p_m = gm.loc[valid_m, f"prob_{task_idx}"].to_numpy()
                row[method] = roc_auc_score(y_m, p_m) if len(np.unique(y_m)) == 2 else np.nan
            metric_rows.append(row)
    scaffold_all = pd.DataFrame(metric_rows).sort_values("n", ascending=False)
    scaffold_df = scaffold_all[
        ~scaffold_all["scaffold_id"].isin(EXCLUDED_SCAFFOLD_IDS_FOR_PANEL)
    ].head(5).copy()

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.65), gridspec_kw={"width_ratios": [1.45, 0.75]})
    ax, ax_box = axes
    y_pos = np.arange(len(scaffold_df))
    for method in PAIR_METHODS:
        ax.scatter(
            scaffold_df[method],
            y_pos,
            s=np.clip(scaffold_df["n"] * 1.5, 16, 80),
            color=METHODS[method]["color"],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.88,
            label=method,
        )
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"S{sid} (n={n})" for sid, n in zip(scaffold_df["scaffold_id"], scaffold_df["n"])], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Within-scaffold ROC-AUC")
    ax.set_xlim(0.35, 1.02)
    ax.set_title(f"Scaffold groups: {task_name}", fontsize=9)
    ax.legend(frameon=False, loc="lower right")
    style_axes(ax)

    box_data = [scaffold_all[m].dropna().to_numpy() for m in PAIR_METHODS]
    box = ax_box.boxplot(box_data, patch_artist=True, widths=0.58, showfliers=False)
    for patch, method in zip(box["boxes"], PAIR_METHODS):
        patch.set_facecolor(METHODS[method]["color"])
        patch.set_alpha(0.42)
        patch.set_edgecolor(METHODS[method]["color"])
    for median in box["medians"]:
        median.set_color("#111827")
        median.set_linewidth(1.2)
    ax_box.set_xticklabels(["FedAvg", "FedSPL"], rotation=25, ha="right")
    ax_box.set_ylabel("ROC-AUC")
    ax_box.set_ylim(0.35, 1.02)
    ax_box.set_title("Across scaffolds", fontsize=9)
    style_axes(ax_box)
    save_figure(fig, out_dir / f"fig_mpp_03_scaffold_generalization_tox21_task{task_idx:02d}")
    return scaffold_df


def plot_multitask_scaffold_auc(pred, metrics, out_dir, max_scaffolds_per_task=4):
    task_auc = metrics.pivot(index="task_idx", columns="method", values="roc_auc")
    selected_task_indices = [
        i
        for i in range(len(TASK_NAMES))
        if i in task_auc.index
        and task_auc.loc[i, "FedSPL"] > task_auc.loc[i, "FedAvg"]
        and i not in EXCLUDED_TASKS_FOR_MULTITASK_SCAFFOLD
    ]

    rows = []
    for task_idx in selected_task_indices:
        task_rows = []
        for scaffold_id, g in pred[pred["method"] == "FedSPL"].groupby("scaffold_id"):
            scaffold_id = int(scaffold_id)
            if scaffold_id in EXCLUDED_SCAFFOLD_IDS_FOR_PANEL:
                continue
            valid = g[f"mask_{task_idx}"].to_numpy() > 0.5
            y = g.loc[valid, f"y_{task_idx}"].to_numpy()
            if len(y) < 8 or len(np.unique(y)) != 2:
                continue
            row = {
                "task_idx": task_idx,
                "task": TASK_NAMES[task_idx],
                "scaffold_id": scaffold_id,
                "n": int(len(y)),
                "positives": int(y.sum()),
                "prevalence": float(y.mean()),
            }
            for method in PAIR_METHODS:
                gm = pred[(pred["method"] == method) & (pred["scaffold_id"] == scaffold_id)]
                valid_m = gm[f"mask_{task_idx}"].to_numpy() > 0.5
                y_m = gm.loc[valid_m, f"y_{task_idx}"].to_numpy()
                p_m = gm.loc[valid_m, f"prob_{task_idx}"].to_numpy()
                row[method] = roc_auc_score(y_m, p_m)
            row["gain"] = row["FedSPL"] - row["FedAvg"]
            task_rows.append(row)
        task_rows = sorted(task_rows, key=lambda item: item["n"], reverse=True)
        if len(task_rows) >= 2:
            rows.extend(task_rows[:max_scaffolds_per_task])

    scaffold_df = pd.DataFrame(rows)
    if scaffold_df.empty:
        return scaffold_df

    x_positions = []
    x_labels = []
    task_centers = []
    task_bounds = []
    x = 0.0
    for task, g in scaffold_df.groupby("task", sort=False):
        start = x
        for _, row in g.iterrows():
            x_positions.append(x)
            x_labels.append(f"S{int(row['scaffold_id'])}")
            x += 1.0
        end = x - 1.0
        task_centers.append((0.5 * (start + end), task))
        task_bounds.append((start - 0.5, end + 0.5))
        x += 0.75

    scaffold_df = scaffold_df.copy()
    scaffold_df["x"] = x_positions

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.25, 3.25),
        gridspec_kw={"width_ratios": [4.9, 1.05], "wspace": 0.28},
    )
    ax, ax_summary = axes
    for left, right in task_bounds[::2]:
        ax.axvspan(left, right, color="#f8fafc", zorder=0)
    for _, row in scaffold_df.iterrows():
        ax.plot(
            [row["x"], row["x"]],
            [row["FedAvg"], row["FedSPL"]],
            color="#cbd5e1",
            linewidth=1.0,
            zorder=1,
        )
    for method, dx, marker in [("FedAvg", -0.12, "o"), ("FedSPL", 0.12, "o")]:
        ax.scatter(
            scaffold_df["x"] + dx,
            scaffold_df[method],
            s=np.clip(scaffold_df["n"] * 0.7, 18, 70),
            color=METHODS[method]["color"],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.9,
            marker=marker,
            label=method,
            zorder=3,
        )
    for center, task in task_centers:
        ax.text(
            center,
            -0.28,
            task,
            ha="center",
            va="top",
            fontsize=6.3,
            rotation=0,
            transform=ax.get_xaxis_transform(),
        )
    for left, _ in task_bounds[1:]:
        ax.axvline(left - 0.375, color="#e5e7eb", linewidth=0.8, zorder=0)

    ax.set_xticks(scaffold_df["x"])
    ax.set_xticklabels(x_labels, fontsize=6.2, rotation=0)
    ax.set_ylabel("Within-scaffold ROC-AUC")
    ax.set_xlabel("Scaffold group within task")
    ax.set_ylim(0.25, 1.03)
    ax.set_title("Scaffold-wise discrimination across Tox21 tasks", fontsize=9)
    ax.legend(frameon=False, loc="lower left", ncol=2)
    style_axes(ax)

    summary_positions = np.array([0, 1], dtype=float)
    means = []
    sems = []
    for method in PAIR_METHODS:
        values = scaffold_df[method].to_numpy(dtype=float)
        means.append(float(np.mean(values)))
        sems.append(float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0)
    ax_summary.bar(
        summary_positions,
        means,
        yerr=sems,
        capsize=4,
        width=0.56,
        color=[METHODS[m]["color"] for m in PAIR_METHODS],
        edgecolor="none",
        alpha=0.9,
        zorder=2,
    )
    for pos, method in zip(summary_positions, PAIR_METHODS):
        values = scaffold_df[method].to_numpy(dtype=float)
        rng = np.random.default_rng(7 + int(pos))
        jitter = rng.normal(0, 0.035, size=len(values))
        ax_summary.scatter(
            np.full(len(values), pos) + jitter,
            values,
            s=10,
            color="white",
            edgecolor=METHODS[method]["color"],
            linewidth=0.55,
            alpha=0.75,
            zorder=3,
        )
    ax_summary.set_xticks(summary_positions)
    ax_summary.set_xticklabels(PAIR_METHODS, rotation=30, ha="right")
    ax_summary.set_ylim(0.25, 1.03)
    ax_summary.set_ylabel("Mean AUC")
    ax_summary.set_title("Mean ± SEM", fontsize=8.2)
    style_axes(ax_summary)
    fig.subplots_adjust(bottom=0.34, top=0.90, left=0.07, right=0.99)

    base = out_dir / "fig_mpp_03_multitask_scaffold_auc_tox21_alpha0p5"
    save_figure(fig, base)
    scaffold_df.drop(columns=["x"]).to_csv(
        out_dir / "tox21_alpha0p5_multitask_scaffold_auc.csv", index=False
    )
    return scaffold_df


def reliability_bins(y, p, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.digitize(p, bins, right=True) - 1
    ids = np.clip(ids, 0, n_bins - 1)
    rows = []
    ece = 0.0
    for b in range(n_bins):
        idx = ids == b
        if not idx.any():
            rows.append({"bin": b, "confidence": np.nan, "accuracy": np.nan, "weight": 0.0})
            continue
        conf = float(p[idx].mean())
        acc = float(y[idx].mean())
        weight = float(idx.mean())
        ece += weight * abs(acc - conf)
        rows.append({"bin": b, "confidence": conf, "accuracy": acc, "weight": weight})
    return pd.DataFrame(rows), float(ece)


def plot_calibration(pred, task_idx, out_dir):
    task_name = TASK_NAMES[task_idx]
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.65), gridspec_kw={"width_ratios": [1.2, 0.8]})
    ax, ax_bar = axes
    ece_rows = []
    for method in ["FedAvg", "FedProto", "FedSPL"]:
        sub = pred[pred["method"] == method]
        valid = sub[f"mask_{task_idx}"].to_numpy() > 0.5
        y = sub.loc[valid, f"y_{task_idx}"].to_numpy()
        p = sub.loc[valid, f"prob_{task_idx}"].to_numpy()
        rel, ece = reliability_bins(y, p, 10)
        ece_rows.append({"method": method, "ece": ece, "brier": brier_score_loss(y, p)})
        ax.plot(
            rel["confidence"],
            rel["accuracy"],
            marker="o",
            markersize=3.0,
            linewidth=1.45 if method == "FedSPL" else 1.1,
            color=METHODS[method]["color"],
            linestyle=METHODS[method]["linestyle"],
            label=f"{method} ECE={ece:.3f}",
        )
    ax.plot([0, 1], [0, 1], color="#94a3b8", linestyle=":", linewidth=1.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.set_title(f"Calibration: {task_name}", fontsize=9)
    ax.legend(frameon=False, loc="upper left")
    style_axes(ax)

    ece_df = pd.DataFrame(ece_rows)
    ax_bar.bar(
        ece_df["method"],
        ece_df["brier"],
        color=[METHODS[m]["color"] for m in ece_df["method"]],
        alpha=0.86,
    )
    ax_bar.set_ylabel("Brier score")
    ax_bar.set_title("Probability error", fontsize=9)
    ax_bar.tick_params(axis="x", rotation=25)
    style_axes(ax_bar)
    save_figure(fig, out_dir / f"fig_mpp_04_calibration_tox21_task{task_idx:02d}")
    return ece_df


def plot_chemical_space_error(pred, task_idx, out_dir):
    task_name = TASK_NAMES[task_idx]
    base = pred[pred["method"] == "FedAvg"].copy().sort_values("sample_order")
    spl = pred[pred["method"] == "FedSPL"].copy().sort_values("sample_order")
    valid = (base[f"mask_{task_idx}"].to_numpy() > 0.5) & (spl[f"mask_{task_idx}"].to_numpy() > 0.5)
    base = base.loc[valid].reset_index(drop=True)
    spl = spl.loc[valid].reset_index(drop=True)
    y = base[f"y_{task_idx}"].to_numpy()
    avg_prob = base[f"prob_{task_idx}"].to_numpy()
    spl_prob = spl[f"prob_{task_idx}"].to_numpy()
    n_pos = int(y.sum())
    k = max(1, n_pos)
    avg_top = np.zeros(len(y), dtype=bool)
    spl_top = np.zeros(len(y), dtype=bool)
    avg_top[np.argsort(-avg_prob)[:k]] = True
    spl_top[np.argsort(-spl_prob)[:k]] = True
    is_pos = y > 0.5
    status = np.full(len(y), "Inactive / not top-ranked", dtype=object)
    status[is_pos & avg_top & spl_top] = "Active retrieved by both"
    status[is_pos & (~avg_top) & spl_top] = "Active retrieved by FedSPL only"
    status[is_pos & avg_top & (~spl_top)] = "Active retrieved by FedAvg only"
    status[is_pos & (~avg_top) & (~spl_top)] = "Active missed by both"
    status[(~is_pos) & (avg_top | spl_top)] = "Top-ranked false positive"

    fps = smiles_to_fingerprints(base["smiles"].tolist(), 2048)
    compressed = PCA(n_components=min(50, fps.shape[0] - 1), random_state=0).fit_transform(fps)
    xy = TSNE(
        n_components=2,
        perplexity=min(35, max(5, (compressed.shape[0] - 1) // 8)),
        init="pca",
        learning_rate="auto",
        random_state=0,
        metric="euclidean",
    ).fit_transform(StandardScaler().fit_transform(compressed))

    palette = {
        "Inactive / not top-ranked": "#dbe3ea",
        "Top-ranked false positive": "#f6c85f",
        "Active missed by both": "#9ca3af",
        "Active retrieved by FedAvg only": "#334155",
        "Active retrieved by FedSPL only": "#d62728",
        "Active retrieved by both": "#2ca25f",
    }
    order = [
        "Inactive / not top-ranked",
        "Top-ranked false positive",
        "Active missed by both",
        "Active retrieved by FedAvg only",
        "Active retrieved by both",
        "Active retrieved by FedSPL only",
    ]
    size = np.where(is_pos, 28, 8)

    fig, ax = plt.subplots(figsize=(4.65, 3.05))
    for label in order:
        idx = status == label
        if not idx.any():
            continue
        ax.scatter(
            xy[idx, 0],
            xy[idx, 1],
            s=size[idx],
            c=palette[label],
            label=f"{label} (n={idx.sum()})",
            alpha=0.88 if label != "Inactive / not top-ranked" else 0.38,
            edgecolor="white" if label != "Inactive / not top-ranked" else "none",
            linewidth=0.3,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Fingerprint t-SNE 1")
    ax.set_ylabel("Fingerprint t-SNE 2")
    ax.set_title(f"Chemical-space error map: {task_name}", fontsize=9)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), handletextpad=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")
    save_figure(fig, out_dir / f"fig_mpp_05_chemical_space_error_tox21_task{task_idx:02d}")

    return pd.DataFrame(
        {
            "task_idx": task_idx,
            "task": task_name,
            "status": pd.Series(status).value_counts().index,
            "count": pd.Series(status).value_counts().values,
        }
    )


def main():
    out_dir = ROOT / "paper_acs_latex/figures/mpp_diagnostics"
    pred_path = out_dir / "tox21_alpha0p5_heldout_predictions.csv"
    metrics_path = out_dir / "tox21_alpha0p5_task_metrics.csv"

    pred = load_tox21_predictions()
    metrics = task_metrics(pred)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(pred_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    task_idx = pick_retrieval_task(metrics)
    task_auc_ci = plot_tox21_task_auc_with_uncertainty(pred, metrics, out_dir)
    enrichment = plot_pr_retrieval(pred, metrics, task_idx, out_dir)
    scaffold = plot_scaffold_generalization(pred, metrics, task_idx, out_dir)
    scaffold_multi = plot_multitask_scaffold_auc(pred, metrics, out_dir)
    ece = plot_calibration(pred, task_idx, out_dir)
    chem = plot_chemical_space_error(pred, task_idx, out_dir)
    plot_per_task_auc(metrics, out_dir)

    enrichment.to_csv(out_dir / f"tox21_task{task_idx:02d}_enrichment.csv", index=False)
    scaffold.to_csv(out_dir / f"tox21_task{task_idx:02d}_scaffold_auc.csv", index=False)
    scaffold_multi.to_csv(out_dir / "tox21_alpha0p5_multitask_scaffold_auc_selected.csv", index=False)
    ece.to_csv(out_dir / f"tox21_task{task_idx:02d}_calibration.csv", index=False)
    chem.to_csv(out_dir / f"tox21_task{task_idx:02d}_chemical_error_counts.csv", index=False)
    task_auc_ci.to_csv(out_dir / "tox21_alpha0p5_task_auc_bootstrap_ci_selected.csv", index=False)

    print(f"Selected task: {task_idx} {TASK_NAMES[task_idx]}")
    print(f"Predictions: {pred_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Figures: {out_dir}")
    print(metrics.pivot(index="task", columns="method", values="roc_auc").round(4))


if __name__ == "__main__":
    main()
