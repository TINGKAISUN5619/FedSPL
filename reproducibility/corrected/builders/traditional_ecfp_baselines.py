#!/usr/bin/env python3
"""Pooled ECFP baselines under the manuscript's exact held-out splits."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import BernoulliNB


DATASETS = ("BBBP", "SIDER", "Tox21", "HIV", "ToxCast")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--methods", nargs="+", choices=["logreg", "nb", "rf"], default=["logreg", "nb"])
    parser.add_argument("--rf-trees", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--n-bits", type=int, default=2048)
    return parser.parse_args()


def load_dataset(code_root: Path, name: str):
    sys.path.insert(0, str(code_root))
    from dgllife.utils import (  # pylint: disable=import-outside-toplevel
        CanonicalAtomFeaturizer,
        CanonicalBondFeaturizer,
        smiles_to_bigraph,
    )
    from functools import partial
    from data import BBBP, HIV, SIDER, Tox21, ToxCast  # pylint: disable=import-outside-toplevel

    classes = {
        "BBBP": BBBP,
        "SIDER": SIDER,
        "Tox21": Tox21,
        "HIV": HIV,
        "ToxCast": ToxCast,
    }
    return classes[name](
        smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
        node_featurizer=CanonicalAtomFeaturizer(),
        edge_featurizer=CanonicalBondFeaturizer(self_loop=True),
        n_jobs=1,
        load=True,
    )


def fingerprints(smiles: list[str], n_bits: int) -> np.ndarray:
    result = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    for row, text in enumerate(smiles):
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
        result[row, list(fp.GetOnBits())] = 1
    return result


def split_indices(size: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    permutation = rng.permutation(np.arange(size))
    train_size = int(0.8 * size)
    train = permutation[:train_size]
    heldout = permutation[train_size:]
    generator = torch.Generator().manual_seed(seed + 104729)
    heldout_order = torch.randperm(len(heldout), generator=generator).numpy()
    heldout = heldout[heldout_order]
    val_size = int(0.5 * len(heldout))
    return train, heldout[:val_size], heldout[val_size:]


def make_model(method: str, seed: int, rf_trees: int, n_jobs: int):
    if method == "logreg":
        return LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            solver="liblinear",
            random_state=seed,
        )
    if method == "nb":
        return BernoulliNB(alpha=1.0)
    return RandomForestClassifier(
        n_estimators=rf_trees,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=n_jobs,
        random_state=seed,
    )


def evaluate_method(
    x: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    method: str,
    seed: int,
    rf_trees: int,
    n_jobs: int,
) -> tuple[float, list[tuple[int, int, float]]]:
    task_rows = []
    for task_idx in range(labels.shape[1]):
        train_observed = train_idx[masks[train_idx, task_idx] > 0]
        test_observed = test_idx[masks[test_idx, task_idx] > 0]
        y_train = labels[train_observed, task_idx].astype(np.int64)
        y_test = labels[test_observed, task_idx].astype(np.int64)
        if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
            continue
        model = make_model(method, seed + task_idx, rf_trees, n_jobs)
        model.fit(x[train_observed], y_train)
        probability = model.predict_proba(x[test_observed])[:, 1]
        task_rows.append((task_idx, len(test_observed), float(roc_auc_score(y_test, probability))))
    macro_auc = float(np.mean([row[2] for row in task_rows])) if task_rows else float("nan")
    return macro_auc, task_rows


def append_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "fingerprint_cache"
    cache_dir.mkdir(exist_ok=True)
    summary_path = args.output_dir / "traditional_ecfp_summary.csv"
    task_path = args.output_dir / "traditional_ecfp_per_task.csv"

    for dataset_name in args.datasets:
        dataset = load_dataset(args.code_root, dataset_name)
        labels = dataset.labels.detach().cpu().numpy()
        masks = dataset.mask.detach().cpu().numpy() if dataset.mask is not None else np.ones_like(labels)
        cache_path = cache_dir / f"{dataset_name}_morgan_r2_{args.n_bits}.npz"
        if cache_path.exists():
            x = np.load(cache_path)["x"]
        else:
            x = fingerprints(dataset.smiles, args.n_bits)
            np.savez_compressed(cache_path, x=x)

        for seed in args.seeds:
            train_idx, val_idx, test_idx = split_indices(len(dataset), seed)
            for method in args.methods:
                macro_auc, task_rows = evaluate_method(
                    x, labels, masks, train_idx, test_idx, method, seed,
                    args.rf_trees, args.n_jobs
                )
                append_rows(
                    summary_path,
                    ["dataset", "method", "seed", "split_seed", "train_n", "val_n", "test_n", "valid_tasks", "test_auc"],
                    [{
                        "dataset": dataset_name,
                        "method": f"ECFP-{method.upper()}",
                        "seed": seed,
                        "split_seed": seed,
                        "train_n": len(train_idx),
                        "val_n": len(val_idx),
                        "test_n": len(test_idx),
                        "valid_tasks": len(task_rows),
                        "test_auc": macro_auc,
                    }],
                )
                append_rows(
                    task_path,
                    ["dataset", "method", "seed", "task", "test_observed_n", "test_auc"],
                    [{
                        "dataset": dataset_name,
                        "method": f"ECFP-{method.upper()}",
                        "seed": seed,
                        "task": task_idx,
                        "test_observed_n": observed_n,
                        "test_auc": auc,
                    } for task_idx, observed_n, auc in task_rows],
                )
                print(
                    f"dataset={dataset_name} method={method} seed={seed} "
                    f"valid_tasks={len(task_rows)} test_auc={macro_auc:.6f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
