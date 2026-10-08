#!/usr/bin/env python3
"""Client-local ECFP baselines under the fair scaffold partitions.

Each client fits an independent conventional classifier using only its local
training molecules. Evaluation uses the same held-out global test split as the
federated server model. The script reports both the mean standalone-client AUC
and a support-weighted probability ensemble of the available local models.
"""

from __future__ import annotations

import argparse
import csv
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
METHODS = ("logreg", "nb", "rf")
SUMMARY_FIELDS = [
    "dataset", "alpha", "method", "seed", "split_seed", "partition_seed",
    "clients", "train_n", "val_n", "test_n", "client_n_min", "client_n_max",
    "valid_client_task_models", "skipped_single_class_models",
    "valid_ensemble_tasks", "mean_local_test_auc", "ensemble_test_auc",
]
TASK_FIELDS = [
    "dataset", "alpha", "method", "seed", "split_seed", "partition_seed",
    "task", "client_idx", "train_observed_n", "test_observed_n", "test_auc",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fingerprint-cache-dir", type=Path)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.1, 0.5, 1.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--min-client-size", type=int, default=64)
    parser.add_argument("--rf-trees", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--n-bits", type=int, default=2048)
    return parser.parse_args()


def load_dataset(code_root: Path, name: str):
    sys.path.insert(0, str(code_root))
    from functools import partial

    from dgllife.utils import (  # pylint: disable=import-outside-toplevel
        CanonicalAtomFeaturizer,
        CanonicalBondFeaturizer,
        smiles_to_bigraph,
    )
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
    heldout = heldout[torch.randperm(len(heldout), generator=generator).numpy()]
    val_size = int(0.5 * len(heldout))
    return train, heldout[:val_size], heldout[val_size:]


def scaffold_partition(
    scaffold_labels: np.ndarray,
    train_idx: np.ndarray,
    alpha: float,
    seed: int,
    clients: int,
    min_client_size: int,
) -> list[np.ndarray]:
    train_scaffolds = scaffold_labels[train_idx].reshape(-1)
    _, remapped = np.unique(train_scaffolds, return_inverse=True)
    rng = np.random.RandomState(seed)
    n_train = len(train_idx)

    min_size = 0
    attempts = 0
    while min_size < min_client_size:
        local_positions = [[] for _ in range(clients)]
        for scaffold_id in range(int(remapped.max()) + 1):
            scaffold_positions = np.flatnonzero(remapped == scaffold_id)
            rng.shuffle(scaffold_positions)
            proportions = rng.dirichlet(np.repeat(alpha, clients))
            balanced = np.asarray([
                weight * (len(current) < n_train / clients)
                for weight, current in zip(proportions, local_positions)
            ])
            if balanced.sum() <= 0:
                balanced = proportions
            split_points = (np.cumsum(balanced / balanced.sum()) * len(scaffold_positions)).astype(int)[:-1]
            local_positions = [
                current + values.tolist()
                for current, values in zip(local_positions, np.split(scaffold_positions, split_points))
            ]
        min_size = min(map(len, local_positions))
        attempts += 1
        if attempts > 1000:
            raise RuntimeError(
                f"Unable to produce partition with min_client_size={min_client_size}; "
                f"observed minimum={min_size}"
            )

    partitions = []
    for positions in local_positions:
        values = np.asarray(positions, dtype=np.int64)
        rng.shuffle(values)
        partitions.append(train_idx[values])

    assigned = np.concatenate(partitions)
    if len(assigned) != len(train_idx) or not np.array_equal(np.sort(assigned), np.sort(train_idx)):
        raise RuntimeError("Client partition does not cover every training molecule exactly once")
    return partitions


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


def evaluate_decentralized(
    x: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
    client_indices: list[np.ndarray],
    test_idx: np.ndarray,
    method: str,
    seed: int,
    rf_trees: int,
    n_jobs: int,
) -> tuple[dict, list[dict]]:
    task_ensemble_aucs = []
    client_task_rows = []
    valid_models = 0
    skipped_single_class = 0

    for task_idx in range(labels.shape[1]):
        test_observed = test_idx[masks[test_idx, task_idx] > 0]
        y_test = labels[test_observed, task_idx].astype(np.int64)
        if len(test_observed) == 0 or np.unique(y_test).size < 2:
            continue

        probabilities = []
        supports = []
        for client_idx, local_idx in enumerate(client_indices):
            train_observed = local_idx[masks[local_idx, task_idx] > 0]
            y_train = labels[train_observed, task_idx].astype(np.int64)
            if len(train_observed) == 0 or np.unique(y_train).size < 2:
                skipped_single_class += 1
                continue
            model = make_model(
                method,
                seed + task_idx * len(client_indices) + client_idx,
                rf_trees,
                n_jobs,
            )
            model.fit(x[train_observed], y_train)
            probability = model.predict_proba(x[test_observed])[:, 1]
            auc = float(roc_auc_score(y_test, probability))
            probabilities.append(probability)
            supports.append(len(train_observed))
            valid_models += 1
            client_task_rows.append({
                "task": task_idx,
                "client_idx": client_idx,
                "train_observed_n": len(train_observed),
                "test_observed_n": len(test_observed),
                "test_auc": auc,
            })

        if probabilities:
            weights = np.asarray(supports, dtype=np.float64)
            weights /= weights.sum()
            ensemble_probability = np.average(np.stack(probabilities), axis=0, weights=weights)
            task_ensemble_aucs.append(float(roc_auc_score(y_test, ensemble_probability)))

    local_auc = float(np.mean([row["test_auc"] for row in client_task_rows])) if client_task_rows else float("nan")
    ensemble_auc = float(np.mean(task_ensemble_aucs)) if task_ensemble_aucs else float("nan")
    return {
        "valid_client_task_models": valid_models,
        "skipped_single_class_models": skipped_single_class,
        "valid_ensemble_tasks": len(task_ensemble_aucs),
        "mean_local_test_auc": local_auc,
        "ensemble_test_auc": ensemble_auc,
    }, client_task_rows


def write_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.fingerprint_cache_dir or (args.output_dir / "fingerprint_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "traditional_ecfp_decentralized_summary.csv"
    task_path = args.output_dir / "traditional_ecfp_decentralized_per_client_task.csv"
    summary_rows = read_rows(summary_path)
    task_rows = read_rows(task_path)
    completed = {
        (row["dataset"], str(float(row["alpha"])), row["method"], str(int(row["seed"])))
        for row in summary_rows
    }

    for dataset_name in args.datasets:
        dataset = load_dataset(args.code_root, dataset_name)
        labels = dataset.labels.detach().cpu().numpy()
        masks = dataset.mask.detach().cpu().numpy() if dataset.mask is not None else np.ones_like(labels)
        scaffold_path = args.code_root / "data" / "scaffold_result" / f"scffoldLabel_{dataset_name}.pt"
        scaffold_labels = torch.load(scaffold_path).detach().cpu().numpy()
        cache_path = cache_dir / f"{dataset_name}_morgan_r2_{args.n_bits}.npz"
        x = np.load(cache_path)["x"] if cache_path.exists() else fingerprints(dataset.smiles, args.n_bits)
        if not cache_path.exists():
            np.savez_compressed(cache_path, x=x)

        for alpha in args.alphas:
            for seed in args.seeds:
                train_idx, val_idx, test_idx = split_indices(len(dataset), seed)
                client_indices = scaffold_partition(
                    scaffold_labels,
                    train_idx,
                    alpha,
                    seed,
                    args.clients,
                    args.min_client_size,
                )
                for method in args.methods:
                    method_name = f"LOCAL-ECFP-{method.upper()}"
                    run_key = (dataset_name, str(float(alpha)), method_name, str(seed))
                    if run_key in completed:
                        print(f"skip complete: {run_key}", flush=True)
                        continue
                    summary, per_client_task = evaluate_decentralized(
                        x,
                        labels,
                        masks,
                        client_indices,
                        test_idx,
                        method,
                        seed,
                        args.rf_trees,
                        args.n_jobs,
                    )
                    common = {
                        "dataset": dataset_name,
                        "alpha": alpha,
                        "method": method_name,
                        "seed": seed,
                        "split_seed": seed,
                        "partition_seed": seed,
                    }
                    summary_rows.append({
                        **common,
                        "clients": args.clients,
                        "train_n": len(train_idx),
                        "val_n": len(val_idx),
                        "test_n": len(test_idx),
                        "client_n_min": min(map(len, client_indices)),
                        "client_n_max": max(map(len, client_indices)),
                        **summary,
                    })
                    task_rows.extend({**common, **row} for row in per_client_task)
                    completed.add(run_key)
                    write_rows(summary_path, SUMMARY_FIELDS, summary_rows)
                    write_rows(task_path, TASK_FIELDS, task_rows)
                    print(
                        f"dataset={dataset_name} alpha={alpha} method={method} seed={seed} "
                        f"local_auc={summary['mean_local_test_auc']:.6f} "
                        f"ensemble_auc={summary['ensemble_test_auc']:.6f}",
                        flush=True,
                    )

    write_rows(summary_path, SUMMARY_FIELDS, summary_rows)
    write_rows(task_path, TASK_FIELDS, task_rows)


if __name__ == "__main__":
    main()
