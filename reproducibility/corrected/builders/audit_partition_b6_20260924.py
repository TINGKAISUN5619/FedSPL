"""Offline duplicate and size-preserving scaffold-allocation audit; no training."""
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from dgl.data.utils import load_labels
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/reviewer_revision_20260831'
OUT = BASE / 'revision_approval_20260924/partition_audit'
BUNDLE = BASE / 'deadline_queue_20260920/central_mpnn/bundle'
CODE = BASE / 'local_execution_20260917/strong_encoder/code_snapshot'
sys.path.insert(0, str(Path(__file__).parent / 'deadline_queue_20260920'))
from central_mpnn import digest, split_indices, SOURCE_HASHES, DATASETS

SCAFFOLD_HASHES = dict(
    BBBP='5e5546e06066ae062e6fac6ef1f501625a82b61d603d9b8140f459888472123d',
    SIDER='b68aa83de577cd939c4b6f00d7b7c74ee95cebc0fb551319c1af5269e77d7c1a',
    Tox21='23a873cc6678254cd0605b1aeb47c4bd66838aaf50bf0302663344d8d621fd7f',
    HIV='383beb500c23b8d831d2f8fa179f139d39a24ecc45c603743e4acbadf4908b91',
    ToxCast='e5e1005151b747bc95f1ed5d9cff71fa22bc999e0c9aa9d34dec7abc596f027e')


def scaffold_file(dataset):
    data_root = BASE / 'local_execution_20260917/datasets' if dataset == 'ToxCast' else ROOT
    path = data_root / 'data/scaffold_result' / ('scffoldLabel_' + dataset + '.pt')
    assert digest(path) == SCAFFOLD_HASHES[dataset], 'Scaffold must match server input'
    return path


def save_csv(name, rows):
    if rows:
        with (OUT / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def frozen_allocator():
    path = CODE / 'data_loader.py'
    assert digest(path) == SOURCE_HASHES['data_loader.py']
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'partition_data')
    rearrange = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'rearrangeLabel')
    # Execute the unchanged allocation tail with metadata-only train/heldout placeholders.
    start = next(i for i, n in enumerate(function.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'n_train' for t in n.targets))
    allocation = ast.FunctionDef(name='allocate', args=ast.arguments(posonlyargs=[],
        args=[ast.arg(arg=n) for n in ('train_dataset', 'val_dataset', 'y_train', 'partition', 'n_nets', 'alpha', 'args')],
        kwonlyargs=[], kw_defaults=[], defaults=[]), body=function.body[start:], decorator_list=[])
    module = ast.Module(body=[rearrange, allocation], type_ignores=[])
    namespace = dict(np=np, torch=torch, logging=logging)
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return namespace['allocate']


def weighted_js(labels, assignment, clients=4):
    labels = np.asarray(labels, dtype=np.int64)
    assignment = np.asarray(assignment, dtype=np.int64)
    assert labels.shape == assignment.shape and labels.size > 0
    width = int(labels.max()) + 1
    count = np.bincount(assignment * width + labels, minlength=clients * width).reshape(clients, width)
    expected = count.sum(1)[:, None] * count.sum(0)[None, :] / len(labels)
    present = count > 0
    return float((count[present] * np.log(count[present] / expected[present])).sum() / len(labels))


def duplicate_summary(keys, labels, masks, splits):
    groups = defaultdict(list)
    for i, key in enumerate(keys):
        if key is not None:
            groups[key].append(i)
    duplicate_groups = {k: ids for k, ids in groups.items() if len(ids) > 1}
    conflict_groups = 0
    for ids in duplicate_groups.values():
        values, valid = labels[ids], masks[ids] > 0
        if any(len(np.unique(values[valid[:, t], t])) > 1 for t in range(values.shape[1])):
            conflict_groups += 1
    result = dict(duplicate_groups=len(duplicate_groups),
                  duplicate_rows=sum(map(len, duplicate_groups.values())),
                  redundant_rows=sum(len(v) - 1 for v in duplicate_groups.values()),
                  observed_label_conflict_groups=conflict_groups)
    sets = {s: {keys[i] for i in ids if keys[i] is not None} for s, ids in splits.items()}
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        shared = sets[a] & sets[b]
        result[a + '_' + b + '_identities'] = len(shared)
        result[a + '_' + b + '_' + b + '_rows'] = sum(keys[i] in shared for i in splits[b])
    for name, ids in splits.items():
        counts = Counter(keys[i] for i in ids if keys[i] is not None)
        result[name + '_within_redundant_rows'] = sum(v - 1 for v in counts.values())
    return result, duplicate_groups


def main():
    assert not OUT.exists(), 'Preserve previous attempts; choose a new path after inspection'
    OUT.mkdir(parents=True)
    RDLogger.DisableLog('rdApp.warning')
    torch.set_num_threads(1)
    allocate = frozen_allocator()
    protocol = dict(datasets=list(DATASETS), seeds=[0, 1, 2], alphas=[.1, .5, 1.],
        clients=4, permutations=1000, canonicalization='RDKit canonical isomeric SMILES; retain all fragments, charges, isotopes and stereochemistry',
        normalization='No neutralization, tautomer or salt normalization; no deduplication',
        statistic='sample-weighted JS divergence, natural logs, stored allocation scaffold labels',
        null='random permutation of train scaffold labels over fixed client assignment slots',
        empty_scaffold='retained; audit stored-group correspondence separately',
        source_sha256=digest(CODE / 'data_loader.py'), no_training=True,
        no_predictions_or_test_scores_used=True)
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    summaries, members, null_rows, allocations, sources, mapping_rows = [], [], [], [], [], []
    for dataset in DATASETS:
        folder = BUNDLE / 'data' / dataset
        cache = folder / 'graphs.bin'
        public = next(folder.glob('public.csv*'))
        tensors = load_labels(str(cache))
        valid = tensors['valid_ids'].long().numpy()
        frame = pd.read_csv(public)
        smiles = frame['smiles'].iloc[valid].tolist()
        labels, masks = tensors['labels'].numpy(), tensors['mask'].numpy()
        tasks = ['p_np'] if dataset == 'BBBP' else [c for c in frame.columns if c not in ('smiles', 'mol_id', 'activity')]
        values = frame[tasks].values[valid]
        assert np.array_equal(np.nan_to_num(values).astype(np.float32), labels)
        assert np.array_equal((~np.isnan(values)).astype(np.float32), masks)
        assert len(smiles) == DATASETS[dataset][0] and labels.shape[1] == DATASETS[dataset][1]
        keys, scaffold_smiles, failures = [], [], []
        for index, value in enumerate(smiles):
            mol = Chem.MolFromSmiles(value)
            if mol is None:
                keys.append(None)
                scaffold_smiles.append(None)
                failures.append(index)
            else:
                keys.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
                scaffold_smiles.append(Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol), isomericSmiles=False))
        scaffold_path = scaffold_file(dataset)
        scaffold = torch.load(scaffold_path, map_location='cpu', weights_only=False).int()
        assert len(scaffold) == len(keys)
        forward, reverse = defaultdict(set), defaultdict(set)
        for stored, fresh in zip(scaffold.tolist(), scaffold_smiles):
            forward[stored].add(fresh)
            reverse[fresh].add(stored)
        mapping_rows.append(dict(dataset=dataset, stored_groups=len(forward), computed_murcko_groups=len(reverse),
            stored_groups_with_multiple_murcko=sum(len(v) > 1 for v in forward.values()),
            murcko_groups_with_multiple_stored=sum(len(v) > 1 for v in reverse.values()),
            empty_murcko_rows=sum(s == '' for s in scaffold_smiles), parse_failures=len(failures)))
        sources.append(dict(dataset=dataset, n=len(keys), cache=str(cache.relative_to(ROOT)),
            cache_sha256=digest(cache), public_sha256=digest(public), scaffold_sha256=digest(scaffold_path),
            parse_failure_indices=failures))
        for seed in range(3):
            splits = split_indices(len(keys), seed)
            saved = json.loads((folder / ('split%d.json' % seed)).read_text())
            assert saved['dataset'] == dataset and saved['seed'] == seed and splits == saved['ids']
            summary, duplicates = duplicate_summary(keys, labels, masks, splits)
            summaries.append(dict(dataset=dataset, seed=seed, n=len(keys), parse_failures=len(failures), **summary))
            membership = {i: split for split, ids in splits.items() for i in ids}
            for key, ids in duplicates.items():
                for i in ids:
                    members.append(dict(dataset=dataset, seed=seed, canonical_smiles=key,
                        dataset_index=i, public_csv_index=int(valid[i]), split=membership[i]))
            train = splits['train']
            stored_train = scaffold[train]
            _, encoded = np.unique(stored_train.numpy(), return_inverse=True)
            for alpha in (.1, .5, 1.):
                # Suppress original diagnostic prints, not its allocation behavior.
                from contextlib import redirect_stdout
                with redirect_stdout(io.StringIO()):
                    _, _, clients = allocate(train, splits['val'] + splits['test'], stored_train.clone(),
                        'hetero', 4, alpha, SimpleNamespace(partition_seed=seed, batch_size=64))
                assignment = np.full(len(train), -1, dtype=np.int64)
                for c, ids in clients.items():
                    assignment[ids] = c
                assert (assignment >= 0).all()
                record = dict(dataset=dataset, seed=seed, alpha=alpha,
                    client_indices={str(c): [train[i] for i in ids] for c, ids in clients.items()})
                (OUT / ('members_%s_a%s_s%d.json' % (dataset, alpha, seed))).write_text(json.dumps(record) + '\n')
                observed = weighted_js(encoded, assignment)
                rng_seed = int(hashlib.sha256(('%s|%d|%.1f|B6-fixed-v1' % (dataset, seed, alpha)).encode()).hexdigest()[:8], 16)
                rng = np.random.RandomState(rng_seed)
                null = np.array([weighted_js(rng.permutation(encoded), assignment) for _ in range(1000)])
                sizes = [len(clients[c]) for c in range(4)]
                allocations.append(dict(dataset=dataset, seed=seed, alpha=alpha, n_train=len(train),
                    client_sizes=';'.join(map(str, sizes)), scaffold_groups=len(np.unique(encoded)),
                    observed_js_nats=observed, null_mean=float(null.mean()), null_sample_sd=float(null.std(ddof=1)),
                    null_q025=float(np.quantile(null, .025)), null_q975=float(np.quantile(null, .975)),
                    observed_minus_null_mean=float(observed-null.mean()),
                    upper_tail_p=(1+int((null >= observed).sum()))/1001, rng_seed=rng_seed, permutations=1000))
                null_rows.extend(dict(dataset=dataset, seed=seed, alpha=alpha, permutation=i,
                                     js_nats=float(v)) for i, v in enumerate(null))
        print(dataset + ': duplicate and nine allocation audits complete', flush=True)
    save_csv('duplicates_summary.csv', summaries)
    save_csv('duplicate_members.csv', members)
    save_csv('scaffold_mapping.csv', mapping_rows)
    save_csv('allocation_summary.csv', allocations)
    save_csv('permutation_null.csv', null_rows)
    (OUT / 'sources.json').write_text(json.dumps(sources, indent=2) + '\n')
    (OUT / 'completed.json').write_text(json.dumps(dict(status='complete', split_audits=len(summaries),
        allocation_audits=len(allocations), permutations=len(null_rows), source_script_sha256=digest(__file__)), indent=2) + '\n')


if __name__ == '__main__':
    main()
