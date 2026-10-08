#!/usr/bin/env python3
"""Independent clients using the unmodified reviewer-v3 MPNN local trainer."""
import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import runpy
import statistics
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'deadline_queue_20260920'))
import central_mpnn as base

VERSION = 'mpnn_individual_reviewer_v3_20260922_v1'
DATASETS = ('BBBP', 'SIDER', 'Tox21', 'HIV', 'ToxCast')
ALPHAS = (0.1, 0.5, 1.0)


def cases():
    return [dict(case_id='%s_alpha%s_seed%d_individual' % (d, a, s),
                 dataset=d, alpha=a, seed=s)
            for d in DATASETS for a in ALPHAS for s in (0, 1, 2)]


def rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def append(path, records):
    if not records:
        return
    exists = path.exists()
    with path.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(records)


def sources(code):
    files = {code / p for p in ('main.py', 'client.py', 'fedavg_api.py',
                               'data_loader.py', 'utils.py', 'vat.py')}
    for directory in ('network', 'data', 'fedml_api', 'fedml_core'):
        files.update((code / directory).rglob('*.py'))
    actual = {str(p.relative_to(code)): base.digest(p) for p in sorted(files)}
    for name, expected in base.SOURCE_HASHES.items():
        base.require(actual.get(name) == expected, 'Not frozen reviewer-v3 source: ' + name)
    return actual


def inputs(code, datasets):
    return {name: dict(cache=base.digest(code / (name.lower() + '_dglgraph.bin')),
                       scaffold=base.digest(code / 'data/scaffold_result' / ('scffoldLabel_' + name + '.pt')))
            for name in datasets}


def clone(state):
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def state_hash(state):
    result = hashlib.sha256()
    for key, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        result.update(key.encode())
        result.update(str((array.dtype, array.shape)).encode())
        result.update(array.tobytes())
    return result.hexdigest()


@contextmanager
def preserve_rng():
    import numpy as np
    import torch
    saved = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
             torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    try:
        yield
    finally:
        random.setstate(saved[0])
        np.random.set_state(saved[1])
        torch.set_rng_state(saved[2])
        if saved[3] is not None:
            torch.cuda.set_rng_state_all(saved[3])


def subset_ids(dataset):
    import numpy as np
    if hasattr(dataset, 'dataset') and hasattr(dataset, 'indices'):
        parent = subset_ids(dataset.dataset)
        return parent[np.asarray(dataset.indices, dtype=np.int64)]
    return np.arange(len(dataset), dtype=np.int64)


def split_audit(api):
    import numpy as np
    seed = api.args.seed
    train, held = subset_ids(api.train_global.dataset), subset_ids(api.test_global.dataset)
    expected = np.random.RandomState(seed).permutation(len(train) + len(held))
    base.require(np.array_equal(train, expected[:len(train)]) and
                 np.array_equal(held, expected[len(train):]), 'Outer split differs from main cohort')
    groups = {str(c): subset_ids(api.train_data_local_dict[c].dataset).tolist() for c in range(4)}
    joined = sum(groups.values(), [])
    base.require(len(joined) == len(set(joined)) == len(train) and set(joined) == set(train),
                 'Local partition duplicates/misses training molecules')
    positions = positions_for(len(held), seed)
    record = dict(train=train.tolist(), val=held[positions['val']].tolist(),
                  test=held[positions['test']].tolist(), client_indices=groups)
    base.validate_splits({s: record[s] for s in ('train', 'val', 'test')}, len(expected))
    base.require(all(len(g) >= 64 for g in groups.values()), 'Client below training batch size')
    return record


def positions_for(n, seed):
    import torch
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed + 104729)).numpy()
    return dict(val=order[:n//2], test=order[n//2:])


def score(labels, masks, predictions):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    base.require(labels.shape == masks.shape == predictions.shape and labels.ndim == 2,
                 'Expected matching molecule/task matrices')
    base.require(np.isfinite(predictions).all(), 'Nonfinite predictions')
    values = []
    for t in range(labels.shape[1]):
        observed = masks[:, t] > 0
        y = labels[observed, t]
        base.require(np.isin(y, [0, 1]).all(), 'Invalid observed binary label')
        if np.unique(y).size == 2:
            values.append(roc_auc_score(y, predictions[observed, t]))
    base.require(bool(values), 'No eligible tasks')
    return float(np.mean(values)), len(values)


def predict(model, loader, device):
    import torch
    arrays = [[], [], [], []]
    model.to(device).eval()
    with torch.no_grad():
        for _, graph, labels, masks, ids in loader:
            graph = graph.to(device)
            logits, _ = model(graph, graph.ndata['h'], graph.edata['e'])
            for target, value in zip(arrays, (torch.sigmoid(logits).cpu().view_as(labels), labels, masks, ids)):
                target.append(value.cpu())
    model.cpu()
    return tuple(torch.cat(a).numpy() for a in arrays)


class ObservedLoader:
    """Observe the batch actually used by Adam, including any prefetched extra batch."""
    def __init__(self, loader):
        self.loader = loader
        self.last_ids = None

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            self.last_ids = batch[-1].detach().cpu().numpy().astype('<i8')
            yield batch


def train_client(client, state, block, cid, expected_steps):
    import torch
    loader = client.local_training_data
    observer = ObservedLoader(loader)
    client.local_training_data = observer
    trainer = client.model_trainer
    old_method, original_step = trainer.wandbconfig.fedmid, torch.optim.Adam.step
    count, batches = [0], hashlib.sha256()
    parameters = {id(p) for p in trainer.model.parameters()}

    def counted(optimizer, *args, **kwargs):
        base.require({id(p) for g in optimizer.param_groups for p in g['params']} == parameters,
                     'Unexpected second optimizer/model')
        base.require(all(g['lr'] == 1e-4 and g['weight_decay'] == 1e-5 for g in optimizer.param_groups),
                     'Optimizer settings changed')
        base.require(observer.last_ids is not None and len(observer.last_ids) == 64, 'Invalid batch')
        result = original_step(optimizer, *args, **kwargs)
        count[0] += 1
        batches.update(observer.last_ids.tobytes())
        return result

    trainer.wandbconfig.fedmid = 'avg'
    torch.optim.Adam.step = counted
    try:
        result = clone(client.train(state, block, cid))
    finally:
        torch.optim.Adam.step = original_step
        trainer.wandbconfig.fedmid = old_method
        client.local_training_data = loader
    base.require(count[0] == expected_steps, 'Incorrect number of successful local updates')
    base.require(all(torch.isfinite(v).all().item() for v in result.values()), 'Nonfinite model')
    return result, batches.hexdigest()


def independent_train(api, output, spec):
    import numpy as np
    import torch
    a = api.args
    base.require((a.encoder, a.client_num_in_total, a.client_num_per_round, a.batch_size,
                  a.comm_round, a.localStepsPerRound) == ('mpnn', 4, 4, 64, spec['blocks'], spec['steps']),
                 'Main protocol mismatch')
    base.require(str(api.device).startswith(spec['device']), 'Wrong training device')
    audit = split_audit(api)
    base.write_json(output / 'split_audit.json', audit)
    if spec['formal']:
        dataset = api.train_global.dataset
        while hasattr(dataset, 'dataset'):
            dataset = dataset.dataset
        bundle = Path(spec['central_bundle'])
        manifest = base.read_json(bundle / 'manifest.json')
        entry = manifest['datasets'][a.dataset]
        reference = base.load_cached(bundle / entry['cache'], bundle / entry['csv'], a.dataset)
        base.require(torch.equal(dataset.labels, reference.labels) and
                     torch.equal(dataset.mask, reference.mask) and dataset.smiles == reference.smiles,
                     'Loaded molecules/labels/masks differ from frozen main dataset')
        del reference
    initial = clone(api.model_trainer.get_model_params())
    states = {c: clone(initial) for c in range(4)}
    best = {c: -1.0 for c in range(4)}
    best_states, best_rounds = {}, {}
    started = time.monotonic()
    final_arrays = None
    for block in range(spec['blocks']):
        for cid in range(4):
            client = api.client_list[cid]
            before = state_hash(states[cid])
            states[cid], batch_hash = train_client(client, states[cid], block, cid, spec['steps'])
            append(output / 'training_trace.csv', [dict(round=block, client=cid,
                   optimizer_steps=spec['steps'], samples=client.get_sample_number(),
                   initial_sha256=before, final_sha256=state_hash(states[cid]), batch_sha256=batch_hash)])
        metrics, predictions = [], {}
        labels = masks = indices = None
        for cid in range(4):
            model = api.client_list[cid].model_trainer.model
            # One evaluation advances the loader RNG as in main-table FedAvg.
            # Additional observations do not change the next training random stream.
            if cid == 0:
                p, y, m, ids = predict(model, api.test_global, api.device)
            else:
                with preserve_rng():
                    p, y, m, ids = predict(model, api.test_global, api.device)
            if labels is not None:
                base.require(all(np.array_equal(x, z) for x, z in ((labels, y), (masks, m), (indices, ids))),
                             'Clients evaluated different molecules/labels/masks')
            labels, masks, indices = y, m, ids
            predictions['client_%d' % cid] = p
            positions = positions_for(len(ids), a.seed)
            for split, loc in positions.items():
                base.require(ids[loc].tolist() == audit[split], 'Prediction IDs differ from audited split')
                value, eligible = score(y[loc], m[loc], p[loc])
                metrics.append(dict(round=block, model_scope='client_%d' % cid, split=split,
                                    auc=value, valid_tasks=eligible, score_kind='probabilities'))
                if split == 'val' and value > best[cid]:
                    best[cid], best_states[cid], best_rounds[cid] = value, clone(states[cid]), block
            base.require(state_hash(model.state_dict()) == state_hash(states[cid]), 'Evaluation changed weights')
        for split in ('val', 'test'):
            part = [r for r in metrics if r['split'] == split]
            base.require(len({r['valid_tasks'] for r in part}) == 1, 'Eligibility differs by client')
            metrics.append(dict(round=block, model_scope='client_mean', split=split,
                                auc=float(np.mean([r['auc'] for r in part])), valid_tasks=part[0]['valid_tasks'],
                                score_kind='probabilities'))
        append(output / 'metric_curves.csv', metrics)
        print('[Individual block %d/%d] common-heldout mean test AUC=%.6f; %.1fs' %
              (block+1, spec['blocks'], metrics[-1]['auc'], time.monotonic()-started), flush=True)
        final_arrays = dict(labels=labels, masks=masks, dataset_indices=indices,
                            val_positions=positions['val'], test_positions=positions['test'], **predictions)
    torch.save(dict(final=states, validation_selected=best_states, best_rounds=best_rounds,
                    initial_sha256=state_hash(initial)), str(output / 'client_models.pt'))
    np.savez_compressed(str(output / 'final_predictions.npz'), **final_arrays)
    base.write_json(output / 'validation_selection.json', dict(rounds=best_rounds, validation_auc=best))
    base.write_json(output / 'completed.json', dict(protocol=VERSION, case=spec['case'],
                    formal=spec['formal'], blocks=spec['blocks'], steps=spec['steps'],
                    seconds=time.monotonic()-started, files={p: base.digest(output / p) for p in
                    ('spec.json', 'split_audit.json', 'metric_curves.csv', 'training_trace.csv',
                     'client_models.pt', 'final_predictions.npz', 'validation_selection.json')}))


def validate(output, deep=False):
    marker = base.read_json(output / 'completed.json')
    spec = base.read_json(output / 'spec.json')
    base.require(marker['protocol'] == spec['protocol'] == VERSION and marker['case'] == spec['case'], 'Wrong run')
    blocks, steps = marker['blocks'], marker['steps']
    base.require((blocks, steps) == ((50, 200) if marker['formal'] else (2, 2)), 'Wrong budget')
    base.require(spec['formal'] == marker['formal'] and spec['blocks'] == blocks and spec['steps'] == steps,
                 'Spec and completion budget differ')
    base.require(set(marker['files']) == {'spec.json', 'split_audit.json', 'metric_curves.csv',
                 'training_trace.csv', 'client_models.pt', 'final_predictions.npz', 'validation_selection.json'},
                 'Missing audited artifacts')
    base.require(spec['model_averaging'] is False and spec['prototype_exchange'] is False and
                 spec['public_distillation'] is False and spec['case'] in cases(), 'Wrong scientific condition')
    for name, digest in marker['files'].items():
        base.require(base.digest(output / name) == digest, 'Artifact changed: ' + name)
    curves, trace = rows(output / 'metric_curves.csv'), rows(output / 'training_trace.csv')
    keys = [(int(r['round']), r['model_scope'], r['split']) for r in curves]
    expected = {(b, c, s) for b in range(blocks) for c in
                ['client_%d' % c for c in range(4)] + ['client_mean'] for s in ('val', 'test')}
    base.require(len(keys) == len(expected) and set(keys) == expected, 'Missing/duplicate metric rows')
    base.require(all(math.isfinite(float(r['auc'])) and 0 <= float(r['auc']) <= 1
                     and int(r['valid_tasks']) > 0 and r['score_kind'] == 'probabilities' for r in curves),
                 'Invalid AUC or scoring policy')
    lookup = {(int(r['round']), r['model_scope'], r['split']): float(r['auc']) for r in curves}
    for b in range(blocks):
        for s in ('val', 'test'):
            mean = sum(lookup[b, 'client_%d' % c, s] for c in range(4))/4
            base.require(abs(mean-lookup[b, 'client_mean', s]) < 1e-12, 'Not an arithmetic client AUC mean')
    base.require(len(trace) == blocks*4 and {(int(r['round']), int(r['client'])) for r in trace}
                 == {(b, c) for b in range(blocks) for c in range(4)}, 'Invalid training trace')
    lookup_trace = {(int(r['round']), int(r['client'])): r for r in trace}
    base.require(len({lookup_trace[0, c]['initial_sha256'] for c in range(4)}) == 1, 'Unequal initialization')
    for (b, c), r in lookup_trace.items():
        base.require(int(r['optimizer_steps']) == steps, 'Update budget mismatch')
        if b:
            base.require(r['initial_sha256'] == lookup_trace[b-1, c]['final_sha256'], 'Client state not continued')
    if deep:
        import numpy as np
        import torch
        audit = base.read_json(output / 'split_audit.json')
        base.validate_splits({s: audit[s] for s in ('train', 'val', 'test')},
                             sum(len(audit[s]) for s in ('train', 'val', 'test')))
        with np.load(output / 'final_predictions.npz', allow_pickle=False) as data:
            for s in ('val', 'test'):
                pos = data[s + '_positions']
                base.require(data['dataset_indices'][pos].tolist() == audit[s], 'Saved prediction split mismatch')
                for c in range(4):
                    name = 'client_%d' % c
                    value, _ = score(data['labels'][pos], data['masks'][pos], data[name][pos])
                    base.require(abs(value-lookup[blocks-1, name, s]) < 1e-12, 'Saved prediction AUC mismatch')
        states = torch.load(str(output / 'client_models.pt'), map_location='cpu')
        base.require(set(states['final']) == set(range(4)), 'Missing independent model')
        for c in range(4):
            base.require(state_hash(states['final'][c]) == lookup_trace[blocks-1, c]['final_sha256'],
                         'Saved weights differ from final training trace')
            best_round = max(range(blocks), key=lambda b: lookup[b, 'client_%d' % c, 'val'])
            base.require(states['best_rounds'][c] == best_round and
                         state_hash(states['validation_selected'][c]) == lookup_trace[best_round, c]['final_sha256'],
                         'Validation-selected checkpoint not selected by validation')
    return dict(spec['case'], blocks=blocks, steps_per_client=steps, formal=marker['formal'],
                endpoint='client_mean_common_heldout_final',
                final_val=lookup[blocks-1, 'client_mean', 'val'],
                final_test=lookup[blocks-1, 'client_mean', 'test'], path=str(output))


def summaries(results):
    """Only completed, three-seed cells are eligible for the main-table reference."""
    answer = []
    for dataset in DATASETS:
        for alpha in ALPHAS:
            group = [r for r in results if r['dataset'] == dataset and r['alpha'] == alpha]
            base.require(len({r['seed'] for r in group}) == len(group), 'Duplicate seed')
            complete = len(group) == 3 and {r['seed'] for r in group} == {0, 1, 2}
            values = [r['final_test'] for r in group]
            answer.append(dict(dataset=dataset, alpha=alpha, method='Individual', n_seeds=len(group),
                          complete=complete, metric='common-heldout client-mean AUC',
                          mean_auc=statistics.mean(values) if complete else '',
                          sample_sd=statistics.stdev(values) if complete else '',
                          endpoint='final_block49', client_count=4,
                          ranking_group='noncollaborative_client_reference_not_server_model'))
    return answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--code-root', type=Path)
    parser.add_argument('--approval', type=Path)
    parser.add_argument('--dataset', choices=DATASETS, default='BBBP')
    parser.add_argument('--alpha', type=float, choices=ALPHAS, default=0.5)
    parser.add_argument('--seed', type=int, choices=(0, 1, 2), default=0)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--micro', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.check:
        print(json.dumps(validate(output, deep=True)))
        return
    code = args.code_root.resolve()
    source, data = sources(code), inputs(code, (args.dataset,))
    case = next(r for r in cases() if (r['dataset'], r['alpha'], r['seed']) ==
                (args.dataset, args.alpha, args.seed))
    approval = base.read_json(args.approval) if args.approval else None
    if approval:
        base.require(source == approval['source_sha256'] and data[args.dataset] == approval['inputs'][args.dataset]
                     and base.digest(__file__) == approval['worker_sha256'], 'Frozen input/source changed')
    base.require(args.micro or (approval is not None and args.device == 'cuda' and case in approval['cases']),
                 'Formal run requires approved CUDA case')
    base.runtime(1)
    import torch
    if args.device == 'cuda':
        base.require(torch.cuda.is_available(), 'CUDA required; no CPU fallback')
        if not args.micro:
            base.require_formal_runtime(base.versions(), torch.cuda.get_device_name(0))
    else:
        base.require(not torch.cuda.is_available(), 'CPU micro must hide CUDA')
    output.mkdir(parents=True, exist_ok=True)
    spec = dict(protocol=VERSION, case=case, formal=not args.micro, blocks=2 if args.micro else 50,
                steps=2 if args.micro else 200, device=args.device, source_sha256=source,
                inputs=data, worker_sha256=base.digest(__file__), runtime=base.versions(),
                central_bundle=approval['central_bundle'] if approval else None,
                endpoint='client_mean_common_heldout_final', initializer='untrained_shared_random',
                score_kind='sigmoid_probability_masked_task_macro_auc',
                model_averaging=False, prototype_exchange=False, public_distillation=False,
                training='unmodified reviewer-v3 Client.train/MyModelTrainer.train',
                batch_matching='same loader protocol; historical main runs lack paired batch traces')
    base.write_json(output / 'spec.json', spec)
    os.chdir(code)
    sys.path.insert(0, str(code))
    import fedavg_api

    def forbidden(*unused, **kwargs):
        raise RuntimeError('Individual must not aggregate models/prototypes or distill')

    for name in ('_aggregate', '_aggregate_topology_prototypes', '_distill_server', '_aggregate_public_predictions'):
        if hasattr(fedavg_api.FedAvgAPI, name):
            setattr(fedavg_api.FedAvgAPI, name, forbidden)
    fedavg_api.FedAvgAPI.train = lambda api: independent_train(api, output, spec)
    sys.argv = [str(code / 'main.py'), '-dataset', args.dataset, '-fedmid', 'individual',
                '-part_alpha', str(args.alpha), '-seed', str(args.seed), '--split_seed', str(args.seed),
                '--partition_seed', str(args.seed), '-comm_round', str(spec['blocks']), '-numClient', '4',
                '--clients_per_round', '4', '--local_steps_per_round', str(spec['steps']),
                '--encoder', 'mpnn', '--partition_method', 'hetero', '--num_workers', '0',
                '--results_dir', str(output / 'original_logger')]
    runpy.run_path(str(code / 'main.py'), run_name='__main__')
    print(json.dumps(validate(output, deep=True)), flush=True)


if __name__ == '__main__':
    main()
