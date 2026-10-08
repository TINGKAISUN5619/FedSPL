#!/usr/bin/env python3
"""Source-bound, offline centralized MPNN cases; resource scheduling is external."""

import argparse
import ast
import csv
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

sys.dont_write_bytecode = True
PROTOCOL = 'central_mpnn_reviewer_v3_50x800_v1'
_RUNTIME_THREADS = None
DATASETS = {'BBBP': (2039, 1), 'SIDER': (1427, 27), 'Tox21': (7831, 12),
            'HIV': (41127, 1), 'ToxCast': (8577, 617)}
TRAINER = 'fedml_api/standalone/fedavg/my_model_trainer_classification.py'
SOURCE_HASHES = {
    'network/model_factory.py': '6528755240ce390240f71e09b41f89fdc8d00e8a04164e2c2a4029672bf87f37',
    'network/myMPNNPredictor.py': 'cdba79f50a92bcc1cff499ee8ae06adc2415e71dbe8c344ca435f194c6e9b834',
    'data_loader.py': '83c8738c5f065f0e7e96aef992340e7d77724cca0dcfe0a29c770bf762591a43',
    TRAINER: '0b5fb42757a32ea48e6e99acd1723e0abde0e0384fca3e5e221f268f18c3ffde',
    'fedavg_api.py': '37df87f9e2097967ede690faa84a81169d25a538045e2bd26117bb3912d1d979',
    'data/csv_dataset.py': '8e92b90492618849c99b384e0362640ace3022c7d1005dfdbae4c736041403f7',
    'data/bbbp.py': '296af944dcac4ed28cab0d1e48b3b9468c34c856ec7266a79e8e913783b1e565',
    'data/sider.py': '4487a243ad3d944dd57aa6841349aa3cc166a5e3f7a5796d2b913483b672eff1',
    'data/tox21.py': '91a6e1f463e3dcfbd0674003e26db447b8cce469bb70aac48a565bd93702b210',
    'data/hiv.py': 'e073080678917f1c452ad3c7f6a9fe955c6cb4b861a5d08696fb082104944c6c',
    'data/toxcast.py': '1eef77c360421e4cb37dd03ce4a182587be048b00888a9b84dfae49e098dfdf1',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    with Path(path).open('x') as stream:
        json.dump(obj, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def json_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def runtime(threads):
    global _RUNTIME_THREADS
    require(1 <= threads <= 4, 'threads must be 1..4')
    if _RUNTIME_THREADS is not None:
        require(threads == _RUNTIME_THREADS, 'cannot change initialized runtime threads')
        return
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = str(threads)
    os.environ.setdefault('DGLBACKEND', 'pytorch')
    import torch
    torch.set_num_threads(threads)
    # PyTorch 1.8 can abort, rather than raise, if this setter is called twice.
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    _RUNTIME_THREADS = threads


def versions():
    import numpy, pandas, torch, dgl, sklearn
    return dict(python=sys.version, numpy=numpy.__version__, pandas=pandas.__version__,
                torch=torch.__version__, dgl=dgl.__version__, sklearn=sklearn.__version__,
                cuda=torch.version.cuda)


def require_formal_runtime(info, gpu_name):
    require(str(info['torch']).split('+')[0] == '1.8.1', 'formal runtime requires torch 1.8.1')
    require(info['cuda'] == '11.1', 'formal runtime requires CUDA 11.1')
    require('A100' in (gpu_name or '').upper(), 'formal runtime requires an A100 GPU')


def cases():
    return [dict(case_id='central_{}_seed{}'.format(d, s), dataset=d, seed=s,
                 blocks=50, updates_per_block=800, batch_size=64, alpha=0.5)
            for d in DATASETS for s in range(3)]


def split_indices(n, seed):
    import numpy as np
    import torch
    perm = np.random.RandomState(seed).permutation(n)
    train, heldout = perm[:int(.8 * n)], perm[int(.8 * n):]
    order = torch.randperm(len(heldout), generator=torch.Generator().manual_seed(seed + 104729))
    heldout = heldout[order.numpy()]
    cut = len(heldout) // 2
    return dict(train=train.tolist(), val=heldout[:cut].tolist(), test=heldout[cut:].tolist())


def validate_splits(ids, n):
    joined = sum([ids[s] for s in ('train', 'val', 'test')], [])
    require(len(joined) == n and set(joined) == set(range(n)), 'split overlap or missing IDs')


class CachedDataset:
    def __init__(self, graphs, labels, masks, valid_ids, smiles):
        self.graphs, self.labels, self.mask = graphs, labels, masks
        self.valid_ids, self.smiles = valid_ids, smiles
        self.n_tasks = labels.shape[1]

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        return self.smiles[index], self.graphs[index], self.labels[index], self.mask[index], index


def load_cached(cache, csv_path, dataset):
    """Read existing tensors only; verify their row/label alignment with the public CSV."""
    import dgl
    import numpy as np
    import pandas as pd
    import torch
    graphs, tensors = dgl.load_graphs(str(cache))
    frame = pd.read_csv(csv_path)
    if dataset == 'Tox21':
        frame = frame.drop(columns=['mol_id'])
    if dataset == 'HIV':
        frame = frame.drop(columns=['activity'])
    tasks = ['p_np'] if dataset == 'BBBP' else list(frame.columns.drop('smiles'))
    valid = tensors['valid_ids'].long().tolist()
    require(len(valid) == len(set(valid)) == len(graphs), 'invalid cached valid_ids')
    require(all(0 <= i < len(frame) for i in valid), 'cached row outside public CSV')
    values = frame[tasks].values
    labels = torch.from_numpy(np.nan_to_num(values).astype(np.float32))[valid]
    masks = torch.from_numpy((~np.isnan(values)).astype(np.float32))[valid]
    require(torch.equal(labels, tensors['labels']) and torch.equal(masks, tensors['mask']),
            'public CSV and cached labels/masks differ')
    require((len(graphs), len(tasks)) == DATASETS[dataset], 'unexpected dataset shape')
    require(all(g.ndata['h'].shape[1] == 74 and g.edata['e'].shape[1] == 13 for g in graphs),
            'not the canonical reviewer-v3 graph features')
    smiles = frame['smiles'].tolist()
    return CachedDataset(graphs, labels, masks, valid, [smiles[i] for i in valid])


def semantic_hash(data):
    h = hashlib.sha256()
    def add(value):
        value = value.detach().cpu().contiguous().numpy()
        h.update(json.dumps([str(value.dtype), list(value.shape)]).encode())
        h.update(value.tobytes())
    h.update(json.dumps(data.smiles, ensure_ascii=True).encode())
    h.update(json.dumps(data.valid_ids).encode())
    add(data.labels)
    add(data.mask)
    for graph in data.graphs:
        h.update(str(graph.num_nodes()).encode() + b':')
        u, v = graph.edges(order='eid')
        add(u)
        add(v)
        for kind, fields in (('node', graph.ndata), ('edge', graph.edata)):
            for key in sorted(fields):
                h.update((kind + ':' + key).encode())
                add(fields[key])
    return h.hexdigest()


def extracted_function(path, name, globals_dict, class_name=None):
    tree = ast.parse(Path(path).read_text())
    body = tree.body
    if class_name:
        body = next(n.body for n in body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), globals_dict)
    return globals_dict[name]


def source_apis(bundle):
    import dgl
    import torch
    source = (Path(bundle) / 'source').resolve()
    for name, module in list(sys.modules.items()):
        if name.startswith('network.') and getattr(module, '__file__', None):
            require(source in Path(module.__file__).resolve().parents, 'foreign network module loaded')
    sys.path.insert(0, str(source))
    factory = importlib.import_module('network.model_factory')
    require(source in Path(factory.__file__).resolve().parents, 'foreign model factory')
    collate = extracted_function(source / 'data_loader.py', 'collate_molgraphs',
                                 dict(dgl=dgl, torch=torch))
    masked_mean = extracted_function(source / TRAINER, '_masked_mean', {}, 'MyModelTrainer')
    return factory.build_molecular_model, collate, masked_mean


def prepare(bundle, source_root, data_map):
    bundle, source_root = Path(bundle), Path(source_root)
    require(not bundle.exists(), 'bundle exists; refuse overwrite')
    mapping = read_json(data_map)
    require(set(mapping) == set(DATASETS), 'data-map must contain exactly five datasets')
    for name, expected in SOURCE_HASHES.items():
        require(digest(source_root / name) == expected, 'not approved reviewer-v3 source: ' + name)
    for entry in mapping.values():
        require(set(entry) == {'cache', 'csv'}, 'data-map entry needs cache and csv')
        require(all(Path(p).is_file() for p in entry.values()), 'missing offline data file')
    bundle.mkdir(parents=True)
    for name in SOURCE_HASHES:
        target = bundle / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / name, target)
    provenance, metadata = {}, {}
    for dataset, entry in mapping.items():
        folder = bundle / 'data' / dataset
        folder.mkdir(parents=True)
        cache, public = folder / 'graphs.bin', folder / ('public' + ''.join(Path(entry['csv']).suffixes))
        shutil.copyfile(entry['cache'], cache)
        shutil.copyfile(entry['csv'], public)
        data = load_cached(cache, public, dataset)
        semantic = semantic_hash(data)
        metadata[dataset] = dict(cache=str(cache.relative_to(bundle)), csv=str(public.relative_to(bundle)),
                                 semantic_sha256=semantic, n=len(data), tasks=data.n_tasks)
        for seed in range(3):
            ids = split_indices(len(data), seed)
            validate_splits(ids, len(data))
            write_json(folder / ('split{}.json'.format(seed)), dict(
                dataset=dataset, seed=seed, ids=ids,
                raw_csv_ids={k: [data.valid_ids[i] for i in v] for k, v in ids.items()}))
        provenance[dataset] = entry
        del data
    write_json(bundle / 'provenance.json', dict(source_root=str(source_root.resolve()), inputs=provenance,
                                               prepare_runtime=versions()))
    manifest = dict(schema_version=1, protocol=PROTOCOL, cases=cases(), datasets=metadata,
                    worker_sha256=digest(__file__), files={
                        str(p.relative_to(bundle)): digest(p) for p in sorted(bundle.rglob('*')) if p.is_file()})
    write_json(bundle / 'manifest.json', manifest)
    write_json(bundle / 'bundle_lock.json', dict(manifest_sha256=digest(bundle / 'manifest.json'),
                                               protocol=PROTOCOL))
    return dict(status='prepared', cases=15, bundle=str(bundle.resolve()),
                manifest_sha256=digest(bundle / 'manifest.json'))


def verify_bundle(bundle):
    bundle = Path(bundle)
    manifest = read_json(bundle / 'manifest.json')
    require(digest(bundle / 'manifest.json') == read_json(bundle / 'bundle_lock.json')['manifest_sha256'],
            'manifest lock mismatch')
    require(manifest['schema_version'] == 1 and manifest['protocol'] == PROTOCOL, 'protocol mismatch')
    require(manifest['cases'] == cases(), 'case or fixed budget mismatch')
    require(manifest['worker_sha256'] == digest(__file__), 'worker changed after prepare')
    require(set(manifest['datasets']) == set(DATASETS), 'dataset set mismatch')
    actual = {str(p.relative_to(bundle)) for p in bundle.rglob('*') if p.is_file()}
    require(actual == set(manifest['files']) | {'manifest.json', 'bundle_lock.json'}, 'bundle file set changed')
    for name, expected in manifest['files'].items():
        path = (bundle / name).resolve()
        require(bundle.resolve() in path.parents, 'path escapes bundle')
        require(digest(path) == expected, 'bundle file hash mismatch: ' + name)
    for name, expected in SOURCE_HASHES.items():
        require(manifest['files']['source/' + name] == expected, 'reviewer-v3 source mismatch')
    for dataset, shape in DATASETS.items():
        meta = manifest['datasets'][dataset]
        require((meta['n'], meta['tasks']) == shape, 'dataset metadata shape mismatch')
        for name in (meta['cache'], meta['csv']):
            require(name in manifest['files'], 'unbound data path')
        for seed in range(3):
            path = 'data/{}/split{}.json'.format(dataset, seed)
            require(path in manifest['files'], 'missing frozen split')
            split = read_json(bundle / path)
            validate_splits(split['ids'], shape[0])
            require(split['dataset'] == dataset and split['seed'] == seed, 'split identity mismatch')
    return manifest


def auc(labels, probabilities, masks):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    scores = []
    for i in range(labels.shape[1]):
        observed = masks[:, i] > 0
        target = labels[observed, i]
        if len(target) and len(np.unique(target)) > 1:
            scores.append(float(roc_auc_score(target, probabilities[observed, i])))
    require(bool(scores), 'no eligible tasks for masked macro AUC')
    return float(np.mean(scores)), len(scores)


def predict(model, loader, device):
    import numpy as np
    import torch
    all_ids, all_labels, all_masks, all_prob = [], [], [], []
    model.eval()
    with torch.no_grad():
        for _, graph, labels, masks, ids in loader:
            graph = graph.to(device)
            logits, _ = model(graph, graph.ndata['h'], graph.edata['e'])
            require(tuple(logits.shape) == tuple(labels.shape), 'prediction shape mismatch')
            all_ids.extend(ids.tolist())
            all_labels.append(labels.numpy())
            all_masks.append(masks.numpy())
            all_prob.append(torch.sigmoid(logits).cpu().numpy())
    result = dict(ids=np.asarray(all_ids), labels=np.concatenate(all_labels),
                  masks=np.concatenate(all_masks), probabilities=np.concatenate(all_prob))
    require(all(np.isfinite(a).all() for a in result.values()), 'nonfinite predictions')
    return result


def train_block(model, loader, device, steps, masked_mean):
    import torch
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    require(not optimizer.state, 'optimizer must reset at block boundary')
    model.train()
    completed, loss_sum = 0, 0.0
    require(len(loader) > 0, 'no full training batch')
    while completed < steps:
        for _, graph, labels, masks, _ in loader:
            if completed == steps:
                break
            graph = graph.to(device)
            labels, masks = labels.to(device), masks.to(device)
            optimizer.zero_grad()
            logits, _ = model(graph, graph.ndata['h'], graph.edata['e'])
            loss = masked_mean(torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels, reduction='none'), masks)
            require(torch.isfinite(loss).item(), 'nonfinite training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            require(torch.isfinite(norm).item(), 'nonfinite gradient norm')
            optimizer.step()
            completed += 1
            loss_sum += float(loss.detach().cpu())
    adam_steps = [int(s['step']) for s in optimizer.state.values() if 'step' in s]
    require(adam_steps and set(adam_steps) == {steps}, 'Adam successful-update mismatch')
    return loss_sum / completed


def synthetic_data():
    import dgl
    import torch
    graphs = []
    for i in range(160):
        graph = dgl.graph(([0, 1, 0, 1], [1, 0, 0, 1]), num_nodes=2)
        graph.ndata['h'] = torch.full((2, 74), float(i % 7) / 7)
        graph.edata['e'] = torch.ones(4, 13)
        graphs.append(graph)
    labels = torch.tensor([[i % 2, (i // 2) % 2] for i in range(160)], dtype=torch.float32)
    masks = torch.ones_like(labels)
    masks[::3, 1] = 0
    return CachedDataset(graphs, labels, masks, list(range(160)), ['synthetic_{}'.format(i) for i in range(160)])


def run(bundle, case_id, output, device, threads=1, smoke=False, synthetic=False):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    manifest = verify_bundle(bundle)
    case = next((c for c in manifest['cases'] if c['case_id'] == case_id), None)
    require(case is not None, 'unknown case-id')
    output, bundle = Path(output), Path(bundle)
    require(not output.exists(), 'output exists; refuse overwrite (including incomplete runs)')
    dev = torch.device(device)
    require(dev.type in ('cuda', 'cpu'), 'only CPU smoke or explicit CUDA is supported')
    formal = not (smoke or synthetic)
    require(not formal or dev.type == 'cuda', 'formal CPU runs are forbidden')
    if dev.type == 'cuda':
        require(torch.cuda.is_available(), 'CUDA unavailable; no CPU fallback')
        torch.cuda.set_device(dev)
        require_formal_runtime(versions(), torch.cuda.get_device_name(dev))
    runtime(threads)
    random.seed(case['seed'])
    np.random.seed(case['seed'])
    torch.manual_seed(case['seed'])
    if dev.type == 'cuda':
        torch.cuda.manual_seed_all(case['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    build, collate, masked_mean = source_apis(bundle)
    if synthetic:
        data = synthetic_data()
        ids = split_indices(len(data), case['seed'])
    else:
        meta = manifest['datasets'][case['dataset']]
        data = load_cached(bundle / meta['cache'], bundle / meta['csv'], case['dataset'])
        require(semantic_hash(data) == meta['semantic_sha256'], 'loaded data semantic hash mismatch')
        ids = split_indices(len(data), case['seed'])
        frozen = read_json(bundle / 'data' / case['dataset'] / ('split{}.json'.format(case['seed'])))
        require(ids == frozen['ids'], 'runtime split differs from frozen IDs')
    validate_splits(ids, len(data))
    if smoke and not synthetic:
        ids = {k: v[:128] for k, v in ids.items()}
    blocks, steps = (2, 2) if not formal else (50, 800)
    output.mkdir(parents=True)
    spec = dict(protocol=PROTOCOL, case=case, formal=formal, synthetic=synthetic, device=str(dev),
                threads=threads, blocks=blocks, updates_per_block=steps, batch_size=64,
                manifest_sha256=digest(bundle / 'manifest.json'), worker_sha256=digest(__file__),
                runtime=versions(), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                gpu_name=torch.cuda.get_device_name(dev) if dev.type == 'cuda' else None,
                allocator_cap_enabled=False, test_policy='after_selection_and_checkpoint_freeze_only')
    write_json(output / 'spec.json', spec)
    write_json(output / 'split_ids.json', dict(ids=ids, raw_csv_ids={
        k: [data.valid_ids[i] for i in v] for k, v in ids.items()},
        smiles={k: [data.smiles[i] for i in v] for k, v in ids.items()},
        data_semantic_sha256=semantic_hash(data)))
    loaders = {k: DataLoader(Subset(data, v), batch_size=64, shuffle=(k == 'train'),
                             drop_last=(k == 'train'), num_workers=0, collate_fn=collate)
               for k, v in ids.items()}
    model = build(data, encoder='mpnn').to(dev)
    best_score, best_block, best_state, curves = -math.inf, None, None, []
    events = []
    def event(kind, **fields):
        events.append(dict(sequence=len(events), kind=kind, **fields))
    with (output / 'block_curve.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['block', 'successful_updates', 'cumulative_updates',
                              'train_loss', 'val_auc', 'val_tasks', 'optimizer_reset', 'elapsed_seconds'])
        writer.writeheader()
        for block in range(blocks):
            start = time.monotonic()
            loss = train_block(model, loaders['train'], dev, steps, masked_mean)
            event('train_block', block=block, successful_updates=steps)
            val = predict(model, loaders['val'], dev)
            score, tasks = auc(val['labels'], val['probabilities'], val['masks'])
            event('val_evaluation', block=block)
            row = dict(block=block, successful_updates=steps, cumulative_updates=(block + 1) * steps,
                       train_loss=loss, val_auc=score, val_tasks=tasks, optimizer_reset=True,
                       elapsed_seconds=time.monotonic() - start)
            writer.writerow(row)
            stream.flush()
            curves.append(row)
            if score > best_score:
                best_score, best_block = score, block
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(json.dumps(dict(status='running', case_id=case_id, **row)), flush=True)
    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    for name, state, block in [('final', final_state, blocks - 1), ('val_selected', best_state, best_block)]:
        torch.save(dict(state_dict=state, block=block, spec=spec), output / (name + '.pt'))
    checkpoint_hashes = {name: digest(output / (name + '.pt')) for name in ('final', 'val_selected')}
    write_json(output / 'selection.json', dict(best_block=best_block, best_val_auc=best_score,
               final_block=blocks - 1, rule='maximum validation AUC; earliest exact tie',
               checkpoint_sha256=checkpoint_hashes, test_access_count_before_selection=0))
    event('selection_frozen', best_block=best_block)
    endpoints = []
    for name, state, block in [('final', final_state, blocks - 1), ('val_selected', best_state, best_block)]:
        model.load_state_dict(state)
        for split in ('val', 'test'):
            result = predict(model, loaders[split], dev)
            event(split + '_endpoint', checkpoint=name)
            require(result['ids'].tolist() == ids[split], 'prediction IDs not in frozen order')
            result['raw_csv_ids'] = np.asarray([data.valid_ids[i] for i in ids[split]])
            score, tasks = auc(result['labels'], result['probabilities'], result['masks'])
            np.savez_compressed(output / (name + '_' + split + '_predictions.npz'), **result)
            endpoints.append(dict(checkpoint=name, block=block, split=split, auc=score, valid_tasks=tasks,
                                  checkpoint_sha256=checkpoint_hashes[name],
                                  val_reinference_minus_curve=score - curves[block]['val_auc']
                                  if split == 'val' else None))
    write_json(output / 'endpoints.json', endpoints)
    write_json(output / 'events.json', events)
    marker = dict(status='complete', protocol=PROTOCOL, case_id=case_id, formal=formal,
                  blocks=blocks, successful_updates=blocks * steps, manifest_sha256=spec['manifest_sha256'],
                  files={p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()})
    write_json(output / 'completed.json', marker)
    return check_case(bundle, manifest, case_id, output, allow_smoke=not formal)


def check_case(bundle, manifest, case_id, output, allow_smoke=False):
    import numpy as np
    output = Path(output)
    case = next((c for c in manifest['cases'] if c['case_id'] == case_id), None)
    require(case is not None, 'unknown case-id')
    if not output.exists():
        return dict(status='pending', case_id=case_id, formal=not allow_smoke)
    marker = read_json(output / 'completed.json')
    require(marker['status'] == 'complete' and marker['case_id'] == case_id, 'completion identity mismatch')
    require(marker['protocol'] == PROTOCOL, 'completion protocol mismatch')
    require(marker['formal'] == (not allow_smoke), 'smoke cannot satisfy formal completion')
    required = {'spec.json', 'split_ids.json', 'block_curve.csv', 'selection.json', 'final.pt',
                'val_selected.pt', 'events.json', 'endpoints.json'} | {
                    n + '_' + s + '_predictions.npz' for n in ('final', 'val_selected') for s in ('val', 'test')}
    require(set(marker['files']) == required, 'missing or unexpected completion artifacts')
    require({p.name for p in output.iterdir()} == required | {'completed.json'}, 'output file set mismatch')
    for name, expected in marker['files'].items():
        require(digest(output / name) == expected, 'output hash mismatch: ' + name)
    spec = read_json(output / 'spec.json')
    require(spec['case'] == case and spec['formal'] == marker['formal'], 'spec identity mismatch')
    require(spec['worker_sha256'] == manifest['worker_sha256'], 'spec worker mismatch')
    require(spec['manifest_sha256'] == marker['manifest_sha256'] == digest(Path(bundle) / 'manifest.json'),
            'run bound to a different manifest')
    require(allow_smoke or spec['device'].startswith('cuda'), 'formal hardware is not CUDA')
    if spec['device'].startswith('cuda'):
        require_formal_runtime(spec['runtime'], spec['gpu_name'])
    blocks, steps = (2, 2) if allow_smoke else (50, 800)
    require(spec['blocks'] == blocks and spec['updates_per_block'] == steps and spec['batch_size'] == 64,
            'budget spec mismatch')
    require(marker['blocks'] == blocks and marker['successful_updates'] == blocks * steps, 'budget marker mismatch')
    with (output / 'block_curve.csv').open() as stream:
        curve = list(csv.DictReader(stream))
    require([int(r['block']) for r in curve] == list(range(blocks)), 'incomplete or duplicate blocks')
    for i, row in enumerate(curve):
        require(int(row['successful_updates']) == steps and int(row['cumulative_updates']) == (i + 1) * steps
                and row['optimizer_reset'] == 'True', 'successful-update count/reset mismatch')
        require(math.isfinite(float(row['train_loss'])) and 0 <= float(row['val_auc']) <= 1,
                'nonfinite curve')
    selection = read_json(output / 'selection.json')
    best = max(range(blocks), key=lambda i: float(curve[i]['val_auc']))
    require(selection['best_block'] == best and selection['best_val_auc'] == float(curve[best]['val_auc'])
            and selection['final_block'] == blocks - 1 and selection['test_access_count_before_selection'] == 0,
            'validation selection mismatch')
    splits = read_json(output / 'split_ids.json')
    ids = splits['ids']
    if not spec['synthetic']:
        frozen = read_json(Path(bundle) / 'data' / case['dataset'] / ('split{}.json'.format(case['seed'])))
        for field in ('ids', 'raw_csv_ids'):
            expected = frozen[field] if not allow_smoke else {k: v[:128] for k, v in frozen[field].items()}
            require(splits[field] == expected, 'split IDs mismatch')
        require(splits['data_semantic_sha256'] == manifest['datasets'][case['dataset']]['semantic_sha256'],
                'semantic identity mismatch')
    else:
        require(allow_smoke and ids == split_indices(160, case['seed']), 'synthetic split mismatch')
    events = read_json(output / 'events.json')
    require([e['sequence'] for e in events] == list(range(len(events))), 'event sequence mismatch')
    require([e['block'] for e in events if e['kind'] == 'train_block'] == list(range(blocks))
            and all(e['successful_updates'] == steps for e in events if e['kind'] == 'train_block'),
            'training event budget mismatch')
    require([e['block'] for e in events if e['kind'] == 'val_evaluation'] == list(range(blocks)),
            'validation event coverage mismatch')
    frozen_at = next(i for i, e in enumerate(events) if e['kind'] == 'selection_frozen')
    tests = [i for i, e in enumerate(events) if e['kind'] == 'test_endpoint']
    require(len(tests) == 2 and all(i > frozen_at for i in tests), 'test evaluated before selection')
    endpoints = read_json(output / 'endpoints.json')
    require(len(endpoints) == 4 and {(e['checkpoint'], e['split']) for e in endpoints} ==
            {(n, s) for n in ('final', 'val_selected') for s in ('val', 'test')}, 'endpoint set mismatch')
    for endpoint in endpoints:
        name, split = endpoint['checkpoint'], endpoint['split']
        with np.load(output / (name + '_' + split + '_predictions.npz'), allow_pickle=False) as pred:
            require(pred['ids'].tolist() == ids[split] and pred['raw_csv_ids'].tolist() == splits['raw_csv_ids'][split],
                    'endpoint prediction IDs mismatch')
            require(all(np.isfinite(pred[k]).all() for k in pred.files), 'nonfinite endpoint prediction')
            value, tasks = auc(pred['labels'], pred['probabilities'], pred['masks'])
            require(abs(value - endpoint['auc']) < 1e-12 and tasks == endpoint['valid_tasks'], 'endpoint AUC mismatch')
        block = blocks - 1 if name == 'final' else best
        require(endpoint['block'] == block, 'endpoint block mismatch')
        require(endpoint['checkpoint_sha256'] == selection['checkpoint_sha256'][name] ==
                digest(output / (name + '.pt')), 'checkpoint selection identity mismatch')
        if split == 'val':
            delta = endpoint['auc'] - float(curve[block]['val_auc'])
            require(abs(delta - endpoint['val_reinference_minus_curve']) < 1e-12,
                    'val diagnostic incorrectly recorded')
    import torch
    for name, block in [('final', blocks - 1), ('val_selected', best)]:
        checkpoint = torch.load(output / (name + '.pt'), map_location='cpu')
        require(checkpoint['block'] == block and checkpoint['spec'] == spec, 'checkpoint identity mismatch')
        require(checkpoint['state_dict'] and all(torch.isfinite(t).all().item()
                for t in checkpoint['state_dict'].values()), 'nonfinite or empty checkpoint')
    return dict(status='complete', protocol=PROTOCOL, case_id=case_id, formal=marker['formal'], blocks=blocks,
                successful_updates=blocks * steps, endpoints=endpoints)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('prepare', 'check', 'run', 'smoke', 'synthetic-smoke'):
        parser.add_argument('--' + flag, action='store_true')
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--data-map', type=Path)
    parser.add_argument('--case-id', default='central_BBBP_seed0')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args(argv)
    try:
        runtime(args.threads)
        require(sum([args.prepare, args.check, args.run, (args.smoke or args.synthetic_smoke) and not args.check]) == 1,
                'choose prepare, check, run, smoke or synthetic-smoke (check may accompany smoke)')
        if args.prepare:
            require(args.source_root is not None and args.data_map is not None, 'prepare needs source-root and data-map')
            result = prepare(args.bundle, args.source_root, args.data_map)
        elif args.check:
            manifest = verify_bundle(args.bundle)
            result = (check_case(args.bundle, manifest, args.case_id, args.output,
                                 allow_smoke=args.smoke or args.synthetic_smoke) if args.output else
                      dict(status='prepared', cases=len(manifest['cases']), protocol=PROTOCOL,
                           manifest_sha256=digest(args.bundle / 'manifest.json')))
        else:
            require(args.output is not None, 'worker needs a new output directory')
            result = run(args.bundle, args.case_id, args.output, args.device, args.threads,
                         smoke=args.smoke, synthetic=args.synthetic_smoke)
        print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
        return 0
    except Exception as exc:
        print(json.dumps(dict(status='invalid', protocol=PROTOCOL, case_id=args.case_id,
                              formal=not (args.smoke or args.synthetic_smoke),
                              error=type(exc).__name__ + ': ' + str(exc))), flush=True)
        return 2


if __name__ == '__main__':
    sys.exit(main())
