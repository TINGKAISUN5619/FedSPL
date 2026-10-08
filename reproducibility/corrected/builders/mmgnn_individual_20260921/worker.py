#!/usr/bin/env python3
"""No-communication MMGNN control, using the frozen paired training/scoring code."""
import argparse
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import runpy
import statistics
import sys

ROOT = Path(__file__).resolve().parents[3]
FOLLOWUP = ROOT / 'scripts/reviewer_revision/mmgnn_followup_20260918'
sys.path.insert(0, str(FOLLOWUP))
import common

VERSION = 'mmgnn-individual-paired-v1-20260921'


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def mean_rows(records):
    answer = []
    for split in ('val', 'test'):
        for kind in ('logits', 'probabilities'):
            selected = [r for r in records if r['split'] == split and r['score_kind'] == kind]
            if len(selected) != 4 or {r['model_scope'] for r in selected} != {
                    'client_%d' % c for c in range(4)}:
                raise ValueError('Require exactly four individual clients')
            if len({(r['molecules'], r['valid_tasks']) for r in selected}) != 1:
                raise ValueError('Inconsistent shared test population')
            answer.append(dict(selected[0], model_scope='client_mean',
                               auc=statistics.mean(float(r['auc']) for r in selected)))
    return answer


def paired_trace_check(actual, reference, epoch):
    a = {int(r['client']): r for r in actual if int(r['round']) == epoch}
    b = {int(r['client']): r for r in reference if int(r['round']) == epoch}
    if set(a) != set(b) or set(a) != set(range(4)):
        raise ValueError('Missing paired batch trace')
    for c in a:
        for key in ('stream_seed', 'optimizer_steps', 'batch_sha256'):
            if str(a[c][key]) != str(b[c][key]):
                raise ValueError('Individual/FedAvg batch mismatch: ' + key)
        if epoch == 0 and a[c]['start_model_sha256'] != b[c]['start_model_sha256']:
            raise ValueError('Individual/FedAvg initialization mismatch')


def train(api, output, reference=None):
    import torch
    from endpoint_diagnostics import append_rows
    from paired_path_training import cloned_state, state_digest
    from raw_logit_eval import snapshot

    api.fedmid = 'individual'
    model = api.model_trainer.model
    initial = cloned_state(model.state_dict())
    states = {c: cloned_state(initial) for c in range(4)}
    best, best_states, endpoints = {}, {}, []
    ref_trace = rows(reference / 'paired_batch_trace.csv') if reference else None
    if reference and read(output / 'partition_audit.json') != read(reference / 'partition_audit.json'):
        raise ValueError('Individual/FedAvg partition mismatch')
    common.dump(output / 'initialization.json', dict(sha256=state_digest(initial),
                shared_random_initialization=True, pretrained=False, aggregation=False))

    for epoch in range(api.args.comm_round):
        round_rows, hashes = [], []
        for c in range(4):
            client = api.client_list[c]
            client.update_local_dataset(c, api.train_data_local_dict[c],
                api.test_data_local_dict[c], api.train_data_local_num_dict[c])
            # Restore this client's own prior weights despite the shared execution model.
            states[c] = cloned_state(client.train(cloned_state(states[c]), epoch, c))
            digest = state_digest(states[c])
            model.load_state_dict(states[c])
            records = snapshot(model, api, output, 'confirmation', epoch, 'client_%d' % c)
            round_rows.extend(records)
            value = next(r['auc'] for r in records if r['split'] == 'val' and r['score_kind'] == 'logits')
            if c not in best or value > best[c]['value']:
                best[c] = dict(value=value, epoch=epoch, rows=records)
                best_states[c] = cloned_state(states[c])
            if digest != state_digest(states[c]) or digest != state_digest(model.state_dict()):
                raise ValueError('Scoring mutated a local model')
            hashes.append(dict(round=epoch, model_scope='client_%d' % c, sha256=digest))
        if ref_trace:
            paired_trace_check(rows(output / 'paired_batch_trace.csv'), ref_trace, epoch)
        means = mean_rows(round_rows)
        append_rows(output / 'raw_metric_curves.csv', means)
        append_rows(output / 'round_state_hashes.csv', hashes)
        val = next(r['auc'] for r in means if r['split'] == 'val' and r['score_kind'] == 'logits')
        test = next(r['auc'] for r in means if r['split'] == 'test' and r['score_kind'] == 'logits')
        print('[individual round%d] client_mean val=%.9f test=%.9f; no model exchange' %
              (epoch, val, test), flush=True)
        if epoch == api.args.comm_round - 1:
            endpoints.extend(dict(r, checkpoint='final', selected_round=epoch) for r in round_rows + means)

    selected_rows = []
    for c in range(4):
        selected_rows.extend(dict(r, checkpoint='validation_selected_per_client',
                                  selected_round=best[c]['epoch']) for r in best[c]['rows'])
    endpoints.extend(selected_rows)
    endpoints.extend(dict(r, round=-1, checkpoint='validation_selected_per_client',
                          selected_round=-1) for r in mean_rows(selected_rows))
    append_rows(output / 'checkpoint_endpoints.csv', endpoints)
    torch.save(dict(final=states, validation_selected_per_client=best_states), output / 'individual_models.pt')
    common.dump(output / 'no_exchange_audit.json', dict(model_aggregation_calls=0,
                prototype_exchange_calls=0, independent_state_continuation=True,
                same_partition_as_fedavg=bool(reference), paired_batches_checked=bool(reference)))


def validate(output, expected=None):
    output = Path(output)
    marker, spec = read(output / 'completed.json'), read(output / 'spec.json')
    row = spec['case']
    if marker.get('protocol') != VERSION or not marker.get('completed') or marker['case'] != row:
        raise ValueError('Invalid completed identity')
    if expected and row != expected:
        raise ValueError('Case mismatch')
    if spec['protocol'] != VERSION or row['method'] != 'individual':
        raise ValueError('Invalid individual protocol')
    for name, digest in marker['artifact_sha256'].items():
        if common.sha(output / name) != digest:
            raise ValueError('Artifact hash mismatch: ' + name)
    metrics = rows(output / 'raw_metric_curves.csv')
    scopes = {'client_%d' % c for c in range(4)} | {'client_mean'}
    keys = {(int(r['round']), r['model_scope'], r['split'], r['score_kind']) for r in metrics}
    expected_keys = {(t, s, p, k) for t in range(row['rounds']) for s in scopes
                     for p in ('val', 'test') for k in ('logits', 'probabilities')}
    if keys != expected_keys or len(keys) != len(metrics):
        raise ValueError('Missing/duplicate individual metrics')
    if any(not math.isfinite(float(r['auc'])) or not 0 <= float(r['auc']) <= 1
           or r['method'] != 'individual' or int(r['valid_tasks']) <= 0 for r in metrics):
        raise ValueError('Invalid individual metric')
    traces = rows(output / 'paired_batch_trace.csv')
    if len(traces) != 4 * row['rounds'] or any(int(r['optimizer_steps']) != row['steps'] for r in traces):
        raise ValueError('Training budget mismatch')
    for c in range(4):
        subset = [r for r in traces if int(r['client']) == c]
        if [int(r['round']) for r in subset] != list(range(row['rounds'])):
            raise ValueError('Missing client updates')
    endpoints = rows(output / 'checkpoint_endpoints.csv')
    endpoint_keys = {(r['checkpoint'], r['model_scope'], r['split'], r['score_kind']) for r in endpoints}
    required_endpoints = {(e, s, p, k) for e in ('final', 'validation_selected_per_client')
                          for s in scopes for p in ('val', 'test') for k in ('logits', 'probabilities')}
    if endpoint_keys != required_endpoints or len(endpoints) != 40:
        raise ValueError('Missing individual endpoints')
    indexed = {(int(r['round']), r['model_scope'], r['split'], r['score_kind']): r for r in metrics}
    for t in range(row['rounds']):
        originals = [r for r in metrics if int(r['round']) == t and r['model_scope'] != 'client_mean']
        for r in mean_rows(originals):
            actual = indexed[(t, 'client_mean', r['split'], r['score_kind'])]
            if abs(float(actual['auc']) - float(r['auc'])) > 1e-12:
                raise ValueError('Incorrect client mean')
        for c in range(4):
            if not (output / 'predictions' / ('round_%03d_client_%d.npz' % (t, c))).is_file():
                raise ValueError('Missing heldout prediction snapshot')
    for r in endpoints:
        if r['checkpoint'] == 'validation_selected_per_client' and r['model_scope'] == 'client_mean':
            continue
        t = int(r['selected_round'])
        if r['checkpoint'] == 'final' and t != row['rounds']-1:
            raise ValueError('Final must use the last fixed budget block')
        if r['checkpoint'] == 'validation_selected_per_client':
            scope = r['model_scope']
            best = max(range(row['rounds']), key=lambda j: float(indexed[(j, scope, 'val', 'logits')]['auc']))
            if t != best:
                raise ValueError('Invalid validation-only checkpoint selection')
        if abs(float(r['auc']) - float(indexed[(t, r['model_scope'], r['split'], r['score_kind'])]['auc'])) > 1e-12:
            raise ValueError('Endpoint does not match the recorded round')
    final = [r for r in endpoints if r['checkpoint'] == 'final' and r['model_scope'] == 'client_mean'
             and r['score_kind'] == 'logits']
    if len(final) != 2:
        raise ValueError('Missing final client mean')
    if not spec['micro']:
        reference = Path(spec['reference_avg'])
        ref_spec = read(reference / 'spec.json')
        if (ref_spec['source_sha256'] != spec['source_sha256'] or ref_spec['inputs_sha256'] != spec['inputs_sha256']
                or ref_spec['case']['seed'] != row['seed'] or ref_spec['case']['dataset'] != row['dataset']
                or ref_spec['case']['peak_lr'] != row['peak_lr']):
            raise ValueError('FedAvg reference identity mismatch')
        if read(reference / 'partition_audit.json') != read(output / 'partition_audit.json'):
            raise ValueError('FedAvg reference partition mismatch')
        reference_traces = rows(reference / 'paired_batch_trace.csv')
        for t in range(row['rounds']):
            paired_trace_check(traces, reference_traces, t)
    return dict(case_id=row['case_id'], dataset=row['dataset'], seed=row['seed'],
                final_val=next(r['auc'] for r in final if r['split'] == 'val'),
                final_test=next(r['auc'] for r in final if r['split'] == 'test'),
                rounds=row['rounds'], steps_per_client=row['steps'], metric='mean_client_global_auc')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--approval', type=Path, required=True)
    parser.add_argument('--dataset', choices=['BBBP', 'Tox21', 'HIV'], required=True)
    parser.add_argument('--seed', type=int, choices=[0, 1, 2], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--micro', action='store_true')
    cli = parser.parse_args()
    parent, output = cli.parent.resolve(), cli.output.resolve()
    config, approval = read(parent / 'approval.json'), read(cli.approval)
    if not approval.get('approved') or approval['protocol'] != VERSION:
        raise ValueError('Explicit supplement approval required')
    if approval['worker_sha256'] != common.sha(__file__) or approval['parent_approval_sha256'] != common.sha(parent / 'approval.json'):
        raise ValueError('Approved source binding changed')
    if output.exists() and any(p.name != 'run.log' for p in output.iterdir()):
        raise ValueError('Preserve previous attempt')
    if cli.device == 'cpu':
        if not cli.micro:
            raise ValueError('Formal experiments require CUDA')
        config.update(code_root=str(common.CODE), official_repo=str(common.OFFICIAL),
                      reference_root=str(common.OLD), helper_root=str(common.HELPERS),
                      data_root=str(ROOT), download_root=str(Path.home() / '.dgl'))
    paths = {k: Path(config[k]) for k in ('code_root', 'official_repo', 'reference_root',
                                        'helper_root', 'data_root', 'download_root')}
    sources = common.verify_sources(paths['code_root'], paths['official_repo'], paths['reference_root'], paths['helper_root'])
    if sources != config['source_sha256']:
        raise ValueError('Frozen dependencies changed')
    selection = read(parent / (cli.dataset + '_selection.json'))
    if selection['test_used'] or selection['source_sha256'] != sources:
        raise ValueError('Unmatched validation selection')
    inputs = dict(graphs=paths['data_root'] / (cli.dataset.lower() + '_dglgraph.bin'),
        scaffolds=paths['data_root'] / 'data/scaffold_result' / ('scffoldLabel_' + cli.dataset + '.pt'),
        raw_csv=paths['download_root'] / {'BBBP':'bbbp/BBBP.csv','Tox21':'tox21.csv.gz','HIV':'hiv/HIV.csv'}[cli.dataset])
    hashes = {k: common.sha(p) for k, p in inputs.items()}
    if hashes != selection['inputs_sha256'] and not (cli.micro and cli.device == 'cpu'):
        raise ValueError('Input dataset mismatch')
    reference = parent / 'cases' / ('%s_seed%d_avg' % (cli.dataset, cli.seed))
    if not cli.micro and not read(reference / 'completed.json')['completed']:
        raise ValueError('Wait for paired FedAvg reference')
    if cli.micro and (cli.dataset, cli.seed) != ('BBBP', 0):
        raise ValueError('Only bounded BBBP seed0 smoke allowed')
    row = dict(case_id='%s_seed%d_individual' % (cli.dataset, cli.seed), dataset=cli.dataset,
               seed=cli.seed, method='individual', alpha=.5, local_k=16, global_k=16,
               lambda_proto=0., rounds=2 if cli.micro else 50, steps=2 if cli.micro else 200,
               peak_lr=selection['peak_lr'])
    torch = common.setup_runtime(cli.device)
    if not cli.micro and (torch.__version__ != selection['runtime_torch'] or cli.device != selection['runtime_device']):
        raise ValueError('Reference runtime mismatch')
    for key in ('helper_root', 'code_root'):
        sys.path.insert(0, str(paths[key]))
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(paths['data_root'])
    import fedavg_api, paired_path_training, strong_encoder_training
    from mmgnn_adapter import install as install_encoder
    from run_closure_case import install_controls
    from validation_tuning_queue import main_args
    install_encoder(paths['code_root'], paths['official_repo'], output=output,
                    hidden_size=300, depth=3, dropout=0., aggregation='sum', cache_size=2048)
    def offline_download(url, path, **kwargs):
        if not Path(path).is_file():
            raise ValueError('Missing offline data: ' + str(path))
        return path
    def offline_extract(path, target_dir, **kwargs):
        if not Path(target_dir).is_dir():
            raise ValueError('Missing extracted data')
    for name in ('bbbp', 'tox21', 'hiv'):
        module = importlib.import_module('data.' + name)
        module.get_download_dir = lambda: str(paths['download_root'])
        module.download = offline_download
        if hasattr(module, 'extract_archive'):
            module.extract_archive = offline_extract
    paired_path_training.train_instrumented = strong_encoder_training.train_shared
    install_controls(paths['code_root'], output, 'none')
    paired_path_training.install(output)
    original_init = fedavg_api.FedAvgAPI.__init__
    def initialize(self, dataset, device, args, trainer, input_args):
        args.strong_peak_lr = row['peak_lr']
        args.encoder = input_args.encoder = 'mmgnn2d_official_rawlogit_v2'
        if (args.split_seed, args.partition_seed, args.client_num_per_round, args.batch_size) != (0, 0, 4, 64):
            raise ValueError('Pairing configuration changed')
        if device.type != cli.device:
            raise ValueError('No silent device fallback')
        return original_init(self, dataset, device, args, trainer, input_args)
    def forbidden_exchange(*args, **kwargs):
        raise RuntimeError('Individual cannot aggregate model weights or prototypes')
    fedavg_api.FedAvgAPI.__init__ = initialize
    fedavg_api.FedAvgAPI._aggregate = forbidden_exchange
    fedavg_api.FedAvgAPI._aggregate_topology_prototypes = forbidden_exchange
    fedavg_api.FedAvgAPI.train = lambda self: train(self, output, None if cli.micro else reference)
    common.dump(output / 'spec.json', dict(protocol=VERSION, case=row, micro=cli.micro,
        source_sha256=sources, inputs_sha256=hashes, worker_sha256=common.sha(__file__),
        reference_avg=str(reference), selection_sha256=common.sha(parent / (cli.dataset + '_selection.json')),
        torch=torch.__version__, device=cli.device, no_exchange=True,
        checkpoint_selection='per-client shared heldout validation only; final is primary',
        lr_policy='shared FedAvg-development-selected schedule, not independently tuned Individual'))
    arguments = main_args(row)
    for flag in ('--split_seed', '--partition_seed'):
        arguments[arguments.index(flag) + 1] = '0'
    sys.argv = [str(paths['code_root'] / 'main.py')] + arguments + ['--results_dir', str(output)]
    runpy.run_path(str(paths['code_root'] / 'main.py'), run_name='__main__')
    artifacts = ['spec.json', 'raw_metric_curves.csv', 'checkpoint_endpoints.csv',
                 'paired_batch_trace.csv', 'round_state_hashes.csv', 'individual_models.pt',
                 'initialization.json', 'partition_audit.json', 'no_exchange_audit.json']
    common.dump(output / 'completed.json', dict(protocol=VERSION, completed=True, case=row,
                artifact_sha256={name: common.sha(output / name) for name in artifacts}))
    print(json.dumps(validate(output)), flush=True)


if __name__ == '__main__':
    main()
