#!/usr/bin/env python3
"""Read-only integrity checks supplementary to the frozen MMGNN dispatcher."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import mmgnn_queue as queue


def validate_records(directory, row):
    metrics = queue.rows(directory / 'raw_metric_curves.csv')
    endpoints = queue.rows(directory / 'checkpoint_endpoints.csv')
    for record in metrics + endpoints:
        if (record['protocol'] != queue.VERSION or record['method'] != row['method']
                or record['score_kind'] not in ('logits', 'probabilities')
                or record['split'] not in ('val', 'test')
                or int(record['valid_tasks']) < 1 or int(record['molecules']) < 2
                or record['test_selection'] != 'False'
                or not math.isfinite(float(record['auc']))
                or not 0 <= float(record['auc']) <= 1):
            raise ValueError('Invalid metric identity, eligibility or score')
    development = row['stage'] == 'development'
    central = row['method'] == 'centralized'
    scope = 'centralized' if central else 'server'
    validation = [r for r in metrics if r['model_scope'] == scope and r['split'] == 'val'
                  and r['score_kind'] == 'logits'
                  and (not central or r['stage'] == 'development')]
    if [int(r['round']) for r in validation] != list(range(row['rounds'])):
        raise ValueError('Expected exactly one validation record per training round')
    best = max(validation, key=lambda r: float(r['auc']))
    splits = ('val',) if development else ('val', 'test')
    expected = {(checkpoint, split, kind) for checkpoint in ('final', 'validation_selected')
                for split in splits for kind in ('logits', 'probabilities')}
    actual = [(r['checkpoint'], r['split'], r['score_kind']) for r in endpoints]
    if Counter(actual) != Counter(expected):
        raise ValueError('Incomplete or duplicated checkpoint endpoints')
    for record in endpoints:
        wanted = row['rounds'] - 1 if record['checkpoint'] == 'final' else int(best['round'])
        if (record['model_scope'] != scope or int(record['selected_round']) != wanted
                or int(record['round']) != wanted):
            raise ValueError('Endpoint is not final or validation-only selected')
        if record['split'] == 'val' and record['score_kind'] == 'logits':
            if float(record['auc']) != float(validation[wanted]['auc']) and not central:
                raise ValueError('FL endpoint differs from recorded selected validation')
    if development and any(r['split'] != 'val' for r in metrics):
        raise ValueError('Development must never score test labels')
    if not central:
        traces = queue.rows(directory / 'paired_batch_trace.csv')
        keys = [(int(r['round']), int(r['client'])) for r in traces]
        expected_keys = {(r, c) for r in range(row['rounds']) for c in range(4)}
        if Counter(keys) != Counter(expected_keys):
            raise ValueError('Missing or duplicate round/client trace')
        if any(int(r['optimizer_steps']) != row['steps'] for r in traces):
            raise ValueError('Successful optimizer-step count differs')
        for epoch in range(row['rounds']):
            hashes = {r['start_model_sha256'] for r in traces if int(r['round']) == epoch}
            if len(hashes) != 1:
                raise ValueError('Clients did not start from one common model')
        required = ['final_server_and_clients.pt', 'validation_selected_server.pt']
        required.append('predictions/round_%03d_server.npz' % (row['rounds'] - 1))
    else:
        required = ['centralized_models.pt']
        for checkpoint, epoch in [('final', row['rounds'] - 1),
                                  ('validation_selected', int(best['round']))]:
            required.append('predictions/%s_%03d_centralized.npz' % (checkpoint, epoch))
    if any(not (directory / name).is_file() or (directory / name).stat().st_size == 0
           for name in required):
        raise ValueError('Missing or empty checkpoint/prediction artifact')
    return dict(case_id=row['case_id'], records_verified=True, selected_round=int(best['round']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = queue.read_json(args.output / 'approval.json')
    queue.source_check(config)
    verified, pending, failures = [], [], []
    for original in queue.ordered(config['cases']):
        directory = args.output / 'cases' / original['case_id']
        if not (directory / 'completed.json').exists():
            pending.append(original['case_id'])
            continue
        try:
            row = dict(original)
            if row['stage'] != 'development':
                row['peak_lr'] = queue.read_json(args.output / (row['dataset'] + '_selection.json'))['peak_lr']
            queue.validate(directory, row, config)
            verified.append(validate_records(directory, row))
        except (ValueError, KeyError, OSError) as error:
            failures.append(dict(case_id=original['case_id'], error=repr(error)))
    print(json.dumps(dict(read_only=True, completed=len(verified), total=42,
                          verified=verified, pending=pending, failures=failures), indent=2))
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
