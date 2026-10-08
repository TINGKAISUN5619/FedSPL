"""Recompute approved B1 evidence from complete immutable source curves."""
import argparse
import csv
import hashlib
import io
import itertools
import json
from pathlib import Path
import statistics

from build_main_classification_table_20260920 import paired_statistics, require, csv_text

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/reviewer_revision_20260831'
STATUS = BASE / 'robustness_audit_20260910/summaries/expected_run_status.csv'
OUT = BASE / 'revision_approval_20260924/robustness'
PAPER = ROOT / 'paper_acs_latex'


def read_csv(path):
    return list(csv.DictReader(io.StringIO(path.read_text())))


def condition(row):
    return row['dataset'], row['encoder'], row['method'], row['descriptor'], int(row['seed'])


def selected(row):
    return (row['suite'] in ('encoder', 'descriptor') or
            (row['suite'] == 'main_classification' and row['dataset'] in ('SIDER', 'Tox21')
             and row['alpha'] == '0.5' and row['method'] in ('avg', 'fedavg_proto')))


def validate(curve, expected):
    require([int(r['round']) for r in curve] == list(range(50)), 'Incomplete/duplicate rounds')
    for row in curve:
        require(all(row[k] == expected[k] for k in ('dataset', 'encoder', 'method', 'descriptor')),
                'Wrong condition')
        require(row['scope'] == 'global' and row['metric'] == 'auc'
                and row['partition_method'] == 'hetero', 'Wrong endpoint')
        require(all(int(row[k]) == int(expected['seed']) for k in
                    ('model_seed', 'split_seed', 'partition_seed')), 'Unpaired seeds')
        require(all(0 <= float(row[k]) <= 1 for k in ('test', 'val')), 'Invalid AUC')
        if row['method'] == 'fedavg_proto':
            require(float(row['lambda_proto']) == .1 and row['local_clusters'] == '16'
                    and row['global_clusters'] == '16', 'Wrong SPL parameters')
    require(all(float(curve[-1][k]) == float(expected['final_' + k]) for k in ('val', 'test')),
            'Manifest does not match actual final endpoint')


def table(caption, label, header, rows, note):
    return '\n'.join([r'\begin{table}[H]', '\\caption{' + caption + '}',
        '\\label{' + label + '}', r'\centering\small',
        '\\begin{tabular}{' + 'l' * len(header) + '}', r'\toprule',
        ' & '.join(header) + r' \\', r'\midrule',
        *[' & '.join(row) + r' \\' for row in rows], r'\bottomrule', r'\end{tabular}',
        '\\par\\smallskip{\\footnotesize\\textit{Note:} ' + note + '}', r'\end{table}', ''])


def mean_sd(values):
    return '$%.2f \\pm %.2f$' % (100 * statistics.mean(values), 100 * statistics.stdev(values))


def build():
    manifest = [r for r in read_csv(STATUS) if selected(r)]
    expected = set()
    for d, s in itertools.product(('SIDER', 'Tox21'), range(3)):
        for enc in ('mpnn', 'attentivefp'):
            expected.add((d, enc, 'avg', '', s))
            expected.add((d, enc, 'fedavg_proto', 'fingerprint', s))
        for desc in ('maccs', 'physchem', 'random'):
            expected.add((d, 'mpnn', 'fedavg_proto', desc, s))
    require(len(manifest) == len(expected) == 42 and {condition(r) for r in manifest} == expected,
            'Expected exact 42 unique source runs')
    sources, lookup = [], {}
    for row in manifest:
        require(row['status'] == 'complete' and row['alpha'] == '0.5', 'Incomplete or wrong alpha')
        marker = 'results/reviewer_revision_20260831/server_synced/formal_v3/'
        require(marker in row['csv_file'], 'Unexpected source root')
        path = BASE / 'server_synced/formal_v3' / row['csv_file'].split(marker, 1)[1]
        try:
            path.resolve().relative_to((BASE / 'server_synced/formal_v3').resolve())
        except ValueError:
            raise ValueError('Path escape')
        curve = read_csv(path)
        validate(curve, row)
        record = dict(dataset=row['dataset'], encoder=row['encoder'], method=row['method'],
            descriptor=row['descriptor'], alpha=.5, seed=int(row['seed']), final_test=float(curve[-1]['test']),
            final_val=float(curve[-1]['val']), source=str(path.relative_to(ROOT)),
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), suite=row['suite'])
        lookup[condition(row)] = record
        sources.append(record)
    require(len({r['source'] for r in sources}) == 42, 'Duplicate source')
    summaries = []
    for d, enc, method, desc in sorted({k[:4] for k in expected}):
        vals = [lookup[d, enc, method, desc, s]['final_test'] for s in range(3)]
        summaries.append(dict(dataset=d, encoder=enc, method=method, descriptor=desc, alpha=.5,
            n=3, seed0=vals[0], seed1=vals[1], seed2=vals[2], mean_auc=statistics.mean(vals),
            sample_sd_auc=statistics.stdev(vals)))
    contrasts = []
    for r in summaries:
        if r['method'] != 'fedavg_proto':
            continue
        d, enc, desc = r['dataset'], r['encoder'], r['descriptor']
        for control in (['avg', 'random'] if desc == 'fingerprint' and enc == 'mpnn' else ['avg']):
            diffs = [100 * (lookup[d, enc, 'fedavg_proto', desc, s]['final_test'] -
                     lookup[d, enc, 'avg' if control == 'avg' else 'fedavg_proto',
                            '' if control == 'avg' else 'random', s]['final_test']) for s in range(3)]
            mean, sd, lo, hi = paired_statistics(diffs)
            contrasts.append(dict(dataset=d, encoder=enc, descriptor=desc, control=control,
                n=3, seed0_pp=diffs[0], seed1_pp=diffs[1], seed2_pp=diffs[2],
                mean_pp=mean, sample_sd_pp=sd, ci95_low_pp=lo, ci95_high_pp=hi))
    def values(d, enc, method, desc):
        return [lookup[d, enc, method, desc, s]['final_test'] for s in range(3)]
    enc_rows = []
    for d, enc in itertools.product(('SIDER', 'Tox21'), ('mpnn', 'attentivefp')):
        c = next(r for r in contrasts if r['dataset'] == d and r['encoder'] == enc
                 and r['descriptor'] == 'fingerprint' and r['control'] == 'avg')
        enc_rows.append([d, 'MPNN' if enc == 'mpnn' else 'AttentiveFP',
            mean_sd(values(d, enc, 'avg', '')), mean_sd(values(d, enc, 'fedavg_proto', 'fingerprint')),
            r'$%+.2f\;[%.2f,%.2f]$' % (c['mean_pp'], c['ci95_low_pp'], c['ci95_high_pp'])])
    desc_rows = []
    for d in ('SIDER', 'Tox21'):
        desc_rows.append([d, 'FedAvg (no prototypes)', mean_sd(values(d, 'mpnn', 'avg', '')), '--'])
        for desc, label in [('fingerprint', 'Morgan'), ('maccs', 'MACCS'),
                            ('physchem', 'Physicochemical'), ('random', 'Random')]:
            c = next(r for r in contrasts if r['dataset'] == d and r['encoder'] == 'mpnn'
                     and r['descriptor'] == desc and r['control'] == 'avg')
            desc_rows.append([d, 'FedSPL: ' + label, mean_sd(values(d, 'mpnn', 'fedavg_proto', desc)),
                r'$%+.2f\;[%.2f,%.2f]$' % (c['mean_pp'], c['ci95_low_pp'], c['ci95_high_pp'])])
    note = ('Final masked server-model test ROC-AUC (percent), mean $\\pm$ sample SD over three '
            'joint split/allocation/initialization seeds. Differences are paired AUC percentage points; '
            'brackets give exploratory, unadjusted 95\\% paired-$t$ intervals (two degrees of freedom). '
            'No best-round, test-selected descriptor, or client-average substitution is used.')
    files = {OUT / 'sources.csv': csv_text(sources), OUT / 'summary.csv': csv_text(summaries),
             OUT / 'paired_contrasts.csv': csv_text(contrasts),
             PAPER / 'robustness_encoder_generated.tex': table('Encoder-specific paired comparisons.',
                 'tab:robustness_encoder', ['Dataset', 'Encoder', 'FedAvg', 'FedSPL', 'Difference [95\\% CI]'], enc_rows, note),
             PAPER / 'robustness_descriptor_generated.tex': table('Descriptor-specific comparisons with MPNN.',
                 'tab:robustness_descriptor', ['Dataset', 'Method / descriptor', 'ROC-AUC (\\%)', 'Difference [95\\% CI]'], desc_rows, note)}
    evidence = dict(unique_runs=42, new_training_runs=0, endpoint='masked_server_test_round49',
        main_table_changed=False, status_sha256=hashlib.sha256(STATUS.read_bytes()).hexdigest(),
        source_sha256={r['source']: r['source_sha256'] for r in sources})
    files[OUT / 'verification.json'] = json.dumps(evidence, indent=2, sort_keys=True) + '\n'
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    for path, content in build().items():
        if args.check:
            require(path.read_text() == content, 'Stale output: ' + str(path))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    print('Verified 42 unique complete curves; 14 summaries and 12 paired contrasts; no training.')


if __name__ == '__main__':
    main()
