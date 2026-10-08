"""Build approved mechanism tables from frozen curves without running training."""
import argparse
import csv
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/reviewer_revision_20260831'
AUDIT = BASE / 'revision_readiness_20260918'
OUT = BASE / 'revision_approval_20260924/mechanism'
PAPER = ROOT / 'paper_acs_latex'
DATASETS = ('SIDER', 'Tox21')
SPLITS = ('test', 'val')
SCOPES = ('server', 'clients_mean')
CONDITIONS = ('hetero_avg', 'hetero_spl', 'permuted_spl', 'iid_avg', 'iid_spl')
COMPARISONS = ('hetero_spl_minus_avg', 'true_minus_permuted_structure',
               'iid_spl_minus_avg', 'hetero_minus_iid_gain')
SYMBOLS = ('H', 'C', 'I', 'D')
T_CRITICAL = 4.302652729911275


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    require(math.isclose(actual, expected, rel_tol=0, abs_tol=1e-9), label)


def paired_statistics(values):
    require(len(values) == 3 and all(math.isfinite(v) for v in values), 'Need three finite seed differences')
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half = T_CRITICAL * sd / math.sqrt(3)
    return dict(mean_auc_pp=mean, sd_auc_pp=sd, ci95_low=mean-half, ci95_high=mean+half)


def contrasts(v):
    hetero = v['hetero_spl'] - v['hetero_avg']
    iid = v['iid_spl'] - v['iid_avg']
    return dict(zip(COMPARISONS, (100*hetero, 100*(v['hetero_spl']-v['permuted_spl']),
                                 100*iid, 100*(hetero-iid))))


def load_endpoints():
    checks = rows(AUDIT / 'MECHANISM_RUN_CHECKS.csv')
    inventory = {r['path']: r['sha256'] for r in rows(AUDIT / 'MECHANISM_SOURCE_INVENTORY.csv')}
    expected = set(itertools.product(DATASETS, range(3), CONDITIONS))
    keys = [(r['dataset'], int(r['seed']), r['condition']) for r in checks]
    require(len(keys) == 30 and set(keys) == expected, 'Need exactly 30 unique condition runs')
    endpoints = []
    sources = []
    for check in checks:
        curve = Path(check['source_directory']) / 'endpoint_curves.csv'
        require(sha(curve) == inventory[str(curve)], 'Source curve changed: ' + str(curve))
        raw = rows(curve)
        require(len(raw) == 800, 'Unexpected curve row count')
        for r in raw:
            require(r['dataset'] == check['dataset'] and int(r['seed']) == int(check['seed'])
                    and float(r['alpha']) == .5 and int(r['local_steps']) == 200
                    and int(r['local_k']) == int(r['global_k']) == 16,
                    'Unexpected run protocol')
            expected_method = 'avg' if check['condition'].endswith('_avg') else 'fedavg_proto'
            require(r['method'] == expected_method and
                    float(r['lambda_proto']) == (0 if expected_method == 'avg' else .1),
                    'Unexpected method/loss weight')
            require(math.isfinite(float(r['auc'])) and 0 <= float(r['auc']) <= 1, 'Invalid AUC')
        for split, scope in itertools.product(SPLITS, SCOPES):
            values = [r for r in raw if r['split'] == split and r['model_scope'] == scope]
            require([int(r['round']) for r in values] == list(range(50)), 'Incomplete or duplicate curve')
            final = values[-1]
            endpoints.append(dict(dataset=check['dataset'], seed=int(check['seed']),
                condition=check['condition'], split=split, model_scope=scope,
                auc=float(final['auc']), valid_tasks=int(final['valid_tasks']),
                source_curve=str(curve.relative_to(ROOT)), source_sha256=inventory[str(curve)]))
        sources.append(dict(dataset=check['dataset'], seed=int(check['seed']), condition=check['condition'],
                            path=str(curve.relative_to(ROOT)), sha256=inventory[str(curve)]))
    return endpoints, checks, sources


def summarize(endpoints):
    expected = set(itertools.product(DATASETS, range(3), CONDITIONS, SPLITS, SCOPES))
    lookup = {(r['dataset'], r['seed'], r['condition'], r['split'], r['model_scope']): r['auc']
              for r in endpoints}
    require(len(endpoints) == 120 and set(lookup) == expected, 'Need all 120 unique endpoints')
    differences, summary = [], []
    for dataset, split, scope in itertools.product(DATASETS, SPLITS, SCOPES):
        per_seed = [contrasts({c: lookup[dataset, s, c, split, scope] for c in CONDITIONS}) for s in range(3)]
        for comparison in COMPARISONS:
            values = [r[comparison] for r in per_seed]
            summary.append(dict(dataset=dataset, split=split, model_scope=scope, comparison=comparison,
                                n=3, **{'seed%d_auc_pp' % s: values[s] for s in range(3)},
                                **paired_statistics(values)))
            for seed, value in enumerate(values):
                differences.append(dict(dataset=dataset, split=split, model_scope=scope,
                                        comparison=comparison, seed=seed, difference_auc_pp=value))
    return differences, summary


def verify_audit(differences, summary):
    audited = {(r['dataset'], r['split'], r['model_scope'], r['comparison']): r
               for r in rows(AUDIT / 'MECHANISM_PAIRED_SUMMARY.csv')}
    require(len(audited) == 32, 'Incomplete audited summaries')
    for row in summary:
        original = audited[row['dataset'], row['split'], row['model_scope'], row['comparison']]
        for field in ('mean_auc_pp', 'sd_auc_pp', 'ci95_low', 'ci95_high',
                      'seed0_auc_pp', 'seed1_auc_pp', 'seed2_auc_pp'):
            close(row[field], float(original[field]), 'Audited summary mismatch: ' + field)
    audited_d = {(r['dataset'], r['split'], r['model_scope'], r['comparison'], int(r['seed'])): r
                 for r in rows(AUDIT / 'MECHANISM_PAIRED_DIFFERENCES.csv')}
    require(len(audited_d) == 96, 'Incomplete audited differences')
    for row in differences:
        original = audited_d[row['dataset'], row['split'], row['model_scope'], row['comparison'], row['seed']]
        close(row['difference_auc_pp'], float(original['difference_auc_pp']), 'Audited difference mismatch')


def table_start(caption, label, columns, header):
    return [r'\begin{table}[H]', r'\centering', r'\small',
            r'\caption{' + caption + '}', r'\label{' + label + '}',
            r'\setlength{\tabcolsep}{7pt}', r'\renewcommand{\arraystretch}{1.2}',
            r'\begin{tabular}{' + columns + '}', r'\toprule', header + r' \\', r'\midrule']


def table_end(note):
    return [r'\bottomrule', r'\end{tabular}', r'\vspace{0.5em}',
            r'\parbox{\linewidth}{\footnotesize\textit{Note:} ' + note + '}', r'\end{table}', '']


def endpoints_tex(endpoints, checks):
    names = ('Heterogeneous Avg', 'Heterogeneous SPL', 'Permuted SPL', 'Size-matched IID Avg', 'Size-matched IID SPL')
    lines = table_start('Final AUC in the mechanism-control experiments.', 'tab:mechanism_endpoints',
                        'llcccc', r'Dataset & Condition & Server val & Server test & Client val & Client test')
    for dataset in DATASETS:
        for ci, condition in enumerate(CONDITIONS):
            cells = []
            for scope, split in itertools.product(SCOPES, ('val', 'test')):
                values = [r['auc']*100 for r in endpoints if
                          (r['dataset'], r['condition'], r['split'], r['model_scope']) == (dataset, condition, split, scope)]
                cells.append('$%.2f \\pm %.2f$' % (statistics.mean(values), statistics.stdev(values)))
            lines.append(' & '.join([dataset if ci == 0 else '', names[ci]] + cells) + r' \\')
        if dataset != DATASETS[-1]:
            lines.append(r'\midrule')
    lines += table_end(r'AUC (\%) is mean $\pm$ sample SD over three joint split/allocation/initialization seeds, evaluated at round 49. Client scores are the arithmetic mean of four client-model AUCs on the same held-out set, not prediction-ensemble AUC or personalized test performance. This diagnostic cohort is reported separately from the main-table cohort; no best-round selection is used.')
    lines += table_start('Training-client sizes and eligible assay tasks.', 'tab:mechanism_sizes',
                         'lccccccc', r'Dataset & Seed & Client 1 & Client 2 & Client 3 & Client 4 & Val tasks & Test tasks')
    for dataset, seed in itertools.product(DATASETS, range(3)):
        matching = [r for r in checks if r['dataset'] == dataset and int(r['seed']) == seed]
        require(len({r['client_sizes_json'] for r in matching}) == 1, 'Client sizes not held fixed')
        r = next(r for r in matching if r['condition'] == 'hetero_avg')
        lines.append(' & '.join([dataset, str(seed)] + [str(n) for n in json.loads(r['client_sizes_json'])] +
                               [r['val_eligible_tasks'], r['test_eligible_tasks']]) + r' \\')
    lines += table_end(r'The size-matched IID control preserves these client counts, not their molecular membership. SIDER uses 1141 training molecules and 143 molecules each for validation and testing; Tox21 uses 6264 training molecules, 783 validation molecules, and 784 test molecules. Eligibility requires observed positive and negative labels. Within each dataset and seed, all five conditions share the same held-out molecules and eligible tasks.')
    return '\n'.join(lines)


def contrasts_tex(summary, scope):
    title = 'server-model' if scope == 'server' else 'client-mean'
    lines = table_start('Paired ' + title + ' contrasts in the mechanism controls.', 'tab:mechanism_' + scope,
                        'llccccc', r'Dataset & Contrast & Seed 0 & Seed 1 & Seed 2 & Mean $\pm$ SD & 95\% CI')
    lookup = {(r['dataset'], r['split'], r['model_scope'], r['comparison']): r for r in summary}
    for split in SPLITS:
        lines.append(r'\multicolumn{7}{l}{' + ('Test' if split == 'test' else 'Validation') + r'} \\')
        for dataset in DATASETS:
            for ci, comparison in enumerate(COMPARISONS):
                r = lookup[dataset, split, scope, comparison]
                cells = [dataset if ci == 0 else '', '$' + SYMBOLS[ci] + '$']
                cells += ['$%+.2f$' % r['seed%d_auc_pp' % s] for s in range(3)]
                cells += ['$%+.2f \\pm %.2f$' % (r['mean_auc_pp'], r['sd_auc_pp']),
                          '$[%+.2f, %+.2f]$' % (r['ci95_low'], r['ci95_high'])]
                lines.append(' & '.join(cells) + r' \\')
            lines.append(r'\addlinespace')
        if split == 'test':
            lines.append(r'\midrule')
    lines += table_end(r'All values are AUC percentage-point differences. $H$: heterogeneous SPL minus heterogeneous Avg; $C$: true-correspondence SPL minus permuted SPL; $I$: size-matched IID SPL minus IID Avg; $D=H-I$: difference in the method gains between allocations. SD is computed from three paired seed differences, not by subtracting model SDs. Intervals are exploratory, unadjusted paired-$t$ intervals with two degrees of freedom. Every interval includes zero; this does not establish equivalence or absence of an effect. Validation and test results are kept separate.')
    return '\n'.join(lines)


def csv_text(records):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(records)
    return stream.getvalue()


def emit(path, text, check):
    if check:
        require(path.is_file() and path.read_text() == text, 'Stale output: ' + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def main(check=False):
    endpoints, checks, sources = load_endpoints()
    differences, summary = summarize(endpoints)
    verify_audit(differences, summary)
    emit(PAPER / 'mechanism_endpoints_generated.tex', endpoints_tex(endpoints, checks), check)
    for scope in SCOPES:
        emit(PAPER / ('mechanism_' + scope + '_generated.tex'), contrasts_tex(summary, scope), check)
    for name, records in (('endpoints', endpoints), ('paired_differences', differences),
                          ('paired_summary', summary), ('source_curves', sources)):
        emit(OUT / (name + '.csv'), csv_text(records), check)
    audit = dict(source_curves=30, audited_endpoints=120, paired_differences=96,
                 contrast_summaries=32, new_training=0, original_results_changed=False,
                 source_hashes={str(p.relative_to(ROOT)): sha(p) for p in (
                     AUDIT/'MECHANISM_RUN_CHECKS.csv', AUDIT/'MECHANISM_SOURCE_INVENTORY.csv',
                     AUDIT/'MECHANISM_PAIRED_SUMMARY.csv', AUDIT/'MECHANISM_PAIRED_DIFFERENCES.csv')},
                 limitation='Separate paired diagnostic cohort; no main-table replacement or causal proof')
    emit(OUT / 'table_audit.json', json.dumps(audit, indent=2, sort_keys=True)+'\n', check)
    print(json.dumps(dict(verified_curves=30, endpoints=120, paired_differences=96, summaries=32, check=check)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    main(parser.parse_args().check)
