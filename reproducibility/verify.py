"""Offline, standard-library result verification; never imports training code."""
import ast
import csv
import hashlib
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREFIX = 'results/reviewer_revision_20260831/'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def canonical(value):
    starts = [value.index(marker) for marker in ('results/', 'paper_acs_latex/', 'TEST/') if marker in value]
    if starts:
        return value[min(starts):]
    return value


def source(name):
    name = canonical(name)
    return ROOT / INDEX[name]['archive_path']


def data(name):
    return rows(source(PREFIX + name))


def close(a, b, label):
    require(math.isfinite(float(a)) and math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-12), label)


def final(curve, field='test', round_field='round'):
    require([int(r[round_field]) for r in curve] == list(range(50)), 'Require exactly ordered rounds 0..49')
    value = float(curve[-1][field])
    require(math.isfinite(value), 'Nonfinite final endpoint')
    return value


def integrity():
    manifest = {}
    for line in (ROOT / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        require(name not in manifest and not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe/duplicate path')
        manifest[name] = digest
        require(sha(ROOT / name) == digest, 'Integrity failure: ' + name)
    actual = {str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and p.name != 'SHA256SUMS'}
    require(actual == set(manifest), 'Unexpected or missing archive files')
    for original, entry in INDEX.items():
        require(sha(ROOT / entry['archive_path']) == entry['archive_sha256'], 'Provenance payload mismatch')
        if entry['transform'] == 'byte_identical':
            require(entry['archive_sha256'] == entry['source_sha256'], 'Byte-identical provenance mismatch')
        if original.endswith('.py'):
            require(entry['transform'] == 'byte_identical', 'Scientific implementation was patched')
    for p in ROOT.rglob('*.py'):
        ast.parse(p.read_text(encoding='utf-8-sig'), filename=str(p))
    require(not list(ROOT.rglob('*.tex')), 'TeX excluded')
    require(not any('queue' in p.name.lower() and p.suffix == '.py' for p in ROOT.rglob('*')), 'Queue excluded')
    return len(manifest)


def runs(which):
    if which == 'classification':
        records = data('revision_approved_20260920/main_classification_source_runs.csv')
        expected = set(itertools.product(('BBBP', 'SIDER', 'Tox21', 'HIV', 'ToxCast'),
            ('0.1', '0.5', '1.0'), ('avg', 'fedprox', 'oursvatFLITPLUS', 'fedproto', 'moon', 'fpl', 'fedtgp', 'fedavg_proto'), range(3)))
        summary = data('revision_approved_20260920/main_classification_summary.csv')
        endpoint, mean, sd = 'final_test_auc', 'mean_auc', 'sample_sd_auc'
    else:
        records = data('revision_approval_20260924/regression/regression_table_source_runs.csv')
        expected = set(itertools.product(('esol', 'freesolv', 'lipo'), ('0.1', '0.5', '1.0'),
            ('FedAvg', 'FedProx', 'FLIT+', 'MOON-MSE', 'FedSPL (fingerprint)', 'FedSPL (physchem)'), range(3)))
        summary = data('revision_approval_20260924/regression/regression_table_summary.csv')
        endpoint, mean, sd = 'final_test_rmse', 'mean_rmse', 'sd_rmse'
    keys = [(r['dataset'], r['alpha'], r['method'], int(r['seed'])) for r in records]
    require(len(keys) == len(expected) and set(keys) == expected, 'Incomplete/duplicate cohort')
    groups = defaultdict(list)
    for r in records:
        p = source(r['source'])
        require(INDEX[canonical(r['source'])]['namespace'] == 'corrected', 'Mixed historical cohort')
        require(sha(p) == r['source_sha256'], 'Original numerical CSV hash mismatch')
        curve = [v for v in rows(p) if v.get('scope') == 'global']
        close(final(curve), r[endpoint], 'Final round mismatch')
        for v in curve:
            require(v['dataset'] == r['dataset'] and v['metric'] == ('auc' if which == 'classification' else 'rmse'), 'Wrong dataset/metric')
            require(all(int(v[k]) == int(r['seed']) for k in ('model_seed', 'split_seed', 'partition_seed')), 'Seed mismatch')
            require(v['encoder'] == 'mpnn' and v['partition_method'] == 'hetero', 'Wrong encoder/allocation')
        groups[r['dataset'], r['alpha'], r['method']].append(float(r[endpoint]))
    require(len(summary) == len(groups), 'Summary coverage')
    for r in summary:
        values = groups[r['dataset'], r['alpha'], r['method']]
        close(statistics.mean(values), r[mean], 'Mean mismatch')
        close(statistics.stdev(values), r[sd], 'Sample SD mismatch')
    return dict(runs=len(records), cells=len(summary), rows=len(records)*50)


def splits():
    saved = {}
    for p in sorted((ROOT / 'datasets').glob('*/split*.json')):
        record = json.loads(p.read_text())
        sets = {k: set(v) for k, v in record['ids'].items()}
        require(set(sets) == {'train', 'val', 'test'}, 'Three split names required')
        require(all(len(sets[k]) == len(record['ids'][k]) for k in sets), 'Duplicate split index')
        require(not any(sets[a] & sets[b] for a,b in itertools.combinations(sets, 2)), 'Split index leakage')
        union = set.union(*sets.values())
        require(union == set(range(len(union))), 'Missing/noncontiguous dataset IDs')
        saved[record['dataset'], int(record['seed'])] = sets
    require(len(saved) == 15, 'Need 15 classification split records')
    members = [k for k in INDEX if '/partition_audit/members_' in k]
    require(len(members) == 45, 'Need 45 allocation records')
    for key in members:
        r = json.loads(source(key).read_text())
        ids = [i for group in r['client_indices'].values() for i in group]
        require(len(ids) == len(set(ids)), 'Overlapping clients')
        require(set(ids) == saved[r['dataset'], int(r['seed'])]['train'], 'Clients must cover exactly training IDs')
    return dict(split_records=15, allocation_records=45, identity_disjointness_claim=False)


def secondary():
    central = data('revision_approval_20260924/central/central_source_runs.csv')
    for r in central:
        p = source(r['source'])
        endpoints = json.loads(p.read_text())
        selected = [v for v in endpoints if v['checkpoint'] == 'final' and v['split'] == 'test']
        require(len(selected) == 1 and selected[0]['block'] == 49, 'Central final endpoint')
        close(selected[0]['auc'], r['final_test_auc'], 'Central endpoint mismatch')
        curve = rows(p.parent / 'block_curve.csv')
        final(curve, field='cumulative_updates', round_field='block')
        require(all(int(v['successful_updates']) == 800 for v in curve), 'Central update budget')
    mech = data('revision_approval_20260924/mechanism/endpoints.csv')
    for r in mech:
        p = source(r['source_curve'])
        require(sha(p) == r['source_sha256'], 'Mechanism raw curve hash')
        curve = [v for v in rows(p) if v['split'] == r['split'] and v['model_scope'] == r['model_scope']]
        close(final(curve, 'auc'), r['auc'], 'Mechanism endpoint')
    robust = data('revision_approval_20260924/robustness/sources.csv')
    for r in robust:
        p = source(r['source'])
        require(sha(p) == r['source_sha256'], 'Robustness curve hash')
        close(final([v for v in rows(p) if v['scope'] == 'global']), r['final_test'], 'Robustness endpoint')
    mmgnn = data('deadline_queue_20260920/priority_20260923/mmgnn_minimal_endpoints.csv')
    for r in mmgnn:
        raw = rows(source(r['source']))
        curve = [v for v in raw if v['split'] == 'test' and v['score_kind'] == 'logits'
                 and v['model_scope'] == r['scope']]
        if not curve and r['scope'] == 'client_mean':
            values = [final([v for v in raw if v['split'] == 'test' and v['score_kind'] == 'logits'
                      and v['model_scope'] == 'client_%d' % c], 'auc') for c in range(4)]
            value = statistics.mean(values)
        else:
            value = final(curve, 'auc')
        close(value, r['auc'], 'MMGNN raw-logit final endpoint')
    local = data('revision_approval_20260924/traditional/local_source_runs.csv')
    groups = defaultdict(list)
    for r in local:
        require(sha(source(r['source_csv'])) == r['source_sha256'], 'Traditional source hash')
        groups[r['dataset'], float(r['alpha']), r['method']].append(float(r['primary_test_auc']))
    for r in data('revision_approval_20260924/traditional/local_mean_sd.csv'):
        values = groups[r['dataset'], float(r['alpha']), r['method']]
        require(len(values) == 3, 'Traditional seeds')
        close(statistics.mean(values), r['mean_auc'], 'Traditional mean')
        close(statistics.stdev(values), r['sample_sd_auc'], 'Traditional SD')
    individual = data('deadline_queue_20260920/priority_20260923/mpnn_individual_seed0_summary.csv')
    for r in individual:
        raw = rows(source(canonical(r['source']) + '/metric_curves.csv'))
        curve = [v for v in raw if v['split'] == 'test' and v['model_scope'] == 'client_mean']
        close(final(curve, 'auc'), r['final_auc'], 'Individual final endpoint')
    return dict(central_runs=len(central), mechanism_endpoints=len(mech), robustness_runs=len(robust),
                mmgnn_endpoints=len(mmgnn), traditional_runs=len(local), individual_seed0_runs=len(individual))


def selection_negative_tests():
    fake = [dict(round=i, test=0.99 if i == 3 else 0.5, best_test=1.0) for i in range(50)]
    require(final(fake) == 0.5, 'Best-test selection occurred')
    for bad in (fake[:-1], fake + [fake[-1]], fake[::-1], fake[:5] + fake[6:]):
        try:
            final(bad)
        except ValueError:
            continue
        raise ValueError('Incomplete/duplicate/reordered rounds accepted')
    return 5


INDEX = json.loads((ROOT / 'PROVENANCE.json').read_text()) if (ROOT / 'PROVENANCE.json').exists() else {}

if __name__ == '__main__':
    report = dict(integrity_files=integrity(), classification=runs('classification'), regression=runs('regression'),
                  disjointness=splits(), secondary=secondary(), final_selection_tests=selection_negative_tests(),
                  training_started=False, CUDA_replay=False)
    print(json.dumps(report, indent=2))
