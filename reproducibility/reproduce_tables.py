"""Regenerate numeric result tables from verified portable source records."""
import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path
import verify as v


def aggregate(records, keys, value):
    groups = defaultdict(list)
    for row in records:
        groups[tuple(row[k] for k in keys)].append(float(row[value]))
    output = []
    for key, values in sorted(groups.items()):
        r = dict(zip(keys, key))
        r.update(n=len(values), mean=statistics.mean(values),
                 sample_sd=statistics.stdev(values) if len(values) > 1 else '')
        output.append(r)
    return output


def write(out, name, records):
    with (out / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() or out == v.ROOT or v.ROOT in out.parents:
        raise ValueError('Use a new output directory outside the immutable archive')
    v.integrity()
    v.runs('classification')
    v.runs('regression')
    v.secondary()
    out.mkdir(parents=True)
    definitions = [
        ('classification', 'revision_approved_20260920/main_classification_source_runs.csv',
         ['dataset', 'alpha', 'method'], 'final_test_auc'),
        ('regression', 'revision_approval_20260924/regression/regression_table_source_runs.csv',
         ['dataset', 'alpha', 'method'], 'final_test_rmse'),
        ('central', 'revision_approval_20260924/central/central_source_runs.csv', ['dataset'], 'final_test_auc'),
        ('traditional', 'revision_approval_20260924/traditional/local_source_runs.csv',
         ['dataset', 'alpha', 'method'], 'primary_test_auc'),
        ('robustness', 'revision_approval_20260924/robustness/sources.csv',
         ['dataset', 'alpha', 'encoder', 'method', 'descriptor'], 'final_test'),
        ('mechanism', 'revision_approval_20260924/mechanism/endpoints.csv',
         ['dataset', 'condition', 'model_scope', 'split'], 'auc'),
        ('mmgnn', 'deadline_queue_20260920/priority_20260923/mmgnn_minimal_endpoints.csv',
         ['dataset', 'method', 'scope'], 'auc'),
    ]
    for name, path, keys, value in definitions:
        write(out, name + '.csv', aggregate(v.data(path), keys, value))
    # These tables have already defined contrasts or seed-specific rows, not new averaging units.
    for name, path in [
        ('classification_paired_ci', 'revision_approved_20260920/main_classification_paired_ci.csv'),
        ('regression_paired', 'revision_approval_20260924/regression/regression_paired_differences.csv'),
        ('mechanism_contrasts', 'revision_approval_20260924/mechanism/paired_summary.csv'),
        ('mechanism_sizes', 'revision_readiness_20260918/MECHANISM_RUN_CHECKS.csv'),
        ('individual_seed0', 'deadline_queue_20260920/priority_20260923/mpnn_individual_seed0_summary.csv'),
        ('official_audit_seeds', 'revision_approval_20260925/official_reference_replay/official_sources.csv')]:
        write(out, name + '.csv', v.data(path))
    print('PASS: seven mean/sample-SD numeric tables regenerated; six explicit contrast/seed source tables exported.')
    print('Source-table exports are labeled, not claimed as independent recomputations of stored contrasts.')


if __name__ == '__main__':
    main()
