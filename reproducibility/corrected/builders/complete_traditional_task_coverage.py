#!/usr/bin/env python3
"""Include untrainable client-task models as constant predictors, without refits.

A constant has AUC=0.5 on a binary held-out task. Adding a constant local
prediction to a support-weighted ensemble is a positive affine transformation
of its nonconstant predictions, so its ranking/AUC is unchanged. Tasks with no
trainable local model also receive 0.5. Original results remain unmodified.
"""

import argparse
from pathlib import Path

from audit_claims_20260910 import save, summarize
from summarize_revision_results import read_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for r in read_csv(args.input):
        trained = int(r['valid_client_task_models'])
        skipped = int(r['skipped_single_class_models'])
        clients = int(r['clients'])
        eligible = trained + skipped
        assert eligible > 0 and eligible % clients == 0, r
        tasks = eligible // clients
        ensemble_tasks = int(r['valid_ensemble_tasks'])
        assert 0 <= ensemble_tasks <= tasks
        local_auc = (float(r['mean_local_test_auc']) * trained + 0.5 * skipped) / eligible
        ensemble_auc = (float(r['ensemble_test_auc']) * ensemble_tasks +
                        0.5 * (tasks-ensemble_tasks)) / tasks
        rows.append(dict(dataset=r['dataset'], alpha=r['alpha'], method=r['method'], seed=r['seed'],
                         eligible_test_tasks=tasks, eligible_client_tasks=eligible,
                         constant_fallback_client_tasks=skipped,
                         original_trainable_coverage=trained/eligible,
                         mean_local_test_auc_all_tasks=local_auc,
                         ensemble_test_auc_all_tasks=ensemble_auc,
                         derivation='analytical_constant_fallback_no_refit'))
    save(args.output / 'traditional_full_coverage_runs.csv', rows)
    save(args.output / 'traditional_full_coverage_summary.csv', summarize(rows,
         ['dataset', 'alpha', 'method'],
         ['mean_local_test_auc_all_tasks', 'ensemble_test_auc_all_tasks', 'original_trainable_coverage']))
    print('Complete-coverage derivation for %d runs; %d constant client-task fallbacks.' %
          (len(rows), sum(r['constant_fallback_client_tasks'] for r in rows)))


if __name__ == '__main__':
    main()
