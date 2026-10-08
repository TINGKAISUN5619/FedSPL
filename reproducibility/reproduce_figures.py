"""Portable path adapter for frozen plot functions; no training or code rewriting."""
import argparse
import importlib.util
import os
import shutil
from pathlib import Path
import result_paths as paths


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True, help='New output directory outside the archive')
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() or paths.ROOT == out or paths.ROOT in out.parents:
        raise ValueError('Use a new directory outside the source package')
    out.mkdir(parents=True)
    os.environ.setdefault('MPLCONFIGDIR', str(out / 'matplotlib_config'))
    base = load('plot_original', paths.ROOT / 'corrected/builders/plot_approved_revision_20260920.py')
    base.OUTPUT = out / 'corrected'
    base.FIGURES = out / 'corrected'
    base.OUTPUT.mkdir()
    base.SUPPORT = paths.source(paths.PREFIX + 'cluster_support_review_20260916/k_summary.csv')
    original_read = base.read_rows
    status = paths.data('robustness_audit_20260910/summaries/expected_run_status.csv')
    selected = []
    for record in status:
        if record['suite'] == 'main_classification':
            record = dict(record)
            record['csv_file'] = str(paths.source(record['csv_file']))
            selected.append(record)
    base.read_rows = lambda p: selected if p == base.STATUS else original_read(p)
    base.prototype_support()
    base.classification_curves()
    # The historical figure builder expects its original relative directory layout.
    work = out / 'historical_task'
    records = paths.ROOT / 'historical_task/records'
    for original in records.rglob('*'):
        key = str(original.relative_to(records))
        if original.is_file() and (key.startswith('TEST/results/') or '/mpp_diagnostics/' in key):
            target = work / key
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
    hist = load('historical_plot_original', paths.ROOT / 'historical_task/builders/restore_historical_task_panel_20260925.py')
    hist.ROOT = work
    hist.SOURCE = work / 'paper_acs_latex/figures/mpp_diagnostics'
    hist.OUT = work / 'reports'
    hist.FIG = work / 'figures/tox21_historical_single_seed_tasks'
    hist.FIG.parent.mkdir(parents=True)
    hist.main()
    print('Corrected convergence/support and historical two-panel Figure 5 exported.')
    print('Official mixed-protocol display is retained as audit-only PDF and numerical curve data, not pooled here.')


if __name__ == '__main__':
    main()
