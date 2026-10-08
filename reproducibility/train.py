"""Explicit opt-in local entry for frozen main.py; only filesystem I/O is adapted.

Default prepares a new workspace and command. --import-smoke parses main --help;
only --execute calls training. Neither smoke nor preparation constructs graphs.
"""
import argparse
import importlib
import json
import os
import runpy
import shutil
import sys
import zipfile
from pathlib import Path
from result_paths import ROOT

DATASETS = {
    'BBBP': ('bbbp', 'bbbp.zip', 'BBBP.csv'),
    'SIDER': ('sider', 'sider.zip', 'sider.csv'),
    'Tox21': ('tox21', None, 'tox21.csv.gz'),
    'HIV': ('hiv', 'hiv.zip', 'HIV.csv'),
    'ToxCast': ('toxcast', 'toxcast.zip', 'toxcast_data.csv'),
    'esol': ('esol', 'ESOL.zip', 'delaney-processed.csv'),
    'freesolv': ('freesolv', 'FreeSolv.zip', 'SAMPL.csv'),
    'lipo': ('lipophilicity', 'lipophilicity.zip', 'Lipophilicity.csv'),
}


def command(config, output):
    dataset, method, seed, alpha = (config[k] for k in ('dataset', 'method', 'seed', 'alpha'))
    if dataset not in DATASETS or type(seed) is not int or seed not in (0, 1, 2) or alpha not in (.1, .5, 1.):
        raise ValueError('Use an approved dataset, seed 0/1/2 and alpha 0.1/0.5/1.0')
    if method not in ('avg', 'fedavg_proto', 'fedprox', 'oursvatFLITPLUS', 'moon'):
        raise ValueError('Portable example supports Avg/SPL/Prox/FLIT+/MOON only')
    workers = config.get('num_workers', 0)
    if type(workers) is not int or workers < 0:
        raise ValueError('num_workers must be a nonnegative integer')
    k = 32 if dataset == 'ToxCast' else 16
    return ['-dataset', dataset, '-fedmid', method, '-part_alpha', str(alpha), '-seed', str(seed),
        '--split_seed', str(seed), '--partition_seed', str(seed), '--partition_method', 'hetero',
        '--encoder', 'mpnn', '-comm_round', '50', '-numClient', '4', '--clients_per_round', '4',
        '--local_steps_per_round', '200', '--topoproto_clusters', str(k), '--topoproto_global_clusters', str(k),
        '--lambda_proto', '0.1', '--proto_descriptor_space', 'fingerprint', '--global_only_eval',
        '--num_workers', str(workers), '--results_dir', str(output)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True, help='New directory outside the archive')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true', help='Explicitly start one full local training run')
    mode.add_argument('--import-smoke', action='store_true', help='Import frozen entry and print help; no graphs/training')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    work = args.workspace.resolve()
    argv = command(config, work / 'new_results')
    if work.exists() or work == ROOT or ROOT in work.parents:
        raise ValueError('Use a new directory outside the source package')
    dataset = config['dataset']
    variant = 'regression_frozen' if dataset in ('esol', 'freesolv', 'lipo') else 'reviewer_v3_snapshot'
    original = ROOT / 'corrected/source' / variant
    target = work / 'code'
    shutil.copytree(original, target)
    downloads = work / 'public_inputs'
    downloads.mkdir()
    module, archive, filename = DATASETS[dataset]
    source = next((ROOT / 'datasets' / dataset).glob('public.csv*'))
    if archive:
        with zipfile.ZipFile(downloads / archive, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(source, filename)
    else:
        shutil.copyfile(source, downloads / filename)
    scaffold = target / 'data/scaffold_result' / ('scffoldLabel_' + dataset + '.pt')
    scaffold.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / 'datasets' / dataset / 'scaffold_labels.pt', scaffold)
    prepared = dict(config=config, argv=argv, frozen_variant=variant,
        adapter='new offline filesystem adapter; unchanged scientific implementation',
        graph_construction_tested=False, training_executed=args.execute,
        warning='New execution, not claimed bitwise historical reproduction; CUDA chosen by frozen entry when available')
    (work / 'PREPARED.json').write_text(json.dumps(prepared, indent=2) + '\n')
    if not args.execute and not args.import_smoke:
        print(json.dumps(prepared, indent=2))
        return
    sys.path.insert(0, str(target))
    dataset_module = importlib.import_module('data.' + module)
    expected = downloads / (archive or filename)

    def local_download(url, path, **kwargs):
        if Path(path) != expected or not expected.is_file():
            raise ValueError('Unexpected data download path; network acquisition disabled')
        return str(expected)

    dataset_module.get_download_dir = lambda: str(downloads)
    dataset_module.download = local_download
    os.chdir(target)
    sys.argv = [str(target / 'main.py')] + (['--help'] if args.import_smoke else argv)
    try:
        runpy.run_path(str(target / 'main.py'), run_name='__main__')
    except SystemExit as exc:
        if not args.import_smoke or exc.code not in (0, None):
            raise
    if args.import_smoke:
        print('PASS: frozen main.py and scientific imports loaded; help parsed; no graph construction or training.')


if __name__ == '__main__':
    main()
