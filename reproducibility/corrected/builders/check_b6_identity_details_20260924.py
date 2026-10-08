"""Resolve audit findings without changing molecular identities or split membership."""
from collections import defaultdict
import json

import numpy as np
import pandas as pd
import torch
from dgl.data.utils import load_labels
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

from audit_partition_b6_20260924 import BUNDLE, OUT, digest, scaffold_file, split_indices


def main():
    destination = OUT / 'identity_followup_v2.json'
    assert not destination.exists(), 'Do not overwrite a completed audit'
    RDLogger.DisableLog('rdApp.warning')
    folder = BUNDLE / 'data/ToxCast'
    tensors = load_labels(str(folder / 'graphs.bin'))
    frame = pd.read_csv(next(folder.glob('public.csv*')))
    stored = torch.load(scaffold_file('ToxCast'), map_location='cpu', weights_only=False).tolist()
    forward, reverse = defaultdict(set), defaultdict(set)
    empty_rows = 0
    for group, smi in zip(stored, frame['smiles'].iloc[tensors['valid_ids'].long().numpy()]):
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None
        key = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=True)
        empty_rows += int(key == '')
        forward[group].add(key)
        reverse[key].add(group)
    stereo = dict(stored_groups=len(forward), fresh_isomeric_murcko_groups=len(reverse),
        stored_to_multiple_fresh=sum(len(v) > 1 for v in forward.values()),
        fresh_to_multiple_stored=sum(len(v) > 1 for v in reverse.values()),
        empty_scaffold_rows=empty_rows,
        empty_scaffold_stored_groups=len(reverse.get('', set())))
    # Count held-out labels sharing identity with training, including contradictions.
    folder = BUNDLE / 'data/BBBP'
    tensors = load_labels(str(folder / 'graphs.bin'))
    frame = pd.read_csv(next(folder.glob('public.csv*')))
    raw_ids = tensors['valid_ids'].long().numpy()
    keys = [Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True, isomericSmiles=True)
            for s in frame['smiles'].iloc[raw_ids]]
    labels, masks = tensors['labels'].numpy(), tensors['mask'].numpy()
    rows = []
    for seed in range(3):
        split = split_indices(len(keys), seed)
        train = defaultdict(list)
        for i in split['train']:
            train[keys[i]].append(i)
        seen = [i for i in split['test'] if keys[i] in train]
        conflict, matching = [], []
        for i in seen:
            values = {float(labels[j, 0]) for j in train[keys[i]] if masks[j, 0] > 0}
            if masks[i, 0] > 0 and values:
                if float(labels[i, 0]) in values:
                    matching.append(i)
                if any(v != float(labels[i, 0]) for v in values):
                    conflict.append(i)
        rows.append(dict(seed=seed, test_n=len(split['test']), train_seen_test_rows=len(seen),
            unseen_test_n=len(split['test'])-len(seen),
            train_seen_test_fraction=len(seen)/len(split['test']),
            matching_training_label_test_rows=matching, conflicting_training_label_test_rows=conflict,
            seen_test_dataset_indices=seen, seen_test_public_indices=raw_ids[seen].tolist(),
            unseen_test_label_counts={str(int(v)): int(n) for v, n in
                zip(*np.unique(labels[[i for i in split['test'] if i not in seen], 0], return_counts=True))}))
    result = dict(ToxCast_stereochemical_mapping=stereo, BBBP_heldout_identity=rows,
        supersedes='identity_followup.json: empty_scaffold_rows counted stored groups, not rows; other findings unchanged',
        no_scores_or_models_used=True, no_data_or_split_changes=True,
        script_sha256=digest(__file__))
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
