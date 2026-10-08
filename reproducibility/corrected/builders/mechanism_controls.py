"""Descriptor-correspondence control with exactly preserved cluster geometry."""

from collections import defaultdict
import hashlib
import json

import numpy as np

from run_closure_case import descriptor_permutation


def donor_positions(smiles, seed):
    mapping = descriptor_permutation(smiles, seed)
    positions = defaultdict(list)
    for i, molecule in enumerate(smiles):
        positions[molecule].append(i)
    used = defaultdict(int)
    donors = []
    for molecule in smiles:
        destination = mapping[molecule]
        donors.append(positions[destination][used[destination]])
        used[destination] += 1
    donors = np.asarray(donors, dtype=np.int64)
    if sorted(donors.tolist()) != list(range(len(smiles))):
        raise RuntimeError('Descriptor correspondence must be a bijection')
    return mapping, donors


def install_permuted_correspondence(output):
    import torch
    from sklearn.cluster import KMeans
    from client import Client

    original_descriptor = Client._descriptor_from_smiles_and_embeddings
    original_collect = Client.collect_topology_prototypes

    def prepare(self):
        if hasattr(self, '_mechanism_correspondence'):
            return self._mechanism_correspondence
        if self.args.proto_descriptor_space != 'fingerprint':
            raise RuntimeError('This prespecified control uses fingerprints only')
        items = sorted((int(item[4]), item[0]) for item in self.local_training_data.dataset)
        indices = np.asarray([item[0] for item in items], dtype=np.int64)
        if len(np.unique(indices)) != len(indices):
            raise RuntimeError('Local training molecule IDs must be unique')
        smiles = [item[1] for item in items]
        seed = int(self.args.partition_seed) + 7919 * (int(self.client_idx) + 1)
        mapping, donors = donor_positions(smiles, seed)
        # Reconstruct the unpermuted descriptor fit in the exact original ID
        # order. Permuting row order before KMeans would change its initialization.
        descriptors = original_descriptor(self, smiles, torch.zeros(len(smiles), 1)).cpu().numpy()
        self._mechanism_correspondence = (indices, smiles, mapping, donors, descriptors)
        record = dict(client=int(self.client_idx), seed=seed,
                      reassigned_molecule_fraction=float(np.mean([mapping[s] != s for s in smiles])),
                      descriptor_multiset_preserved=True, cluster_counts_preserved=True,
                      fixed_training_only_mapping=True,
                      mapping_sha256=hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest())
        (output / ('correspondence_client%d.json' % self.client_idx)).write_text(json.dumps(record, indent=2)+'\n')
        return self._mechanism_correspondence

    def descriptor(self, smiles, embeddings):
        _, _, mapping, _, _ = prepare(self)
        return original_descriptor(self, [mapping[s] for s in smiles], embeddings)

    def collect(self, n_clusters):
        indices, smiles, mapping, donors, descriptors = prepare(self)
        k = min(max(1, int(n_clusters)), len(indices))
        key = ('fingerprint', k, hashlib.sha256(indices.tobytes()).hexdigest())
        if key not in self.prototype_cluster_cache:
            if k == 1:
                original_assignments = np.zeros(len(indices), dtype=np.int64)
            else:
                original_assignments = KMeans(n_clusters=k, n_init=10,
                    random_state=int(self.args.partition_seed)+int(self.client_idx)).fit_predict(descriptors)
            centers = {int(c): descriptors[original_assignments == c].mean(axis=0)
                       for c in np.unique(original_assignments)}
            shuffled = original_assignments[donors]
            if not np.array_equal(np.bincount(shuffled), np.bincount(original_assignments)):
                raise RuntimeError('Correspondence control changed cluster occupancy')
            self.prototype_cluster_cache[key] = dict(indices=indices.copy(), assignments=shuffled,
                                                     cluster_descriptors=centers)
            np.savez_compressed(str(output / ('correspondence_geometry_client%d.npz' % self.client_idx)),
                dataset_indices=indices, donors=donors, original_assignments=original_assignments,
                permuted_assignments=shuffled, descriptors=np.stack([centers[c] for c in sorted(centers)]),
                counts=np.asarray([(original_assignments == c).sum() for c in sorted(centers)]))
        result = original_collect(self, n_clusters)
        if sum(item['count'] for item in result) != len(indices):
            raise RuntimeError('Prototype collection omitted local training molecules')
        return result

    Client._descriptor_from_smiles_and_embeddings = descriptor
    Client.collect_topology_prototypes = collect
