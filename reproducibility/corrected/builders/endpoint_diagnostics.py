"""Observation-only common-test evaluation before and after model averaging."""

import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from paired_path_training import cloned_state, local_random_stream, state_digest


def append_rows(path, rows):
    if not rows:
        return
    new = not path.exists()
    with path.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        if new:
            writer.writeheader()
        writer.writerows(rows)


def split_positions(n, seed):
    generator = torch.Generator().manual_seed(int(seed) + 104729)
    indices = torch.randperm(n, generator=generator).numpy()
    return {'val': indices[:n//2], 'test': indices[n//2:]}


def score_tasks(labels, masks, predictions):
    labels, masks, predictions = map(np.asarray, (labels, masks, predictions))
    if labels.shape != predictions.shape or masks.shape != labels.shape or labels.ndim != 2:
        raise ValueError('Expected identical N-by-task arrays')
    if not np.isfinite(predictions).all():
        raise ValueError('Never silently drop a model with nonfinite predictions')
    tasks = []
    for task in range(labels.shape[1]):
        observed = masks[:, task] > 0
        y = labels[observed, task]
        if not np.isin(y, [0, 1]).all():
            raise ValueError('Observed classification labels must be binary')
        positives, negatives = int((y == 1).sum()), int((y == 0).sum())
        eligible = positives > 0 and negatives > 0
        tasks.append(dict(task=task, positives=positives, negatives=negatives,
                          observed_n=int(observed.sum()), eligible=eligible,
                          auc=float(roc_auc_score(y, predictions[observed, task])) if eligible else None))
    eligible = [r['auc'] for r in tasks if r['eligible']]
    if not eligible:
        raise ValueError('No eligible tasks on this predeclared split')
    return float(np.mean(eligible)), tasks


def compare_endpoints(labels, masks, probabilities, client_sizes, positions):
    names = ['server'] + ['client_%d' % i for i in range(len(client_sizes))]
    if set(probabilities) != set(names):
        raise ValueError('All client models and the server model are required')
    weights = np.asarray(client_sizes, dtype=float)
    if (weights <= 0).any():
        raise ValueError('Each client must have a positive training sample count')
    weights /= weights.sum()
    summary, detail = [], []
    for split, indices in positions.items():
        model_scores, eligibility = {}, None
        for name in names:
            auc, tasks = score_tasks(labels[indices], masks[indices], probabilities[name][indices])
            valid = [r['task'] for r in tasks if r['eligible']]
            if eligibility is not None and valid != eligibility:
                raise ValueError('Eligibility cannot depend on the evaluated model')
            eligibility = valid
            model_scores[name] = auc
            summary.append(dict(split=split, model_scope=name, auc=auc, valid_tasks=len(valid),
                                test_molecules=len(indices)))
            detail.extend(dict(split=split, model_scope=name, **r) for r in tasks)
        local = [model_scores[name] for name in names[1:]]
        # This is mean AUC, not the AUC of averaged predictions.
        derived = {'clients_mean': float(np.mean(local)),
                   'clients_train_weighted_mean': float(np.dot(weights, local)),
                   'clients_min': float(np.min(local))}
        for name, value in derived.items():
            summary.append(dict(split=split, model_scope=name, auc=value, valid_tasks=len(eligibility),
                                test_molecules=len(indices)))
    return summary, detail


def predict(model, loader, device):
    probabilities, labels, masks, indices = [], [], [], []
    model.to(device).eval()
    with torch.no_grad():
        for _, graph, y, mask, index in loader:
            graph = graph.to(device)
            logits, _ = model(graph, graph.ndata['h'], graph.edata.get('e', None))
            probabilities.append(torch.sigmoid(logits).cpu().view_as(y))
            labels.append(y.cpu())
            masks.append(mask.cpu())
            indices.append(index.cpu())
    return tuple(torch.cat(items).numpy() for items in (probabilities, labels, masks, indices))


def install(output):
    from client import Client
    from fedavg_api import FedAvgAPI
    import paired_path_training

    original_evaluate = FedAvgAPI.validateGlobal
    paired_path_training.install(output)
    original_avg, original_spl = Client.train, Client.train_topofedproto

    def capture_avg(self, weights, round_idx, client_idx):
        result = original_avg(self, weights, round_idx, client_idx)
        self._endpoint_round = round_idx
        self._endpoint_local_weights = cloned_state(result)
        return result

    def capture_spl(self, weights, prototypes, round_idx, client_idx):
        result = original_spl(self, weights, prototypes, round_idx, client_idx)
        self._endpoint_round = round_idx
        self._endpoint_local_weights = cloned_state(result)
        return result

    Client.train, Client.train_topofedproto = capture_avg, capture_spl

    def evaluate(self, epoch):
        original_evaluate(self, epoch)
        clients = self.client_list
        if len(clients) != self.args.client_num_per_round or any(c._endpoint_round != epoch for c in clients):
            raise RuntimeError('This diagnostic requires all clients to update in the same round')
        server_weights = cloned_state(self.model_trainer.model.state_dict())
        states = {'server': server_weights}
        states.update({'client_%d' % i: c._endpoint_local_weights for i, c in enumerate(clients)})
        hashes = {name: state_digest(value) for name, value in states.items()}
        probabilities = {}
        labels, masks, indices = None, None, None
        # Clone evaluation models and restore every RNG stream, so observation
        # does not alter future training batches, dropout, modes or BN buffers.
        with local_random_stream(19000000 + int(self.args.seed)*1000 + epoch):
            template = copy.deepcopy(self.model_trainer.model)
            for name, weights in states.items():
                template.load_state_dict(weights)
                p, y, m, ids = predict(template, self.test_global, self.device)
                if labels is not None:
                    if not all(np.array_equal(a, b) for a, b in ((labels, y), (masks, m), (indices, ids))):
                        raise RuntimeError('Model evaluations did not use the same molecules/labels/masks')
                probabilities[name], labels, masks, indices = p, y, m, ids
            del template
        if state_digest(self.model_trainer.model.state_dict()) != hashes['server']:
            raise RuntimeError('Observation changed the server model')
        if len(np.unique(indices)) != len(indices):
            raise RuntimeError('Held-out molecule indices are not unique')
        audit = json.loads((output / 'partition_audit.json').read_text())
        train_indices = {i for group in audit['client_indices'].values() for i in group}
        if train_indices.intersection(map(int, indices)):
            raise RuntimeError('Training/test molecule overlap')
        positions = split_positions(len(indices), self.args.split_seed)
        metadata = dict(dataset=self.args.dataset, method=self.fedmid, alpha=self.args.part_alpha,
                        seed=self.args.seed, round=epoch, local_steps=self.args.localStepsPerRound,
                        local_k=self.args.topoproto_clusters, global_k=self.args.topoproto_global_clusters,
                        lambda_proto=self.args.lambda_proto)
        summary, detail = compare_endpoints(labels, masks, probabilities,
                                            [c.get_sample_number() for c in clients], positions)
        append_rows(output / 'endpoint_curves.csv', [dict(metadata, **r) for r in summary])
        append_rows(output / 'endpoint_task_curves.csv', [dict(metadata, **r) for r in detail])
        if epoch == 0:
            record = dict(train_holdout_disjoint=True, identical_model_evaluation_rows=True,
                          heldout_indices_sha256=hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest(),
                          validation_indices=indices[positions['val']].tolist(),
                          test_indices=indices[positions['test']].tolist(),
                          aggregation_before_or_after='Client returned weights before server averaging; common held-out test',
                          local_personalization_claim=False,
                          client_weights=[c.get_sample_number() for c in clients])
            (output / 'endpoint_split_audit.json').write_text(json.dumps(record, indent=2)+'\n')
        append_rows(output / 'endpoint_model_hashes.csv',
                    [dict(round=epoch, model_scope=name, sha256=value) for name, value in hashes.items()])
        if epoch == self.args.comm_round-1:
            torch.save(states, str(output / 'final_server_and_clients.pt'))
            np.savez_compressed(str(output / 'final_predictions.npz'), labels=labels, masks=masks,
                                dataset_indices=indices, val_positions=positions['val'],
                                test_positions=positions['test'], **probabilities)
            # Audit-only counterfactuals. Neither value is used to select a model
            # or represented as legitimate held-out-test evidence.
            audit_rows = []
            for name, p in probabilities.items():
                for policy, y, m, pred in (
                    ('heldout_val_plus_test_NOT_primary', labels, masks, p),
                    ('missing_labels_as_zero_INVALID', np.where(masks[positions['test']]>0, labels[positions['test']], 0),
                     np.ones_like(masks[positions['test']]), p[positions['test']])):
                    value, _ = score_tasks(y, m, pred)
                    audit_rows.append(dict(metadata, model_scope=name, audit_policy=policy,
                                           auc=value, eligible_for_performance_claim=False))
            append_rows(output / 'metric_sensitivity_AUDIT_ONLY.csv', audit_rows)
        test_scores = {r['model_scope']: r['auc'] for r in summary if r['split']=='test'}
        print('[Endpoint round %d] server=%.6f clients_mean=%.6f gap=%.6f; common masked test' %
              (epoch, test_scores['server'], test_scores['clients_mean'],
               test_scores['clients_mean']-test_scores['server']), flush=True)

    FedAvgAPI.validateGlobal = evaluate
