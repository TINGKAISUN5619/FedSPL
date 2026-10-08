"""Regression test against the actual reviewer-v3 masked evaluation function."""

import argparse
import contextlib
import io
import sys
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--code-root', required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.code_root)
    import torch
    from fedavg_api import FedAvgAPI
    from fedml_api.standalone.fedavg.my_model_trainer_classification import MyModelTrainer

    logits = torch.stack([torch.sin(torch.arange(40).float()),
                          torch.cos(torch.arange(40).float()),
                          torch.arange(40).float() / 40], dim=1)
    observed_labels = (logits > 0).float()
    masks = torch.ones_like(logits)
    masks[::3, 0] = 0
    masks[::4, 1] = 0
    masks[:, 2] = 0

    class Graph:
        ndata = {'h': logits}
        edata = {'e': torch.ones(40, 1)}

        def to(self, device):
            return self

    class Model(torch.nn.Module):
        def forward(self, graph, nodes, edges):
            return nodes, nodes

    class Loader:
        dataset = range(40)

        def __init__(self, labels):
            self.labels = labels

        def __len__(self):
            return 1

        def __iter__(self):
            yield ['C'] * 40, Graph(), self.labels, masks, torch.arange(40)

    class Logger:
        path = 'memory-only'

        def log(self, row):
            self.row = row

    results = []
    for missing_fill in (0, 1):
        labels = torch.where(masks > 0, observed_labels,
                             torch.full_like(observed_labels, missing_fill))
        api = FedAvgAPI.__new__(FedAvgAPI)
        api.model_trainer = SimpleNamespace(model=Model())
        api.device = torch.device('cpu')
        api.args = SimpleNamespace(dataset='Tox21', split_seed=13)
        api.test_global = Loader(labels)
        api.bestVal, api.bestTest = float('-inf'), float('-inf')
        api.bestQm9EveryTask = []
        api.fedmid = 'test'
        api.csv_logger = Logger()
        with contextlib.redirect_stdout(io.StringIO()):
            api.validateGlobal(49)
        results.append(api.csv_logger.row)
        losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        expected = losses[masks > 0].mean()
        actual = MyModelTrainer._masked_mean(losses, masks)
        assert torch.allclose(actual, expected), (actual, expected)
    for row in results:
        assert row['test'] == 1.0 and row['val'] == 1.0, row
    assert results[0]['test'] == results[1]['test']
    print('PASS: actual validateGlobal ignores missing-label fill values and all-missing tasks.')
    print('PASS: actual supervised masked mean equals observed-entry-only mean.')


if __name__ == '__main__':
    main()
