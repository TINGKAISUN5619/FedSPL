"""CPU-only tests of byte-identical reviewer-v3 functions. No optimizer steps."""
import contextlib
import importlib.util
import io
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
CODE = ROOT / 'corrected/source/reviewer_v3_snapshot'
sys.path.insert(0, str(CODE))
import torch
import dgl
import data_loader
from fedml_api.standalone.fedavg.my_model_trainer_classification import MyModelTrainer


class ProtocolTests(unittest.TestCase):
    def test_observed_loss_and_gradient_fill_invariance(self):
        labels = torch.tensor([[0., 1.], [1., 0.], [0., 1.]])
        masks = torch.tensor([[1., 0.], [1., 1.], [0., 1.]])
        results = []
        for fill in (0., 1., -23., 17.):
            logits = torch.tensor([[.2, -.4], [-.1, .8], [1., -.3]], requires_grad=True)
            targets = torch.where(masks > 0, labels, torch.full_like(labels, fill))
            losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
            actual = MyModelTrainer._masked_mean(losses, masks)
            self.assertTrue(torch.allclose(actual, losses[masks > 0].mean()))
            actual.backward()
            self.assertTrue(torch.equal(logits.grad[masks == 0], torch.zeros_like(logits.grad[masks == 0])))
            results.append((actual.detach(), logits.grad.clone()))
        for value, grad in results[1:]:
            self.assertTrue(torch.allclose(value, results[0][0]))
            self.assertTrue(torch.equal(grad, results[0][1]))
        self.assertEqual(MyModelTrainer._masked_mean(torch.ones(2, 2), torch.zeros(2, 2)).item(), 0.)

    def test_actual_masked_evaluator(self):
        p = ROOT / 'corrected/builders/test_fair_evaluation.py'
        spec = importlib.util.spec_from_file_location('frozen_evaluation_test', p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with patch.object(sys, 'argv', [str(p), '--code-root', str(CODE)]):
            mod.main()

    def test_labels_masks_stable_ids_separate(self):
        graph = dgl.graph(([0], [0]))
        a = ('C', graph, torch.tensor([0., 1.]), torch.tensor([1., 0.]), 107)
        b = ('N', graph, torch.tensor([1., 0.]), torch.tensor([0., 1.]), 23)
        _, _, labels, masks, ids = data_loader.collate_molgraphs([a, b])
        self.assertEqual(ids.tolist(), [107, 23])
        self.assertEqual(labels.tolist(), [[0., 1.], [1., 0.]])
        self.assertEqual(masks.tolist(), [[1., 0.], [0., 1.]])
        reverse = data_loader.collate_molgraphs([b, a])
        self.assertEqual(reverse[-1].tolist(), [23, 107])
        self.assertNotEqual(labels.data_ptr(), masks.data_ptr())
        # The dataset-specific override supplies the ID; the base CSV fallback does not.
        from data import Tox21
        dataset = Tox21.__new__(Tox21)
        dataset.load_full = False
        dataset.smiles, dataset.graphs = ['C', 'N'], [graph, graph]
        dataset.labels, dataset.mask = labels, masks
        self.assertEqual(dataset[1][-1], 1)
        self.assertTrue(torch.equal(dataset[1][3], masks[1]))

    def test_corrected_loader_no_localtrain_alias(self):
        graph = dgl.graph(([0], [0]))
        records = [('C', graph, torch.tensor([i % 2.]), torch.ones(1), i) for i in range(12)]
        train, holdout = records[:8], records[8:]
        allocation = {0: [0, 2, 4, 6], 1: [1, 3, 5, 7]}
        args = SimpleNamespace(dataset='Tox21', partition_method='hetero', partition_alpha=.5,
                               client_num_in_total=2, batch_size=2, numWorker=0)
        with patch.object(data_loader, 'partition_data', return_value=(train, holdout, allocation)):
            result = data_loader.load_partition_data(args)
        self.assertTrue(all(loader is None for loader in result[6].values()))
        self.assertTrue(all(loader is not result[3] for loader in result[5].values()))
        self.assertFalse({r[-1] for r in result[2].dataset} & {r[-1] for r in result[3].dataset})


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
