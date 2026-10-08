"""Isolated paired-path controls; do not alter the frozen reviewer-v3 code."""

from contextlib import contextmanager
import csv
import hashlib
from pathlib import Path
import random

import numpy as np
import torch

from sider_transfer_training import train_instrumented


@contextmanager
def local_random_stream(seed):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if devices:
                torch.cuda.manual_seed_all(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def training_seed(seed, round_idx, client_idx):
    payload = '%s:%s:%s:paired-path-v1' % (seed, round_idx, client_idx)
    return int(hashlib.sha256(payload.encode()).hexdigest()[:8], 16)


def cloned_state(state):
    return {name: tensor.detach().cpu().clone() for name, tensor in state.items()}


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class TracedLoader:
    """Track actual consumed batches without drawing any additional samples."""

    def __init__(self, loader):
        self.loader = loader
        self.batch_hash = hashlib.sha256()
        self.used_batches = 0
        self.skipped_singletons = 0

    def __iter__(self):
        for batch in self.loader:
            if len(batch[0]) > 1:
                self.used_batches += 1
                indices = batch[4].detach().cpu().numpy().astype('<i8')
                self.batch_hash.update(len(indices).to_bytes(8, 'little'))
                self.batch_hash.update(indices.tobytes())
            else:
                self.skipped_singletons += 1
            yield batch


def train_paired(self, start_weights, prototypes, round_idx, client_idx, baseline=False):
    original_loader = self.local_training_data
    original_weight = self.args.lambda_proto
    self.local_training_data = TracedLoader(original_loader)
    seed = training_seed(self.args.seed, round_idx, client_idx)
    start_digest = state_digest(start_weights)
    try:
        if baseline:
            self.args.lambda_proto = 0.0
        # The same helper counts successful optimizer steps for both methods.
        with local_random_stream(seed):
            outcome = train_instrumented(self, cloned_state(start_weights),
                                         None if baseline else prototypes, round_idx, client_idx)
        trace = self.local_training_data
        assert trace.used_batches == self.args.localStepsPerRound
        assert state_digest(start_weights) == start_digest, 'Global starting snapshot was mutated'
        if not all(torch.isfinite(value).all() for value in outcome.values()):
            raise RuntimeError('Nonfinite model state; zero times a nonfinite loss is not a valid control')
        record = dict(round=round_idx, client=client_idx, stream_seed=seed,
                      optimizer_steps=trace.used_batches, skipped_singletons=trace.skipped_singletons,
                      batch_sha256=trace.batch_hash.hexdigest(), start_model_sha256=start_digest)
        path = Path(self._diagnostics_dir) / 'paired_batch_trace.csv'
        new_file = not path.exists()
        with path.open('a', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(record))
            if new_file:
                writer.writeheader()
            writer.writerow(record)
        return cloned_state(outcome)
    finally:
        self.local_training_data = original_loader
        self.args.lambda_proto = original_weight


def train_avg(self, start_weights, round_idx, client_idx):
    return train_paired(self, start_weights, None, round_idx, client_idx, baseline=True)


def train_spl(self, start_weights, prototypes, round_idx, client_idx):
    return train_paired(self, start_weights, prototypes, round_idx, client_idx)


def install(output):
    from client import Client
    from fedavg_api import FedAvgAPI
    from fedml_api.standalone.fedavg.my_model_trainer_classification import MyModelTrainer

    Client.train = train_avg
    Client.train_topofedproto = train_spl
    Client._diagnostics_dir = str(output)
    Client._diagnostics_interval = 50
    Client._require_finite_losses = True
    # Still compute/validate prototype targets at zero weight, but differentiate
    # exactly the supervised graph instead of adding an unused zero branch.
    Client._zero_weight_supervised_graph = True
    original_get = MyModelTrainer.get_model_params
    MyModelTrainer.get_model_params = lambda self: cloned_state(original_get(self))
    original_aggregate = FedAvgAPI._aggregate
    FedAvgAPI._aggregate = lambda self, updates: cloned_state(original_aggregate(
        self, [(count, cloned_state(state)) for count, state in updates]))
    original_evaluate = FedAvgAPI.validateGlobal

    def evaluate_and_snapshot(self, epoch):
        original_evaluate(self, epoch)
        # Every-round Avg/zero states permit a numerical parity gate, not just AUC comparison.
        if self.fedmid == 'avg' or self.args.lambda_proto == 0 or epoch == 0:
            checkpoint = Path(output) / 'checkpoints' / ('round_%03d.pt' % epoch)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(cloned_state(self.model_trainer.model.state_dict()), str(checkpoint))
        if epoch == self.args.comm_round - 1:
            torch.save(cloned_state(self.model_trainer.model.state_dict()),
                       str(Path(output) / 'final_model.pt'))

    FedAvgAPI.validateGlobal = evaluate_and_snapshot
