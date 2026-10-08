import copy
import csv
import hashlib
import logging
import os
import time
from tqdm import tqdm
import numpy as np
import torch
from datetime import datetime
from client import Client
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from torch.utils.data import Subset, DataLoader
from random import sample
import torch.nn.functional as F
import dgl


MOLECULAR_CLASSIFICATION_DATASETS = ['MUV', 'BACE', 'BBBP', 'ClinTox', 'SIDER',
                                     'ToxCast', 'HIV', 'PCBA', 'Tox21']
MOLECULAR_REGRESSION_DATASETS = ['esol', 'lipo', 'freesolv']


class CSVResultLogger:
    def __init__(self, results_dir, run_name, metadata=None):
        os.makedirs(results_dir, exist_ok=True)
        self.path = os.path.join(results_dir, run_name + ".csv")
        self.metadata = metadata or {}
        self.fieldnames = [
            "time", "method", "dataset", "round", "scope", "client_idx",
            "metric", "val", "test", "best_val", "best_test",
            "local_metric", "global_metric", "client_indexes",
            "model_seed", "split_seed", "partition_seed", "partition_method",
            "encoder", "descriptor", "lambda_proto", "proto_support_tau",
            "proto_weighted_kmeans",
            "local_clusters", "global_clusters"
        ]
        with open(self.path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def log(self, row):
        merged = dict(self.metadata)
        merged.update(row)
        complete = {name: merged.get(name, "") for name in self.fieldnames}
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(complete)

def sample_public_data_from_clients(train_data_local_dict, total_sample_size):
    all_indices = []
    client_indices = {}
    total_data_size = sum(len(local_loader.dataset) for local_loader in train_data_local_dict.values())

    quotas = {}
    remainders = []
    for client_id, local_loader in train_data_local_dict.items():
        if hasattr(local_loader.dataset, 'indices'):
            base_indices = [int(idx) for idx in local_loader.dataset.indices]
        else:
            base_indices = list(range(len(local_loader.dataset)))
        client_indices[client_id] = base_indices
        proportion = len(base_indices) / total_data_size
        raw_quota = total_sample_size * proportion
        quotas[client_id] = min(len(base_indices), int(raw_quota))
        remainders.append((raw_quota - int(raw_quota), client_id))

    remaining = max(0, total_sample_size - sum(quotas.values()))
    for _, client_id in sorted(remainders, reverse=True):
        if remaining == 0:
            break
        capacity = len(client_indices[client_id]) - quotas[client_id]
        if capacity <= 0:
            continue
        add = min(capacity, remaining)
        quotas[client_id] += add
        remaining -= add

    for client_id, base_indices in client_indices.items():
        client_sample_size = quotas[client_id]
        selected = sample(list(base_indices), min(client_sample_size, len(base_indices)))
        logging.info(f"[Client {client_id}] Base size = {len(base_indices)}, Selected = {len(selected)}")
        all_indices.extend(selected)

    logging.info(f"Total sampled public data size: {len(all_indices)}")
    return all_indices

def collate_molgraphs(data):
    smiles = [item[0] for item in data]
    graphs = [item[1] for item in data]
    raw_labels = [item[2] for item in data]
    raw_masks = []
    raw_indices = []
    for fallback_idx, item in enumerate(data):
        label = item[2]
        if len(item) >= 5:
            mask, index = item[3], item[4]
        elif len(item) == 4 and torch.is_tensor(item[3]) and item[3].dtype.is_floating_point:
            mask, index = item[3], fallback_idx
        elif len(item) == 4:
            mask, index = torch.ones_like(label), item[3]
        else:
            mask, index = torch.ones_like(label), fallback_idx
        raw_masks.append(mask)
        raw_indices.append(int(index))
    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(raw_labels, dim=0)
    masks = torch.stack(raw_masks, dim=0).float()
    indices = torch.tensor(raw_indices, dtype=torch.long)
    return smiles, bg, labels, masks, indices

class FedAvgAPI(object):
    def __init__(self, dataset, device, args, model_trainer, wandbConfig):
        self.device = device
        self.args = args
        self.wandbConfig = wandbConfig
        self.fedmid = self.wandbConfig.fedmid
        is_structure_proto = self.fedmid in {
            "fedavg_proto", "topofedproto", "fedkd_proto", "individual_proto"
        }
        proto_tag = ""
        if is_structure_proto:
            proto_tag = "_desc{}".format(getattr(self.args, "proto_descriptor_space", "fingerprint"))
        run_name = "{}_{}_alpha{}_seed{}_split{}_part{}_enc{}{}_{}".format(
            self.fedmid, self.args.dataset, self.args.partition_alpha,
            self.args.seed, getattr(self.args, 'split_seed', self.args.seed),
            getattr(self.args, 'partition_seed', self.args.seed),
            getattr(self.args, 'encoder', 'mpnn'), proto_tag,
            datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        self.csv_logger = CSVResultLogger(
            getattr(self.args, "results_dir", "results"),
            run_name,
            metadata={
                "model_seed": self.args.seed,
                "split_seed": getattr(self.args, 'split_seed', self.args.seed),
                "partition_seed": getattr(self.args, 'partition_seed', self.args.seed),
                "partition_method": getattr(self.args, 'partition_method', 'hetero'),
                "encoder": getattr(self.args, 'encoder', 'mpnn'),
                "descriptor": (
                    getattr(self.args, 'proto_descriptor_space', '')
                    if is_structure_proto else ''
                ),
                "lambda_proto": (
                    getattr(self.args, 'lambda_proto', '')
                    if is_structure_proto else ''
                ),
                "proto_support_tau": (
                    getattr(self.args, 'proto_support_tau', '')
                    if is_structure_proto else ''
                ),
                "proto_weighted_kmeans": (
                    getattr(self.args, 'proto_weighted_kmeans', False)
                    if is_structure_proto else ''
                ),
                "local_clusters": (
                    getattr(self.args, 'topoproto_clusters', '')
                    if is_structure_proto else ''
                ),
                "global_clusters": (
                    getattr(self.args, 'topoproto_global_clusters', '')
                    if is_structure_proto else ''
                ),
            },
        )
        self.prototype_occupancy_path = None
        if self.fedmid == 'fedavg_proto':
            self.prototype_occupancy_path = os.path.join(
                getattr(self.args, "results_dir", "results"),
                run_name + "_prototype_occupancy.csv"
            )
            with open(self.prototype_occupancy_path, "w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow([
                    "time", "dataset", "round", "scope", "client_idx",
                    "requested_k", "realized_k", "samples", "min_count",
                    "p25_count", "median_count", "mean_count", "max_count",
                    "singleton_clusters", "clusters_lt3", "clusters_lt5",
                    "cluster_cache_hit", "cluster_fit_seconds", "collection_seconds",
                    "model_seed", "split_seed", "partition_seed", "encoder", "descriptor"
                ])
        logging.info("Local CSV results will be written to %s", self.csv_logger.path)
        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num
        self.client_list = []
        self.server_cluster_cache = {}
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.model_trainer = model_trainer
        self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer)
        if self.args.dataset in MOLECULAR_CLASSIFICATION_DATASETS:
            self.bestTest = -1e7
            self.bestVal = -1e7
            self.bestQm9EveryTask = []
        else:
            self.bestTest = 1e7
            self.bestVal = 1e7
            self.bestQm9EveryTask = []
            
    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_in_total):
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, model_trainer,
                       global_test_data=self.test_global)  # 传递全局测试集
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    def _exclude_public_indices_from_local_training(self, public_indices):
        public_index_set = {int(idx) for idx in public_indices}
        for client_idx, local_loader in self.train_data_local_dict.items():
            local_dataset = local_loader.dataset
            if hasattr(local_dataset, "indices") and hasattr(local_dataset, "dataset"):
                base_dataset = local_dataset.dataset
                base_indices = [int(idx) for idx in local_dataset.indices]
            else:
                base_dataset = self.train_global.dataset
                base_indices = list(range(len(local_dataset)))

            kept_indices = [idx for idx in base_indices if idx not in public_index_set]
            if len(kept_indices) == len(base_indices):
                continue

            filtered_dataset = Subset(base_dataset, kept_indices)
            filtered_loader = DataLoader(
                filtered_dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=True,
                pin_memory=True,
                collate_fn=collate_molgraphs
            )
            self.train_data_local_dict[client_idx] = filtered_loader
            self.test_data_local_dict[client_idx] = filtered_loader
            self.train_data_local_num_dict[client_idx] = len(kept_indices)
            logging.info(
                "[Client %s] Public samples removed from private training: before=%d, after=%d",
                client_idx, len(base_indices), len(kept_indices)
            )

    def _instanciate_opt(self):
        self.opt = torch.optim.Adam(
            self.model_trainer.model.parameters(), lr=self.wandbConfig.weightFed)
    def train(self):
        if self.fedmid == 'topofedproto':
            global_topology_prototypes = None
            for round_idx in range(self.args.comm_round):
                logging.info(f"########## topofedproto Communication round : {round_idx} ##########")
                client_indexes = self._client_sampling(
                    round_idx, self.args.client_num_in_total, self.args.client_num_per_round
                )

                local_topology_prototypes = []
                for client_idx in client_indexes:
                    client = self.client_list[client_idx]
                    client.update_local_dataset(
                        client_idx,
                        self.train_data_local_dict[client_idx],
                        self.test_data_local_dict[client_idx],
                        self.train_data_local_num_dict[client_idx]
                    )
                    client.train_topofedproto(None, global_topology_prototypes, round_idx, client_idx)
                    local_topology_prototypes.extend(
                        client.collect_topology_prototypes(getattr(self.args, 'topoproto_clusters', 16))
                    )

                global_topology_prototypes = self._aggregate_topology_prototypes(local_topology_prototypes)
                if global_topology_prototypes is not None:
                    logging.info(
                        "Round %d aggregated %d global topology prototypes",
                        round_idx, len(global_topology_prototypes['prototypes'])
                    )
                self._log_prototype_occupancy(
                    round_idx, local_topology_prototypes, global_topology_prototypes
                )

                if round_idx % self.args.frequency_of_the_test == 0:
                    self.validateClients(round_idx, client_indexes)

        elif self.fedmid == 'centralized':
            logging.info(
                "Centralized learning uses fixed steps per round: %d steps",
                self.args.localStepsPerRound
            )
            for round_idx in range(self.args.comm_round):
                logging.info("########## centralized training round : %s ##########", round_idx)
                old_fedmid = self.model_trainer.wandbconfig.fedmid
                self.model_trainer.wandbconfig.fedmid = 'avg'
                try:
                    self.model_trainer.train(self.train_global, self.device, self.args, round_idx, -1)
                finally:
                    self.model_trainer.wandbconfig.fedmid = old_fedmid

                if round_idx % self.args.frequency_of_the_test == 0:
                    self.validateGlobal(round_idx)

        elif self.fedmid == 'moon':
            previous_local_params = {}
            for round_idx in range(self.args.comm_round):
                logging.info("########## MOON Communication round : %s ##########", round_idx)
                w_global = self.model_trainer.get_model_params()
                client_indexes = self._client_sampling(
                    round_idx, self.args.client_num_in_total, self.args.client_num_per_round
                )

                w_locals = []
                for client_idx in client_indexes:
                    client = self.client_list[client_idx]
                    client.update_local_dataset(
                        client_idx,
                        self.train_data_local_dict[client_idx],
                        self.test_data_local_dict[client_idx],
                        self.train_data_local_num_dict[client_idx]
                    )
                    w = client.train_moon(
                        w_global,
                        previous_local_params.get(client_idx),
                        round_idx,
                        client_idx
                    )
                    previous_local_params[client_idx] = copy.deepcopy(w)
                    w_locals.append((client.get_sample_number(), copy.deepcopy(w)))

                w_global = self._aggregate(w_locals)
                self.model_trainer.set_model_params(w_global)
                if round_idx % self.args.frequency_of_the_test == 0:
                    self.validateGlobal(round_idx)

        elif self.fedmid in ['fedproto', 'fpl', 'fedtgp']:
            global_task_prototypes = None
            for round_idx in range(self.args.comm_round):
                logging.info(
                    "########## %s Communication round : %s ##########",
                    self.fedmid, round_idx
                )
                w_global = self.model_trainer.get_model_params()
                client_indexes = self._client_sampling(
                    round_idx, self.args.client_num_in_total, self.args.client_num_per_round
                )

                w_locals = []
                local_task_prototypes = []
                for client_idx in client_indexes:
                    client = self.client_list[client_idx]
                    client.update_local_dataset(
                        client_idx,
                        self.train_data_local_dict[client_idx],
                        self.test_data_local_dict[client_idx],
                        self.train_data_local_num_dict[client_idx]
                    )
                    w = client.train_task_proto(
                        w_global, global_task_prototypes, round_idx, client_idx
                    )
                    w_locals.append((client.get_sample_number(), copy.deepcopy(w)))
                    local_proto = client.collect_task_prototypes()
                    if local_proto is not None:
                        local_task_prototypes.append(local_proto)

                w_global = self._aggregate(w_locals)
                self.model_trainer.set_model_params(w_global)

                if local_task_prototypes:
                    if self.fedmid == 'fpl':
                        global_task_prototypes = self._aggregate_task_prototypes_fpl(
                            local_task_prototypes
                        )
                    elif self.fedmid == 'fedtgp':
                        init_prototypes = self._aggregate_task_prototypes(local_task_prototypes)
                        global_task_prototypes = self._train_global_task_prototypes(
                            local_task_prototypes, init_prototypes
                        )
                    else:
                        global_task_prototypes = self._aggregate_task_prototypes(
                            local_task_prototypes
                        )
                    if global_task_prototypes is not None:
                        logging.info(
                            "Round %d aggregated %s task prototypes with %d observed task/classes",
                            round_idx, self.fedmid,
                            int((global_task_prototypes['counts'] > 0).sum().item())
                        )

                if round_idx % self.args.frequency_of_the_test == 0:
                    self.validateGlobal(round_idx)

        elif self.fedmid in ['fedkd', 'topofedmd', 'topofedrep', 'fedkd_proto']:
            subset_indices = sample_public_data_from_clients(
                self.train_data_local_dict,
                total_sample_size=getattr(self.args, 'public_sample_size', 512)
            )
            self._exclude_public_indices_from_local_training(subset_indices)
            public_subset = Subset(self.train_global.dataset, subset_indices)
            public_loader = DataLoader(public_subset, batch_size=32, shuffle=False, drop_last=False,
                                    collate_fn=collate_molgraphs, num_workers=0)
            global_task_prototypes = None
            global_topology_prototypes = None
            topology = None
            if self.fedmid == 'topofedmd':
                topology = self._build_public_topology(public_loader)

            for round_idx in range(self.args.comm_round):
                logging.info(f"########## {self.fedmid} Communication round : {round_idx} ##########")
                w_global = self.model_trainer.get_model_params()
                client_indexes = self._client_sampling(round_idx, self.args.client_num_in_total,
                                                    self.args.client_num_per_round)

                client_logits = []
                local_task_prototypes = []
                local_topology_prototypes = []
                w_locals = []
                for client_idx in client_indexes:
                    client = self.client_list[client_idx]
                    client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                                self.test_data_local_dict[client_idx],
                                                self.train_data_local_num_dict[client_idx])
                    old_fedmid = client.model_trainer.wandbconfig.fedmid
                    client.model_trainer.wandbconfig.fedmid = 'avg'
                    if getattr(self.args, 'distill_target', 'client') == 'server':
                        start_weights = w_global
                    else:
                        start_weights = client.model_trainer.get_model_params()
                    if self.fedmid == 'fedkd_proto':
                        w = client.train_topofedproto(start_weights, global_topology_prototypes, round_idx, client_idx)
                        local_proto = client.collect_topology_prototypes(getattr(self.args, 'topoproto_clusters', 16))
                        if local_proto is not None:
                            local_topology_prototypes.extend(local_proto)
                    elif getattr(self.args, 'lambda_task_proto', 0.0) > 0:
                        w = client.train_task_proto(start_weights, global_task_prototypes, round_idx, client_idx)
                        local_proto = client.collect_task_prototypes()
                        if local_proto is not None:
                            local_task_prototypes.append(local_proto)
                    else:
                        w = client.train(start_weights, round_idx, client_idx)
                    client.model_trainer.wandbconfig.fedmid = old_fedmid
                    w_locals.append((client.get_sample_number(), copy.deepcopy(w)))
                    if self.fedmid == 'topofedrep':
                        public_preds, public_embeddings, public_smiles = self._predict_public(client, public_loader, return_embeddings=True)
                        client_logits.append((client_idx, client.get_sample_number(), public_preds, public_embeddings, public_smiles))
                    else:
                        public_preds, public_smiles = self._predict_public(client, public_loader)
                        client_logits.append((client_idx, client.get_sample_number(), public_preds, None, public_smiles))

                if self.fedmid == 'topofedrep':
                    selective_weights = None
                    if getattr(self.args, 'teacher_weighting', 'sample') == 'selective':
                        selective_weights = self._compute_selective_teacher_weights(client_logits)
                    elif getattr(self.args, 'teacher_weighting', 'sample') == 'scaffold':
                        selective_weights = self._compute_scaffold_teacher_weights(client_logits)

                    teacher = self._aggregate_public_predictions(
                        [
                            (client_idx, sample_num, predictions, selective_weights.get(client_idx, 1.0) if selective_weights else 1.0)
                            for client_idx, sample_num, predictions, _, _ in client_logits
                        ]
                    )
                    rep_teacher = self._aggregate_public_representations(client_logits, client_weights=selective_weights)
                    feat_teacher = self._aggregate_public_feature_targets(client_logits, client_weights=selective_weights)
                else:
                    teacher = self._aggregate_public_predictions([
                        (client_idx, sample_num, predictions, 1.0)
                        for client_idx, sample_num, predictions, _, _ in client_logits
                    ])
                    rep_teacher = None
                    feat_teacher = None
                if self.fedmid == 'fedkd_proto' and local_topology_prototypes:
                    global_topology_prototypes = self._aggregate_topology_prototypes(local_topology_prototypes)
                    if global_topology_prototypes is not None:
                        logging.info(
                            "Round %d aggregated %d global topology prototypes",
                            round_idx, len(global_topology_prototypes['prototypes'])
                        )
                if local_task_prototypes:
                    global_task_prototypes = self._aggregate_task_prototypes(local_task_prototypes)
                    if global_task_prototypes is not None:
                        logging.info(
                            "Round %d aggregated task prototypes with %d observed task/classes",
                            round_idx, int((global_task_prototypes['counts'] > 0).sum().item())
                        )
                if topology is not None:
                    beta = getattr(self.args, 'topo_beta', 0.3)
                    teacher = (1.0 - beta) * teacher + beta * torch.matmul(topology, teacher)

                entropy = self._teacher_uncertainty(teacher)
                logging.info("Round %d public teacher uncertainty = %.4f", round_idx, entropy)
                if rep_teacher is not None:
                    logging.info("Round %d public SVD representation teacher shape = %s", round_idx, tuple(rep_teacher.shape))

                distill_target = getattr(self.args, 'distill_target', 'client')
                teacher_mode = getattr(self.args, 'teacher_mode', 'global')
                if distill_target == 'server':
                    server_init = getattr(self.args, 'server_init', 'previous')
                    if server_init == 'fedavg':
                        self.model_trainer.set_model_params(self._aggregate(w_locals))
                    else:
                        self.model_trainer.set_model_params(w_global)
                    self._distill_server(public_loader, teacher, rep_teacher, feat_teacher)
                elif self.fedmid == 'topofedrep' and teacher_mode == 'leave_one_out':
                    for client_idx in client_indexes:
                        loo_predictions = [
                            (cid, sample_num, predictions, selective_weights.get(cid, 1.0) if selective_weights else 1.0)
                            for cid, sample_num, predictions, _, _ in client_logits
                            if cid != client_idx
                        ]
                        if not loo_predictions:
                            loo_predictions = [
                                (cid, sample_num, predictions, selective_weights.get(cid, 1.0) if selective_weights else 1.0)
                                for cid, sample_num, predictions, _, _ in client_logits
                            ]
                        client_teacher = self._aggregate_public_predictions(loo_predictions)
                        client_rep_teacher = self._aggregate_public_representations(
                            client_logits, excluded_client_idx=client_idx, client_weights=selective_weights
                        )
                        logging.info(
                            "Round %d client %d leave-one-out teacher uncertainty = %.4f",
                            round_idx, client_idx, self._teacher_uncertainty(client_teacher)
                        )
                        client_feat_teacher = self._aggregate_public_feature_targets(
                            client_logits, excluded_client_idx=client_idx, client_weights=selective_weights
                        )
                        distill_batches = self._make_distill_batches(
                            public_loader, client_teacher, client_rep_teacher, client_feat_teacher
                        )
                        client = self.client_list[client_idx]
                        client.receive_soft_targets(distill_batches)
                        client.distill(round_idx, client_idx)
                else:
                    distill_batches = self._make_distill_batches(public_loader, teacher, rep_teacher, feat_teacher)
                    for client_idx in client_indexes:
                        client = self.client_list[client_idx]
                        client.receive_soft_targets(distill_batches)
                        client.distill(round_idx, client_idx)  # 移除 global_step

                if round_idx % self.args.frequency_of_the_test == 0:
                    if distill_target == 'server':
                        self.validateGlobal(round_idx)
                    else:
                        self.validateClients(round_idx, client_indexes)

        elif self.fedmid == 'opt':
            for round_idx in range(self.args.comm_round):
                w_global = self.model_trainer.get_model_params()
                logging.info("################ Communication round : {}".format(round_idx))

                w_locals = []

                """
                for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
                Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
                """
                client_indexes = self._client_sampling(round_idx, self.args.client_num_in_total,
                                                       self.args.client_num_per_round)
                logging.info("client_indexes = " + str(client_indexes))

                for client_idx in client_indexes:
                    client = self.client_list[client_idx]
                    client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                                self.test_data_local_dict[client_idx],
                                                self.train_data_local_num_dict[client_idx])
                    w = client.train(w_global, round_idx, client_idx)
                    w_locals.append((client.get_sample_number(), copy.deepcopy(w)))
                    #
                # reset weight after standalone simulation
                self.model_trainer.set_model_params(w_global)
                # update global weights
                w_avg = self._aggregate(w_locals)
                # server optimizer
                self.opt.zero_grad()
                opt_state = self.opt.state_dict()
                self._set_model_global_grads(w_avg)
                self._instanciate_opt()
                self.opt.load_state_dict(opt_state)
                self.opt.step()
                if round_idx % self.args.frequency_of_the_test == 0:
                    self.validateGlobal(round_idx)



        elif self.fedmid == 'fedavg_proto':
            global_topology_prototypes = None
            for round_idx in range(self.args.comm_round):
                w_global = self.model_trainer.get_model_params()
                logging.info("########## fedavg_proto Communication round : {} ##########".format(round_idx))

                w_locals = []
                local_topology_prototypes = []

                client_indexes = self._client_sampling(round_idx, self.args.client_num_in_total,
                                                       self.args.client_num_per_round)
                logging.info("client_indexes = " + str(client_indexes))

                for client_idx in client_indexes:
                    client = self.client_list[client_idx]
                    client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                                self.test_data_local_dict[client_idx],
                                                self.train_data_local_num_dict[client_idx])
                    
                    w = client.train_topofedproto(w_global, global_topology_prototypes, round_idx, client_idx)
                    local_proto = client.collect_topology_prototypes(getattr(self.args, 'topoproto_clusters', 16))
                    if local_proto is not None:
                        local_topology_prototypes.extend(local_proto)
                        
                    w_locals.append((client.get_sample_number(), copy.deepcopy(w)))

                w_global = self._aggregate(w_locals)
                self.model_trainer.set_model_params(w_global)
                
                global_topology_prototypes = self._aggregate_topology_prototypes(local_topology_prototypes)
                if global_topology_prototypes is not None:
                    logging.info(
                        "Round %d aggregated %d global topology prototypes",
                        round_idx, len(global_topology_prototypes['prototypes'])
                    )
                self._log_prototype_occupancy(
                    round_idx, local_topology_prototypes, global_topology_prototypes
                )

                if round_idx % self.args.frequency_of_the_test == 0:
                    server_global_eval = (
                        getattr(self.args, "global_only_eval", False)
                        or getattr(self.args, "proto_validate_global", False)
                    )
                    if server_global_eval:
                        self.validateGlobal(round_idx)
                    if not server_global_eval and not getattr(self.args, "skip_client_eval", False):
                        self.validateClients(round_idx, client_indexes)

        elif self.fedmid in ['individual', 'individual_proto']:
            client_weights = {
                client_idx: copy.deepcopy(self.model_trainer.get_model_params())
                for client_idx in range(self.args.client_num_in_total)
            }
            client_topology_prototypes = {
                client_idx: None for client_idx in range(self.args.client_num_in_total)
            }

            for round_idx in range(self.args.comm_round):
                logging.info("########## %s local-only round : %s ##########", self.fedmid, round_idx)
                client_indexes = self._client_sampling(
                    round_idx, self.args.client_num_in_total, self.args.client_num_per_round
                )
                logging.info("client_indexes = " + str(client_indexes))

                for client_idx in client_indexes:
                    client = self.client_list[client_idx]
                    client.update_local_dataset(
                        client_idx,
                        self.train_data_local_dict[client_idx],
                        self.test_data_local_dict[client_idx],
                        self.train_data_local_num_dict[client_idx]
                    )
                    if self.fedmid == 'individual_proto':
                        w = client.train_topofedproto(
                            client_weights[client_idx],
                            client_topology_prototypes[client_idx],
                            round_idx,
                            client_idx
                        )
                        local_proto = client.collect_topology_prototypes(
                            getattr(self.args, 'topoproto_clusters', 16)
                        )
                        client_topology_prototypes[client_idx] = self._pack_topology_prototypes(local_proto)
                    else:
                        old_fedmid = client.model_trainer.wandbconfig.fedmid
                        client.model_trainer.wandbconfig.fedmid = 'avg'
                        try:
                            w = client.train(client_weights[client_idx], round_idx, client_idx)
                        finally:
                            client.model_trainer.wandbconfig.fedmid = old_fedmid
                    client_weights[client_idx] = copy.deepcopy(w)

                if round_idx % self.args.frequency_of_the_test == 0:
                    self.validateClients(round_idx, client_indexes)

        else:
            
            for round_idx in range(self.args.comm_round):
                w_global = self.model_trainer.get_model_params()
                logging.info("################Communication round : {}".format(round_idx))

                w_locals = []

                """
                for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
                Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
                """
                client_indexes = self._client_sampling(round_idx, self.args.client_num_in_total,
                                                       self.args.client_num_per_round)
                logging.info("client_indexes = " + str(client_indexes))

                for client_idx in client_indexes:
                    client = self.client_list[client_idx]
                    client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                                self.test_data_local_dict[client_idx],
                                                self.train_data_local_num_dict[client_idx])
                    w = client.train(w_global, round_idx, client_idx)
                    w_locals.append((client.get_sample_number(), copy.deepcopy(w)))
                    #
                w_global = self._aggregate(w_locals)
                self.model_trainer.set_model_params(w_global)

                # test results
                # at last round
                if round_idx % self.args.frequency_of_the_test == 0:
                    self.validateGlobal(round_idx)
                    if getattr(self.args, "fedavg_validate_clients", False):
                        self.validateClients(round_idx, client_indexes)
                    
    def _set_model_global_grads(self, new_state):
        new_model = copy.deepcopy(self.model_trainer.model)
        new_model.load_state_dict(new_state)
        with torch.no_grad():
            for parameter, new_parameter in zip(
                self.model_trainer.model.parameters(), new_model.parameters()
            ):
                parameter.grad = parameter.data - new_parameter.data
                # because we go to the opposite direction of the gradient
        model_state_dict = self.model_trainer.model.state_dict()
        new_model_state_dict = new_model.state_dict()
        for k in dict(self.model_trainer.model.named_parameters()).keys():
            new_model_state_dict[k] = model_state_dict[k]
        self.model_trainer.set_model_params(new_model_state_dict)

    def _predict_public(self, client, public_loader, return_embeddings=False):
        model = client.model_trainer.model
        model.to(self.device)
        model.eval()
        predictions = []
        embeddings_list = []
        all_smiles = []
        with torch.no_grad():
            for batch in public_loader:
                smiles, bg, labels, masks, indices = batch
                bg = bg.to(self.device)
                node_feats = bg.ndata['h'].to(self.device)
                edge_feats = bg.edata.get('e', None)
                if edge_feats is not None:
                    edge_feats = edge_feats.to(self.device)
                logits, embeddings = model(bg, node_feats, edge_feats)
                if logits.dim() == 1:
                    logits = logits.view(bg.batch_size, -1)
                if self.args.dataset in MOLECULAR_CLASSIFICATION_DATASETS:
                    pred = torch.sigmoid(logits / getattr(self.args, 'temperature', 1.0))
                else:
                    pred = logits
                predictions.append(pred.detach().cpu())
                if return_embeddings:
                    embeddings_list.append(embeddings.detach().cpu())
                all_smiles.extend(smiles)
        if return_embeddings:
            return torch.cat(predictions, dim=0), torch.cat(embeddings_list, dim=0), all_smiles
        return torch.cat(predictions, dim=0), all_smiles

    def _aggregate_public_predictions(self, client_predictions):
        if not client_predictions:
            raise ValueError("client_predictions must not be empty")
        if (
            getattr(self.args, 'logit_teacher_mode', 'avg') == 'selective_confidence' and
            self.args.dataset in MOLECULAR_CLASSIFICATION_DATASETS
        ):
            weighted_sum = None
            total_weight = None
            sample_rho = float(getattr(self.args, 'teacher_sample_rho', 1.0))
            gamma = max(float(getattr(self.args, 'selective_logit_gamma', 2.0)), 0.0)
            floor = max(float(getattr(self.args, 'selective_logit_floor', 0.02)), 0.0)
            for item in client_predictions:
                if len(item) == 2:
                    sample_num, predictions = item
                    reliability = 1.0
                elif len(item) == 3:
                    sample_num, predictions, reliability = item
                elif len(item) == 4:
                    client_idx, sample_num, predictions, reliability = item
                elif len(item) == 5:
                    client_idx, sample_num, predictions, _, _ = item
                    reliability = 1.0
                else:
                    raise ValueError("client_predictions items must be formatted correctly")
                confidence = (predictions.float() - 0.5).abs().mul(2.0).clamp(0.0, 1.0)
                confidence = confidence.pow(gamma)
                weight = confidence + floor
                weight = weight * (float(sample_num) ** sample_rho) * float(reliability)
                weighted = predictions.float() * weight
                weighted_sum = weighted if weighted_sum is None else weighted_sum + weighted
                total_weight = weight if total_weight is None else total_weight + weight
            teacher = weighted_sum / total_weight.clamp_min(1e-12)
            teacher = teacher.clamp(1e-6, 1 - 1e-6)
            logging.info(
                "Selective logit teacher: gamma=%.3f, floor=%.3f, mean_conf_weight=%.4f",
                gamma, floor, float((total_weight / max(1, len(client_predictions))).mean().item())
            )
            return teacher
        weighted_predictions = []
        total_weight = 0.0
        for item in client_predictions:
            if len(item) == 2:
                sample_num, predictions = item
                reliability = 1.0
            elif len(item) == 3:
                sample_num, predictions, reliability = item
            elif len(item) == 4:
                client_idx, sample_num, predictions, reliability = item
            elif len(item) == 5:
                client_idx, sample_num, predictions, _, _ = item
                reliability = 1.0
            else:
                raise ValueError("client_predictions items must be formatted correctly")
            sample_rho = float(getattr(self.args, 'teacher_sample_rho', 1.0))
            weight = float(sample_num) ** sample_rho * float(reliability)
            weighted_predictions.append((weight, predictions))
            total_weight += weight
        if total_weight <= 0:
            total_weight = float(len(weighted_predictions))
            weighted_predictions = [(1.0, predictions) for _, predictions in weighted_predictions]
        teacher = None
        for weight_value, predictions in weighted_predictions:
            weight = weight_value / total_weight
            weighted = predictions * weight
            teacher = weighted if teacher is None else teacher + weighted
        if self.args.dataset in MOLECULAR_CLASSIFICATION_DATASETS:
            teacher = teacher.clamp(1e-6, 1 - 1e-6)
        return teacher

    def _public_svd_coordinates(self, embeddings):
        rank = max(1, getattr(self.args, 'svd_rank', 16))
        eta = getattr(self.args, 'svd_eta', 0.5)
        centered = embeddings.float() - embeddings.float().mean(dim=0, keepdim=True)
        u, s, _ = torch.linalg.svd(centered, full_matrices=False)
        rank = min(rank, u.shape[1], s.shape[0])
        coords = u[:, :rank] * torch.pow(s[:rank].clamp_min(1e-12), eta).unsqueeze(0)
        return coords

    def _align_public_coordinates(self, coords, reference):
        if coords.shape[1] != reference.shape[1]:
            dim = min(coords.shape[1], reference.shape[1])
            coords = coords[:, :dim]
            reference = reference[:, :dim]
        matrix = coords.t().matmul(reference)
        u, _, vh = torch.linalg.svd(matrix, full_matrices=False)
        return coords.matmul(u.matmul(vh))

    def _compute_selective_teacher_weights(self, client_predictions, excluded_client_idx=None):
        selected = [
            item for item in client_predictions
            if excluded_client_idx is None or item[0] != excluded_client_idx
        ]
        if not selected:
            selected = client_predictions
        if len(selected) == 1:
            return {selected[0][0]: 1.0}

        eps = 1e-12
        topo_weight = float(getattr(self.args, 'selective_topo_weight', 0.5))
        topo_weight = min(1.0, max(0.0, topo_weight))
        logit_weight = 1.0 - topo_weight
        tau = max(float(getattr(self.args, 'selective_tau', 1.0)), eps)
        sample_rho = float(getattr(self.args, 'selective_sample_rho', 1.0))
        min_weight = float(getattr(self.args, 'selective_min_weight', 0.05))
        min_weight = min(1.0, max(0.0, min_weight))

        client_ids = [item[0] for item in selected]
        sample_nums = torch.tensor([float(item[1]) for item in selected], dtype=torch.float32)
        predictions = [item[2].float() for item in selected]
        coords = [self._public_svd_coordinates(item[3]).float() for item in selected]
        relations = []
        for coord in coords:
            normalized = F.normalize(coord, p=2, dim=1)
            relations.append(normalized.matmul(normalized.t()))

        num_clients = len(selected)
        logit_dist = torch.zeros(num_clients, num_clients)
        topo_dist = torch.zeros(num_clients, num_clients)
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                logit_value = torch.mean((predictions[i] - predictions[j]) ** 2)
                topo_value = torch.mean((relations[i] - relations[j]) ** 2)
                logit_dist[i, j] = logit_dist[j, i] = logit_value
                topo_dist[i, j] = topo_dist[j, i] = topo_value

        def normalize_distance(matrix):
            off_diag = matrix[matrix > 0]
            if off_diag.numel() == 0:
                return matrix
            return matrix / off_diag.mean().clamp_min(eps)

        combined = (
            logit_weight * normalize_distance(logit_dist) +
            topo_weight * normalize_distance(topo_dist)
        )
        kernel = torch.exp(-combined / tau)
        kernel.fill_diagonal_(0.0)
        density = kernel.sum(dim=1) / max(1, num_clients - 1)
        density = min_weight + (1.0 - min_weight) * density
        reliability = sample_nums.clamp_min(1.0).pow(sample_rho - 1.0) * density
        raw_weights = sample_nums * reliability
        normalized_weights = raw_weights / raw_weights.sum().clamp_min(eps)
        result = {
            client_id: float(weight.item())
            for client_id, weight in zip(client_ids, reliability)
        }
        logging.info(
            "Selective teacher weights: %s",
            {client_id: round(float(weight.item()), 4) for client_id, weight in zip(client_ids, normalized_weights)}
        )
        logging.info(
            "Selective density scores: %s",
            {client_id: round(float(score.item()), 4) for client_id, score in zip(client_ids, density)}
        )
        return result

    def _compute_scaffold_teacher_weights(self, client_logits):
        import numpy as np
        from client import get_scaffolds
        
        selected = client_logits
        client_ids = [item[0] for item in selected]
        smiles_list = selected[0][4]  # all clients have same public smiles
        
        public_scaffolds = [list(get_scaffolds([s])) for s in smiles_list]
        public_scaffolds = [s[0] if len(s) > 0 else "" for s in public_scaffolds]
        
        weights = {}
        for client_idx, sample_num, predictions, _, _ in selected:
            client = self.client_list[client_idx]
            vocab = client.scaffold_vocab
            
            # Count how many public molecules have scaffolds in the client's local vocabulary
            overlap_count = sum(1 for s in public_scaffolds if s in vocab and s != "")
            
            # Simple weighting: base density is overlap ratio
            ratio = overlap_count / max(1, len(public_scaffolds))
            min_weight = float(getattr(self.args, 'selective_min_weight', 0.05))
            weight = min_weight + (1.0 - min_weight) * ratio
            
            weights[client_idx] = float(weight)
        
        return weights

    def _aggregate_public_representations(self, client_predictions, excluded_client_idx=None, client_weights=None):
        selected = [
            item for item in client_predictions
            if excluded_client_idx is None or item[0] != excluded_client_idx
        ]
        if not selected:
            selected = client_predictions
        total_weight = 0.0
        representations = []
        weights = []
        sample_rho = float(getattr(self.args, 'teacher_sample_rho', 1.0))
        teacher_space = getattr(self.args, 'rep_teacher_space', 'svd')
        for client_idx, sample_num, _, embeddings, _ in selected:
            if teacher_space == 'raw_relation':
                normalized = F.normalize(embeddings.float(), p=2, dim=1)
                representations.append(normalized.matmul(normalized.t()))
            else:
                representations.append(self._public_svd_coordinates(embeddings))
            reliability = 1.0 if client_weights is None else client_weights.get(client_idx, 1.0)
            weight = float(sample_num) ** sample_rho * float(reliability)
            weights.append(weight)
            total_weight += weight
        if total_weight <= 0:
            total_weight = float(len(weights))
            weights = [1.0 for _ in weights]
        weights = [weight / total_weight for weight in weights]
        reference = representations[0]
        teacher = None
        for weight, representation in zip(weights, representations):
            if teacher_space == 'raw_relation':
                aligned = representation
            else:
                aligned = representation if teacher is None else self._align_public_coordinates(representation, reference)
            weighted = aligned * weight
            teacher = weighted if teacher is None else teacher + weighted
        return teacher

    def _aggregate_public_feature_targets(self, client_predictions, excluded_client_idx=None, client_weights=None):
        selected = [
            item for item in client_predictions
            if excluded_client_idx is None or item[0] != excluded_client_idx
        ]
        if not selected:
            selected = client_predictions

        sample_rho = float(getattr(self.args, 'teacher_sample_rho', 1.0))
        weighted_embeddings = []
        total_weight = 0.0
        for client_idx, sample_num, _, embeddings, _ in selected:
            reliability = 1.0 if client_weights is None else client_weights.get(client_idx, 1.0)
            weight = float(sample_num) ** sample_rho * float(reliability)
            weighted_embeddings.append((weight, embeddings.float()))
            total_weight += weight

        if total_weight <= 0:
            total_weight = float(len(weighted_embeddings))
            weighted_embeddings = [(1.0, embeddings) for _, embeddings in weighted_embeddings]

        reference = weighted_embeddings[0][1]
        teacher = None
        for weight_value, embeddings in weighted_embeddings:
            weight = weight_value / total_weight
            aligned = embeddings if teacher is None else self._align_public_coordinates(embeddings, reference)
            weighted = aligned * weight
            teacher = weighted if teacher is None else teacher + weighted
        return teacher

    def _make_distill_batches(self, public_loader, teacher, rep_teacher=None, feat_teacher=None):
        distill_batches = []
        start = 0
        for batch in public_loader:
            smiles, bg, labels, masks, indices = batch
            batch_size = labels.shape[0]
            soft_labels = teacher[start:start + batch_size]
            if rep_teacher is None:
                rep_labels = None
            elif rep_teacher.dim() == 2 and rep_teacher.shape[0] == rep_teacher.shape[1] == teacher.shape[0]:
                rep_labels = rep_teacher[start:start + batch_size, start:start + batch_size]
            else:
                rep_labels = rep_teacher[start:start + batch_size]
            feat_labels = None if feat_teacher is None else feat_teacher[start:start + batch_size]
            start += batch_size
            if rep_labels is None and feat_labels is None:
                distill_batches.append((bg, soft_labels, labels, masks))
            elif feat_labels is None:
                distill_batches.append((bg, soft_labels, labels, masks, rep_labels))
            else:
                distill_batches.append((bg, soft_labels, labels, masks, rep_labels, feat_labels))
        return distill_batches

    def _distill_server(self, public_loader, teacher, rep_teacher=None, feat_teacher=None):
        model = self.model_trainer.model
        model.to(self.device)
        model.train()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=getattr(self.args, 'server_distill_lr', 1e-4),
            weight_decay=1e-4
        )
        temperature = getattr(self.args, 'temperature', 1.0)
        alpha_kd = getattr(self.args, 'alpha_kd', 0.7)
        lambda_rep = getattr(self.args, 'lambda_rep', 0.1)
        lambda_feat = getattr(self.args, 'lambda_feat', 0.0)
        epochs = max(1, getattr(self.args, 'server_distill_epochs', 1))
        is_regression = self.args.dataset in MOLECULAR_REGRESSION_DATASETS

        total_loss = 0.0
        num_batches = 0
        for _ in range(epochs):
            start = 0
            for batch in public_loader:
                smiles, bg, labels, masks, indices = batch
                batch_size = labels.shape[0]
                soft_target = teacher[start:start + batch_size].to(self.device)
                rep_target = None if rep_teacher is None else rep_teacher[start:start + batch_size].to(self.device)
                feat_target = None if feat_teacher is None else feat_teacher[start:start + batch_size].to(self.device)
                start += batch_size

                bg = bg.to(self.device)
                node_feats = bg.ndata['h'].to(self.device)
                edge_feats = bg.edata.get('e', None)
                
                if getattr(self.args, 'distill_node_dropout', 0.0) > 0:
                    drop_rate = getattr(self.args, 'distill_node_dropout', 0.0)
                    mask = torch.empty((node_feats.size(0), 1), device=self.device).bernoulli_(1 - drop_rate)
                    node_feats = node_feats * mask / (1 - drop_rate)
                if edge_feats is not None:
                    edge_feats = edge_feats.to(self.device)

                optimizer.zero_grad()
                logits, embeddings = model(bg, node_feats, edge_feats)
                if soft_target.shape != logits.shape:
                    soft_target = soft_target.view_as(logits)

                if is_regression:
                    loss_kd = F.mse_loss(logits, soft_target)
                else:
                    teacher_probs = soft_target.clamp(1e-6, 1 - 1e-6)
                    loss_kd = F.binary_cross_entropy_with_logits(logits / temperature, teacher_probs)

                loss_rep = torch.tensor(0.0, device=self.device)
                if rep_target is not None and embeddings.shape[0] > 1:
                    tau = max(getattr(self.args, 'rep_tau', 0.2), 1e-6)
                    student_sim = torch.matmul(
                        F.normalize(embeddings, p=2, dim=1),
                        F.normalize(embeddings, p=2, dim=1).t()
                    ) / tau
                    if rep_target.dim() == 2 and rep_target.shape[0] == rep_target.shape[1] == embeddings.shape[0]:
                        teacher_sim = rep_target / tau
                    else:
                        teacher_sim = torch.matmul(
                            F.normalize(rep_target, p=2, dim=1),
                            F.normalize(rep_target, p=2, dim=1).t()
                        ) / tau
                    rep_top_k = int(getattr(self.args, 'rep_top_k', 0))
                    if rep_top_k > 0 and embeddings.shape[0] > 2:
                        masked_teacher = teacher_sim.detach().clone()
                        masked_teacher.fill_diagonal_(-1e9)
                        top_k = min(rep_top_k, embeddings.shape[0] - 1)
                        _, top_indices = torch.topk(masked_teacher, k=top_k, dim=1)
                        pair_mask = torch.zeros_like(teacher_sim, dtype=torch.bool)
                        pair_mask.scatter_(1, top_indices, True)
                        teacher_sim_for_loss = teacher_sim.detach().masked_fill(~pair_mask, -1e9)
                        student_sim_for_loss = student_sim.masked_fill(~pair_mask, -1e9)
                    else:
                        teacher_sim_for_loss = teacher_sim.detach()
                        student_sim_for_loss = student_sim
                    teacher_rel = F.softmax(teacher_sim_for_loss, dim=1)
                    student_log_rel = F.log_softmax(student_sim_for_loss, dim=1)
                    loss_rep = F.kl_div(student_log_rel, teacher_rel, reduction='batchmean')

                loss_feat = torch.tensor(0.0, device=self.device)
                if feat_target is not None and feat_target.shape == embeddings.shape:
                    loss_feat = 1.0 - F.cosine_similarity(
                        F.normalize(embeddings, p=2, dim=1),
                        F.normalize(feat_target.detach(), p=2, dim=1),
                        dim=1
                    ).mean()

                loss = alpha_kd * loss_kd + lambda_rep * loss_rep + lambda_feat * loss_feat
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        self.model_trainer.set_model_params(model.state_dict())
        if num_batches > 0:
            logging.info("Server distillation finished: loss=%.6f, batches=%d", total_loss / num_batches, num_batches)

    def _teacher_uncertainty(self, teacher):
        if self.args.dataset in MOLECULAR_CLASSIFICATION_DATASETS:
            p = teacher.clamp(1e-6, 1 - 1e-6)
            entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
            return entropy.mean().item()
        return teacher.std().item()

    def _build_public_topology(self, public_loader):
        smiles_list = []
        for batch in public_loader:
            smiles, bg, labels, masks, indices = batch
            smiles_list.extend(smiles)
        num_public = len(smiles_list)
        try:
            from rdkit import Chem, DataStructs
            from rdkit.Chem import AllChem
        except ImportError:
            logging.warning("RDKit is not available; using identity topology for topofedmd.")
            return torch.eye(num_public)

        fps = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) if mol is not None else None)

        topology = torch.zeros(num_public, num_public, dtype=torch.float32)
        for i, fp in enumerate(fps):
            if fp is None:
                topology[i, i] = 1.0
                continue
            valid = [(j, other) for j, other in enumerate(fps) if other is not None]
            sims = DataStructs.BulkTanimotoSimilarity(fp, [other for _, other in valid])
            for (j, _), sim in zip(valid, sims):
                topology[i, j] = float(sim)
            topology[i, i] = 1.0

        top_k = getattr(self.args, 'topo_top_k', 20)
        if top_k and 0 < top_k < num_public:
            values, indices = torch.topk(topology, k=top_k, dim=1)
            sparse_topology = torch.zeros_like(topology)
            sparse_topology.scatter_(1, indices, values)
            topology = sparse_topology
        row_sums = topology.sum(dim=1, keepdim=True).clamp_min(1e-12)
        topology = topology / row_sums
        logging.info("Built Morgan/Tanimoto public topology: molecules=%d, top_k=%s, beta=%.3f",
                     num_public, top_k, getattr(self.args, 'topo_beta', 0.3))
        return topology

    def _aggregate_topology_prototypes(self, local_prototypes):
        aggregation_start = time.perf_counter()
        if not local_prototypes:
            logging.warning("No local topology prototypes were uploaded.")
            return None

        descriptors = torch.stack([item['descriptor'] for item in local_prototypes]).float()
        embeddings = torch.stack([item['prototype'] for item in local_prototypes]).float()
        counts = torch.tensor([item['count'] for item in local_prototypes], dtype=torch.float32)

        global_clusters = min(
            max(1, getattr(self.args, 'topoproto_global_clusters', 16)),
            descriptors.shape[0]
        )
        descriptor_array = descriptors.numpy()
        descriptor_signature = hashlib.sha256(descriptor_array.tobytes()).hexdigest()
        cache_key = (global_clusters, descriptor_signature)
        cache_hit = cache_key in self.server_cluster_cache
        cluster_fit_seconds = 0.0
        if cache_hit:
            assignments = self.server_cluster_cache[cache_key]
        elif global_clusters == 1:
            assignments = np.zeros(descriptors.shape[0], dtype=np.int64)
        else:
            kmeans = KMeans(
                n_clusters=global_clusters,
                random_state=int(getattr(self.args, 'partition_seed', self.args.seed)),
                n_init=10
            )
            cluster_fit_start = time.perf_counter()
            sample_weight = (
                counts.numpy()
                if getattr(self.args, 'proto_weighted_kmeans', False)
                else None
            )
            assignments = kmeans.fit_predict(
                descriptor_array,
                sample_weight=sample_weight,
            )
            cluster_fit_seconds = time.perf_counter() - cluster_fit_start
        if not cache_hit:
            self.server_cluster_cache[cache_key] = assignments.copy()
        logging.info(
            "Server descriptor clustering: requested=%d, uploaded=%d, cache_hit=%s, weighted=%s",
            global_clusters, descriptors.shape[0], cache_hit,
            bool(getattr(self.args, 'proto_weighted_kmeans', False))
        )

        global_embeddings = []
        global_descriptors = []
        global_counts = []
        for cluster_id in range(global_clusters):
            mask_np = assignments == cluster_id
            if not mask_np.any():
                continue
            mask = torch.from_numpy(mask_np)
            weights = counts[mask]
            weights = weights / weights.sum().clamp_min(1.0)
            global_embeddings.append((embeddings[mask] * weights[:, None]).sum(dim=0))
            global_descriptors.append((descriptors[mask] * weights[:, None]).sum(dim=0))
            global_counts.append(int(counts[mask].sum().item()))

        if not global_embeddings:
            return None

        return {
            'prototypes': torch.stack(global_embeddings).cpu(),
            'descriptors': torch.stack(global_descriptors).cpu(),
            'counts': torch.tensor(global_counts, dtype=torch.float32),
            'cluster_cache_hit': bool(cache_hit),
            'cluster_fit_seconds': float(cluster_fit_seconds),
            'collection_seconds': float(time.perf_counter() - aggregation_start),
        }

    def _pack_topology_prototypes(self, local_prototypes):
        if not local_prototypes:
            return None
        return {
            'prototypes': torch.stack([item['prototype'] for item in local_prototypes]).float().cpu(),
            'descriptors': torch.stack([item['descriptor'] for item in local_prototypes]).float().cpu(),
            'counts': torch.tensor([item['count'] for item in local_prototypes], dtype=torch.float32)
        }

    def _log_prototype_occupancy(self, round_idx, local_prototypes, global_prototypes):
        if self.prototype_occupancy_path is None:
            return

        grouped = {}
        for item in local_prototypes or []:
            client_idx = int(item.get('client_idx', -1))
            grouped.setdefault(client_idx, []).append(item)

        rows = []
        for client_idx, items in sorted(grouped.items()):
            counts = [float(item['count']) for item in items]
            metadata = items[0]
            requested_k = int(getattr(self.args, 'topoproto_clusters', len(counts)))
            rows.append(self._prototype_occupancy_row(
                round_idx, 'client', client_idx, requested_k, counts,
                metadata.get('cluster_cache_hit', ''),
                metadata.get('cluster_fit_seconds', ''),
                metadata.get('collection_seconds', '')
            ))

        if global_prototypes is not None:
            global_counts = global_prototypes['counts'].detach().cpu().numpy().astype(float).tolist()
            requested_k = int(getattr(self.args, 'topoproto_global_clusters', len(global_counts)))
            rows.append(self._prototype_occupancy_row(
                round_idx, 'server', 'server', requested_k, global_counts,
                global_prototypes.get('cluster_cache_hit', ''),
                global_prototypes.get('cluster_fit_seconds', ''),
                global_prototypes.get('collection_seconds', '')
            ))

        if rows:
            with open(self.prototype_occupancy_path, 'a', newline='') as stream:
                csv.writer(stream).writerows(rows)

    def _prototype_occupancy_row(
        self, round_idx, scope, client_idx, requested_k, counts,
        cluster_cache_hit='', cluster_fit_seconds='', collection_seconds=''
    ):
        counts_array = np.asarray(counts, dtype=np.float64)
        if counts_array.size == 0:
            counts_array = np.asarray([0.0], dtype=np.float64)
        return [
            datetime.now().isoformat(timespec='seconds'),
            self.args.dataset,
            round_idx,
            scope,
            client_idx,
            requested_k,
            int(len(counts)),
            int(counts_array.sum()),
            float(counts_array.min()),
            float(np.percentile(counts_array, 25)),
            float(np.median(counts_array)),
            float(counts_array.mean()),
            float(counts_array.max()),
            int((counts_array == 1).sum()),
            int((counts_array < 3).sum()),
            int((counts_array < 5).sum()),
            cluster_cache_hit,
            cluster_fit_seconds,
            collection_seconds,
            self.args.seed,
            getattr(self.args, 'split_seed', self.args.seed),
            getattr(self.args, 'partition_seed', self.args.seed),
            getattr(self.args, 'encoder', 'mpnn'),
            getattr(self.args, 'proto_descriptor_space', 'fingerprint'),
        ]

    def _aggregate_task_prototypes(self, local_task_prototypes):
        if not local_task_prototypes:
            logging.warning("No local task prototypes were uploaded.")
            return None
        total_sums = None
        total_counts = None
        for item in local_task_prototypes:
            sums = item['sums'].float()
            counts = item['counts'].float()
            total_sums = sums if total_sums is None else total_sums + sums
            total_counts = counts if total_counts is None else total_counts + counts
        prototypes = total_sums / total_counts.clamp_min(1.0).unsqueeze(-1)
        prototypes = torch.where(
            total_counts.unsqueeze(-1) > 0,
            prototypes,
            torch.zeros_like(prototypes)
        )
        return {
            'prototypes': prototypes.cpu(),
            'counts': total_counts.cpu()
        }

    def _aggregate_task_prototypes_fpl(self, local_task_prototypes):
        if not local_task_prototypes:
            logging.warning("No local task prototypes were uploaded.")
            return None
        proto_sum = None
        client_votes = None
        total_counts = None
        for item in local_task_prototypes:
            sums = item['sums'].float()
            counts = item['counts'].float()
            local_proto = sums / counts.clamp_min(1.0).unsqueeze(-1)
            observed = counts > 0
            masked_proto = local_proto * observed.unsqueeze(-1).float()
            proto_sum = masked_proto if proto_sum is None else proto_sum + masked_proto
            client_votes = observed.float() if client_votes is None else client_votes + observed.float()
            total_counts = counts if total_counts is None else total_counts + counts
        prototypes = proto_sum / client_votes.clamp_min(1.0).unsqueeze(-1)
        prototypes = torch.where(
            client_votes.unsqueeze(-1) > 0,
            prototypes,
            torch.zeros_like(prototypes)
        )
        return {
            'prototypes': prototypes.cpu(),
            'counts': total_counts.cpu()
        }

    def _train_global_task_prototypes(self, local_task_prototypes, init_prototypes):
        if init_prototypes is None:
            return None
        init = init_prototypes['prototypes'].float().to(self.device)
        counts = init_prototypes['counts'].float().to(self.device)
        if init.numel() == 0:
            return init_prototypes

        train_task_indices = []
        train_class_indices = []
        train_embeddings = []
        train_weights = []
        for item in local_task_prototypes:
            sums = item['sums'].float()
            local_counts = item['counts'].float()
            local_proto = sums / local_counts.clamp_min(1.0).unsqueeze(-1)
            for task_idx in range(local_counts.shape[0]):
                for class_idx in range(local_counts.shape[1]):
                    count = float(local_counts[task_idx, class_idx].item())
                    if count <= 0:
                        continue
                    train_task_indices.append(task_idx)
                    train_class_indices.append(class_idx)
                    train_embeddings.append(local_proto[task_idx, class_idx])
                    train_weights.append(max(1.0, np.log1p(count)))
        if not train_embeddings:
            return init_prototypes

        task_indices = torch.tensor(
            train_task_indices, dtype=torch.long, device=self.device
        )
        class_indices = torch.tensor(
            train_class_indices, dtype=torch.long, device=self.device
        )
        local_embeddings = F.normalize(
            torch.stack(train_embeddings).to(self.device), p=2, dim=1
        )
        item_weights = torch.tensor(
            train_weights, dtype=torch.float32, device=self.device
        )

        prototypes = torch.nn.Parameter(init.clone())
        optimizer = torch.optim.Adam(
            [prototypes], lr=float(getattr(self.args, 'fedtgp_lr', 0.01))
        )
        tau = max(float(getattr(self.args, 'fedtgp_tau', 0.2)), 1e-6)
        steps = max(1, int(getattr(self.args, 'fedtgp_steps', 20)))
        init_detached = init.detach()
        for _ in range(steps):
            optimizer.zero_grad()
            task_prototypes = F.normalize(prototypes[task_indices], p=2, dim=2)
            logits = torch.einsum(
                'nd,ncd->nc', local_embeddings, task_prototypes
            ) / tau
            item_losses = F.cross_entropy(logits, class_indices, reduction='none')
            weighted_loss = (item_losses * item_weights).sum() / item_weights.sum().clamp_min(1.0)
            anchor_loss = F.mse_loss(prototypes, init_detached)
            loss = weighted_loss + 0.01 * anchor_loss
            loss.backward()
            optimizer.step()

        trained = prototypes.detach()
        trained = torch.where(
            counts.unsqueeze(-1) > 0,
            trained,
            torch.zeros_like(trained)
        )
        logging.info(
            "FedTGP server prototype training finished: items=%d, steps=%d, tau=%.4f",
            len(train_embeddings), steps, tau
        )
        return {
            'prototypes': trained.cpu(),
            'counts': counts.cpu()
        }
                    
    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params
   
    def validateClients(self, round_idx, client_indexes):
        logging.info(f"########## Evaluating Clients at Round {round_idx} ##########")
        if getattr(self.args, "eval_all_clients", False):
            client_indexes = list(range(self.args.client_num_in_total))
        local_results = []
        global_results = []
        for client_idx in client_indexes:
            client = self.client_list[client_idx]
            metrics = {} if getattr(self.args, "global_only_eval", False) else client.local_test(b_use_test_dataset=True)
            global_metrics= client.global_test()
            if self.args.dataset in MOLECULAR_REGRESSION_DATASETS:
                metric_name = 'rmse'
                val_result = metrics.get('rmse', float('nan'))
                global_result=global_metrics.get('rmse', float('inf'))
            elif self.args.dataset in MOLECULAR_CLASSIFICATION_DATASETS:
                metric_name = 'auc'
                val_result = metrics.get('roc_auc_score', float('nan'))
                global_result=global_metrics.get('roc_auc_score', -float('inf'))
            elif self.args.dataset == 'qm9':
                metric_name = 'mae'
                val_result = metrics.get('mae', float('nan'))
                global_result=global_metrics.get('mae', float('inf'))
            else:
                metric_name = 'unknown'
                val_result = metrics.get('loss', float('nan'))
                global_result=global_metrics.get('loss', float('inf'))

            if not np.isnan(val_result):
                local_results.append(val_result)
            global_results.append(global_result)
            self.csv_logger.log({
                "time": datetime.now().isoformat(timespec="seconds"),
                "method": self.fedmid,
                "dataset": self.args.dataset,
                "round": round_idx,
                "scope": "client",
                "client_idx": client_idx,
                "metric": metric_name,
                "local_metric": val_result,
                "global_metric": global_result,
                "client_indexes": list(client_indexes)
            })
            if not np.isnan(val_result):
                logging.info(f"[Client {client_idx}] {metric_name.upper()}: {val_result:.4f}")
            logging.info(f"[Client {client_idx}] global {metric_name.upper()}: {global_result:.4f}")
        avg_local = float(np.mean(local_results)) if local_results else float('nan')
        avg_global = float(np.mean(global_results)) if global_results else float('nan')
        self.csv_logger.log({
            "time": datetime.now().isoformat(timespec="seconds"),
            "method": self.fedmid,
            "dataset": self.args.dataset,
            "round": round_idx,
            "scope": "clients_avg",
            "metric": metric_name,
            "local_metric": avg_local,
            "global_metric": avg_global,
            "client_indexes": list(client_indexes)
        })
        logging.info("[Round %d] avg client %s=%.4f, avg global %s=%.4f",
                     round_idx, metric_name.upper(), avg_local, metric_name.upper(), avg_global)
        print("[Round {}] {} avg_local_{:.0f}clients={:.4f}, avg_global={:.4f}, csv={}".format(
            round_idx, self.fedmid, len(client_indexes), avg_local, avg_global, self.csv_logger.path
        ), flush=True)

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def validateGlobal(self, epoch):
        model = self.model_trainer.model
        tbar = tqdm(self.test_global)
        device = self.device
        model.to(device)
        model.eval()
        predList = []
        labelList = []
        maskList = []
        with torch.no_grad():
            for batch_idx, data in enumerate(tbar):
                if self.args.dataset == 'qm9':
                    z, pos, batch, y = data.z.to(device), data.pos.to(device), data.batch.to(device), data.y.to(device)
                    pred, latentEmb = model(z, pos, batch)
                    predList.append(pred.squeeze())
                    labelList.append(y.squeeze())
                elif self.args.dataset in MOLECULAR_REGRESSION_DATASETS:
                    smiles, bg, labels, masks, indices = data
                    labels, masks = labels.to(device), masks.to(device)
                    bg = bg.to(device)
                    node_feats = bg.ndata['h'].to(device)
                    edge_feats = bg.edata.get('e', None)
                    pred, latentEmb = model(bg, node_feats, edge_feats)
                    predList.append(pred.view(labels.shape))
                    labelList.append(labels)
                    maskList.append(masks.float().view(labels.shape))
                elif self.args.dataset in MOLECULAR_CLASSIFICATION_DATASETS:
                    smiles, bg, labels, masks, indices = data
                    labels, masks = labels.to(device), masks.to(device)
                    bg = bg.to(device)
                    node_feats = bg.ndata['h'].to(device)
                    edge_feats = bg.edata.get('e', None)
                    pred, latentEmb = model(bg, node_feats, edge_feats)
                    predList.append(torch.sigmoid(pred).view(labels.shape))
                    labelList.append(labels)
                    maskList.append(masks.float().view(labels.shape))
                tbar.set_description('Round: {:d} Iter: {:d} / {:d}'.format(epoch, batch_idx, len(self.test_global)))
            if self.args.dataset == 'qm9':
                valSize = 10000
            else:
                valSize = int(0.5*len(self.test_global.dataset))
            eval_generator = torch.Generator()
            eval_generator.manual_seed(int(getattr(self.args, 'split_seed', 0)) + 104729)
            indexShuffle = torch.randperm(
                len(self.test_global.dataset), generator=eval_generator
            ).to(device)
            if predList[-1].size()==torch.Size([]):
                predList[-1] = predList[-1].unsqueeze(0)
                labelList[-1] = labelList[-1].unsqueeze(0)
            predAll = torch.cat(predList, dim=0)[indexShuffle]
            labelAll = torch.cat(labelList, dim=0)[indexShuffle]
            maskAll = torch.cat(maskList, dim=0)[indexShuffle] if maskList else None
            if self.args.dataset == 'qm9':
                maeAll = (predAll - labelAll).abs()
                valResult = maeAll[:valSize].mean().item()
                valResultStd = maeAll[:valSize].std().item()
                testResult = maeAll[valSize:].mean().item()
                testResultStd = maeAll[valSize:].std().item()
                resultsNoMean = (predAll - labelAll).abs().mean(dim=0)
                metricName = 'mae'
            elif self.args.dataset in MOLECULAR_REGRESSION_DATASETS:
                squaredError = (predAll - labelAll) ** 2
                valMask = maskAll[:valSize]
                testMask = maskAll[valSize:]
                valMse = (squaredError[:valSize] * valMask).sum() / valMask.sum().clamp_min(1.0)
                testMse = (squaredError[valSize:] * testMask).sum() / testMask.sum().clamp_min(1.0)
                valResult = torch.sqrt(valMse).item()
                valResultStd = squaredError[:valSize][valMask > 0].std().item()
                testResult = torch.sqrt(testMse).item()
                testResultStd = squaredError[valSize:][testMask > 0].std().item()
                metricName = 'rmse'
            elif self.args.dataset in MOLECULAR_CLASSIFICATION_DATASETS:
                predVal = predAll[:valSize]
                labelVal = labelAll[:valSize]
                maskVal = maskAll[:valSize]
                predTest = predAll[valSize:]
                labelTest = labelAll[valSize:]
                maskTest = maskAll[valSize:]
                valResultsList = []
                testResultsList = []
                for itask in range(predAll.shape[1]):
                    valObserved = maskVal[:, itask] > 0
                    testObserved = maskTest[:, itask] > 0
                    labelValTask = labelVal[valObserved, itask].detach().cpu()
                    predValTask = predVal[valObserved, itask].detach().cpu()
                    labelTestTask = labelTest[testObserved, itask].detach().cpu()
                    predTestTask = predTest[testObserved, itask].detach().cpu()
                    if labelValTask.numel() > 0 and torch.unique(labelValTask).numel() > 1:
                        valResultsList.append(roc_auc_score(labelValTask.numpy(), predValTask.numpy()))
                    if labelTestTask.numel() > 0 and torch.unique(labelTestTask).numel() > 1:
                        testResultsList.append(roc_auc_score(labelTestTask.numpy(), predTestTask.numpy()))
                if len(valResultsList) == 0 or len(testResultsList) == 0:
                    logging.warning("No valid binary tasks for ROC-AUC at round %s", epoch)
                    return
                valResult = float(np.mean(valResultsList))
                valResultStd = float(np.std(valResultsList))
                testResult = float(np.mean(testResultsList))
                testResultStd = float(np.std(testResultsList))
                metricName = 'auc'
            if metricName == 'auc':
                if valResult > self.bestVal:
                    self.bestVal = valResult
                    self.bestTest = testResult
            else:
                if valResult < self.bestVal:
                    self.bestVal = valResult
                    self.bestTest = testResult
                    if self.args.dataset == 'qm9':
                        self.bestQm9EveryTask = resultsNoMean.tolist()
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            curValResult = 'cur Val Steps: ' + str(epoch) + ' ' + metricName + ' ' + str(valResult) + ' std ' + str(valResultStd) + '\n'
            curTestResult = 'cur Test Steps: ' + str(epoch) + ' ' + metricName + ' ' + str(testResult) + ' std ' + str(testResultStd) + '\n'
            bestValResult = 'best Val Steps: ' + str(epoch) + ' ' + metricName + ' ' + str(self.bestVal) + '\n'
            bestTestResult = 'best Test Steps: ' + str(epoch) + ' ' + metricName + ' ' + str(self.bestTest) + '\n'
            self.csv_logger.log({
                "time": datetime.now().isoformat(timespec="seconds"),
                "method": self.fedmid,
                "dataset": self.args.dataset,
                "round": epoch,
                "scope": "global",
                "metric": metricName,
                "val": valResult,
                "test": testResult,
                "best_val": self.bestVal,
                "best_test": self.bestTest
            })
            res = dt_string + '\n' + curValResult + curTestResult + bestValResult + bestTestResult + 'detail results' + str(self.bestQm9EveryTask) + '\n'
            logging.info(res)
            print("[Round {}] {} val_{}={:.4f}, test_{}={:.4f}, best_test={:.4f}, csv={}".format(
                epoch, self.fedmid, metricName, valResult, metricName, testResult, self.bestTest, self.csv_logger.path
            ), flush=True)
