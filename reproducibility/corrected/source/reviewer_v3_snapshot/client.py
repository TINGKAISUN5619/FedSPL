import logging
import numpy as np
import torch
import torch.nn.functional as F
import dgl
import copy
import hashlib
import time
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader


def smiles_to_morgan_arrays(smiles_list, n_bits=2048):
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        logging.warning("RDKit is not available; topology fingerprints are zero vectors.")
        return np.zeros((len(smiles_list), n_bits), dtype=np.float32)

    fps = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
        on_bits = list(fp.GetOnBits())
        fps[i, on_bits] = 1.0
    return fps


def smiles_to_maccs_arrays(smiles_list):
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import MACCSkeys
    except ImportError:
        logging.warning("RDKit is not available; MACCS descriptors are zero vectors.")
        return np.zeros((len(smiles_list), 167), dtype=np.float32)

    fps = np.zeros((len(smiles_list), 167), dtype=np.float32)
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        fp = MACCSkeys.GenMACCSKeys(mol)
        DataStructs.ConvertToNumpyArray(fp, fps[i])
    return fps


def smiles_to_random_arrays(smiles_list, n_bits=2048):
    """Deterministic random descriptors used as a negative-control prototype space."""
    fps = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
    for i, smiles in enumerate(smiles_list):
        digest = hashlib.sha256(smiles.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], byteorder="little", signed=False) % (2 ** 32)
        rng = np.random.default_rng(seed)
        fps[i] = rng.standard_normal(n_bits).astype(np.float32)
    return fps


def smiles_to_physchem_arrays(smiles_list):
    """RDKit physicochemical descriptors for regression-oriented prototype matching."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
    except ImportError:
        logging.warning("RDKit is not available; physicochemical descriptors are zero vectors.")
        return np.zeros((len(smiles_list), 19), dtype=np.float32)

    descs = np.zeros((len(smiles_list), 19), dtype=np.float32)
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        try:
            formal_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
            hetero_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in (1, 6))
            raw = np.array([
                Descriptors.MolWt(mol) / 500.0,
                Descriptors.ExactMolWt(mol) / 500.0,
                Crippen.MolLogP(mol) / 5.0,
                rdMolDescriptors.CalcTPSA(mol) / 200.0,
                Lipinski.NumHDonors(mol) / 10.0,
                Lipinski.NumHAcceptors(mol) / 15.0,
                Lipinski.NumRotatableBonds(mol) / 20.0,
                Lipinski.RingCount(mol) / 10.0,
                Lipinski.NumAromaticRings(mol) / 6.0,
                Lipinski.NumAliphaticRings(mol) / 6.0,
                Lipinski.HeavyAtomCount(mol) / 80.0,
                hetero_atoms / 40.0,
                formal_charge / 5.0,
                rdMolDescriptors.CalcFractionCSP3(mol),
                rdMolDescriptors.CalcLabuteASA(mol) / 300.0,
                Crippen.MolMR(mol) / 150.0,
                Descriptors.NumValenceElectrons(mol) / 400.0,
                Lipinski.NOCount(mol) / 20.0,
                Lipinski.NHOHCount(mol) / 10.0,
            ], dtype=np.float32)
            descs[i] = np.nan_to_num(raw, nan=0.0, posinf=5.0, neginf=-5.0)
        except Exception:
            continue
    return np.clip(descs, -5.0, 5.0).astype(np.float32)


def qm9_batch_descriptors(z, pos, batch):
    """Composition and geometry descriptors for QM9 PyG batches."""
    z = z.detach().cpu()
    pos = pos.detach().cpu()
    batch = batch.detach().cpu()
    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    descriptors = np.zeros((num_graphs, 16), dtype=np.float32)
    tracked_atomic_numbers = [1, 6, 7, 8, 9]
    for graph_idx in range(num_graphs):
        mask = batch == graph_idx
        atoms = z[mask]
        coords = pos[mask]
        if atoms.numel() == 0:
            continue
        atom_count = float(atoms.numel())
        heavy_count = float((atoms > 1).sum().item())
        desc = []
        desc.append(atom_count / 30.0)
        desc.append(heavy_count / 10.0)
        for atomic_number in tracked_atomic_numbers:
            desc.append(float((atoms == atomic_number).sum().item()) / max(atom_count, 1.0))
        desc.append(float(atoms.float().mean().item()) / 10.0)
        desc.append(float(atoms.float().std(unbiased=False).item()) / 10.0)
        centered = coords - coords.mean(dim=0, keepdim=True)
        radius = torch.sqrt((centered.pow(2).sum(dim=1)).mean()).item()
        desc.append(radius / 5.0)
        if atom_count > 1:
            dists = torch.pdist(coords)
            desc.extend([
                float(dists.mean().item()) / 5.0,
                float(dists.std(unbiased=False).item()) / 5.0,
                float(dists.max().item()) / 10.0,
                float(dists.min().item()) / 2.0,
            ])
        else:
            desc.extend([0.0, 0.0, 0.0, 0.0])
        extent = coords.max(dim=0).values - coords.min(dim=0).values
        desc.extend((extent / 10.0).tolist())
        descriptors[graph_idx] = np.array(desc[:16], dtype=np.float32)
    return np.clip(descriptors, -5.0, 5.0)

def get_scaffolds(smiles_list):
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError:
        return set()

    scaffolds = set()
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            try:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffolds.add(Chem.MolToSmiles(scaffold))
            except Exception:
                pass
    return scaffolds


class Client:
    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, args, device, model_trainer, global_test_data=None):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        logging.info("self.local_sample_number = " + str(self.local_sample_number))
        self.args = args
        self.device = device
        self.model_trainer = copy.deepcopy(model_trainer)
        self.distill_batches  = None
        self.global_test_data = global_test_data  # 保存全局测试集
        self.fp_cache = {}
        self.prototype_cluster_cache = {}
        self.scaffold_vocab = self._compute_scaffold_vocab()

    def _compute_scaffold_vocab(self):
        if getattr(self.args, 'dataset', None) == 'qm9':
            return set()
        # Build local scaffold vocabulary
        vocab = set()
        for batch in self.local_training_data:
            smiles = batch[0]
            vocab.update(get_scaffolds(smiles))
        return vocab

    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    def get_sample_number(self):
        return self.local_sample_number

    def _full_local_training_loader(self):
        collate_fn = getattr(self.local_training_data, 'collate_fn', None)
        return DataLoader(
            self.local_training_data.dataset,
            batch_size=getattr(self.args, 'batch_size', 64),
            shuffle=False,
            num_workers=0,
            drop_last=False,
            pin_memory=False,
            collate_fn=collate_fn,
        )

    def train(self, w_global, round_idx, client_idx):
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.train(self.local_training_data, self.device, self.args, round_idx, client_idx)
        weights = self.model_trainer.get_model_params()
        return weights

    def train_moon(self, w_global, previous_local_params, round_idx, client_idx):
        self.model_trainer.set_model_params(w_global)
        model = self.model_trainer.model
        model.to(self.device)
        model.train()

        global_model = copy.deepcopy(model)
        global_model.load_state_dict(w_global)
        global_model.to(self.device)
        global_model.eval()

        previous_model = None
        if previous_local_params is not None and round_idx >= 1:
            previous_model = copy.deepcopy(model)
            previous_model.load_state_dict(previous_local_params)
            previous_model.to(self.device)
            previous_model.eval()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
        localsteps = self.args.localStepsPerRound
        mu = float(getattr(self.args, 'moon_mu', 1.0))
        tau = max(float(getattr(self.args, 'moon_tau', 0.5)), 1e-6)
        tmpstep = 0
        data_iter = iter(self.local_training_data)
        total_moon_loss = 0.0
        total_sup_loss = 0.0
        num_batches = 0

        while tmpstep < localsteps:
            try:
                smiles, bg, labels, masks, indices = next(data_iter)
            except StopIteration:
                data_iter = iter(self.local_training_data)
                smiles, bg, labels, masks, indices = next(data_iter)
            if len(smiles) <= 1:
                continue

            labels = labels.to(self.device)
            masks = masks.to(self.device).float()
            bg = bg.to(self.device)
            node_feats = bg.ndata['h'].to(self.device)
            edge_feats = bg.edata.get('e', None)
            if edge_feats is not None:
                edge_feats = edge_feats.to(self.device)

            optimizer.zero_grad()
            logits, embeddings = model(bg, node_feats, edge_feats)
            if labels.shape != logits.shape:
                labels = labels.view_as(logits)
            if masks.shape != logits.shape:
                masks = masks.view_as(logits)

            per_task_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
            supervised_loss = (per_task_loss * masks).sum() / masks.sum().clamp_min(1.0)

            moon_loss = torch.tensor(0.0, device=self.device)
            if previous_model is not None and mu > 0:
                with torch.no_grad():
                    _, global_embeddings = global_model(bg, node_feats, edge_feats)
                    _, previous_embeddings = previous_model(bg, node_feats, edge_feats)
                current_norm = F.normalize(embeddings, p=2, dim=1)
                global_norm = F.normalize(global_embeddings.detach(), p=2, dim=1)
                previous_norm = F.normalize(previous_embeddings.detach(), p=2, dim=1)
                positive = F.cosine_similarity(current_norm, global_norm, dim=1)
                negative = F.cosine_similarity(current_norm, previous_norm, dim=1)
                contrast_logits = torch.stack([positive, negative], dim=1) / tau
                contrast_labels = torch.zeros(
                    contrast_logits.shape[0], dtype=torch.long, device=self.device
                )
                moon_loss = F.cross_entropy(contrast_logits, contrast_labels)

            loss = supervised_loss + mu * moon_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_sup_loss += float(supervised_loss.detach().item())
            total_moon_loss += float(moon_loss.detach().item())
            num_batches += 1
            tmpstep += 1

        logging.info(
            "[Client %s] MOON train finished: round=%s, use_prev=%s, mu=%.4f, sup=%.4f, moon=%.4f",
            client_idx, round_idx, previous_model is not None, mu,
            total_sup_loss / max(1, num_batches),
            total_moon_loss / max(1, num_batches)
        )
        return self.model_trainer.get_model_params()

    def local_test(self, b_use_test_dataset):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        if test_data is None:
            return {}
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics

    def global_test(self):
        if self.global_test_data is None:
            logging.error(f"[Client {self.client_idx}] Global test data is not available")
            raise ValueError("Global test data not initialized")
        metrics = self.model_trainer.test(self.global_test_data, self.device, self.args)
        return metrics

    def receive_soft_targets(self, distill_batches):
        self.distill_batches = distill_batches  # Store as list of tuples
        logging.info(f"[Client {self.client_idx}] Received {len(distill_batches)} distill samples")
        for i, batch in enumerate(distill_batches):
            logging.debug(f"[Client {self.client_idx}] Batch {i}: {[type(b) for b in batch]}")

    def distill(self, round_idx, client_idx):
        if not self.distill_batches or len(self.distill_batches) == 0:
            logging.warning(f"[Client {client_idx}] No soft targets received, skipping distillation")
            return

        model = self.model_trainer.model
        model.to(self.device)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
        temperature = getattr(self.args, 'temperature', 1.0)
        alpha_kd = getattr(self.args, 'alpha_kd', 0.7)
        is_regression = self.args.dataset in ['esol', 'lipo', 'freesolv']

        total_loss_kd = 0
        total_loss_rep = 0
        total_loss_feat = 0
        num_batches = 0

        for batch_idx, batch in enumerate(self.distill_batches):
            if len(batch) == 6:
                bg, soft_target, labels, masks, rep_target, feat_target = batch
            elif len(batch) == 5:
                bg, soft_target, labels, masks, rep_target = batch
                feat_target = None
            else:
                bg, soft_target, labels, masks = batch
                rep_target = None
                feat_target = None
            if not isinstance(bg, dgl.DGLGraph):
                logging.error(f"[Client {client_idx}] bg is not a DGLGraph, got {type(bg)}")
                continue
            bg = bg.to(self.device)
            soft_target = soft_target.to(self.device)
            if rep_target is not None:
                rep_target = rep_target.to(self.device)
            if feat_target is not None:
                feat_target = feat_target.to(self.device)

            node_feats = bg.ndata['h']
            edge_feats = bg.edata.get('e', None)
            
            # Apply Graph Augmentation (Node feature dropout) if enabled
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

            if logits.shape != soft_target.shape:
                logging.warning(f"[Client {client_idx}] Shape mismatch: logits={logits.shape}, soft_target={soft_target.shape}")
                continue

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

            loss = alpha_kd * loss_kd
            loss = loss + getattr(self.args, 'lambda_rep', 0.1) * loss_rep
            loss = loss + getattr(self.args, 'lambda_feat', 0.0) * loss_feat

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss_kd += loss_kd.item()
            total_loss_rep += loss_rep.item()
            total_loss_feat += loss_feat.item()
            num_batches += 1

        if num_batches > 0:
            logging.info(
                "[Client %d] Distillation finished with alpha_kd=%.2f, kd=%.4f, rep=%.4f, feat=%.4f",
                client_idx,
                alpha_kd,
                total_loss_kd / num_batches,
                total_loss_rep / num_batches,
                total_loss_feat / num_batches
            )
        else:
            logging.info(f"[Client {client_idx}] Distillation finished with alpha_kd={alpha_kd:.2f}")
        self.model_trainer.set_model_params(model.state_dict())

    def _batch_fingerprints(self, smiles):
        missing = [s for s in smiles if s not in self.fp_cache]
        if missing:
            fps = smiles_to_morgan_arrays(missing)
            for smi, fp in zip(missing, fps):
                self.fp_cache[smi] = fp
        return np.stack([self.fp_cache[s] for s in smiles]).astype(np.float32)

    def _batch_random_descriptors(self, smiles):
        dim = int(getattr(self.args, 'random_descriptor_dim', 2048))
        cache_name = '_random_descriptor_cache'
        if not hasattr(self, cache_name):
            setattr(self, cache_name, {})
        cache = getattr(self, cache_name)
        missing = [s for s in smiles if s not in cache]
        if missing:
            fps = smiles_to_random_arrays(missing, dim)
            for smi, fp in zip(missing, fps):
                cache[smi] = fp
        return np.stack([cache[s] for s in smiles]).astype(np.float32)

    def _batch_maccs_descriptors(self, smiles):
        cache_name = '_maccs_descriptor_cache'
        if not hasattr(self, cache_name):
            setattr(self, cache_name, {})
        cache = getattr(self, cache_name)
        missing = [s for s in smiles if s not in cache]
        if missing:
            fps = smiles_to_maccs_arrays(missing)
            for smi, fp in zip(missing, fps):
                cache[smi] = fp
        return np.stack([cache[s] for s in smiles]).astype(np.float32)

    def _batch_physchem_descriptors(self, smiles):
        cache_name = '_physchem_descriptor_cache'
        if not hasattr(self, cache_name):
            setattr(self, cache_name, {})
        cache = getattr(self, cache_name)
        missing = [s for s in smiles if s not in cache]
        if missing:
            descs = smiles_to_physchem_arrays(missing)
            for smi, desc in zip(missing, descs):
                cache[smi] = desc
        return np.stack([cache[s] for s in smiles]).astype(np.float32)

    def _descriptor_from_smiles_and_embeddings(self, smiles, embeddings):
        descriptor_space = getattr(self.args, 'proto_descriptor_space', 'fingerprint')
        if descriptor_space == 'embedding':
            return embeddings.detach()
        if descriptor_space == 'random':
            return torch.from_numpy(self._batch_random_descriptors(smiles)).to(embeddings.device)
        if descriptor_space == 'maccs':
            return torch.from_numpy(self._batch_maccs_descriptors(smiles)).to(embeddings.device)
        if descriptor_space == 'physchem':
            return torch.from_numpy(self._batch_physchem_descriptors(smiles)).to(embeddings.device)
        return torch.from_numpy(self._batch_fingerprints(smiles)).to(embeddings.device)

    def _qm9_descriptor_from_batch(self, data, embeddings):
        descriptor_space = getattr(self.args, 'proto_descriptor_space', 'physchem')
        if descriptor_space == 'embedding':
            return embeddings.detach()
        descriptors = qm9_batch_descriptors(data.z, data.pos, data.batch)
        return torch.from_numpy(descriptors).to(embeddings.device)

    def _observed_label_masks(self, labels, aux):
        if aux.shape == labels.shape:
            return aux.to(labels.device).float()
        if aux.dim() == 1 and aux.numel() == labels.shape[0]:
            dataset = self.local_training_data.dataset
            while hasattr(dataset, 'dataset') and not hasattr(dataset, 'mask'):
                dataset = dataset.dataset
            if hasattr(dataset, 'mask'):
                indices = aux.detach().cpu().long()
                return dataset.mask[indices].to(labels.device).float()
        return torch.ones_like(labels, dtype=torch.float32, device=labels.device)

    def train_task_proto(self, w_global, global_task_prototypes, round_idx, client_idx):
        self.model_trainer.set_model_params(w_global)
        model = self.model_trainer.model
        model.to(self.device)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
        localsteps = self.args.localStepsPerRound
        lambda_task_proto = float(getattr(self.args, 'lambda_task_proto', 0.0))
        min_count = int(getattr(self.args, 'task_proto_min_count', 2))
        use_proto = (
            global_task_prototypes is not None and
            lambda_task_proto > 0 and
            round_idx >= 1
        )
        if use_proto:
            proto_embeddings = global_task_prototypes['prototypes'].to(self.device)
            proto_counts = global_task_prototypes['counts'].to(self.device)
            proto_available = proto_counts >= min_count
        else:
            proto_embeddings = None
            proto_available = None

        tmpstep = 0
        data_iter = iter(self.local_training_data)
        total_proto_loss = 0.0
        num_proto_batches = 0
        while tmpstep < localsteps:
            try:
                smiles, bg, labels, masks, indices = next(data_iter)
            except StopIteration:
                data_iter = iter(self.local_training_data)
                smiles, bg, labels, masks, indices = next(data_iter)
            if len(smiles) <= 1:
                continue

            labels = labels.to(self.device)
            masks = masks.to(self.device).float()
            bg = bg.to(self.device)
            node_feats = bg.ndata['h'].to(self.device)
            edge_feats = bg.edata.get('e', None)
            if edge_feats is not None:
                edge_feats = edge_feats.to(self.device)

            optimizer.zero_grad()
            logits, embeddings = model(bg, node_feats, edge_feats)
            if labels.shape != logits.shape:
                labels = labels.view_as(logits)
            if masks.shape != logits.shape:
                if masks.numel() == logits.shape[0]:
                    masks = masks.view(-1, 1).expand_as(logits)
                else:
                    masks = masks.view_as(logits)

            per_task_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
            supervised_loss = (per_task_loss * masks).sum() / masks.sum().clamp_min(1.0)

            proto_loss = torch.tensor(0.0, device=self.device)
            if use_proto and embeddings.shape[0] > 0:
                batch_size, n_tasks = labels.shape
                label_idx = labels.long().clamp(0, 1)
                task_idx = torch.arange(n_tasks, device=self.device).unsqueeze(0).expand(batch_size, n_tasks)
                selected = proto_embeddings[task_idx, label_idx]
                valid = (masks > 0) & proto_available[task_idx, label_idx]
                valid_count = valid.sum(dim=1, keepdim=True)
                has_target = valid_count.squeeze(1) > 0
                if has_target.any():
                    targets = (selected * valid.unsqueeze(-1).float()).sum(dim=1)
                    targets = targets / valid_count.clamp_min(1).float()
                    proto_loss = 1.0 - F.cosine_similarity(
                        F.normalize(embeddings[has_target], p=2, dim=1),
                        F.normalize(targets[has_target].detach(), p=2, dim=1),
                        dim=1
                    ).mean()

            loss = supervised_loss + lambda_task_proto * proto_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_proto_loss += proto_loss.item()
            num_proto_batches += 1
            tmpstep += 1

        logging.info(
            "[Client %s] TaskProto train finished: round=%s, use_proto=%s, lambda=%.4f, proto_loss=%.4f",
            client_idx, round_idx, use_proto, lambda_task_proto,
            total_proto_loss / max(1, num_proto_batches)
        )
        weights = self.model_trainer.get_model_params()
        return weights

    def collect_task_prototypes(self):
        model = self.model_trainer.model
        model.to(self.device)
        model.eval()
        sums = None
        counts = None
        with torch.no_grad():
            for smiles, bg, labels, masks, indices in self._full_local_training_loader():
                if len(smiles) == 0:
                    continue
                labels = labels.to(self.device)
                masks = masks.to(self.device).float()
                bg = bg.to(self.device)
                node_feats = bg.ndata['h'].to(self.device)
                edge_feats = bg.edata.get('e', None)
                if edge_feats is not None:
                    edge_feats = edge_feats.to(self.device)
                _, embeddings = model(bg, node_feats, edge_feats)
                if sums is None:
                    n_tasks = labels.shape[1]
                    emb_dim = embeddings.shape[1]
                    sums = torch.zeros(n_tasks, 2, emb_dim, device=self.device)
                    counts = torch.zeros(n_tasks, 2, device=self.device)
                label_idx = labels.long().clamp(0, 1)
                for task_idx in range(labels.shape[1]):
                    valid = masks[:, task_idx] > 0
                    if not valid.any():
                        continue
                    for class_idx in (0, 1):
                        selected = valid & (label_idx[:, task_idx] == class_idx)
                        if selected.any():
                            sums[task_idx, class_idx] += embeddings[selected].sum(dim=0)
                            counts[task_idx, class_idx] += selected.sum()
        if sums is None:
            return None
        logging.info(
            "[Client %s] Collected task prototypes with %d observed task/classes",
            self.client_idx, int((counts > 0).sum().item())
        )
        return {'sums': sums.detach().cpu(), 'counts': counts.detach().cpu()}

    def train_topofedproto(self, start_weights, global_topology_prototypes, round_idx, client_idx):
        if start_weights is not None:
            self.model_trainer.set_model_params(start_weights)
        model = self.model_trainer.model
        model.to(self.device)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
        localsteps = self.args.localStepsPerRound
        lambda_proto = getattr(self.args, 'lambda_proto', 0.1)
        use_proto = (
            global_topology_prototypes is not None and
            len(global_topology_prototypes.get('prototypes', [])) > 0 and
            round_idx >= getattr(self.args, 'distill_warmup_rounds', 1)
        )

        if use_proto:
            proto_embeddings = global_topology_prototypes['prototypes'].to(self.device)
            proto_descriptors = global_topology_prototypes['descriptors'].to(self.device)
            proto_descriptors = F.normalize(proto_descriptors, p=2, dim=1)
            proto_counts = global_topology_prototypes['counts'].to(self.device).float().clamp_min(0.0)
            support_tau = max(float(getattr(self.args, 'proto_support_tau', 0.0)), 0.0)
            if support_tau > 0:
                proto_reliability = proto_counts / (proto_counts + support_tau)
            else:
                proto_reliability = torch.ones_like(proto_counts)
        else:
            proto_embeddings = None
            proto_descriptors = None
            proto_reliability = None
            support_tau = 0.0

        tmpstep = 0
        data_iter = iter(self.local_training_data)
        while tmpstep < localsteps:
            try:
                data = next(data_iter)
            except StopIteration:
                data_iter = iter(self.local_training_data)
                data = next(data_iter)

            optimizer.zero_grad()
            if self.args.dataset == 'qm9':
                data = data.to(self.device)
                if data.y.size(0) <= 1:
                    continue
                labels = data.y.to(self.device)
                logits, embeddings = model(data.z, data.pos, data.batch)
                if labels.shape != logits.shape:
                    labels = labels.view_as(logits)
                supervised_loss = F.l1_loss(logits, labels)
                descriptor_batch = self._qm9_descriptor_from_batch(data, embeddings)
            else:
                smiles, bg, labels, masks, indices = data
                if len(smiles) <= 1:
                    continue

                labels = labels.to(self.device)
                masks = masks.to(self.device)
                bg = bg.to(self.device)
                node_feats = bg.ndata['h'].to(self.device)
                edge_feats = bg.edata.get('e', None)
                if edge_feats is not None:
                    edge_feats = edge_feats.to(self.device)

                logits, embeddings = model(bg, node_feats, edge_feats)
                if labels.shape != logits.shape:
                    labels = labels.view_as(logits)
                if masks.shape != logits.shape:
                    if masks.numel() == logits.shape[0]:
                        masks = masks.view(-1, 1).expand_as(logits)
                    else:
                        masks = masks.view_as(logits)

                if self.args.dataset in ['esol', 'lipo', 'freesolv']:
                    supervised_loss = F.mse_loss(logits.squeeze(), labels.squeeze())
                else:
                    per_task_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
                    supervised_loss = (per_task_loss * masks).sum() / masks.sum().clamp_min(1.0)
                descriptor_batch = self._descriptor_from_smiles_and_embeddings(smiles, embeddings)

            if use_proto:
                descriptor_batch = F.normalize(descriptor_batch.float(), p=2, dim=1)
                similarities = torch.matmul(descriptor_batch, proto_descriptors.t())
                match_rule = getattr(self.args, 'topoproto_match', 'hard')
                if match_rule == 'contrastive':
                    tau = max(getattr(self.args, 'topoproto_tau', 0.2), 1e-6)
                    matched = similarities.argmax(dim=1)
                    emb_norm = F.normalize(embeddings, p=2, dim=1)
                    proto_norm = F.normalize(proto_embeddings.detach(), p=2, dim=1)
                    contrastive_logits = torch.matmul(emb_norm, proto_norm.t()) / tau
                    if support_tau > 0:
                        per_sample_proto_loss = F.cross_entropy(
                            contrastive_logits, matched, reduction='none'
                        )
                        confidence = proto_reliability[matched]
                        proto_loss = (per_sample_proto_loss * confidence).mean()
                    else:
                        proto_loss = F.cross_entropy(contrastive_logits, matched)
                elif match_rule == 'soft':
                    tau = max(getattr(self.args, 'topoproto_tau', 0.2), 1e-6)
                    weights = F.softmax(similarities / tau, dim=1)
                    if support_tau > 0:
                        weights = weights * proto_reliability.unsqueeze(0)
                        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
                    proto_targets = torch.matmul(weights, proto_embeddings)
                    proto_loss = F.mse_loss(embeddings, proto_targets.detach())
                else:
                    matched = similarities.argmax(dim=1)
                    proto_targets = proto_embeddings[matched]
                    if support_tau > 0:
                        per_sample_proto_loss = F.mse_loss(
                            embeddings, proto_targets.detach(), reduction='none'
                        ).mean(dim=1)
                        confidence = proto_reliability[matched]
                        proto_loss = (per_sample_proto_loss * confidence).mean()
                    else:
                        proto_loss = F.mse_loss(embeddings, proto_targets.detach())
                loss = supervised_loss + lambda_proto * proto_loss
            else:
                proto_loss = torch.tensor(0.0, device=self.device)
                loss = supervised_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tmpstep += 1

        logging.info(
            "[Client %s] TopoFedProto train finished: round=%s, use_proto=%s, lambda=%.4f, support_tau=%.2f",
            client_idx, round_idx, use_proto, lambda_proto, support_tau
        )
        self.model_trainer.set_model_params(model.state_dict())
        return model.state_dict()

    def collect_topology_prototypes(self, n_clusters):
        collection_start = time.perf_counter()
        model = self.model_trainer.model
        model.to(self.device)
        model.eval()
        all_descriptors = []
        all_embeddings = []
        all_indices = []
        prototype_loader = (
            self.local_training_data
            if self.args.dataset == 'qm9'
            else self._full_local_training_loader()
        )
        with torch.no_grad():
            for data in prototype_loader:
                if self.args.dataset == 'qm9':
                    data = data.to(self.device)
                    if data.y.size(0) == 0:
                        continue
                    _, embeddings = model(data.z, data.pos, data.batch)
                    descriptors = self._qm9_descriptor_from_batch(data, embeddings)
                    indices = data.idx.detach().cpu().view(-1)
                else:
                    smiles, bg, labels, masks, indices = data
                    if len(smiles) == 0:
                        continue
                    bg = bg.to(self.device)
                    node_feats = bg.ndata['h'].to(self.device)
                    edge_feats = bg.edata.get('e', None)
                    if edge_feats is not None:
                        edge_feats = edge_feats.to(self.device)
                    _, embeddings = model(bg, node_feats, edge_feats)
                    descriptors = self._descriptor_from_smiles_and_embeddings(smiles, embeddings)
                all_embeddings.append(embeddings.detach().cpu())
                all_descriptors.append(descriptors.detach().cpu().numpy())
                all_indices.append(indices.detach().cpu().view(-1))

        if not all_embeddings:
            return []

        embeddings = torch.cat(all_embeddings, dim=0)
        descriptors = np.concatenate(all_descriptors, axis=0)
        indices = torch.cat(all_indices, dim=0).numpy().astype(np.int64)
        order = np.argsort(indices, kind='stable')
        indices = indices[order]
        embeddings = embeddings[torch.from_numpy(order).long()]
        descriptors = descriptors[order]

        cluster_num = min(max(1, int(n_clusters)), descriptors.shape[0])
        descriptor_space = getattr(self.args, 'proto_descriptor_space', 'fingerprint')
        index_signature = hashlib.sha256(indices.tobytes()).hexdigest()
        cache_key = (descriptor_space, cluster_num, index_signature)
        fixed_descriptor = descriptor_space != 'embedding'
        cache_hit = fixed_descriptor and cache_key in self.prototype_cluster_cache

        cluster_fit_seconds = 0.0
        if cache_hit:
            cache_entry = self.prototype_cluster_cache[cache_key]
            assignments = cache_entry['assignments']
            cluster_descriptors = cache_entry['cluster_descriptors']
        elif cluster_num == 1:
            assignments = np.zeros(descriptors.shape[0], dtype=np.int64)
            cluster_descriptors = {0: descriptors.mean(axis=0)}
        else:
            kmeans_seed = int(getattr(self.args, 'partition_seed', self.args.seed)) + int(self.client_idx)
            kmeans = KMeans(n_clusters=cluster_num, random_state=kmeans_seed, n_init=10)
            cluster_fit_start = time.perf_counter()
            assignments = kmeans.fit_predict(descriptors)
            cluster_fit_seconds = time.perf_counter() - cluster_fit_start
            cluster_descriptors = {
                cluster_id: descriptors[assignments == cluster_id].mean(axis=0)
                for cluster_id in np.unique(assignments)
            }

        if fixed_descriptor and not cache_hit:
            self.prototype_cluster_cache[cache_key] = {
                'indices': indices.copy(),
                'assignments': assignments.copy(),
                'cluster_descriptors': {
                    int(cluster_id): value.copy()
                    for cluster_id, value in cluster_descriptors.items()
                },
            }

        collection_seconds = time.perf_counter() - collection_start
        prototypes = []
        for cluster_id in sorted(np.unique(assignments).tolist()):
            mask = assignments == cluster_id
            count = int(mask.sum())
            if count == 0:
                continue
            proto_embedding = embeddings[mask].mean(dim=0)
            descriptor = torch.from_numpy(cluster_descriptors[int(cluster_id)]).float()
            prototypes.append({
                'prototype': proto_embedding,
                'descriptor': descriptor,
                'count': count,
                'client_idx': int(self.client_idx),
                'cluster_id': int(cluster_id),
                'requested_clusters': int(n_clusters),
                'cluster_cache_hit': bool(cache_hit),
                'cluster_fit_seconds': float(cluster_fit_seconds),
                'collection_seconds': float(collection_seconds),
            })
        logging.info(
            "[Client %s] Collected %d/%d local topology prototypes from %d samples (descriptor=%s, cache_hit=%s)",
            self.client_idx, len(prototypes), int(n_clusters), len(indices), descriptor_space, cache_hit
        )
        return prototypes
