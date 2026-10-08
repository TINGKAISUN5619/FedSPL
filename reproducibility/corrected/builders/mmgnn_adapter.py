"""Pinned official MMGNN-2D with a DGL-style (logits, embedding) interface.

This is an adapter, not a reimplementation. The upstream graph featurizer,
colored-subgraph generator, CMPNN, reconstruction and prediction head run
unchanged. Eager training/scaffold-splitting package imports are bypassed.

Attach the *ordered batch* SMILES after DGL batching/device transfer with
``bind_mmgnn_smiles(graph, smiles, sample_ids)``. DGL node/edge feature values
are intentionally not used: upstream MMGNN featurizes those SMILES itself.
The returned embedding is the official molecular encoder output, before FFN.
Logits are returned in both train/eval mode; sigmoid belongs in evaluation.
"""

from argparse import Namespace
from collections import OrderedDict
from contextlib import nullcontext
import hashlib
import importlib
import functools
import json
from pathlib import Path
import subprocess
import sys
import types

import torch
from torch import nn


UPSTREAM_URL = "https://github.com/MathIntelligence/MMGNN"
UPSTREAM_COMMIT = "78dcbff3a3d576506434241e4f26c70416453d9e"
DEFAULT_SOURCE = Path(__file__).resolve().parents[2] / "external/MMGNN_20260916"
_LOADED_SOURCE = None
_INSTALLED_HOOK = None


def source_provenance(source_dir=None):
    """Require the pinned, unchanged tracked checkout; do not invent a license."""
    root = Path(source_dir or DEFAULT_SOURCE).resolve()

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(root)] + list(args), text=True
        ).strip()

    commit = git("rev-parse", "HEAD")
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError("MMGNN commit mismatch: " + commit)
    dirty = git("diff", "--name-only", "HEAD", "--")
    if dirty:
        raise RuntimeError("Official MMGNN tracked files were modified: " + dirty)
    files = git("ls-tree", "-r", "--name-only", "HEAD").splitlines()
    hashes = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in files
    }
    licenses = [name for name in files if any(
        token in Path(name).name.lower() for token in ("license", "licence", "copying")
    )]
    return {
        "repository": UPSTREAM_URL,
        "commit": commit,
        "source_dir": str(root),
        "tracked_sha256": hashes,
        "license_files": licenses,
        "license_status": "present" if licenses else "absent_in_pinned_upstream",
    }


def load_official_core(source_dir=None):
    """Load exact official modules without importing the unrelated train CLI.

    Upstream absolute imports require the `mmgnn` namespace. Reserve it for
    this checkout and reject another installation, rather than replace one.
    No function, layer, dependency stub or upstream source is monkey-patched.
    Use one pinned checkout per worker process.
    """
    global _LOADED_SOURCE
    root = Path(source_dir or DEFAULT_SOURCE).resolve()
    if _LOADED_SOURCE is not None:
        if root != _LOADED_SOURCE:
            raise RuntimeError("Another MMGNN checkout is already loaded")
        return
    source_provenance(root)
    if any(name == "mmgnn" or name.startswith("mmgnn.") for name in sys.modules):
        raise RuntimeError("Load this adapter in a fresh process: mmgnn already imported")
    def package_namespace(name, directory):
        package = types.ModuleType(name)
        package.__path__ = [str(directory)]
        package.__package__ = name
        package.__file__ = str(directory / "__init__.py")
        sys.modules[name] = package
        return package

    package = package_namespace("mmgnn", root / "mmgnn")
    try:
        importlib.import_module("mmgnn.features")
        # data/__init__.py eagerly imports scaffold splitting and numba. The
        # encoder needs only these exact official classes, not the splitter.
        data = package_namespace("mmgnn.data", root / "mmgnn/data")
        package.data = data
        data_leaf = importlib.import_module("mmgnn.data.data")
        data.MoleculeDataset = data_leaf.MoleculeDataset
        data.MoleculeDatapoint = data_leaf.MoleculeDatapoint
        data.StandardScaler = importlib.import_module("mmgnn.data.scaler").StandardScaler
        for name in ("models", "utils", "features.subgraph"):
            importlib.import_module("mmgnn." + name)
    except Exception:
        for name in list(sys.modules):
            if name == "mmgnn" or name.startswith("mmgnn."):
                del sys.modules[name]
        raise
    _LOADED_SOURCE = root


def _validated_smiles(smiles):
    from rdkit import Chem

    if isinstance(smiles, str):
        raise ValueError("Expected an ordered sequence of SMILES, not one string")
    values = tuple(smiles)
    if not values:
        raise ValueError("Empty MMGNN batch")
    counts = []
    for position, value in enumerate(values):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Missing SMILES at batch position %d" % position)
        mol = Chem.MolFromSmiles(value)
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError("Invalid/empty molecule at batch position %d" % position)
        counts.append(mol.GetNumAtoms())
    return values, tuple(counts)


def bind_mmgnn_smiles(graph, smiles, sample_ids=None):
    """Bind metadata to the *batched* DGL graph, preserving sample order.

    Labels/masks/targets are neither accepted nor needed. Atom counts catch
    accidental batch mismatches, not same-size molecule swaps: callers must
    pass SMILES from the very same collate result, not reconstruct the order.
    """
    values, counts = _validated_smiles(smiles)
    observed = tuple(int(n) for n in graph.batch_num_nodes().tolist())
    if observed != counts:
        raise ValueError("DGL/SMILES atom-count or batch-order mismatch")
    ids = None if sample_ids is None else tuple(int(i) for i in sample_ids)
    if ids is not None and len(ids) != len(values):
        raise ValueError("sample_ids length does not match SMILES")
    graph.mmgnn_metadata = {"smiles": values, "atom_counts": counts, "sample_ids": ids}
    return graph


def _metadata_collate(original):
    @functools.wraps(original)
    def wrapped(samples):
        batch = original(samples)
        if len(batch) < 4:
            raise ValueError("Expected (smiles, graph, labels, masks[, indices])")
        bind_mmgnn_smiles(batch[1], batch[0], batch[4] if len(batch) > 4 else None)
        return batch
    return wrapped


def _replace_loaded_references(code_root, original, replacement):
    replaced = []
    for module in list(sys.modules.values()):
        location = getattr(module, "__file__", None)
        if not location:
            continue
        path = Path(location).resolve()
        if code_root not in path.parents:
            continue
        for name, value in list(vars(module).items()):
            if value is original:
                setattr(module, name, replacement)
                replaced.append(module.__name__ + "." + name)
    return replaced


def install(code_root, official_repo=None, output=None, **model_options):
    """Process-local factory/collate hooks; never edits the frozen source files.

    Call once in a dedicated worker, before creating any models/DataLoaders.
    Hooks network.model_factory and BOTH project collates, plus already-loaded
    aliases inside code_root. Later `from ... import ...` sees the replacement.
    Use num_workers=0; spawned workers do not inherit these process-local hooks.
    The runner must also set its parsed/logged encoder name to `mmgnn2d` (the
    old argparse choices do not contain it). No CLI, optimizer or loss is patched.
    """
    global _INSTALLED_HOOK
    if _INSTALLED_HOOK is not None:
        raise RuntimeError("MMGNN hooks already installed; use a fresh worker")
    root = Path(code_root).resolve()
    if "num_tasks" in model_options or "source_dir" in model_options:
        raise ValueError("Task count comes from the factory; use official_repo for source")
    if not all((root / name).is_file() for name in
               ("network/model_factory.py", "data_loader.py", "fedavg_api.py")):
        raise ValueError("code_root lacks the expected molecular project interfaces")
    for name in ("network", "network.model_factory", "data_loader", "fedavg_api", "client"):
        module = sys.modules.get(name)
        if module is not None:
            location = getattr(module, "__file__", None)
            if location is None and name == "network":
                correct = root / "network" in [Path(p).resolve() for p in module.__path__]
            else:
                correct = location is not None and root in Path(location).resolve().parents
            if not correct:
                raise RuntimeError("Conflicting pre-imported project module: " + name)
    load_official_core(official_repo)
    sys.path.insert(0, str(root))
    factory = importlib.import_module("network.model_factory")
    data = importlib.import_module("data_loader")
    api = importlib.import_module("fedavg_api")
    old_factory = factory.build_molecular_model

    def build(dataset_or_loader, encoder="mmgnn2d", n_tasks=None):
        dataset = factory.unwrap_dataset(dataset_or_loader)
        tasks = n_tasks if n_tasks is not None else getattr(dataset, "n_tasks", None)
        if tasks is None:
            raise ValueError("Cannot infer n_tasks; supply it explicitly to model_factory")
        return MMGNN2DAdapter(num_tasks=int(tasks), source_dir=official_repo, **model_options)

    replacements = _replace_loaded_references(root, old_factory, build)
    for module in (data, api):
        original = module.collate_molgraphs
        replacements.extend(_replace_loaded_references(root, original, _metadata_collate(original)))
    _INSTALLED_HOOK = {
        "code_root": str(root), "official": source_provenance(official_repo),
        "factory_output": "MMGNN2DAdapter", "encoder_log_name_required": "mmgnn2d",
        "model_options": model_options, "replaced_references": replacements,
        "num_workers_required": 0, "disk_source_modified": False,
        "training_started": False,
    }
    if output is not None:
        directory = Path(output)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "mmgnn_adapter_install.json").write_text(
            json.dumps(_INSTALLED_HOOK, indent=2, sort_keys=True) + "\n")
    return dict(_INSTALLED_HOOK)


class MMGNN2DAdapter(nn.Module):
    """Official 2D local reconstruction, sum/mean aggregation, float32 CPU/CUDA.

    Defaults follow upstream parsing.py, except 2D binary/multitask mode and
    disabled process-global featurizer caching. `cache_size` bounds our
    per-instance CPU graph cache; it contains no learned representations.
    Attention is deliberately not exposed: upstream has a version-dependent
    scatter_reduce fallback that is not covered by this old-PyTorch adapter.
    """

    def __init__(self, num_tasks, source_dir=None, hidden_size=300, depth=3,
                 dropout=0.0, aggregation="sum", ffn_hidden_size=None,
                 ffn_num_layers=2, activation="ReLU", bias=False,
                 cache_size=1024):
        super().__init__()
        if num_tasks < 1 or hidden_size < 1 or depth < 1 or ffn_num_layers < 1:
            raise ValueError("Model dimensions and depths must be positive")
        if aggregation not in ("sum", "mean"):
            raise ValueError("Only verified official sum/mean reconstruction is supported")
        if not 0 <= dropout < 1 or cache_size < 0:
            raise ValueError("Invalid dropout/cache_size")
        if ffn_hidden_size is not None and ffn_hidden_size < 1:
            raise ValueError("ffn_hidden_size must be positive")
        self.source_dir = str(Path(source_dir or DEFAULT_SOURCE).resolve())
        load_official_core(self.source_dir)
        self.config = dict(num_tasks=num_tasks, hidden_size=hidden_size, depth=depth,
                           dropout=dropout, aggregation=aggregation,
                           ffn_hidden_size=ffn_hidden_size, ffn_num_layers=ffn_num_layers,
                           activation=activation, bias=bias, cache_size=cache_size)
        self.args = Namespace(
            mode="2d", dataset_type="classification", num_tasks=num_tasks,
            hidden_size=hidden_size, depth=depth, dropout=dropout, bias=bias,
            activation=activation, atom_messages=False, undirected=False,
            features_only=False, use_input_features=False, features_size=0,
            features_dim=0, features_generator=None, no_cache=True, cuda=False,
            ffn_hidden_size=ffn_hidden_size or hidden_size, ffn_num_layers=ffn_num_layers,
            recon_pool=aggregation, agg=aggregation, local_mode=True,
            global_mode=False, baseline_full=False, recon_only=True, dual_agg="sum",
        )
        build_model = importlib.import_module("mmgnn.models").build_model
        self.official = build_model(self.args)
        # Supply the official BatchMolGraph ourselves, avoiding the global
        # mol2graph cache whose key does not distinguish 2D/3D configurations.
        self.official.encoder.graph_input = True
        self._molecules = OrderedDict()

    def __getstate__(self):
        parent_getstate = getattr(super(), "__getstate__", None)
        state = parent_getstate() if parent_getstate is not None else self.__dict__.copy()
        # Federated deepcopy/checkpoints must not propagate clients' cached data.
        state["_molecules"] = OrderedDict()
        return state

    def clear_cache(self):
        self._molecules.clear()

    def prepare_smiles(self, smiles):
        """Use only official MolGraph/generation/batching, with no model fallback."""
        values, _ = _validated_smiles(smiles)
        features = importlib.import_module("mmgnn.features.featurization")
        subgraph = importlib.import_module("mmgnn.features.subgraph")
        full_graphs, subgraphs, indices = [], [], []
        for index, value in enumerate(values):
            if value in self._molecules:
                full, subs = self._molecules.pop(value)
                self._molecules[value] = (full, subs)
            else:
                full = features.MolGraph(value, self.args, atom_coordinates=None)
                subs = subgraph.generate_subgraphs_from_molgraph(full, self.args)
                if not subs:
                    raise ValueError("Official subgraph generation returned no graphs")
                if self.config["cache_size"]:
                    self._molecules[value] = (full, subs)
                    while len(self._molecules) > self.config["cache_size"]:
                        self._molecules.popitem(last=False)
            full_graphs.append(full)
            subgraphs.extend(subs)
            indices.extend([index] * len(subs))
        full_batch = features.BatchMolGraph(full_graphs, self.args)
        sub_batch, mapping = subgraph.batch_subgraphs(subgraphs, indices, self.args)
        if sub_batch is None or set(mapping) != set(range(len(values))):
            raise RuntimeError("Incomplete official MMGNN subgraph coverage")
        return full_batch, sub_batch, mapping

    def forward_smiles(self, smiles):
        parameters = list(self.parameters())
        device = parameters[0].device
        if device.type not in ("cpu", "cuda"):
            raise ValueError("Official MMGNN adapter supports only CPU/CUDA")
        if any(p.dtype != torch.float32 or p.device != device for p in parameters):
            raise ValueError("Official MMGNN adapter requires uniform float32 parameters")
        full, subs, mapping = self.prepare_smiles(smiles)
        # Upstream uses .cuda() without a device argument. Scope it to this
        # model's device rather than silently transferring to cuda:0.
        context = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
        with context:
            embedding = self.official.encoder(full, None, subs, mapping)
            logits = self.official.ffn(embedding)
        return logits, embedding

    def forward(self, graph, node_feats, edge_feats):
        metadata = getattr(graph, "mmgnn_metadata", None)
        if not isinstance(metadata, dict) or "smiles" not in metadata:
            raise ValueError("Missing MMGNN metadata; call bind_mmgnn_smiles after batching")
        counts = tuple(int(n) for n in graph.batch_num_nodes().tolist())
        if counts != metadata.get("atom_counts"):
            raise ValueError("MMGNN metadata is stale after graph rebatching")
        values, expected_counts = _validated_smiles(metadata["smiles"])
        if expected_counts != counts:
            raise ValueError("SMILES no longer match the bound batch")
        if node_feats is not None and node_feats.shape[0] != graph.num_nodes():
            raise ValueError("Node feature rows do not match DGL graph")
        if edge_feats is not None and edge_feats.shape[0] != graph.num_edges():
            raise ValueError("Edge feature rows do not match DGL graph")
        return self.forward_smiles(values)
