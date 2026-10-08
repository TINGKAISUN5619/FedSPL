from typing import List
from argparse import Namespace
import torch
import torch.nn as nn
from tqdm import trange

from mmgnn.data import MoleculeDataset, StandardScaler


def predict(model: nn.Module,
            data: MoleculeDataset,
            batch_size: int,
            scaler: StandardScaler = None,
            args: Namespace = None) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A model.
    :param data: A MoleculeDataset.
    :param batch_size: Batch size.
    :param scaler: A StandardScaler object fit on the training targets.
    :param args: Arguments (for subgraph mode detection).
    :return: A list of lists of predictions. The outer list is examples
    while the inner list is tasks.
    """
    model.eval()

    preds = []

    num_iters, iter_step = len(data), batch_size
    
    # Always use local subgraph reconstruction when subgraphs are available
    use_subgraphs = True

    for i in range(0, num_iters, iter_step):
        # Prepare batch
        mol_batch = MoleculeDataset(data[i:i + batch_size])
        smiles_batch, features_batch = mol_batch.smiles(), mol_batch.features()

        # Run model
        batch = smiles_batch
        
        # Prepare subgraphs if needed
        batched_sub = None
        sub_to_mol = None
        if use_subgraphs and args is not None:
            from mmgnn.features.subgraph import batch_subgraphs
            
            all_subgraphs = []
            mol_indices = []
            for mol_idx, datapoint in enumerate(mol_batch.data):
                if datapoint.subgraphs is not None:
                    all_subgraphs.extend(datapoint.subgraphs)
                    mol_indices.extend([mol_idx] * len(datapoint.subgraphs))
            
            if all_subgraphs:
                batched_sub, sub_to_mol = batch_subgraphs(all_subgraphs, mol_indices, args)

        with torch.no_grad():
            if batched_sub is not None and sub_to_mol is not None:
                batch_preds = model(batch, features_batch, batched_sub, sub_to_mol)
            else:
                batch_preds = model(batch, features_batch)

        batch_preds = batch_preds.data.cpu().numpy()

        # Inverse scale if regression
        if scaler is not None:
            batch_preds = scaler.inverse_transform(batch_preds)

        # Collect vectors
        batch_preds = batch_preds.tolist()
        preds.extend(batch_preds)

    return preds
