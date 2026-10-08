import csv
import os
import pickle
from typing import List

import numpy as np


def save_features(path: str, features: List[np.ndarray]):
    """
    Saves features to a compressed .npz file with array name "features".

    :param path: Path to a .npz file where the features will be saved.
    :param features: A list of 1D numpy arrays containing the features for molecules.
    """
    np.savez_compressed(path, features=features)


def load_features(path: str) -> np.ndarray:
    """
    Loads features saved in a variety of formats.

    Supported formats:
    - .npz compressed (assumes features are saved with name "features")
    - .npz (assumes features are saved with name "features")
    - .npy
    - .csv/.txt (assumes comma-separated features with a header and with one line per molecule)
    - .pkl/.pckl/.pickle containing a sparse numpy array (TODO: remove this option once we are no longer dependent on it)

    All formats assume that the SMILES strings loaded elsewhere in the code are in the same
    order as the features loaded here.

    :param path: Path to a file containing features.
    :return: A 2D numpy array of size (num_molecules, features_size) containing the features.
    """
    extension = os.path.splitext(path)[1]

    if extension == '.npz':
        features = np.load(path)['features']
    elif extension == '.npy':
        features = np.load(path)
    elif extension in ['.csv', '.txt']:
        with open(path) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            features = np.array([[float(value) for value in row] for row in reader])
    elif extension in ['.pkl', '.pckl', '.pickle']:
        with open(path, 'rb') as f:
            features = np.array([np.squeeze(np.array(feat.todense())) for feat in pickle.load(f)])
    else:
        raise ValueError(f'Features path extension {extension} not supported.')

    return features


def load_atom_features(path: str) -> List[np.ndarray]:
    """
    Loads external atom-level features from a .npz file (used in 2D mode).

    Expected format:
    - .npz with keys arr_0, arr_1, ..., one array per molecule
    - Each array has shape (n_atoms, feature_dim) or (n_atoms,) which will be reshaped to (n_atoms, 1)

    :param path: Path to the .npz file containing per-molecule atom features.
    :return: A list where each element is a 2D numpy array (n_atoms, feature_dim) for one molecule.
    """
    extension = os.path.splitext(path)[1]
    if extension != '.npz':
        raise ValueError(f'Atom features must be provided in .npz format; got {extension}')
    loaded = np.load(path)
    keys = list(loaded.keys())
    if len(keys) == 0:
        raise ValueError(f'No arrays found in atom features file: {path}')
    atom_features_list: List[np.ndarray] = []
    if 'features' in loaded and loaded['features'].dtype == object:
        for arr in loaded['features']:
            arr = np.asarray(arr)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            atom_features_list.append(arr)
    else:
        arr_keys = [k for k in keys if k.startswith('arr_')]
        def key_index(k: str) -> int:
            try:
                return int(k.split('_')[1])
            except Exception:
                return -1
        arr_keys = sorted(arr_keys, key=key_index)
        if not arr_keys:
            arr = loaded[keys[0]]
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            atom_features_list.append(arr)
        else:
            for k in arr_keys:
                arr = loaded[k]
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                atom_features_list.append(arr)
    return atom_features_list
