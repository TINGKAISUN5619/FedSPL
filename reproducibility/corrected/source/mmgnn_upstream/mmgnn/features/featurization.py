from argparse import Namespace
from typing import List, Tuple, Union
from rdkit import Chem
import torch
import numpy as np

# Atom feature sizes
MAX_ATOMIC_NUM = 100
ATOM_FEATURES = {
    'atomic_num': list(range(MAX_ATOMIC_NUM)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

# Distance feature sizes
PATH_DISTANCE_BINS = list(range(10))
THREE_D_DISTANCE_MAX = 20
THREE_D_DISTANCE_STEP = 1
THREE_D_DISTANCE_BINS = list(range(0, THREE_D_DISTANCE_MAX + 1, THREE_D_DISTANCE_STEP))

# 2D mode: chemical bond features only
BOND_FDIM_2D = 14

# 3D mode: concatenate RDKit chemical bond features with RBF distance features and angular CBF features
BOND_FDIM_BASE = 14  # length of RDKit chemical bond feature vector (type, conjugation, ring, stereo)
BOND_DIST_DIM = 64   # length of RBF distance expansion
RBF_CUTOFF = 8.0
RBF_CENTERS = np.linspace(0.0, RBF_CUTOFF, BOND_DIST_DIM, dtype=np.float32)
RBF_DELTA = RBF_CENTERS[1] - RBF_CENTERS[0] if len(RBF_CENTERS) > 1 else 1.0
RBF_GAMMA = 1.0 / (RBF_DELTA ** 2) if RBF_DELTA != 0 else 1.0

# Angular Circular Basis Functions (CBF) 
CBF_ANG_DIM = 8  # Set to 0 to disable
CBF_CENTERS = np.linspace(-1.0, 1.0, CBF_ANG_DIM, dtype=np.float32) if CBF_ANG_DIM > 0 else np.array([])
CBF_DELTA = CBF_CENTERS[1] - CBF_CENTERS[0] if len(CBF_CENTERS) > 1 else 1.0
CBF_GAMMA = 1.0 / (4.0 * CBF_DELTA ** 2) if CBF_DELTA != 0 else 1.0

# Dihedral Angle Features 
DIHEDRAL_DIM = 8  # Set to 0 to disable

# len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2
BOND_FDIM = BOND_FDIM_BASE + BOND_DIST_DIM + CBF_ANG_DIM + DIHEDRAL_DIM  # Chem + distance + angular CBF + dihedral

# Memoization
SMILES_TO_GRAPH = {}
SMILES_TO_COORDINATES = {}  # Cache for 3D coordinates


def clear_cache():
    """Clears featurization cache."""
    global SMILES_TO_GRAPH, SMILES_TO_COORDINATES
    SMILES_TO_GRAPH = {}
    SMILES_TO_COORDINATES = {}


def get_atom_fdim(args: Namespace) -> int:
    """
    Gets the dimensionality of atom features.

    :param: Arguments.
    """
    return ATOM_FDIM


def get_bond_fdim(args: Namespace) -> int:
    """
    Gets the dimensionality of bond features. 2D mode uses chemical-only (14); 3D uses chemical + RBF + angular + dihedral.
    """
    if getattr(args, 'mode', '3d') == '2d':
        return BOND_FDIM_2D
    return BOND_FDIM


def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the value in a list of length len(choices) + 1.
    If value is not in the list of choices, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def atom_features(atom: Chem.rdchem.Atom, functional_groups: List[int] = None) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.

    :param atom: An RDKit atom.
    :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    :return: A list containing the atom features.
    """
    features = onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
           onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
           onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
           onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
           onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
           onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
           [1 if atom.GetIsAromatic() else 0] + \
           [atom.GetMass() * 0.01]  # scaled to about the same range as other features
    if functional_groups is not None:
        features += functional_groups
    return features


def bond_features_2d(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """
    Builds the chemical-only bond feature vector (2D mode, length BOND_FDIM_2D).
    Same as standard CMPNN bond features: type, conjugation, ring, stereo.
    """
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM_2D - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond


def bond_distance_rbf(distance: float, cutoff: float = RBF_CUTOFF, num_centers: int = BOND_DIST_DIM) -> List[float]:
    """
    Radial basis expansion of a scalar distance using Gaussian kernels.
    
    :param distance: The 3D distance between two atoms.
    :param cutoff: Maximum distance cutoff for RBF expansion.
    :param num_centers: Number of RBF centers.
    :return: List of RBF-expanded distance features.
    """
    if num_centers <= 0:
        return []
    if num_centers == BOND_DIST_DIM and cutoff == RBF_CUTOFF:
        centers = RBF_CENTERS
        gamma = RBF_GAMMA
    else:
        centers = np.linspace(0.0, cutoff, num_centers, dtype=np.float32)
        delta = centers[1] - centers[0] if len(centers) > 1 else 1.0
        gamma = 1.0 / (delta ** 2) if delta != 0 else 1.0

    dist = float(distance)
    if np.isnan(dist):
        dist = 0.0
    if cutoff is not None:
        dist = dist = max(dist, 0.0)

    values = np.exp(-gamma * (dist - centers) ** 2)
    return values.astype(np.float32).tolist()


def angle_cbf(cos_theta: float, num_centers: int = CBF_ANG_DIM) -> List[float]:
    """
    Circular basis expansion over cos(theta) using fixed Gaussian centers in [-1, 1].

    :param cos_theta: Cosine of the angle.
    :param num_centers: Number of angular basis functions.
    :return: List of length num_centers with CBF values.
    """
    if num_centers <= 0:
        return []
    if num_centers == CBF_ANG_DIM:
        centers = CBF_CENTERS
        gamma = CBF_GAMMA
    else:
        centers = np.linspace(-1.0, 1.0, num_centers, dtype=np.float32)
        delta = centers[1] - centers[0] if len(centers) > 1 else 1.0
        gamma = 1.0 / (delta ** 2) if delta != 0 else 1.0

    c = float(np.clip(cos_theta, -1.0, 1.0))
    values = np.exp(-gamma * (c - centers) ** 2)
    return values.astype(np.float32).tolist()


def aggregate_angle_features(central_idx: int,
                                        target_idx: int,
                                        neighbors: List[List[int]],
                                        coordinates: Union[np.ndarray, List[Tuple[float, float, float]]],
                                        num_centers: int = CBF_ANG_DIM) -> List[float]:
    
    # Setup vectors
    coords_np = np.asarray(coordinates, dtype=np.float32)
    if central_idx >= len(coords_np) or target_idx >= len(coords_np):
        return [0.0] * num_centers

    r_i = coords_np[central_idx]
    r_k = coords_np[target_idx]
    v_ik = r_k - r_i
    norm_ik = np.linalg.norm(v_ik)
    
    if norm_ik < 1e-6:
        return [0.0] * num_centers

    # Get all neighbor indices (excluding target)
    neighbor_indices = [j for j in neighbors[central_idx] if j != target_idx and j < len(coords_np)]
    
    if not neighbor_indices:
        return [0.0] * num_centers

    neighbor_indices = np.array(neighbor_indices, dtype=np.int64)

    # Vectorized Calculation
    r_j_all = coords_np[neighbor_indices]
    
    # Vectors from central(i) to all neighbors(j)
    v_ij_all = r_j_all - r_i  
    
    # Calculate norms for all neighbors
    norm_ij_all = np.linalg.norm(v_ij_all, axis=1)
    
    valid_mask = norm_ij_all > 1e-6
    if not np.any(valid_mask):
         return [0.0] * num_centers
         
    # Filter valid neighbors
    v_ij_all = v_ij_all[valid_mask]
    norm_ij_all = norm_ij_all[valid_mask]

    # Compute Cosines
    dot_products = np.dot(v_ij_all, v_ik)
    cos_thetas = dot_products / (norm_ij_all * norm_ik)
    
    # Clip
    cos_thetas = np.clip(cos_thetas, -1.0, 1.0)

    # Compute CBF
    if num_centers == CBF_ANG_DIM:
        centers = CBF_CENTERS
        gamma = CBF_GAMMA
    else:
        centers = np.linspace(-1.0, 1.0, num_centers, dtype=np.float32)
        delta = centers[1] - centers[0]
        gamma = 1.0 / (4.0 * delta ** 2)

    # Broadcasting
    diff = cos_thetas[:, np.newaxis] - centers
    cbf_values = np.exp(-gamma * diff**2)

    # Sum (Aggregate) across the neighbor dimension
    final_feature = np.sum(cbf_values, axis=0)
    
    return final_feature.astype(np.float32).tolist()


def compute_dihedral_angle(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Compute dihedral (torsion) angle for 4 points p0-p1-p2-p3.
    
    :param p0, p1, p2, p3: 3D coordinates as numpy arrays
    :return: Dihedral angle in radians [-pi, pi]
    """
    # Vectors along bonds
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    
    # Normal vectors to planes
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    
    # Normalize
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    
    if n1_norm < 1e-6 or n2_norm < 1e-6:
        return 0.0
    
    n1 = n1 / n1_norm
    n2 = n2 / n2_norm
    
    # Dihedral angle
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    
    return np.arctan2(y, x)


def dihedral_circular_basis(angle: float, num_centers: int = DIHEDRAL_DIM) -> List[float]:
    """
    Circular basis expansion of dihedral angle in [-pi, pi].
    
    :param angle: Dihedral angle in radians
    :param num_centers: Number of circular basis functions
    :return: List of CBF values
    """
    if num_centers <= 0:
        return []
    
    # Centers uniformly distributed in [-pi, pi]
    centers = np.linspace(-np.pi, np.pi, num_centers, dtype=np.float32)
    delta = centers[1] - centers[0] if len(centers) > 1 else 1.0
    gamma = 1.0 / (4.0 * delta ** 2)  # wider Gaussians
    
    # Handle periodicity: compute circular distance
    diff = angle - centers
    diff = np.arctan2(np.sin(diff), np.cos(diff))  # wrap to [-pi, pi]
    
    values = np.exp(-gamma * diff ** 2)
    return values.astype(np.float32).tolist()


def aggregate_dihedral_features(central_idx: int,
                                target_idx: int,
                                neighbors: List[List[int]],
                                coordinates: Union[np.ndarray, List[Tuple[float, float, float]]],
                                num_centers: int = DIHEDRAL_DIM) -> List[float]:
    """
    Vectorized aggregation of dihedral angle features for edge central->target.
    For edge i->k, find all 4-body paths j-i-k-l and compute torsion angles.
    
    :param central_idx: Index i (central atom of edge i->k)
    :param target_idx: Index k (target atom of edge i->k)
    :param neighbors: Adjacency list
    :param coordinates: 3D coordinates
    :param num_centers: Number of dihedral basis functions
    :return: Aggregated dihedral features (mean over all valid 4-body paths)
    """
    if num_centers <= 0:
        return []

    coords_np = np.asarray(coordinates, dtype=np.float32)
    if central_idx >= len(coords_np) or target_idx >= len(coords_np):
        return [0.0] * num_centers

    # Get Neighbor Indices
    idx_j = [n for n in neighbors[central_idx] if n != target_idx]
    idx_l = [n for n in neighbors[target_idx] if n != central_idx]

    if not idx_j or not idx_l:
        return [0.0] * num_centers

    # Prepare Coordinates
    J, L = np.meshgrid(idx_j, idx_l, indexing='ij')
    J = J.flatten()
    L = L.flatten()
    
    valid_mask = (J < len(coords_np)) & (L < len(coords_np))
    J = J[valid_mask]
    L = L[valid_mask]
    
    if len(J) == 0:
        return [0.0] * num_centers
    
    # Get all coordinates for the 4-body paths
    p_j = coords_np[J]         # shape (N_paths, 3)
    p_i = coords_np[central_idx]  # scalar broadcasted
    p_k = coords_np[target_idx]   # scalar broadcasted
    p_l = coords_np[L]         # shape (N_paths, 3)

    # Vectorized Dihedral Calculation
    b1 = p_i - p_j              # shape (N_paths, 3)
    b2 = p_k - p_i              # shape (3,) broadcasted to (N_paths, 3)
    b3 = p_l - p_k              # shape (N_paths, 3)

    # Cross products
    n1 = np.cross(b1, b2)       # shape (N_paths, 3)
    n2 = np.cross(b2, b3)       # shape (N_paths, 3)

    # Normalize
    n1_norm = np.linalg.norm(n1, axis=1, keepdims=True) + 1e-6
    n2_norm = np.linalg.norm(n2, axis=1, keepdims=True) + 1e-6
    n1 = n1 / n1_norm
    n2 = n2 / n2_norm

    # Angle calculation
    b2_norm = np.linalg.norm(b2) + 1e-6  # scalar
    b2_dir = b2 / b2_norm                 # shape (3,) or (1, 3)
    
    m1 = np.cross(n1, b2_dir)             # shape (N_paths, 3)
    
    # Dot products
    x = np.sum(n1 * n2, axis=1)           # shape (N_paths,)
    y = np.sum(m1 * n2, axis=1)           # shape (N_paths,)
    
    angles = np.arctan2(y, x)             # shape (N_paths,)

    # Vectorized Circular Basis Expansion
    centers = np.linspace(-np.pi, np.pi, num_centers, dtype=np.float32)
    delta = centers[1] - centers[0] if len(centers) > 1 else 1.0
    gamma = 1.0 / (4.0 * delta ** 2)

    # Broadcasting
    diff = angles[:, np.newaxis] - centers
    # Handle periodicity
    diff = np.arctan2(np.sin(diff), np.cos(diff))
    
    cbf_values = np.exp(-gamma * diff ** 2)  # shape (N_paths, num_centers)

    # Aggregate
    final_feature = np.sum(cbf_values, axis=0)

    return final_feature.astype(np.float32).tolist()


def bond_features(bond: Chem.rdchem.Bond,
                  distance: float = None,
                  central_idx: int = None,
                  target_idx: int = None,
                  neighbors: List[List[int]] = None,
                  coordinates: List[Tuple[float, float, float]] = None,
                  cbf_dim: int = CBF_ANG_DIM,
                  dihedral_dim: int = DIHEDRAL_DIM) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a directed edge as [chem || distance RBF || angular CBF || dihedral].
    
    :param bond: A RDKit bond (can be None for non-bonded pairs).
    :param distance: 3D distance between atoms.
    :param central_idx: Index of central atom i (edge i->k).
    :param target_idx: Index of target atom k (edge i->k).
    :param neighbors: Adjacency list for the molecule.
    :param coordinates: 3D coordinates for all atoms.
    :param cbf_dim: Length of angular CBF vector (0 to disable).
    :param dihedral_dim: Length of dihedral CBF vector (0 to disable).
    :return: List of length BOND_FDIM.
    """
    # RDKit chemical bond features
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM_BASE - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))

    # Distance features (RBF expansion)
    dist_feats = bond_distance_rbf(distance) if distance is not None else [0.0] * BOND_DIST_DIM

    # Angular CBF features aggregated over neighbors of central atom
    ang_feats = aggregate_angle_features(
        central_idx=central_idx,
        target_idx=target_idx,
        neighbors=neighbors,
        coordinates=coordinates,
        num_centers=cbf_dim
    ) if cbf_dim > 0 and coordinates is not None and neighbors is not None else [0.0] * cbf_dim

    # Dihedral angle features
    dihedral_feats = aggregate_dihedral_features(
        central_idx=central_idx,
        target_idx=target_idx,
        neighbors=neighbors,
        coordinates=coordinates,
        num_centers=dihedral_dim
    ) if dihedral_dim > 0 and coordinates is not None and neighbors is not None else [0.0] * dihedral_dim

    return fbond + dist_feats + ang_feats + dihedral_feats


class MolGraph:
    """
    A MolGraph represents the graph structure and featurization of a single molecule.

    A MolGraph computes the following attributes:
    - smiles: Smiles string.
    - n_atoms: The number of atoms in the molecule.
    - n_bonds: The number of bonds in the molecule.
    - f_atoms: A mapping from an atom index to a list atom features.
    - f_bonds: A mapping from a bond index to a list of bond features.
    - a2b: A mapping from an atom index to a list of incoming bond indices.
    - b2a: A mapping from a bond index to the index of the atom the bond originates from.
    - b2revb: A mapping from a bond index to the index of the reverse bond.
    - atom_coordinates: 3D coordinates for each atom (List of (x, y, z) tuples).
    """

    def __init__(self, smiles: str, args: Namespace, atom_coordinates: List[Tuple[float, float, float]] = None):
        """
        Computes the graph structure and featurization of a molecule.

        :param smiles: A smiles string.
        :param args: Arguments.
        :param atom_coordinates: Optional list of (x, y, z) tuples for 3D coordinates of each atom.
        """
        self.smiles = smiles
        self.n_atoms = 0  # number of atoms
        self.n_bonds = 0  # number of bonds
        self.f_atoms = []  # mapping from atom index to atom features
        self.f_bonds = []  # mapping from bond index to concat(in_atom, bond) features
        self.a2b = []  # mapping from atom index to incoming bond indices
        self.b2a = []  # mapping from bond index to the index of the atom the bond is coming from
        self.b2revb = []  # mapping from bond index to the index of the reverse bond
        self.bonds = []
        # Convert smiles to molecule
        mol = Chem.MolFromSmiles(smiles)

        # fake the number of "atoms" if we are collapsing substructures
        self.n_atoms = mol.GetNumAtoms()
        self.atom_coordinates = atom_coordinates  # 3D coordinates for each atom

        # Get atom features
        for i, atom in enumerate(mol.GetAtoms()):
            self.f_atoms.append(atom_features(atom))
        self.f_atoms = [self.f_atoms[i] for i in range(self.n_atoms)]

        # Build adjacency for angular features
        self.neighbors = [[] for _ in range(self.n_atoms)]
        for bond in mol.GetBonds():
            a = bond.GetBeginAtomIdx()
            b = bond.GetEndAtomIdx()
            self.neighbors[a].append(b)
            self.neighbors[b].append(a)

        for _ in range(self.n_atoms):
            self.a2b.append([])

        # Get bond features (directional: a1->a2 and a2->a1)
        use_2d_bonds = self.atom_coordinates is None  # 2D mode: chemical-only bond features
        for a1 in range(self.n_atoms):
            for a2 in range(a1 + 1, self.n_atoms):
                bond = mol.GetBondBetweenAtoms(a1, a2)

                if bond is None:
                    continue

                if use_2d_bonds:
                    # 2D mode: chemical bond features only
                    f_bond = bond_features_2d(bond)
                    if args.atom_messages:
                        self.f_bonds.append(f_bond)
                        self.f_bonds.append(f_bond)
                    else:
                        self.f_bonds.append(self.f_atoms[a1] + f_bond)
                        self.f_bonds.append(self.f_atoms[a2] + f_bond)
                else:
                    # 3D mode: chemical + distance RBF + angular CBF + dihedral
                    distance = None
                    if len(self.atom_coordinates) > max(a1, a2):
                        from mmgnn.utils import calculate_3d_distance
                        distance = calculate_3d_distance(
                            self.atom_coordinates[a1],
                            self.atom_coordinates[a2]
                        )
                    f_bond_fwd = bond_features(
                        bond,
                        distance=distance,
                        central_idx=a1,
                        target_idx=a2,
                        neighbors=self.neighbors,
                        coordinates=self.atom_coordinates
                    )
                    f_bond_rev = bond_features(
                        bond,
                        distance=distance,
                        central_idx=a2,
                        target_idx=a1,
                        neighbors=self.neighbors,
                        coordinates=self.atom_coordinates
                    )
                    if args.atom_messages:
                        self.f_bonds.append(f_bond_fwd)
                        self.f_bonds.append(f_bond_rev)
                    else:
                        self.f_bonds.append(self.f_atoms[a1] + f_bond_fwd)
                        self.f_bonds.append(self.f_atoms[a2] + f_bond_rev)

                # Update index mappings
                b1 = self.n_bonds
                b2 = b1 + 1
                self.a2b[a2].append(b1)  # b1 = a1 --> a2
                self.b2a.append(a1)
                self.a2b[a1].append(b2)  # b2 = a2 --> a1
                self.b2a.append(a2)
                self.b2revb.append(b2)
                self.b2revb.append(b1)
                self.n_bonds += 2
                self.bonds.append(np.array([a1, a2]))
        # rectify a2b
# =============================================================================
#         for ix in range(len(self.a2b)):
#             if len(self.a2b[ix]) <= 1:
#                 continue
#             if len(self.a2b[ix]) == 2:
#                 self.a2b[ix] = [self.a2b[ix][0], -1, self.a2b[ix][1]]
# =============================================================================
# =============================================================================
#         for ix in range(len(self.a2b)):
#             self.a2b[ix] = sorted(self.a2b[ix])
# =============================================================================

class BatchMolGraph:
    """
    A BatchMolGraph represents the graph structure and featurization of a batch of molecules.

    A BatchMolGraph contains the attributes of a MolGraph plus:
    - smiles_batch: A list of smiles strings.
    - n_mols: The number of molecules in the batch.
    - atom_fdim: The dimensionality of the atom features.
    - bond_fdim: The dimensionality of the bond features (technically the combined atom/bond features).
    - a_scope: A list of tuples indicating the start and end atom indices for each molecule.
    - b_scope: A list of tuples indicating the start and end bond indices for each molecule.
    - max_num_bonds: The maximum number of bonds neighboring an atom in this batch.
    - b2b: (Optional) A mapping from a bond index to incoming bond indices.
    - a2a: (Optional): A mapping from an atom index to neighboring atom indices.
    - atom_coordinates_batch: List of 3D coordinates for each molecule in the batch.
    """

    def __init__(self, mol_graphs: List[MolGraph], args: Namespace):
        self.smiles_batch = [mol_graph.smiles for mol_graph in mol_graphs]
        self.n_mols = len(self.smiles_batch)

        self.atom_fdim = get_atom_fdim(args)
        self.bond_fdim = get_bond_fdim(args) + (not args.atom_messages) * self.atom_fdim # * 2

        # Start n_atoms and n_bonds at 1 b/c zero padding
        self.n_atoms = 1  # number of atoms (start at 1 b/c need index 0 as padding)
        self.n_bonds = 1  # number of bonds (start at 1 b/c need index 0 as padding)
        self.a_scope = []  # list of tuples indicating (start_atom_index, num_atoms) for each molecule
        self.b_scope = []  # list of tuples indicating (start_bond_index, num_bonds) for each molecule

        # Store 3D coordinates for each molecule (for future bond feature calculations)
        self.atom_coordinates_batch = []  # List of coordinate lists for each molecule

        # All start with zero padding so that indexing with zero padding returns zeros
        f_atoms = [[0] * self.atom_fdim]  # atom features
        f_bonds = [[0] * self.bond_fdim]  # combined atom/bond features
        a2b = [[]]  # mapping from atom index to incoming bond indices
        b2a = [0]  # mapping from bond index to the index of the atom the bond is coming from
        b2revb = [0]  # mapping from bond index to the index of the reverse bond
        bonds = [[0,0]]
        for mol_graph in mol_graphs:
            # Store 3D coordinates
            self.atom_coordinates_batch.append(mol_graph.atom_coordinates)
            f_atoms.extend(mol_graph.f_atoms)
            f_bonds.extend(mol_graph.f_bonds)

            for a in range(mol_graph.n_atoms):
                a2b.append([b + self.n_bonds for b in mol_graph.a2b[a]]) #  if b!=-1 else 0

            for b in range(mol_graph.n_bonds):
                b2a.append(self.n_atoms + mol_graph.b2a[b])
                b2revb.append(self.n_bonds + mol_graph.b2revb[b])
                bonds.append([b2a[-1], 
                              self.n_atoms + mol_graph.b2a[mol_graph.b2revb[b]]])
            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
            self.b_scope.append((self.n_bonds, mol_graph.n_bonds))
            self.n_atoms += mol_graph.n_atoms
            self.n_bonds += mol_graph.n_bonds
        
        bonds = np.array(bonds).transpose(1,0)
        
        self.max_num_bonds = max(1, max(len(in_bonds) for in_bonds in a2b)) # max with 1 to fix a crash in rare case of all single-heavy-atom mols
        
        self.f_atoms = torch.FloatTensor(f_atoms)
        self.f_bonds = torch.FloatTensor(f_bonds)
        self.a2b = torch.LongTensor([a2b[a][:self.max_num_bonds] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)])
        self.b2a = torch.LongTensor(b2a)
        self.bonds = torch.LongTensor(bonds)
        self.b2revb = torch.LongTensor(b2revb)
        self.b2b = None  # try to avoid computing b2b b/c O(n_atoms^3)
        self.a2a = None  # only needed if using atom messages

    def get_components(self) -> Tuple[torch.FloatTensor, torch.FloatTensor,
                                      torch.LongTensor, torch.LongTensor, torch.LongTensor,
                                      List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Returns the components of the BatchMolGraph.

        :return: A tuple containing PyTorch tensors with the atom features, bond features, and graph structure
        and two lists indicating the scope of the atoms and bonds (i.e. which molecules they belong to).
        """
        return self.f_atoms, self.f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope, self.bonds

    def get_b2b(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each bond index to all the incoming bond indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incoming bond indices.
        """

        if self.b2b is None:
            b2b = self.a2b[self.b2a]  # num_bonds x max_num_bonds
            # b2b includes reverse edge for each bond so need to mask out
            revmask = (b2b != self.b2revb.unsqueeze(1).repeat(1, b2b.size(1))).long()  # num_bonds x max_num_bonds
            self.b2b = b2b * revmask

        return self.b2b

    def get_a2a(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each atom index to all neighboring atom indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incodming bond indices.
        """
        if self.a2a is None:
            # b = a1 --> a2
            # a2b maps a2 to all incoming bonds b
            # b2a maps each bond b to the atom it comes from a1
            # thus b2a[a2b] maps atom a2 to neighboring atoms a1
            self.a2a = self.b2a[self.a2b]  # num_atoms x max_num_bonds

        return self.a2a


def register_molecule_coordinates(smiles: str, coordinates: List[Tuple[float, float, float]]):
    """
    Registers 3D coordinates for a molecule identified by its SMILES string.
    This allows mol2graph to retrieve coordinates when creating molecular graphs.
    
    :param smiles: SMILES string of the molecule.
    :param coordinates: List of (x, y, z) tuples for each atom's 3D coordinates.
    """
    global SMILES_TO_COORDINATES
    if coordinates is not None:
        SMILES_TO_COORDINATES[smiles] = coordinates


def mol2graph(smiles_batch: List[str],
              args: Namespace,
              atom_coordinates_batch: List[List[Tuple[float, float, float]]] = None) -> BatchMolGraph:
    """
    Converts a list of SMILES strings to a BatchMolGraph. In 2D mode coordinates are ignored; in 3D mode
    uses atom_coordinates_batch or SMILES_TO_COORDINATES cache.
    """
    mode = getattr(args, 'mode', '3d')
    mol_graphs = []
    for i, smiles in enumerate(smiles_batch):
        coordinates = None
        if mode == '3d':
            if atom_coordinates_batch is not None and i < len(atom_coordinates_batch):
                coordinates = atom_coordinates_batch[i]
            elif smiles in SMILES_TO_COORDINATES:
                coordinates = SMILES_TO_COORDINATES[smiles]
        # 2D mode: coordinates stay None
        cache_key = smiles
        if cache_key in SMILES_TO_GRAPH:
            mol_graph = SMILES_TO_GRAPH[cache_key]
            if coordinates is not None:
                mol_graph.atom_coordinates = coordinates
        else:
            mol_graph = MolGraph(smiles, args, atom_coordinates=coordinates)
            if not args.no_cache:
                SMILES_TO_GRAPH[cache_key] = mol_graph
        mol_graphs.append(mol_graph)
    return BatchMolGraph(mol_graphs, args)
