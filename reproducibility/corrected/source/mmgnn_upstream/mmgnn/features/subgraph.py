from argparse import Namespace
from typing import List, Tuple, Set, Optional
import torch
import numpy as np
from rdkit import Chem
from mmgnn.utils import calculate_3d_distance
from .featurization import (
    atom_features, bond_features, bond_features_2d, get_atom_fdim, get_bond_fdim,
    BatchMolGraph, MolGraph, RBF_CUTOFF
)


class SimpleMolGraph2D:
    """Simple molecular graph representation for subgraphs."""
    
    def __init__(self, f_atoms, f_bonds, a2b, b2a, b2revb, n_atoms, n_bonds, 
                 global_indices=None, sub_atom_types=None, atom_coordinates=None):
        """Initialize a simple molecular graph."""
        self.f_atoms = f_atoms
        self.f_bonds = f_bonds
        self.a2b = a2b
        self.b2a = b2a
        self.b2revb = b2revb
        self.n_atoms = n_atoms
        self.n_bonds = n_bonds
        self.global_indices = global_indices  # Mapping from local to global atom indices
        self.sub_atom_types = sub_atom_types
        self.atom_coordinates = atom_coordinates  # List of (x, y, z) tuples for each atom


def generate_subgraphs_from_molgraph(mol_graph: MolGraph, args: Namespace) -> List[SimpleMolGraph2D]:
    """
    Generate subgraphs from a MolGraph based on atom types.
    
    Parameters:
    -----------
    mol_graph : MolGraph
        The full molecular graph from chemprop
    args : Namespace
        Arguments for featurization
        
    Returns:
    --------
    List[SimpleMolGraph2D]
        List of subgraphs including the original full graph as the last element
    """
    subgraphs = []
    
    # Get atom types from the molecule
    mol = Chem.MolFromSmiles(mol_graph.smiles)
    if mol is None:
        return []
    
    atoms = list(mol.GetAtoms())
    atom_types = [atom.GetSymbol() for atom in atoms]
    unique_types = list(set(atom_types))
    type_to_indices = {t: [i for i, at in enumerate(atom_types) if at == t] 
                      for t in unique_types}
    
    # Get bond list from MolGraph (used in both 2D and 3D)
    bond_list = []
    n_atoms = mol_graph.n_atoms
    for a1 in range(n_atoms):
        for a2 in range(a1 + 1, n_atoms):
            bond = mol.GetBondBetweenAtoms(a1, a2)
            if bond is not None:
                bond_list.append((a1, a2, bond))

    mode = getattr(args, 'mode', '3d')
    if mode == '2d':
        # 2D: only chemical bonds (bond_list), chemical bond features
        for t in unique_types:
            indices = type_to_indices[t]
            sg = _build_subgraph_2d(mol_graph, indices, {(t, t)}, bond_list, atom_types, args)
            if sg is not None and sg.n_bonds > 0:
                subgraphs.append(sg)
        for i, t1 in enumerate(unique_types):
            for t2 in unique_types[i+1:]:
                indices = type_to_indices[t1] + type_to_indices[t2]
                allowed_pairs = {(t1, t2), (t2, t1)}
                sg = _build_subgraph_2d(mol_graph, indices, allowed_pairs, bond_list, atom_types, args)
                if sg is not None and sg.n_bonds > 0:
                    subgraphs.append(sg)
        full_graph = _build_full_subgraph_2d(mol_graph, atom_types, args)
    else:
        # 3D: all-to-all within type pairs with 3D distance/angle features
        for t in unique_types:
            indices = type_to_indices[t]
            sg = _build_subgraph(mol_graph, indices, {(t, t)}, bond_list, atom_types, args)
            if sg is not None and sg.n_bonds > 0:
                subgraphs.append(sg)
        for i, t1 in enumerate(unique_types):
            for t2 in unique_types[i+1:]:
                indices = type_to_indices[t1] + type_to_indices[t2]
                allowed_pairs = {(t1, t2), (t2, t1)}
                sg = _build_subgraph(mol_graph, indices, allowed_pairs, bond_list, atom_types, args)
                if sg is not None and sg.n_bonds > 0:
                    subgraphs.append(sg)
        full_graph = _build_full_subgraph(mol_graph, atom_types, args)
    subgraphs.append(full_graph)
    return subgraphs


def _build_subgraph_2d(mol_graph: MolGraph, atom_indices: List[int],
                       allowed_pairs: Set[Tuple[str, str]], bond_list: List[Tuple],
                       atom_types: List[str], args: Namespace) -> Optional[SimpleMolGraph2D]:
    """Build a bipartite subgraph (2D mode): edges only between allowed type pairs, chemical bond features only."""
    if not atom_indices:
        return None
    index_set = set(atom_indices)
    local_index_map = {global_idx: local_idx for local_idx, global_idx in enumerate(atom_indices)}
    sub_atom_types = [atom_types[i] for i in atom_indices]
    sub_f_atoms = [mol_graph.f_atoms[i] for i in atom_indices]
    n_atoms = len(sub_f_atoms)
    f_bonds = []
    a2b = [[] for _ in range(n_atoms)]
    b2a = []
    b2revb = []
    n_bonds = 0
    bond_fdim = get_bond_fdim(args)
    if not args.atom_messages:
        bond_fdim += get_atom_fdim(args)
    for a1, a2, bond in bond_list:
        if a1 not in index_set or a2 not in index_set:
            continue
        t_i, t_j = atom_types[a1], atom_types[a2]
        if (t_i, t_j) not in allowed_pairs and (t_j, t_i) not in allowed_pairs:
            continue
        local_i, local_j = local_index_map[a1], local_index_map[a2]
        f_bond = bond_features_2d(bond)
        if not args.atom_messages:
            forward = sub_f_atoms[local_i] + f_bond
            reverse = sub_f_atoms[local_j] + f_bond
        else:
            forward = f_bond
            reverse = f_bond
        if len(forward) < bond_fdim:
            forward = forward + [0] * (bond_fdim - len(forward))
        if len(reverse) < bond_fdim:
            reverse = reverse + [0] * (bond_fdim - len(reverse))
        f_bonds.append(forward)
        f_bonds.append(reverse)
        b1, b2 = n_bonds, n_bonds + 1
        a2b[local_j].append(b1)
        b2a.append(local_i)
        a2b[local_i].append(b2)
        b2a.append(local_j)
        b2revb.append(b2)
        b2revb.append(b1)
        n_bonds += 2
    return SimpleMolGraph2D(sub_f_atoms, f_bonds, a2b, b2a, b2revb, n_atoms, n_bonds,
                            global_indices=atom_indices, sub_atom_types=sub_atom_types)


def _build_full_subgraph_2d(mol_graph: MolGraph, atom_types: List[str], args: Namespace) -> SimpleMolGraph2D:
    """Build the full graph as a subgraph (2D mode)."""
    all_indices = list(range(mol_graph.n_atoms))
    all_pairs = set((t1, t2) for t1 in atom_types for t2 in atom_types)
    mol = Chem.MolFromSmiles(mol_graph.smiles)
    bond_list = []
    if mol is not None:
        for bond in mol.GetBonds():
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_list.append((a1, a2, bond))
    return _build_subgraph_2d(mol_graph, all_indices, all_pairs, bond_list, atom_types, args)


def _build_subgraph(mol_graph: MolGraph, atom_indices: List[int], 
                   allowed_pairs: Set[Tuple[str, str]], bond_list: List[Tuple],
                   atom_types: List[str], args: Namespace) -> Optional[SimpleMolGraph2D]:
    """Build a subgraph (3D mode) with all pairwise connections between allowed type pairs.
    Creates edges between ALL atoms of allowed types; bond features use 3D RBF/angular/dihedral.
    """
    if not atom_indices:
        return None
    
    index_set = set(atom_indices)
    local_index_map = {global_idx: local_idx for local_idx, global_idx in enumerate(atom_indices)}
    
    sub_atom_types = [atom_types[i] for i in atom_indices]
    sub_f_atoms = [mol_graph.f_atoms[i] for i in atom_indices]
    n_atoms = len(sub_f_atoms)
    
    f_bonds = []
    a2b = [[] for _ in range(n_atoms)]
    b2a = []
    b2revb = []
    n_bonds = 0
    
    bond_fdim = get_bond_fdim(args)
    if not args.atom_messages:
        bond_fdim += get_atom_fdim(args)
    
    # Collect valid atom indices by type
    type_to_local = {}
    for global_idx in atom_indices:
        t = atom_types[global_idx]
        if t not in type_to_local:
            type_to_local[t] = []
        type_to_local[t].append((global_idx, local_index_map[global_idx]))
    
    # Check if this is a single-type or multi-type (bipartite) subgraph
    is_single_type = len(type_to_local) == 1
    
    # Create all-to-all connections for allowed type pairs
    atom_pairs_added = set()
    
    for t1 in type_to_local:
        for t2 in type_to_local:
            # Check if this type pair is allowed
            if (t1, t2) not in allowed_pairs and (t2, t1) not in allowed_pairs:
                continue
            
            # Skip reverse direction to avoid duplicate edges
            if t1 != t2 and (t2, t1) in allowed_pairs and t1 > t2:
                continue
            
            # Connect all atoms of type t1 to all atoms of type t2
            for global_i, local_i in type_to_local[t1]:
                for global_j, local_j in type_to_local[t2]:
                    if local_i >= local_j:  # Avoid self-loops and duplicates
                        continue
                    
                    pair_key = tuple(sorted([local_i, local_j]))
                    if pair_key in atom_pairs_added:
                        continue
                    
                    # Calculate 3D distance if coordinates available
                    distance = None
                    if mol_graph.atom_coordinates is not None:
                        try:
                            distance = calculate_3d_distance(
                                mol_graph.atom_coordinates[global_i],
                                mol_graph.atom_coordinates[global_j]
                            )
                        except Exception:
                            pass
                    
                    # Enforce distance cutoff
                    if distance is None or distance > RBF_CUTOFF:
                        continue

                    # Get RBF distance features + angular CBF
                    f_bond = bond_features(
                        None,
                        distance=distance,
                        central_idx=global_i,
                        target_idx=global_j,
                        neighbors=mol_graph.neighbors,
                        coordinates=mol_graph.atom_coordinates
                    )
                    
                    if not args.atom_messages:
                        forward = sub_f_atoms[local_i] + f_bond
                        reverse = sub_f_atoms[local_j] + f_bond
                    else:
                        forward = f_bond
                        reverse = f_bond
                    
                    # Pad to bond_fdim if needed
                    if len(forward) < bond_fdim:
                        forward = forward + [0] * (bond_fdim - len(forward))
                    if len(reverse) < bond_fdim:
                        reverse = reverse + [0] * (bond_fdim - len(reverse))
                    
                    f_bonds.append(forward)
                    f_bonds.append(reverse)
                    b1 = n_bonds
                    b2 = b1 + 1
                    a2b[local_j].append(b1)
                    b2a.append(local_i)
                    a2b[local_i].append(b2)
                    b2a.append(local_j)
                    b2revb.append(b2)
                    b2revb.append(b1)
                    n_bonds += 2
                    atom_pairs_added.add(pair_key)
    
    # Extract atom_coordinates for the subgraph atoms
    sub_atom_coordinates = None
    if mol_graph.atom_coordinates is not None:
        sub_atom_coordinates = [mol_graph.atom_coordinates[idx] for idx in atom_indices]
    
    return SimpleMolGraph2D(sub_f_atoms, f_bonds, a2b, b2a, b2revb, n_atoms, n_bonds,
                           global_indices=atom_indices, sub_atom_types=sub_atom_types,
                           atom_coordinates=sub_atom_coordinates)


def _build_full_subgraph(mol_graph: MolGraph, atom_types: List[str], 
                        args: Namespace) -> SimpleMolGraph2D:
    """Build the full graph as a subgraph (for consistency)."""
    all_indices = list(range(mol_graph.n_atoms))
    all_pairs = set((t1, t2) for t1 in atom_types for t2 in atom_types)
    
    # Get bond list
    mol = Chem.MolFromSmiles(mol_graph.smiles)
    bond_list = []
    if mol is not None:
        for bond in mol.GetBonds():
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_list.append((a1, a2, bond))
    
    return _build_subgraph(mol_graph, all_indices, all_pairs, bond_list, atom_types, args)


def batch_full_graphs(graph_list: List[SimpleMolGraph2D], args: Namespace) -> Optional[BatchMolGraph]:
    """Batch a list of molecular graphs into a single batched graph."""
    if not graph_list:
        return None
    
    return _batch_graphs(graph_list, [], args)


def batch_subgraphs(subgraph_list: List[SimpleMolGraph2D], mol_indices: List[int], 
                   args: Namespace) -> Tuple[Optional[BatchMolGraph], List[int]]:
    """Batch subgraphs and keep track of molecule indices."""
    if not subgraph_list:
        return None, []
    
    batched = _batch_graphs(subgraph_list, mol_indices, args)
    
    # Attach reconstruction indices expected by the model
    atom_mol_index = []
    atom_local_index = []
    
    for sg, mol_idx in zip(subgraph_list, mol_indices):
        if getattr(sg, 'global_indices', None) is not None:
            for global_idx in sg.global_indices:
                atom_mol_index.append(mol_idx)
                atom_local_index.append(global_idx)
        else:
            for local_idx in range(sg.n_atoms):
                atom_mol_index.append(mol_idx)
                atom_local_index.append(local_idx)
    
    batched.atom_mol_index = torch.tensor(atom_mol_index, dtype=torch.long)
    batched.atom_global_index = torch.tensor(atom_local_index, dtype=torch.long)
    
    return batched, mol_indices


def _batch_graphs(graph_list: List[SimpleMolGraph2D], mol_indices: List[int], 
                 args: Namespace) -> BatchMolGraph:
    """Internal function to batch graphs with common logic."""
    if not graph_list:
        # Return empty BatchMolGraph
        from .featurization import MolGraph
        empty_graphs = []
        return BatchMolGraph(empty_graphs, args)
    
    # Create minimal MolGraph-like wrappers for BatchMolGraph constructor
    # BatchMolGraph expects a list of MolGraph objects, so we create wrappers
    class _MolGraphWrapper:
        """Wrapper to make SimpleMolGraph2D compatible with BatchMolGraph constructor."""
        def __init__(self, sg: SimpleMolGraph2D, smiles: str = ""):
            self.smiles = smiles
            self.n_atoms = sg.n_atoms
            self.n_bonds = sg.n_bonds
            self.f_atoms = sg.f_atoms
            self.f_bonds = sg.f_bonds
            self.a2b = sg.a2b
            self.b2a = sg.b2a
            self.b2revb = sg.b2revb
            self.atom_coordinates = sg.atom_coordinates  # 3D coordinates for atoms
    
    # Convert SimpleMolGraph2D to MolGraph-like wrappers
    wrapped_graphs = [_MolGraphWrapper(sg) for sg in graph_list]
    
    # Create BatchMolGraph using the standard constructor
    batched = BatchMolGraph(wrapped_graphs, args)
    
    return batched
