from argparse import Namespace
from typing import List, Union
import torch
import torch.nn as nn
import numpy as np

from mmgnn.features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph
from mmgnn.nn_utils import index_select_ND, get_activation_function
import math
import torch.nn.functional as F

class MPNEncoder(nn.Module):
    def __init__(self, args: Namespace, atom_fdim: int, bond_fdim: int):
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.dropout
        self.layers_per_message = 1
        self.undirected = args.undirected
        self.atom_messages = args.atom_messages
        self.features_only = args.features_only
        self.use_input_features = args.use_input_features
        self.args = args

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Input
        input_dim = self.atom_fdim
        self.W_i_atom = nn.Linear(input_dim, self.hidden_size, bias=self.bias)
        input_dim = self.bond_fdim
        self.W_i_bond = nn.Linear(input_dim, self.hidden_size, bias=self.bias)
        
        
        w_h_input_size_atom = self.hidden_size + self.bond_fdim
        self.W_h_atom = nn.Linear(w_h_input_size_atom, self.hidden_size, bias=self.bias)
        
        w_h_input_size_bond = self.hidden_size
        
        
        for depth in range(self.depth-1):
            self._modules[f'W_h_{depth}'] = nn.Linear(w_h_input_size_bond, self.hidden_size, bias=self.bias)
        
        self.W_o = nn.Linear(
                (self.hidden_size)*2,
                self.hidden_size)
        
        self.gru = BatchGRU(self.hidden_size)
        
        self.lr = nn.Linear(self.hidden_size*3, self.hidden_size, bias=self.bias)
        

    def forward(self,mol_graph: BatchMolGraph, features_batch=None) -> torch.FloatTensor:

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, bonds = mol_graph.get_components()
        if self.args.cuda or next(self.parameters()).is_cuda:
            f_atoms, f_bonds, a2b, b2a, b2revb = (
                    f_atoms.cuda(), f_bonds.cuda(), 
                    a2b.cuda(), b2a.cuda(), b2revb.cuda())
            
        # Input
        input_atom = self.W_i_atom(f_atoms)  # num_atoms x hidden_size
        input_atom = self.act_func(input_atom)
        message_atom = input_atom.clone()
        
        input_bond = self.W_i_bond(f_bonds)  # num_bonds x hidden_size
        message_bond = self.act_func(input_bond)
        input_bond = self.act_func(input_bond)
        # Message passing
        for depth in range(self.depth - 1):
            agg_message = index_select_ND(message_bond, a2b)
            agg_message = agg_message.sum(dim=1) * agg_message.max(dim=1)[0]
            message_atom = message_atom + agg_message
            
            # directed graph
            rev_message = message_bond[b2revb]  # num_bonds x hidden
            message_bond = message_atom[b2a] - rev_message  # num_bonds x hidden
            
            message_bond = self._modules[f'W_h_{depth}'](message_bond)
            message_bond = self.dropout_layer(self.act_func(input_bond + message_bond))
        
        agg_message = index_select_ND(message_bond, a2b)
        agg_message = agg_message.sum(dim=1) * agg_message.max(dim=1)[0]
        agg_message = self.lr(torch.cat([agg_message, message_atom, input_atom], 1))
        agg_message = self.gru(agg_message, a_scope)
        
        atom_hiddens = self.act_func(self.W_o(agg_message))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden
        
        # Readout
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                assert 0
            cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
            mol_vecs.append(cur_hiddens.mean(0))
        mol_vecs = torch.stack(mol_vecs, dim=0)
        
        return mol_vecs  # B x H
    
    def encode_atom_embeddings(self, mol_graph: BatchMolGraph) -> tuple:
        """
        Extract atom-level embeddings from the encoder (before pooling to molecular level).
        
        Returns:
        --------
        torch.Tensor: Atom-level embeddings (num_atoms x hidden_size)
        List[Tuple[int, int]]: Atom scope for each molecule
        """
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, bonds = mol_graph.get_components()
        if self.args.cuda or next(self.parameters()).is_cuda:
            f_atoms, f_bonds, a2b, b2a, b2revb = (
                    f_atoms.cuda(), f_bonds.cuda(), 
                    a2b.cuda(), b2a.cuda(), b2revb.cuda())
            
        # Input
        input_atom = self.W_i_atom(f_atoms)  # num_atoms x hidden_size
        input_atom = self.act_func(input_atom)
        message_atom = input_atom.clone()
        
        input_bond = self.W_i_bond(f_bonds)  # num_bonds x hidden_size
        message_bond = self.act_func(input_bond)
        input_bond = self.act_func(input_bond)
        
        # Message passing
        for depth in range(self.depth - 1):
            agg_message = index_select_ND(message_bond, a2b)
            agg_message = agg_message.sum(dim=1) * agg_message.max(dim=1)[0]
            message_atom = message_atom + agg_message
            
            # directed graph
            rev_message = message_bond[b2revb]  # num_bonds x hidden
            message_bond = message_atom[b2a] - rev_message  # num_bonds x hidden
            
            message_bond = self._modules[f'W_h_{depth}'](message_bond)
            message_bond = self.dropout_layer(self.act_func(input_bond + message_bond))
        
        agg_message = index_select_ND(message_bond, a2b)
        agg_message = agg_message.sum(dim=1) * agg_message.max(dim=1)[0]
        agg_message = self.lr(torch.cat([agg_message, message_atom, input_atom], 1))
        agg_message = self.gru(agg_message, a_scope)
        
        atom_hiddens = self.act_func(self.W_o(agg_message))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden
        
        return atom_hiddens, a_scope  # Return atom embeddings and scope


class BatchGRU(nn.Module):
    def __init__(self, hidden_size=300):
        super(BatchGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru  = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True, 
                           bidirectional=True)
        self.bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.bias.data.uniform_(-1.0 / math.sqrt(self.hidden_size), 
                                1.0 / math.sqrt(self.hidden_size))


    def forward(self, node, a_scope):
        hidden = node
        message = F.relu(node + self.bias)
        MAX_atom_len = max([a_size for a_start, a_size in a_scope])
        # padding
        message_lst = []
        hidden_lst = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                assert 0
            cur_message = message.narrow(0, a_start, a_size)
            cur_hidden = hidden.narrow(0, a_start, a_size)
            hidden_lst.append(cur_hidden.max(0)[0].unsqueeze(0).unsqueeze(0))
            
            cur_message = torch.nn.ZeroPad2d((0,0,0,MAX_atom_len-cur_message.shape[0]))(cur_message)
            message_lst.append(cur_message.unsqueeze(0))
            
        message_lst = torch.cat(message_lst, 0)
        hidden_lst  = torch.cat(hidden_lst, 1)
        hidden_lst = hidden_lst.repeat(2,1,1)
        cur_message, cur_hidden = self.gru(message_lst, hidden_lst)
        
        # unpadding
        cur_message_unpadding = []
        for i, (a_start, a_size) in enumerate(a_scope):
            cur_message_unpadding.append(cur_message[i, :a_size].view(-1, 2*self.hidden_size))
        cur_message_unpadding = torch.cat(cur_message_unpadding, 0)
        
        message = torch.cat([torch.cat([message.narrow(0, 0, 1), message.narrow(0, 0, 1)], 1), 
                             cur_message_unpadding], 0)
        return message


class MPN(nn.Module):
    def __init__(self,
                 args: Namespace,
                 atom_fdim: int = None,
                 bond_fdim: int = None,
                 graph_input: bool = False):
        super(MPN, self).__init__()
        self.args = args
        self.atom_fdim = atom_fdim or get_atom_fdim(args)
        self.bond_fdim = bond_fdim or get_bond_fdim(args) + \
                            (not args.atom_messages) * self.atom_fdim # * 2
        self.graph_input = graph_input
        self.encoder = MPNEncoder(self.args, self.atom_fdim, self.bond_fdim)
        
        # Subgraph reconstruction parameters (local-only; no global/dual modes)
        self.recon_pool = getattr(args, 'recon_pool', 'sum')
        if self.recon_pool == 'attn':
            self.attention = nn.Linear(2 * args.hidden_size, 1)
            self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.attention = None
            self.leaky_relu = None

    def forward(self, batch: Union[List[str], BatchMolGraph],
                features_batch: List[np.ndarray] = None,
                batched_sub: BatchMolGraph = None,
                sub_to_mol: List[int] = None) -> torch.FloatTensor:
        """
        Forward pass with optional subgraph support.
        
        Parameters:
        -----------
        batch : Union[List[str], BatchMolGraph]
            Full molecular graphs
        features_batch : List[np.ndarray], optional
            Additional features
        batched_sub : BatchMolGraph, optional
            Batched subgraphs for reconstruction
        sub_to_mol : List[int], optional
            Subgraph-to-molecule mapping
            
        Returns:
        --------
        torch.FloatTensor
            Molecular embeddings
        """
        if not self.graph_input:  # if features only, batch won't even be used
            batch = mol2graph(batch, self.args)
        
        # Local-only: if subgraphs are provided, reconstruct from subgraphs;
        # otherwise fall back to full-graph embeddings.
        if batched_sub is not None and sub_to_mol is not None:
            return self._reconstruct_full_from_subgraphs(batch, batched_sub, sub_to_mol, local_only=True)
        
        # Fallback: just use full graphs (no subgraph information)
        return self.encoder.forward(batch, features_batch)
    
    def _reconstruct_full_from_subgraphs(self, batched_full: BatchMolGraph,
                                        batched_sub: BatchMolGraph,
                                        sub_to_mol: List[int],
                                        local_only: bool = True) -> torch.FloatTensor:
        """
        Reconstruct full molecular embeddings from subgraphs.
        
        Flow:
        1. Apply message passing to subgraphs (same as chemprop) -> get atom embeddings
        2. (Optional) Get full graph atom embeddings (only needed for attention pooling or dual mode)
        3. Aggregate subgraph atom embeddings back to full graph atoms
        4. Pool reconstructed atoms to molecular embeddings (same as chemprop: mean pooling)
        
        Parameters:
        -----------
        local_only : bool
            If True (local mode), don't compute full graph embeddings (only needed for attention)
        """
        sub_atom_embeddings, sub_atom_scope = self.encoder.encode_atom_embeddings(batched_sub)
        
        if local_only and self.recon_pool in ['sum', 'mean']:
            # Get scope only (needed for atom mapping and final pooling)
            _, _, _, _, _, full_atom_scope, _, _ = batched_full.get_components()
            full_atom_embeddings = None  # Not needed for sum/mean in local mode
        else:
            # For attention pooling, need full graph embeddings
            full_atom_embeddings, full_atom_scope = self.encoder.encode_atom_embeddings(batched_full)
        
        reconstructed_atoms = self._pool_subgraph_contributions(
            sub_atom_embeddings, sub_atom_scope,
            full_atom_embeddings, full_atom_scope,
            batched_sub.atom_mol_index, batched_sub.atom_global_index,
            sub_to_mol,
            local_only=local_only
        )
        
        # Aggregate reconstructed atoms to molecular embeddings
        return self._aggregate_reconstructed_atoms(reconstructed_atoms, full_atom_scope)
    
    def _pool_subgraph_contributions(self, sub_atom_embeddings, sub_atom_scope,
                                    full_atom_embeddings, full_atom_scope,
                                    atom_mol_index, atom_global_index, sub_to_mol,
                                    local_only=True):
        """Pool subgraph contributions per atom using sum/mean/attention."""
        if sub_atom_embeddings is None or sub_atom_embeddings.numel() == 0:
            # If no subgraph embeddings, return full embeddings (if available) or empty tensor
            if full_atom_embeddings is not None:
                return full_atom_embeddings
            else:
                # Fallback: return zero tensor matching expected shape
                if full_atom_scope:
                    total_atoms = max([start + size for start, size in full_atom_scope], default=0)
                    # Use hidden_size from args if available, otherwise default
                    hidden_size = getattr(self.args, 'hidden_size', 300)
                    # Get device from sub_atom_embeddings if available, otherwise use CPU
                    device = sub_atom_embeddings.device if sub_atom_embeddings is not None else torch.device('cpu')
                    dtype = sub_atom_embeddings.dtype if sub_atom_embeddings is not None else torch.float32
                    return torch.zeros((total_atoms, hidden_size), dtype=dtype, device=device)
                return torch.empty(0)
        
        # Get device and dtype from sub_atom_embeddings if full_atom_embeddings is None
        if full_atom_embeddings is not None:
            device = full_atom_embeddings.device
            hidden_size = full_atom_embeddings.size(1)
            target_dtype = full_atom_embeddings.dtype
        else:
            device = sub_atom_embeddings.device
            hidden_size = sub_atom_embeddings.size(1)
            target_dtype = sub_atom_embeddings.dtype
        
        # Ensure tensors are on the right device
        if isinstance(atom_mol_index, list):
            atom_mol_index = torch.tensor(atom_mol_index, dtype=torch.long, device=device)
        else:
            atom_mol_index = atom_mol_index.to(device)
            
        if isinstance(atom_global_index, list):
            atom_global_index = torch.tensor(atom_global_index, dtype=torch.long, device=device)
        else:
            atom_global_index = atom_global_index.to(device)
        
        # Extract only real atoms from sub_atom_embeddings (skip padding at index 0)
        # BatchMolGraph adds padding at index 0, so we need to extract atoms using scope
        if sub_atom_scope and len(sub_atom_scope) > 0:
            real_sub_atom_list = []
            for sg_start, sg_size in sub_atom_scope:
                if sg_size > 0:
                    real_sub_atom_list.append(sub_atom_embeddings[sg_start:sg_start + sg_size])
            if real_sub_atom_list:
                real_sub_atom_embeddings = torch.cat(real_sub_atom_list, dim=0)  # (N_real_sub_atoms, hidden_size)
            else:
                real_sub_atom_embeddings = torch.empty((0, hidden_size), dtype=target_dtype, device=device)
        else:
            real_sub_atom_embeddings = sub_atom_embeddings
        
        # Verify size match
        num_real_sub_atoms = real_sub_atom_embeddings.size(0)
        num_indices = atom_mol_index.size(0)
        if num_real_sub_atoms != num_indices:
            raise ValueError(f"Size mismatch: {num_real_sub_atoms} real subgraph atoms but {num_indices} indices. "
                           f"This suggests padding/scope mismatch.")
        
        # Compute per-molecule atom start offsets
        mol_starts = torch.tensor([start for start, _ in full_atom_scope], dtype=torch.long, device=device)
        
        # Map each sub-atom to global full-graph atom index
        target_indices = mol_starts[atom_mol_index] + atom_global_index  # (N_sub_atoms,)
        
        # Compute total number of full graph atoms from scope 
        if full_atom_embeddings is not None:
            total_full_atoms = full_atom_embeddings.size(0)
        else:
            # Compute from scope: find max (start + size)
            total_full_atoms = max([start + size for start, size in full_atom_scope], default=0)
        
        # Aggregate contributions using index_add (sum pooling)
        pooled_sum = torch.zeros((total_full_atoms, hidden_size), dtype=target_dtype, device=device)
        pooled_sum.index_add_(0, target_indices, real_sub_atom_embeddings.to(target_dtype))
        
        if self.recon_pool == 'sum':
            pooled = pooled_sum
        elif self.recon_pool == 'mean':
            counts = torch.zeros((total_full_atoms,), dtype=target_dtype, device=device)
            ones = torch.ones((atom_global_index.size(0),), dtype=target_dtype, device=device)
            counts.index_add_(0, target_indices, ones)
            denom = counts.clamp_min(1.0).unsqueeze(1)
            pooled = pooled_sum / denom
        elif self.recon_pool == 'attn':
            if self.attention is None:
                # Fallback to mean if attention not initialized
                counts = torch.zeros((total_full_atoms,), dtype=target_dtype, device=device)
                ones = torch.ones((atom_global_index.size(0),), dtype=target_dtype, device=device)
                counts.index_add_(0, target_indices, ones)
                denom = counts.clamp_min(1.0).unsqueeze(1)
                pooled = pooled_sum / denom
            else:
                # Cross-Attention pooling
                h_i_full = full_atom_embeddings[target_indices] 
                h_sub_ik = real_sub_atom_embeddings 
                attn_input = torch.cat([h_i_full, h_sub_ik], dim=1) 
                
                # Compute attention scores: e_{i,k}
                e_ik = self.attention(attn_input.to(torch.float32)).squeeze(1)  
                e_ik = self.leaky_relu(e_ik)  
                
                # Normalization (Softmax per atom i)
                if target_indices.numel() == 0:
                    pooled = full_atom_embeddings
                else:
                    e_ik_f32 = e_ik.to(torch.float32)
                    target_indices_long = target_indices.to(torch.long)
                    
                    # Compute max per target atom i for numerical stability
                    try:
                        max_e_per_atom = torch.full((total_full_atoms,), -1e9, dtype=torch.float32, device=device)
                        max_e_per_atom = max_e_per_atom.scatter_reduce(0, target_indices_long, e_ik_f32, reduce='amax')
                    except Exception:
                        perm = torch.argsort(target_indices_long)
                        sorted_targets = target_indices_long[perm]
                        sorted_scores = e_ik_f32[perm]
                        unique_targets, counts_unique = torch.unique_consecutive(sorted_targets, return_counts=True)
                        last_indices = torch.cumsum(counts_unique, dim=0) - 1
                        group_maxes = torch.cummax(sorted_scores, dim=0).values[last_indices]
                        max_e_per_atom = torch.full((total_full_atoms,), -1e9, dtype=torch.float32, device=device)
                        max_e_per_atom[unique_targets] = group_maxes
                    
                    # Subtract max for numerical stability
                    max_per_sub = max_e_per_atom[target_indices_long].to(torch.float32)
                    stable_e_ik = e_ik_f32 - max_per_sub
                    
                    # Compute exp
                    exp_e_ik = torch.exp(stable_e_ik)  # (N_sub_atoms,)
                    exp_e_ik_dtype = exp_e_ik.to(target_dtype)
                    
                    # Compute normalization denominator per atom
                    sum_exp_per_atom = torch.zeros((total_full_atoms,), dtype=target_dtype, device=device)
                    sum_exp_per_atom.index_add_(0, target_indices, exp_e_ik_dtype)
                    
                    # Normalize to get α_{i,k}
                    sum_exp_per_sub = sum_exp_per_atom[target_indices_long].clamp_min(1e-6)  
                    alpha_ik = exp_e_ik_dtype / sum_exp_per_sub 
                    
                    # Aggregation
                    weighted_sub = h_sub_ik.to(target_dtype) * alpha_ik.unsqueeze(1)  
                    weighted_sum = torch.zeros((total_full_atoms, hidden_size), dtype=target_dtype, device=device)
                    weighted_sum.index_add_(0, target_indices, weighted_sub)
                    
                    pooled = weighted_sum 
        else:
            pooled = pooled_sum
        
        reconstructed = pooled
        return reconstructed
    
    def _aggregate_reconstructed_atoms(self, reconstructed_atoms, full_atom_scope):
        """Aggregate reconstructed atom embeddings to molecular embeddings."""
        mol_embeddings = []
        device = reconstructed_atoms.device
        dtype = reconstructed_atoms.dtype
        
        for mol_start, mol_size in full_atom_scope:
            if mol_size == 0:
                mol_embeddings.append(torch.zeros(
                    reconstructed_atoms.size(1), 
                    device=device,
                    dtype=dtype
                ))
            else:
                mol_atoms = reconstructed_atoms[mol_start:mol_start + mol_size]
                # Use mean pooling
                mol_embedding = mol_atoms.mean(0)
                mol_embeddings.append(mol_embedding)
        
        return torch.stack(mol_embeddings, dim=0)
