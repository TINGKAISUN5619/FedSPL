from argparse import Namespace
import torch
import torch.nn as nn
from .mpn import MPN
from mmgnn.nn_utils import get_activation_function, initialize_weights


class MoleculeModel(nn.Module):
    """A MoleculeModel is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, classification: bool, multiclass: bool):
        """
        Initializes the MoleculeModel.

        :param classification: Whether the model is a classification model.
        """
        super(MoleculeModel, self).__init__()

        self.classification = classification
        if self.classification:
            self.sigmoid = nn.Sigmoid()
        self.multiclass = multiclass
        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)
        assert not (self.classification and self.multiclass)

    def create_encoder(self, args: Namespace):
        """
        Creates the message passing encoder for the model.

        :param args: Arguments.
        """
        self.encoder = MPN(args)

    def create_ffn(self, args: Namespace):
        """
        Creates the feed-forward network for the model.

        :param args: Arguments.
        """
        self.multiclass = args.dataset_type == 'multiclass'
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
        
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            # Encoder always returns a single embedding per molecule of size hidden_size
            first_linear_dim = args.hidden_size
            if args.use_input_features:
                first_linear_dim += args.features_dim
        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)
        if args.ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.ffn_hidden_size)
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, args.output_size),
            ])

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)

    def forward(self, *input):
        """
        Runs the MoleculeModel on input.
        
        Supports three modes:
        - Global: Full graph only (same as chemprop)
        - Local: Subgraph reconstruction only (subgraphs processed same as chemprop, then aggregated back)
        - Dual: Both global (full graph) and local (reconstructed) branches combined

        :param input: Input. Can be:
            - (batch, features_batch) for global mode
            - (batch, features_batch, batched_sub, sub_to_mol) for subgraph modes
        :return: The output of the MoleculeModel.
        """
        # Parse input arguments
        if len(input) == 2:
            # Baseline mode: (batch, features_batch)
            batch, features_batch = input
            batched_sub = None
            sub_to_mol = None
        elif len(input) >= 4:
            # Subgraph mode: (batch, features_batch, batched_sub, sub_to_mol)
            batch, features_batch, batched_sub, sub_to_mol = input[0], input[1], input[2], input[3]
        else:
            batch = input[0]
            features_batch = input[1] if len(input) > 1 else None
            batched_sub = input[2] if len(input) > 2 else None
            sub_to_mol = input[3] if len(input) > 3 else None
        
        # Get embeddings from encoder
        embeddings = self.encoder(batch, features_batch, batched_sub, sub_to_mol)
        output = self.ffn(embeddings)

        if self.classification and not self.training:
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape((output.size(0), -1, self.num_classes)) 
            if not self.training:
                output = self.multiclass_softmax(output) 

        return output


def build_model(args: Namespace) -> nn.Module:
    """
    Builds a MoleculeModel, which is a message passing neural network + feed-forward layers.

    :param args: Arguments.
    :return: A MoleculeModel containing the MPN encoder along with final linear layers with parameters initialized.
    """
    output_size = args.num_tasks
    args.output_size = output_size
    if args.dataset_type == 'multiclass':
        args.output_size *= args.multiclass_num_classes

    model = MoleculeModel(classification=args.dataset_type == 'classification', multiclass=args.dataset_type == 'multiclass')
    model.create_encoder(args)
    model.create_ffn(args)

    initialize_weights(model)

    return model
