import torch.nn as nn

from dgllife.model import AttentiveFPPredictor as DGLAttentiveFPPredictor


class AttentiveFPPredictor(nn.Module):
    """AttentiveFP wrapper exposing both predictions and graph embeddings."""

    def __init__(
        self,
        node_feat_size,
        edge_feat_size,
        n_tasks,
        num_layers=3,
        num_timesteps=3,
        graph_feat_size=128,
        dropout=0.1,
    ):
        super().__init__()
        self.model = DGLAttentiveFPPredictor(
            node_feat_size=node_feat_size,
            edge_feat_size=edge_feat_size,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            graph_feat_size=graph_feat_size,
            n_tasks=n_tasks,
            dropout=dropout,
        )

    def forward(self, graph, node_feats, edge_feats):
        node_embeddings = self.model.gnn(graph, node_feats, edge_feats)
        graph_embeddings = self.model.readout(graph, node_embeddings, False)
        logits = self.model.predict(graph_embeddings)
        return logits, graph_embeddings
