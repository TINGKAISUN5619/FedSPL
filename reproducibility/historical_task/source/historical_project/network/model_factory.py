def unwrap_dataset(dataset_or_loader):
    dataset = getattr(dataset_or_loader, "dataset", dataset_or_loader)
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return dataset


def build_molecular_model(dataset_or_loader, encoder="mpnn", n_tasks=None):
    dataset = unwrap_dataset(dataset_or_loader)
    node_in_feats = dataset.graphs[0].ndata["h"].shape[1]
    edge_in_feats = dataset.graphs[0].edata["e"].shape[1]
    if n_tasks is None:
        n_tasks = getattr(dataset, "n_tasks", 1)

    if encoder == "attentivefp":
        from network.myAttentiveFPPredictor import AttentiveFPPredictor

        return AttentiveFPPredictor(
            node_feat_size=node_in_feats,
            edge_feat_size=edge_in_feats,
            num_layers=3,
            num_timesteps=3,
            graph_feat_size=128,
            n_tasks=n_tasks,
            dropout=0.1,
        )

    if encoder != "mpnn":
        raise ValueError("Unsupported molecular encoder: {}".format(encoder))

    from network.myMPNNPredictor import MPNNPredictor

    return MPNNPredictor(
        node_in_feats=node_in_feats,
        edge_in_feats=edge_in_feats,
        node_out_feats=64,
        edge_hidden_feats=16,
        num_step_message_passing=3,
        num_step_set2set=3,
        num_layer_set2set=3,
        n_tasks=n_tasks,
    )
