import logging
import os.path as osp
import numpy as np
import torch
# import  as data
try:
    import torchvision.transforms as transforms
except ImportError:
    transforms = None

try:
    from torch_geometric.datasets import QM9
except ImportError:
    QM9 = None
import dgl
from dgl.data.utils import Subset

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def collate_molgraphs(data):
    """Batching a list of datapoints for dataloader.

    Parameters
    ----------
    data : list of 3-, 4-, or 5-tuples.
        Each tuple is for a single datapoint, consisting of
        a SMILES, a DGLGraph, all-task labels, an optional binary label
        mask, and an optional stable sample index.

    Returns
    -------
    smiles : list
        List of smiles
    bg : DGLGraph
        The batched DGLGraph.
    labels : Tensor of dtype float32 and shape (B, T)
        Batched datapoint labels. B is len(data) and
        T is the number of total tasks.
    masks : Tensor of dtype float32 and shape (B, T)
        Batched datapoint binary mask, indicating the
        existence of labels.
    indices : Tensor of dtype int64 and shape (B,)
        Stable dataset indices used by algorithms with prediction caches.
    """
    smiles = [item[0] for item in data]
    graphs = [item[1] for item in data]
    raw_labels = [item[2] for item in data]
    raw_masks = []
    raw_indices = []
    for fallback_idx, item in enumerate(data):
        label = item[2]
        if len(item) >= 5:
            mask, index = item[3], item[4]
        elif len(item) == 4 and torch.is_tensor(item[3]) and item[3].dtype.is_floating_point:
            mask, index = item[3], fallback_idx
        elif len(item) == 4:
            mask, index = torch.ones_like(label), item[3]
        else:
            mask, index = torch.ones_like(label), fallback_idx
        raw_masks.append(mask)
        raw_indices.append(int(index))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(raw_labels, dim=0)
    masks = torch.stack(raw_masks, dim=0).float()
    indices = torch.tensor(raw_indices, dtype=torch.long)
    return smiles, bg, labels, masks, indices


def rearrangeLabel(y_train, minClsNum=0):
    unique_labels, inverse, counts = torch.unique(
        y_train, sorted=True, return_inverse=True, return_counts=True
    )
    if minClsNum <= 0:
        return inverse.to(y_train.dtype)

    kept = counts > minClsNum
    remap = torch.full((len(unique_labels),), -1, dtype=torch.long)
    remap[kept] = torch.arange(int(kept.sum().item()))
    mapped = remap[inverse]
    if (mapped < 0).any():
        raise ValueError("Scaffold labels below minClsNum must be filtered before partitioning")
    return mapped.to(y_train.dtype)


def partition_data(partition, n_nets, alpha, args):
    logging.info("*********partition data***************")
    datasetName = args.dataset
    if datasetName == 'qm9':
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9')
        dataset = QM9(path)
        # atomsLabel.pt; scaffoldLabel.pt
        y = torch.load('data/scaffold_result/scffoldLabel_qm9.pt').int()
        idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 11])
        dataset.data.y = dataset.data.y[:, idx]
        random_state = np.random.RandomState(seed=int(getattr(args, 'split_seed', 42)))
        perm = torch.from_numpy(random_state.permutation(np.arange(len(dataset))))
        train_idx = perm[:110000]
        val_idx = perm[110000:]
        y_train = y[train_idx]
        train_dataset, val_dataset = dataset[train_idx], dataset[val_idx]

    elif datasetName in ['esol', 'freesolv', 'lipo','MUV', 'BACE', 'BBBP', 'ClinTox', 'SIDER',
                                                    'ToxCast', 'HIV', 'PCBA', 'Tox21']:
        from dgllife.utils import smiles_to_bigraph
        from functools import partial

        from dgllife.utils import CanonicalAtomFeaturizer
        node_featurizer = CanonicalAtomFeaturizer()
        from dgllife.utils import CanonicalBondFeaturizer
        edge_featurizer = CanonicalBondFeaturizer(self_loop=True)
        if datasetName == 'freesolv':
            from data import FreeSolv
            dataset = FreeSolv(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                               node_featurizer=node_featurizer,
                               edge_featurizer=edge_featurizer,
                               n_jobs=1, load=True)
        elif datasetName == 'lipo':
            from data import Lipophilicity
            dataset = Lipophilicity(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                                    node_featurizer=node_featurizer,
                                    edge_featurizer=edge_featurizer,
                                    n_jobs=1, load=True)
        elif datasetName == 'esol':
            from data import ESOL
            dataset = ESOL(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                           node_featurizer=node_featurizer,
                           edge_featurizer=edge_featurizer,
                           n_jobs=1, load=True)
        elif datasetName == 'MUV':
            from data import MUV
            dataset = MUV(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                          node_featurizer=node_featurizer,
                          edge_featurizer=edge_featurizer,
                          n_jobs=1, load=True)
        elif datasetName == 'BACE':  #
            from data import BACE
            dataset = BACE(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                           node_featurizer=node_featurizer,
                           edge_featurizer=edge_featurizer,
                           n_jobs=1, load=True)
        elif datasetName == 'BBBP':  #
            from data import BBBP
            dataset = BBBP(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                           node_featurizer=node_featurizer,
                           edge_featurizer=edge_featurizer,
                           n_jobs=1, load=True)
        elif datasetName == 'ClinTox':  #
            from data import ClinTox
            dataset = ClinTox(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                              node_featurizer=node_featurizer,
                              edge_featurizer=edge_featurizer,
                              n_jobs=1, load=True)
        elif datasetName == 'SIDER':  #
            from data import SIDER
            dataset = SIDER(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                            node_featurizer=node_featurizer,
                            edge_featurizer=edge_featurizer,
                            n_jobs=1, load=True)
        elif datasetName == 'ToxCast':
            from data import ToxCast
            dataset = ToxCast(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                              node_featurizer=node_featurizer,
                              edge_featurizer=edge_featurizer,
                              n_jobs=1, load=True)
        elif datasetName == 'HIV':
            from data import HIV
            dataset = HIV(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                          node_featurizer=node_featurizer,
                          edge_featurizer=edge_featurizer,
                          n_jobs=1, load=True)
        elif datasetName == 'PCBA':
            from data import PCBA
            dataset = PCBA(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                           node_featurizer=node_featurizer,
                           edge_featurizer=edge_featurizer,
                           n_jobs=1, load=True)
        elif datasetName == 'Tox21':  #
            from data import Tox21
            dataset = Tox21(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                            node_featurizer=node_featurizer,
                            edge_featurizer=edge_featurizer,
                            n_jobs=1, load=True)
        else:
            raise ValueError('Unexpected dataset: {}'.format(datasetName))

        y = torch.load('data/scaffold_result/scffoldLabel_'+datasetName+'.pt').int()
        random_state = np.random.RandomState(seed=int(getattr(args, 'split_seed', 42)))
        perm = torch.from_numpy(random_state.permutation(np.arange(len(dataset))))
        train_idx = perm[:int(len(perm)*0.8)]
        val_idx = perm[int(len(perm)*0.8):]

        y_train = y[train_idx]
        train_dataset, val_dataset = Subset(dataset, train_idx), Subset(dataset, val_idx)

    n_train = len(train_dataset)

    if partition == "homo":
        total_num = n_train
        partition_rng = np.random.RandomState(int(getattr(args, 'partition_seed', 0)))
        idxs = partition_rng.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero":
        min_size = 0
        y_train = rearrangeLabel(y_train, 0)
        y_train = y_train.numpy()
        K = len(np.unique(y_train))
        N = y_train.shape[0]
        logging.info("N = " + str(N))
        net_dataidx_map = {}

        partition_rng = np.random.RandomState(int(getattr(args, 'partition_seed', 0)))
        minNumPerClient = n_train / n_nets / 2
        minNumPerClient = args.batch_size if minNumPerClient<args.batch_size else minNumPerClient
        minNumPerClient = 64
        count = 0
        while min_size < minNumPerClient:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                partition_rng.shuffle(idx_k)
                proportions1 = partition_rng.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions2 = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions1, idx_batch)])
                proportions3 = proportions2 / proportions2.sum()
                proportions = (np.cumsum(proportions3) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            count += 1
            if count>1000:
                break
                raise ValueError('Not valid training')

        for j in range(n_nets):
            partition_rng.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    assigned = np.concatenate([
        np.asarray(net_dataidx_map[client_idx], dtype=np.int64)
        for client_idx in range(n_nets)
    ])
    if len(assigned) != n_train or len(np.unique(assigned)) != n_train:
        raise RuntimeError(
            "Invalid client partition: assigned={} unique={} expected={}".format(
                len(assigned), len(np.unique(assigned)), n_train
            )
        )
    if not np.array_equal(np.sort(assigned), np.arange(n_train)):
        raise RuntimeError("Client partition does not cover every training index exactly once")

    indexlist = []
    index0 = np.where(y_train == 0)[0]
    for k,v in net_dataidx_map.items():
        indexlist = indexlist + v
        print(np.intersect1d(index0, v).shape)


# traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    return train_dataset, val_dataset, net_dataidx_map


def load_partition_data(args):
    datasetName = args.dataset
    partition_method, partition_alpha, client_number = args.partition_method,args.partition_alpha, args.client_num_in_total
    train_dataset, val_dataset, net_dataidx_map = partition_data(partition_method,
                                                                 client_number,
                                                                 partition_alpha, args)
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])
    # trainDL = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=args.numWorker, drop_last=True)
    # valDL = DataLoader(val_dataset, batch_size=args.bs, num_workers=args.numWorker)
    # collate_fn
    # collate_fn
    if datasetName in ['esol', 'freesolv', 'lipo','MUV', 'BACE', 'BBBP', 'ClinTox', 'SIDER',
                                                    'ToxCast', 'HIV', 'PCBA', 'Tox21']:
        from torch.utils.data import DataLoader
        collate_fn1 = collate_molgraphs
    elif datasetName=='qm9':
        from torch_geometric.data import DataLoader
        collate_fn1 = None
    trainDL = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.numWorker,
                         drop_last=True, pin_memory=False, collate_fn=collate_fn1)
    valDL = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.numWorker, pin_memory=False, collate_fn=collate_fn1)
    logging.info("train_dl_global number = " + str(len(trainDL)))
    logging.info("test_dl_global number = " + str(len(valDL)))
    test_data_num = len(val_dataset)

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        dataidxs = net_dataidx_map[client_idx]
        local_data_num = len(dataidxs)
        data_local_num_dict[client_idx] = local_data_num
        logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))
        dataidxs = torch.Tensor(dataidxs).long()
        # training batch size = 64; algorithms batch size = 32

        if datasetName in ['esol', 'freesolv', 'lipo', 'MUV', 'BACE', 'BBBP', 'ClinTox', 'SIDER',
                                                    'ToxCast', 'HIV', 'PCBA', 'Tox21']:
            localDataset = Subset(train_dataset, dataidxs)
        elif datasetName == 'qm9':
            localDataset = train_dataset[dataidxs]

        train_data_local = DataLoader(localDataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=0, drop_last=True, pin_memory=True,collate_fn=collate_fn1)
        test_data_local = None
        logging.info("client_idx = %d, batch_num_train_local = %d; local test disabled" % (
            client_idx, len(train_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local

    return train_data_num, test_data_num, trainDL, valDL, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, 1
