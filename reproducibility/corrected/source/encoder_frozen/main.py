import os
import random
import sys
from datetime import datetime
import numpy as np
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
from data_loader import load_partition_data
from fedavg_api import FedAvgAPI
from fedml_api.standalone.fedavg.my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS
from easydict import EasyDict
import argparse



def load_data(args):
    args_batch_size = args.batch_size
    if args.batch_size <= 0:
        full_batch = True
        args.batch_size = 128
    else:
        full_batch = False

    data_loader = load_partition_data
    train_data_num, test_data_num, train_data_global, test_data_global, \
    train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
    class_num = data_loader(args)

    if full_batch:
        train_data_global = combine_batches(train_data_global)
        test_data_global = combine_batches(test_data_global)
        train_data_local_dict = {cid: combine_batches(train_data_local_dict[cid]) for cid in
                                 train_data_local_dict.keys()}
        test_data_local_dict = {cid: combine_batches(test_data_local_dict[cid]) for cid in test_data_local_dict.keys()}
        args.batch_size = args_batch_size

    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
    return dataset


def combine_batches(batches):
    full_x = torch.from_numpy(np.asarray([])).float()
    full_y = torch.from_numpy(np.asarray([])).long()
    for (batched_x, batched_y) in batches:
        full_x = torch.cat((full_x, batched_x), 0)
        full_y = torch.cat((full_y, batched_y), 0)
    return [(full_x, full_y)]

def custom_model_trainer(args, model):
    return MyModelTrainerCLS(model)


if __name__ == "__main__":
    # test
    now = datetime.now()
    dt_string = now.strftime("%Y/%m/%d %H:%M")

    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, default='freesolv',
                            help='esol, lipo, freesolv, BACE, BBBP, ClinTox, SIDER, Tox21, qm9')
    parser.add_argument('-fedmid', type=str, default='oursvatFLITPLUS',
                        help='avg, moon, centralized, individual, individual_proto, fedprox, fedproto, fpl, fedtgp, fedkd, topofedmd, topofedrep, topofedproto, fedkd_proto, fedavg_proto, oursFLIT, oursvatFLITPLUS')
    parser.add_argument('-comm_round', type=int, default=50, help='number of communication rounds in total')
    parser.add_argument('-numClient', type=int, default=4, help='number of clients in totoal')
    parser.add_argument('-tmpFed', type=float, default=0.5, help='temperature scale for weight, search from [0.5, 1, 2]')
    parser.add_argument('-weightReg', type=float, default=1, help='weight for regulization term, set as 1 for FLIT+')
    parser.add_argument('-lambdavat', type=float, default=0.5, help='parameter for weight of vat, search from [0.01, 0.1, 1]')
    parser.add_argument('-xi', type=float, default=0.001, help='xi for vat')
    parser.add_argument('-part_alpha', type=float, default=0.1, help='partition alpha, vary from [0.1, 0.5, 1]')
    parser.add_argument('-seed', type=int, default=0, help='random seed')
    parser.add_argument('--split_seed', type=int, default=None,
                        help='Seed for the global train/held-out split; defaults to -seed')
    parser.add_argument('--partition_seed', type=int, default=None,
                        help='Seed for client partitioning; defaults to -seed')
    parser.add_argument('--partition_method', type=str, default='hetero', choices=['hetero', 'homo'],
                        help='Scaffold-Dirichlet or IID client partitioning')
    parser.add_argument('--encoder', type=str, default='mpnn', choices=['mpnn', 'attentivefp'],
                        help='Molecular graph encoder used by every compared method')
    parser.add_argument('--distill_warmup_rounds', type=int, default=1, help='Number of initial rounds to skip distillation')
    parser.add_argument('--public_sample_size', type=int, default=512, help='Public proxy molecules sampled and excluded from private training')
    parser.add_argument('--topo_beta', type=float, default=0.3, help='Topology smoothing strength for topofedmd')
    parser.add_argument('--topo_top_k', type=int, default=20, help='Keep top-k topology neighbors per public molecule')
    parser.add_argument('--topoproto_clusters', type=int, default=16, help='Local topology clusters per client for topofedproto')
    parser.add_argument('--topoproto_global_clusters', type=int, default=16, help='Server topology prototype groups for topofedproto')
    parser.add_argument('--lambda_proto', type=float, default=0.1, help='Weight for topology prototype regularization')
    parser.add_argument('--proto_support_tau', type=float, default=0.0,
                        help='Downweight low-support global prototypes by n/(n+tau); 0 disables weighting')
    parser.add_argument('--proto_weighted_kmeans', action='store_true',
                        help='Weight uploaded descriptor centroids by cluster support in server KMeans')
    parser.add_argument('--topoproto_match', type=str, default='hard', choices=['hard', 'soft', 'contrastive'],
                        help='Prototype matching rule for topofedproto')
    parser.add_argument('--topoproto_tau', type=float, default=0.2, help='Temperature for soft topology prototype matching')
    parser.add_argument('--proto_descriptor_space', type=str, default='fingerprint',
                        choices=['fingerprint', 'maccs', 'embedding', 'random', 'physchem'],
                        help='Descriptor space used to form and match structure prototypes')
    parser.add_argument('--random_descriptor_dim', type=int, default=2048,
                        help='Descriptor dimension for the random negative-control prototype space')
    parser.add_argument('--svd_rank', type=int, default=16, help='Public representation SVD rank for topofedrep')
    parser.add_argument('--svd_eta', type=float, default=0.5, help='Power applied to singular values in topofedrep')
    parser.add_argument('--rep_teacher_space', type=str, default='svd', choices=['svd', 'raw_relation'],
                        help='Build representation teacher from SVD coordinates or directly averaged raw embedding relations')
    parser.add_argument('--lambda_rep', type=float, default=0.1, help='Weight for public representation relation distillation')
    parser.add_argument('--lambda_feat', type=float, default=0.0, help='Weight for aligned public embedding feature distillation')
    parser.add_argument('--rep_top_k', type=int, default=0,
                        help='Use only top-k teacher neighbors per public batch for relation distillation; 0 uses all pairs')
    parser.add_argument('--rep_tau', type=float, default=0.2, help='Temperature for public representation relation distillation')
    parser.add_argument('--logit_teacher_mode', type=str, default='avg', choices=['avg', 'selective_confidence'],
                        help='Aggregate public logits by uniform/sample average or per-sample per-task confidence weighting')
    parser.add_argument('--selective_logit_gamma', type=float, default=2.0,
                        help='Power applied to public prediction confidence for selective logit aggregation')
    parser.add_argument('--selective_logit_floor', type=float, default=0.02,
                        help='Minimum confidence weight floor for selective logit aggregation')
    parser.add_argument('--lambda_task_proto', type=float, default=0.0,
                        help='Weight for task-wise positive/negative prototype regularization during local training')
    parser.add_argument('--task_proto_min_count', type=int, default=2,
                        help='Minimum local/global count required for task-wise prototypes')
    parser.add_argument('--fedproto_lambda', type=float, default=0.1,
                        help='FedProto task/class prototype loss weight; copied to lambda_task_proto when fedmid=fedproto')
    parser.add_argument('--moon_mu', type=float, default=1.0,
                        help='MOON model-contrastive loss weight')
    parser.add_argument('--moon_tau', type=float, default=0.5,
                        help='MOON model-contrastive temperature')
    parser.add_argument('--fedtgp_lr', type=float, default=0.01,
                        help='Server learning rate for trainable FedTGP prototypes')
    parser.add_argument('--fedtgp_steps', type=int, default=20,
                        help='Server optimization steps for trainable FedTGP prototypes per round')
    parser.add_argument('--fedtgp_tau', type=float, default=0.2,
                        help='FedTGP prototype contrast temperature')
    parser.add_argument('--fedprox_mu', type=float, default=0.01,
                        help='FedProx proximal regularization coefficient')
    parser.add_argument('--temperature', type=float, default=1.0, help='Distillation temperature')
    parser.add_argument('--alpha_kd', type=float, default=0.7, help='Weight of public-set distillation loss')
    parser.add_argument('--public_supervised_weight', type=float, default=1.0,
                        help='Multiplier for supervised loss on public data during distillation; set 0 for FedMD-style unlabeled public distillation')
    parser.add_argument('--distill_target', type=str, default='client', choices=['client', 'server'],
                        help='Apply public-set distillation on clients or on the server model')
    parser.add_argument('--server_distill_epochs', type=int, default=1,
                        help='Public distillation epochs for the server model')
    parser.add_argument('--server_distill_lr', type=float, default=1e-4,
                        help='Learning rate for server-side public distillation')
    parser.add_argument('--server_init', type=str, default='previous', choices=['previous', 'fedavg'],
                        help='Server model initialization before server-side public distillation')
    parser.add_argument('--teacher_mode', type=str, default='global', choices=['global', 'leave_one_out'],
                        help='Public teacher construction for topofedrep')
    parser.add_argument('--teacher_weighting', type=str, default='sample', choices=['sample', 'selective', 'scaffold'],
                        help='Weight public teacher aggregation by sample count or selective density reliability')
    parser.add_argument('--teacher_sample_rho', type=float, default=1.0,
                        help='Exponent applied to client sample count for non-selective teacher aggregation; 0 gives uniform FedMD-style averaging')
    parser.add_argument('--selective_topo_weight', type=float, default=0.5,
                        help='Weight of SVD relation distance in selective teacher weighting')
    parser.add_argument('--selective_tau', type=float, default=1.0,
                        help='Kernel temperature for selective teacher density weighting')
    parser.add_argument('--selective_sample_rho', type=float, default=1.0,
                        help='Exponent applied to client sample count in selective teacher weighting')
    parser.add_argument('--selective_min_weight', type=float, default=0.05,
                        help='Minimum density floor for selective teacher weighting')
    parser.add_argument('--results_dir', type=str, default='results', help='Directory for local CSV logs')
    parser.add_argument('--local_steps_per_round', type=int, default=None,
                        help='Override local optimization steps per communication round for quick experiments')
    parser.add_argument('--global_only_eval', action='store_true',
                        help='Skip local client evaluation and log only global client AUC each round')
    parser.add_argument('--eval_all_clients', action='store_true',
                        help='Evaluate every client each round while keeping the training client sampling unchanged')
    parser.add_argument('--eval_frequency', type=int, default=1,
                        help='Evaluate every N communication rounds')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers for global train/test loaders')
    parser.add_argument('--clients_per_round', type=int, default=None,
                        help='Clients sampled per round; default follows the earlier Tox21 setup')
    parser.add_argument('--detect_anomaly', action='store_true',
                        help='Enable torch autograd anomaly detection for debugging')
    parser.add_argument('--distill_node_dropout', type=float, default=0.0,
                        help='Node feature dropout rate during public distillation for graph augmentation')
    parser.add_argument('--proto_validate_global', action='store_true',
                        help='Also evaluate the aggregated server model with validateGlobal for prototype-based methods')
    parser.add_argument('--fedavg_validate_clients', action='store_true',
                        help='Also evaluate every selected FedAvg client each communication round')
    parser.add_argument('--skip_client_eval', action='store_true',
                        help='Skip per-client evaluation for prototype methods and only run server/global evaluation')
    argsinput = parser.parse_args()
    
    argsinput = argsinput

    args = EasyDict()
    for k, v in vars(argsinput).items():
        args[k] = v
    if argsinput.fedmid in ['fedproto', 'fpl', 'fedtgp']:
        args.lambda_task_proto = argsinput.fedproto_lambda
        argsinput.lambda_task_proto = argsinput.fedproto_lambda
    args.numWorker = argsinput.num_workers
    args.frequency_of_the_test = max(1, argsinput.eval_frequency)
    args.client_num_in_total = argsinput.numClient
    if argsinput.clients_per_round is None:
        clientSelectdict = {4: 3, 5: 4, 6: 5, 7: 6, 8: 7}
        args.client_num_per_round = clientSelectdict.get(args.client_num_in_total, args.client_num_in_total)
    else:
        args.client_num_per_round = argsinput.clients_per_round
    args.client_optimizer = 'adam'
    args.batch_size = 64
    args.xi = argsinput.xi
    args.seed = argsinput.seed
    args.model_seed = argsinput.seed
    args.split_seed = argsinput.seed if argsinput.split_seed is None else argsinput.split_seed
    args.partition_seed = argsinput.seed if argsinput.partition_seed is None else argsinput.partition_seed
    args.partition_method = argsinput.partition_method
    totalsteps = 10000
    args.epochs = 1000

    device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")

    if argsinput.dataset == 'qm9':
        totalsteps = 100000

    args.comm_round = argsinput.comm_round
    if argsinput.local_steps_per_round is None:
        args.localStepsPerRound = int(totalsteps/args.comm_round)
    else:
        args.localStepsPerRound = argsinput.local_steps_per_round

    args.dataset = argsinput.dataset
    args.partition_alpha = argsinput.part_alpha

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset = load_data(args)

    if args.dataset == 'qm9':
        from network.myschnet import SchNet
        model = SchNet(hidden_channels=128, num_filters=128, num_interactions=6,
                       num_gaussians=50, cutoff=10.0)
    elif args.dataset in ['esol', 'lipo', 'freesolv']:
        from network.model_factory import build_molecular_model
        model = build_molecular_model(dataset[2], encoder=args.encoder, n_tasks=1)
    elif args.dataset in ['MUV', 'BACE', 'BBBP', 'ClinTox', 'SIDER', 'ToxCast', 'HIV', 'PCBA', 'Tox21']:
        from network.model_factory import build_molecular_model
        model = build_molecular_model(dataset[2], encoder=args.encoder)
    else:
        raise ValueError('not found dataset')

    model_trainer = MyModelTrainerCLS(model, args, argsinput)
    print(argsinput)
    fedavgAPI = FedAvgAPI(dataset, device, args, model_trainer, argsinput)
    print(args)
    fedavgAPI.train()
