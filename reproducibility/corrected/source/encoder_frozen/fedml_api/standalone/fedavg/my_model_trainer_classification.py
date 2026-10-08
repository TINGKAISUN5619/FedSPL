import torch
from torch import nn
from tqdm import tqdm
from vat import VATLoss
import torch.nn.functional as F

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class BinaryLogitConsistencyLoss(nn.Module):
    """Cross-entropy consistency between perturbed and detached reference logits."""

    def forward(self, prediction, reference):
        target = torch.sigmoid(reference.detach())
        return F.binary_cross_entropy_with_logits(prediction, target, reduction='none')


class MyModelTrainer(ModelTrainer):
    @staticmethod
    def _base_dataset(train_data):
        dataset = train_data.dataset
        while hasattr(dataset, 'dataset'):
            dataset = dataset.dataset
        return dataset

    @staticmethod
    def _masked_mean(losses, masks=None):
        if masks is None:
            return losses.mean()
        masks = masks.to(losses.device).float()
        if masks.shape != losses.shape:
            masks = masks.view_as(losses)
        return (losses * masks).sum() / masks.sum().clamp_min(1.0)

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)
        
    def receive_soft_targets(self, soft_targets_tensor):
        self.soft_targets = soft_targets_tensor

    def train(self, train_data, device, args, roundidx, clientidx):

        fedmid = self.wandbconfig.fedmid
        weightReg = self.wandbconfig.weightReg
        tmpFed = self.wandbconfig.tmpFed
        lambdavat = self.wandbconfig.lambdavat
        warmupRound = 1

        model = self.model
        model.to(device)
        model.train()
        # init a new model
        if args.dataset == 'qm9':
            from network.myschnet import SchNet
            globalModel = SchNet(hidden_channels=128, num_filters=128, num_interactions=6,
                                 num_gaussians=50, cutoff=10.0)
            lossCriterion = nn.L1Loss(reduction='none')
            vatCriterion = lossCriterion

        elif args.dataset in ['esol', 'lipo', 'freesolv']:
            from network.model_factory import build_molecular_model
            globalModel = build_molecular_model(
                train_data, encoder=getattr(args, 'encoder', 'mpnn'), n_tasks=1
            )
            lossCriterion = nn.MSELoss(reduction='none')
            vatCriterion = lossCriterion
        elif args.dataset in ['MUV', 'BACE', 'BBBP', 'ClinTox', 'SIDER',
                              'ToxCast', 'HIV', 'PCBA', 'Tox21']:
            from network.model_factory import build_molecular_model
            globalModel = build_molecular_model(
                train_data, encoder=getattr(args, 'encoder', 'mpnn')
            )
            lossCriterion = nn.BCEWithLogitsLoss(reduction='none')
            vatCriterion = BinaryLogitConsistencyLoss()
        else:
            raise ValueError('not found dataset')
        globalModel = globalModel.to(device)
        for param_q, param_k in zip(model.parameters(), globalModel.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = True  # not update by gradient

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
        localsteps = args.localStepsPerRound
        tmpstep = 0

        fedprox_global_params = None
        if fedmid == 'fedprox':
            fedprox_global_params = [param.detach().clone() for param in model.parameters()]

        if fedmid in ['avg', 'fedkd', 'fedprox']:
            predGAll, predGAll_emb, predGAll_vat = None, None, None
        else:
            predGAll, predGAll_emb, predGAll_vat = self.globalEpoch(
                train_data, globalModel, args, device, lossCriterion, vatCriterion
            )
        weight_denomaitor = None
        tbar = tqdm(train_data, mininterval=2, disable=True)
        while tmpstep < localsteps:
            tbarLocalAll = tqdm(range(args.epochs), disable=getattr(args, 'quiet_tqdm', True))
            for epoch in enumerate(tbarLocalAll):
                batch_loss = []
                for batch_idx, data in enumerate(tbar):
                    if tmpstep >= localsteps:
                        if fedmid == 'moon':
                            embedding = self.globalEpoch(
                                train_data, self.model, args, device, lossCriterion, vatCriterion
                            )
                            if hasattr(self, 'embedding'):
                                self.embedding[clientidx] = embedding
                            else:
                                self.embedding = {}
                                self.embedding[clientidx] = embedding
                        del globalModel
                        return

                    tmpstep += 1
                    optimizer.zero_grad()
                    masks = None
                    if args.dataset == 'qm9':
                        z, pos, batch, labels = data.z.to(device), data.pos.to(device), data.batch.to(
                            device), data.y.to(
                            device)
                        if 'vat' in fedmid:
                            vatloss = VATLoss(framework='geometric', criterion=vatCriterion, xi=args.xi)  # xi, and eps
                            xCombined = [z, pos, batch]
                            localVATLoss = vatloss(model, xCombined)

                        pred, latentEmb = model(z, pos, batch)
                        index = data.idx.to(device)
                        batch_index = index.squeeze().long()
                        if predGAll is not None:
                            predG = predGAll[batch_index]
                            latentEmbG = predGAll_emb[batch_index]
                            globalVATLoss = predGAll_vat[batch_index]


                    elif args.dataset in ['esol', 'lipo', 'freesolv', 'MUV', 'BACE', 'BBBP', 'ClinTox', 'SIDER',
                                          'ToxCast', 'HIV', 'PCBA', 'Tox21']:
                        smiles, bg, labels, masks, index = data
                        if len(smiles) == 1:
                            continue
                        labels, masks, index = labels.to(device), masks.to(device), index.to(device)
                        batch_index = index.squeeze().long()
                        bg = bg.to(device)
                        node_feats = bg.ndata.pop('h').to(device)
                        edge_feats = bg.edata.pop('e').to(device)

                        if 'vat' in fedmid:
                            vatloss = VATLoss(framework='dgl', criterion=vatCriterion, xi=args.xi)  # xi, and eps
                            xCombined = [bg, node_feats, edge_feats]
                            localVATLoss = vatloss(model, xCombined)

                        pred, latentEmb = model(bg, node_feats, edge_feats)
                        if predGAll is not None:
                            predG = predGAll[batch_index]
                            latentEmbG = predGAll_emb[batch_index]
                            globalVATLoss = predGAll_vat[batch_index]

                    if roundidx < warmupRound:
                        loss = lossCriterion(pred, labels)
                        loss = self._masked_mean(loss, masks)
                    else:
                        if fedmid in ['avg', 'fedprox']:
                            loss = lossCriterion(pred, labels)
                            loss = self._masked_mean(loss, masks)
                        elif fedmid == 'fedkd':
                            temperature = getattr(args, 'temperature', 1.0)
                            alpha_kd = getattr(args, 'alpha_kd', 0.5)

                            loss_ce = lossCriterion(pred, labels)
                            loss_ce = self._masked_mean(loss_ce, masks)


                            if hasattr(self, "soft_targets") and self.soft_targets is not None:
                                soft_targets = self.soft_targets[batch_index].to(device)
                                # safe KL loss calc
                                student_logits = pred / temperature
                                teacher_logits = soft_targets / temperature

                                student_probs = F.log_softmax(student_logits, dim=1)
                                teacher_probs = F.softmax(teacher_logits, dim=1)

                                loss_kd = F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
                                loss = alpha_kd * loss_kd

                            else:
                                loss = loss_ce

                        elif fedmid == 'oursvatFLITPLUS':
                            lossGlobalLabel = lossCriterion(predG, labels)
                            lossLocalLabel = lossCriterion(pred, labels)
                            lossLocalVAT = localVATLoss
                            lossGlobalVAT = globalVATLoss

                            weightloss_loss = lossLocalLabel + torch.relu(lossLocalLabel - lossGlobalLabel.detach())
                            weightloss_vat = (localVATLoss + torch.relu(lossLocalVAT - lossGlobalVAT.detach()))
                            weightloss = weightloss_loss + lambdavat * weightloss_vat
                            factor_ema = 0.8
                            masked_weightloss = weightloss if masks is None else weightloss * masks
                            if weight_denomaitor == None:
                                if masks is None:
                                    weight_denomaitor = weightloss.mean(dim=0, keepdim=True).detach()
                                else:
                                    weight_denomaitor = (
                                        masked_weightloss.sum(dim=0, keepdim=True)
                                        / masks.sum(dim=0, keepdim=True).clamp_min(1.0)
                                    ).detach()
                            else:
                                if masks is None:
                                    batch_weight_mean = weightloss.mean(dim=0, keepdim=True)
                                else:
                                    batch_weight_mean = (
                                        masked_weightloss.sum(dim=0, keepdim=True)
                                        / masks.sum(dim=0, keepdim=True).clamp_min(1.0)
                                    )
                                weight_denomaitor = factor_ema * weight_denomaitor + (1 - factor_ema) * batch_weight_mean.detach()
                            loss = (1 - torch.exp(-weightloss / (weight_denomaitor + 1e-7)) + 1e-7) ** tmpFed * (
                                    lossLocalLabel + weightReg*lossLocalVAT)
                            loss = self._masked_mean(loss, masks)

                        elif fedmid == 'oursFLIT':
                            lossGlobalLabel = lossCriterion(predG, labels)
                            lossLocalLabel = lossCriterion(pred, labels)

                            weightloss = lossLocalLabel + torch.relu(lossLocalLabel - lossGlobalLabel.detach())
                            factor_ema = 0.8
                            masked_weightloss = weightloss if masks is None else weightloss * masks
                            if weight_denomaitor == None:
                                if masks is None:
                                    weight_denomaitor = weightloss.mean(dim=0, keepdim=True).detach()
                                else:
                                    weight_denomaitor = (
                                        masked_weightloss.sum(dim=0, keepdim=True)
                                        / masks.sum(dim=0, keepdim=True).clamp_min(1.0)
                                    ).detach()
                            else:
                                if masks is None:
                                    batch_weight_mean = weightloss.mean(dim=0, keepdim=True)
                                else:
                                    batch_weight_mean = (
                                        masked_weightloss.sum(dim=0, keepdim=True)
                                        / masks.sum(dim=0, keepdim=True).clamp_min(1.0)
                                    )
                                weight_denomaitor = factor_ema * weight_denomaitor + (1 - factor_ema) * batch_weight_mean.detach()

                            loss = (1 - torch.exp(-weightloss / (weight_denomaitor + 1e-7)) + 1e-7) ** tmpFed * (
                                lossLocalLabel)
                            loss = self._masked_mean(loss, masks)
                        else:
                            print(fedmid)
                            raise ValueError('not found fed method')
                    if fedmid == 'fedprox':
                        mu = float(getattr(args, 'fedprox_mu', 0.01))
                        prox_term = torch.tensor(0.0, device=device)
                        for param, global_param in zip(model.parameters(), fedprox_global_params):
                            prox_term = prox_term + torch.sum((param - global_param) ** 2)
                        loss = loss + 0.5 * mu * prox_term
                    torch.autograd.set_detect_anomaly(getattr(args, 'detect_anomaly', False))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    batch_loss.append(loss.item())
                    del loss


    def globalEpoch(self, train_data, globalModel, args, device, criterion, vat_criterion):
        # globalModel.eval()
        if args.dataset == 'qm9':
            numData = train_data.dataset.data.idx[-1]
            numTasks = train_data.dataset.data.y.shape[-1]
        else:
            base_dataset = self._base_dataset(train_data)
            numData = len(base_dataset)
            numTasks = base_dataset.labels.shape[-1]
        predGAll = torch.zeros(numData, numTasks, device=device)
        predGAll_loss = torch.zeros(numData, numTasks, device=device)
        predGAll_vat = torch.zeros(numData, numTasks, device=device)
        predGAll_emb = torch.zeros(numData, 128, device=device)
        tbar1 = tqdm(train_data)

        for batch_idx, data in enumerate(tbar1):
            if args.dataset == 'qm9':
                z, pos, batch, labels = data.z.to(device), data.pos.to(device), data.batch.to(
                    device), data.y.to(
                    device)
                x = [z, pos, batch]
                index = data.idx.to(device)
                batch_index = index.squeeze().long()

                globalModel.zero_grad()
                if True:
                    vatloss = VATLoss(framework='geometric', criterion=vat_criterion, xi=args.xi)  # xi, and eps
                    predGAll_vat[batch_index] = vatloss(globalModel.train(), x).detach()
                with torch.no_grad():
                    predG, latentEmbG = globalModel(z, pos, batch)
                    losstmp = criterion(predG, labels)

                    predGAll_loss[batch_index] = losstmp.detach()
                    predGAll[batch_index] = predG.detach()
                    predGAll_emb[batch_index] = latentEmbG.clone().detach()

            elif args.dataset in ['esol', 'lipo', 'freesolv', 'MUV', 'BACE', 'BBBP', 'ClinTox', 'SIDER',
                                  'ToxCast', 'HIV', 'PCBA', 'Tox21']:
                smiles, bg, labels, masks, index = data
                labels, masks, index = labels.to(device), masks.to(device), index.to(device)
                batch_index = index.squeeze().long()
                bg = bg.to(device)
                node_feats = bg.ndata.pop('h').to(device)
                edge_feats = bg.edata.pop('e').to(device)
                globalModel.zero_grad()
                x = [bg, node_feats, edge_feats]
                if True:
                    vatloss = VATLoss(framework='dgl', criterion=vat_criterion, xi=args.xi)  # xi, and eps
                    predGAll_vat[batch_index] = vatloss(globalModel, x).detach()

                with torch.no_grad():
                    predG, latentEmbG = globalModel(bg, node_feats, edge_feats)

                losstmp = criterion(predG, labels)
                predGAll_loss[batch_index] = losstmp.detach()
                predGAll[batch_index] = predG.detach()
                predGAll_emb[batch_index] = latentEmbG.clone().detach()
        return predGAll, predGAll_emb, predGAll_vat
    
    def test(self, test_data, device, args):
        model = self.model
        model.to(device)
        model.eval()
        pred_list = []
        label_list = []
        mask_list = []
        
        with torch.no_grad():
            for batch_idx, data in enumerate(test_data):
                if args.dataset == 'qm9':
                    z, pos, batch, y = data.z.to(device), data.pos.to(device), data.batch.to(device), data.y.to(device)
                    pred, _ = model(z, pos, batch)
                    pred_list.append(pred.view(-1))
                    label_list.append(y.view(-1))
                elif args.dataset in ['esol', 'lipo', 'freesolv']:
                    smiles, bg, labels, masks, indices = data
                    labels = labels.to(device)
                    bg = bg.to(device)
                    node_feats = bg.ndata['h'].to(device)
                    edge_logits = bg.edata.get('e', None)
                    if edge_logits is not None:
                        edge_logits = edge_logits.to(device)
                    pred, _ = model(bg, node_feats, edge_logits)
                    pred_list.append(pred.view(labels.shape))
                    label_list.append(labels)
                    mask_list.append(masks.to(device).view(labels.shape))
                elif args.dataset in ['MUV', 'BACE', 'BBBP', 'ClinTox', 'SIDER', 'ToxCast', 'HIV', 'PCBA', 'Tox21']:
                    smiles, bg, labels, masks, indices = data
                    labels = labels.to(device)
                    bg = bg.to(device)
                    node_feats = bg.ndata['h'].to(device)
                    edge_logits = bg.edata.get('e', None)
                    if edge_logits is not None:
                        edge_logits = edge_logits.to(device)
                    pred, _ = model(bg, node_feats, edge_logits)
                    pred_list.append(torch.sigmoid(pred).view(labels.shape))
                    label_list.append(labels)
                    mask_list.append(masks.to(device).view(labels.shape))

        pred_all = torch.cat(pred_list)
        label_all = torch.cat(label_list)
        mask_all = torch.cat(mask_list) if mask_list else None

        if args.dataset in ['esol', 'lipo', 'freesolv']:
            squared_error = (pred_all - label_all) ** 2
            mse = self._masked_mean(squared_error, mask_all)
            rmse = torch.sqrt(mse).item()
            return {'rmse': rmse}
        elif args.dataset in ['MUV', 'BACE', 'BBBP', 'ClinTox', 'SIDER', 'ToxCast', 'HIV', 'PCBA', 'Tox21']:
            from sklearn.metrics import roc_auc_score
            task_aucs = []
            for task_idx in range(label_all.shape[1]):
                observed = mask_all[:, task_idx] > 0
                task_labels = label_all[observed, task_idx].detach().cpu()
                task_predictions = pred_all[observed, task_idx].detach().cpu()
                if task_labels.numel() == 0 or torch.unique(task_labels).numel() < 2:
                    continue
                task_aucs.append(roc_auc_score(task_labels.numpy(), task_predictions.numpy()))
            auc = float(sum(task_aucs) / len(task_aucs)) if task_aucs else float('nan')
            return {'roc_auc_score': auc, 'valid_tasks': len(task_aucs)}
        elif args.dataset == 'qm9':
            mae = torch.abs(pred_all - label_all).mean().item()
            return {'mae': mae}
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")
