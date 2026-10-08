"""Shared MMGNN training with a prespecified warmup/decay learning-rate rule."""

import math
import time

import torch
import torch.nn.functional as F


def scheduled_lr(peak, epoch, total=50):
    # Follow upstream's 1:10:1 initial/peak/final ratio, over the declared
    # global budget. Local optimizers reset each communication round for FL.
    if epoch < 2:
        return peak * (0.1 + 0.9 * epoch / 2)
    return peak * math.exp(math.log(0.1) * min(1., (epoch-2)/max(1, total-2)))


def train_shared(self, start_weights, teacher, round_idx, client_idx):
    from endpoint_diagnostics import append_rows
    from pathlib import Path
    if start_weights is not None:
        self.model_trainer.set_model_params(start_weights)
    model = self.model_trainer.model.to(self.device).train()
    rate = scheduled_lr(self.args.strong_peak_lr, round_idx, 50)
    optimizer = torch.optim.Adam(model.parameters(), lr=rate, weight_decay=0.)
    weight = float(self.args.lambda_proto)
    active = teacher is not None and len(teacher['prototypes']) and round_idx >= 1
    if active:
        prototypes = teacher['prototypes'].to(self.device)
        centers = F.normalize(teacher['descriptors'].to(self.device), dim=1)
    iterator = iter(self.local_training_data)
    step, records = 0, []
    while step < self.args.localStepsPerRound:
        try:
            smiles, graph, labels, masks, indices = next(iterator)
        except StopIteration:
            iterator = iter(self.local_training_data)
            smiles, graph, labels, masks, indices = next(iterator)
        if len(smiles) <= 1:
            continue
        graph, labels, masks = graph.to(self.device), labels.to(self.device), masks.to(self.device)
        logits, embedding = model(graph, graph.ndata['h'], graph.edata.get('e'))
        if logits.shape != labels.shape or masks.shape != labels.shape:
            raise RuntimeError('Require the true N-by-task observation mask')
        if not torch.all((masks == 0) | (masks == 1)):
            raise RuntimeError('Observation mask must be binary')
        supervised = (F.binary_cross_entropy_with_logits(logits, labels, reduction='none')*masks).sum()/masks.sum().clamp_min(1)
        proto = embedding.new_tensor(0.)
        if active:
            descriptors = self._descriptor_from_smiles_and_embeddings(smiles, embedding)
            matched = (F.normalize(descriptors.float(), dim=1) @ centers.t()).argmax(1)
            proto = F.mse_loss(embedding, prototypes[matched].detach())
        loss = supervised if weight == 0 else supervised+weight*proto
        if not torch.isfinite(torch.stack([supervised, proto, loss])).all():
            raise RuntimeError('Nonfinite MMGNN losses')
        if step % 50 == 0:
            grad_ratio, cosine = '', ''
            if active and weight:
                gs = torch.autograd.grad(supervised, embedding, retain_graph=True)[0]
                gp = torch.autograd.grad(proto, embedding, retain_graph=True)[0]
                ns, np_ = float(gs.norm()), float(gp.norm())
                grad_ratio = weight*np_/max(ns, 1e-12)
                cosine = float((gs*gp).sum())/max(ns*np_, 1e-12)
            records.append(dict(round=round_idx, client=client_idx, step=step, lr=rate,
                supervised_loss=float(supervised.detach()), prototype_loss=float(proto.detach()),
                lambda_proto=weight, weighted_embedding_grad_ratio=grad_ratio,
                embedding_gradient_cosine=cosine, mean_embedding_norm=float(embedding.detach().norm(dim=1).mean())))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        step += 1
    append_rows(Path(self._diagnostics_dir)/'training_signal.csv', records)
    return model.state_dict()


def centralized(api, output):
    """Full-data epochs, validation-only checkpoint selection, then held-out test."""
    import numpy as np
    from torch.utils.data import DataLoader
    from endpoint_diagnostics import append_rows, predict, score_tasks, split_positions
    from paired_path_training import cloned_state, local_random_stream
    model = api.model_trainer.model.to(api.device)
    loader = DataLoader(api.train_global.dataset, batch_size=64, shuffle=True, drop_last=False,
                        collate_fn=api.train_global.collate_fn, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=api.args.strong_peak_lr/10, weight_decay=0.)
    best, best_epoch, best_weights = -float('inf'), None, None
    for epoch in range(api.args.comm_round):
        started = time.monotonic()
        rate = scheduled_lr(api.args.strong_peak_lr, epoch, api.args.comm_round)
        for group in optimizer.param_groups:
            group['lr'] = rate
        model.train()
        steps = 0
        seen = []
        with local_random_stream(3000000+api.args.seed*100+epoch):
            for smiles, graph, labels, masks, ids in loader:
                graph = graph.to(api.device)
                labels, masks = labels.to(api.device), masks.to(api.device)
                logits, _ = model(graph, graph.ndata['h'], graph.edata.get('e'))
                loss = (F.binary_cross_entropy_with_logits(logits, labels, reduction='none')*masks).sum()/masks.sum().clamp_min(1)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite centralized loss')
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                steps += 1
                seen.extend(ids.tolist())
        if len(seen) != len(loader.dataset) or len(set(seen)) != len(seen):
            raise RuntimeError('Centralized epoch did not visit every training row once')
        with local_random_stream(5000000+epoch):
            p, y, m, ids = predict(model, api.test_global, api.device)
        positions = split_positions(len(ids), api.args.split_seed)
        val, _ = score_tasks(y[positions['val']], m[positions['val']], p[positions['val']])
        if val > best:
            best, best_epoch, best_weights = val, epoch, cloned_state(model.state_dict())
        append_rows(output/'centralized_curves.csv', [dict(epoch=epoch, val_auc=val,
            test_auc='', lr=rate, optimizer_steps=steps, training_rows=len(seen), seconds=time.monotonic()-started)])
        print('Centralized epoch %d validation_auc=%.6f; no test checkpoint selection' % (epoch, val), flush=True)
    final = cloned_state(model.state_dict())
    records, arrays = [], dict(labels=y, masks=m, dataset_indices=ids, **{k+'_positions':v for k,v in positions.items()})
    for name, state in [('final', final), ('validation_selected', best_weights)]:
        model.load_state_dict(state)
        p, y, m, ids = predict(model, api.test_global, api.device)
        arrays[name] = p
        for split, idx in positions.items():
            value, _ = score_tasks(y[idx], m[idx], p[idx])
            records.append(dict(checkpoint=name, selected_epoch=api.args.comm_round-1 if name=='final' else best_epoch,
                                split=split, auc=value))
    append_rows(output/'centralized_endpoints.csv', records)
    torch.save(dict(final=final, validation_selected=best_weights), str(output/'centralized_models.pt'))
    np.savez_compressed(str(output/'final_predictions.npz'), **arrays)
