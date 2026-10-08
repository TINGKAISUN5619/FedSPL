"""DGL 0.x segment-sum backward compatibility with Torch 1.8 deterministic mode.

Only integer segment membership construction moves to CPU. Floating-point
forward reductions and gradient gathering remain the library operations. This
does not certify determinism of other custom DGL kernels.
"""

import torch


def segment_membership(offsets):
    offsets = offsets.detach().cpu().long()
    lengths = offsets[1:] - offsets[:-1]
    assert int(offsets[0]) == 0 and (lengths >= 0).all()
    return torch.repeat_interleave(torch.arange(len(lengths)), lengths)


def install():
    from dgl.backend.pytorch.sparse import SegmentReduce
    original = SegmentReduce.backward

    @torch.cuda.amp.custom_bwd
    def backward(ctx, gradient):
        if ctx.backward_cache != 'sum':
            return original(ctx, gradient)
        _, offsets = ctx.saved_tensors
        membership = segment_membership(offsets).to(gradient.device)
        return None, gradient[membership], None

    SegmentReduce.backward = staticmethod(backward)
