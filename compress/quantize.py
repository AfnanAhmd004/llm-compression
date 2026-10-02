"""Post-training weight quantization: symmetric int8 per output channel and int4 per group."""
from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F


def quantize_tensor(w: torch.Tensor, bits: int, group_size: int | None = None):
    """Symmetric quantization. Per output channel if group_size is None, else per group of input columns."""
    qmax = 2 ** (bits - 1) - 1
    out_f, in_f = w.shape
    if group_size is None:
        scale = w.abs().amax(1, keepdim=True).clamp_min(1e-8) / qmax
        q = torch.clamp(torch.round(w / scale), -qmax - 1, qmax)
        return q.to(torch.int8), scale
    g = w.view(out_f, in_f // group_size, group_size)
    scale = g.abs().amax(-1, keepdim=True).clamp_min(1e-8) / qmax
    q = torch.clamp(torch.round(g / scale), -qmax - 1, qmax)
    return q.to(torch.int8), scale


class QuantLinear(nn.Module):
    def __init__(self, lin: nn.Linear, bits: int = 8, group_size: int | None = None):
        super().__init__()
        self.bits, self.group_size, self.shape = bits, group_size, lin.weight.shape
        q, s = quantize_tensor(lin.weight.detach(), bits, group_size)
        self.register_buffer("q", q)
        self.register_buffer("scale", s.half())
        self.register_buffer("bias", None if lin.bias is None else lin.bias.detach().clone())

    def weight(self) -> torch.Tensor:
        return (self.q.float() * self.scale.float()).view(self.shape)

    def forward(self, x):
        return F.linear(x, self.weight(), self.bias)

    def storage_bytes(self) -> int:
        return self.q.numel() * self.bits // 8 + self.scale.numel() * 2 + (0 if self.bias is None else self.bias.numel() * 4)


def quantize_model(model: nn.Module, bits: int = 8, group_size: int | None = None, skip=("head",)) -> nn.Module:
    m = copy.deepcopy(model)
    for module in list(m.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and name not in skip:
                gs = group_size if group_size and child.in_features % group_size == 0 else None
                setattr(module, name, QuantLinear(child, bits, gs))
    return m


def model_bytes(model: nn.Module) -> int:
    packed = sum(m._packed_params._weight_bias()[0].numel() + m._packed_params._weight_bias()[0].q_per_channel_scales().numel() * 8
                 if m._packed_params._weight_bias()[0].qscheme() in (torch.per_channel_affine, torch.per_channel_symmetric)
                 else m._packed_params._weight_bias()[0].numel()
                 for m in model.modules() if type(m).__name__ == "Linear" and hasattr(m, "_packed_params"))
    q = sum(m.storage_bytes() for m in model.modules() if isinstance(m, QuantLinear))
    qids = {id(b) for m in model.modules() if isinstance(m, QuantLinear) for b in m.buffers()}
    rest = sum(p.numel() * p.element_size() for p in model.parameters())
    rest += sum(b.numel() * b.element_size() for b in model.buffers() if id(b) not in qids)
    return q + rest + packed
