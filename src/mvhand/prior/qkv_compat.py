"""Private frozen-QKV compatibility for released checkpoints.

L0/Z freeze these zero-residual tensors. Their original initializer and forward
arithmetic are retained to preserve checkpoint names and RNG consumption.
This is not a supported backbone-adaptation training API.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LoraConfig:
    last_blocks: int
    rank: int
    alpha: float
    dropout: float = 0.0

    def validate(self) -> None:
        if isinstance(self.last_blocks, bool) or self.last_blocks <= 0:
            raise ValueError("last_blocks must be a positive integer")
        if isinstance(self.rank, bool) or self.rank <= 0:
            raise ValueError("rank must be a positive integer")
        if self.alpha <= 0 or not math.isfinite(self.alpha):
            raise ValueError("alpha must be finite and positive")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")


class LoraFusedQkvLinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        if base.out_features % 3:
            raise ValueError("fused QKV output size must be divisible by three")
        self.base = base
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        self.chunk = base.out_features // 3
        self.scaling = float(alpha) / rank
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.q_a = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.q_b = nn.Parameter(base.weight.new_zeros(self.chunk, rank))
        self.v_a = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.v_b = nn.Parameter(base.weight.new_zeros(self.chunk, rank))
        nn.init.kaiming_uniform_(self.q_a, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.v_a, a=math.sqrt(5))

    def forward(self, value: Tensor) -> Tensor:
        base = self.base(value)
        dropped = self.dropout(value)
        q = F.linear(F.linear(dropped, self.q_a), self.q_b) * self.scaling
        v = F.linear(F.linear(dropped, self.v_a), self.v_b) * self.scaling
        return base + torch.cat((q, torch.zeros_like(q), v), dim=-1)


def trainable_parameter_names(module: nn.Module) -> List[str]:
    return [name for name, parameter in module.named_parameters() if parameter.requires_grad]


def inject_lora_qv(
    model: nn.Module,
    backbone_blocks: Sequence[nn.Module],
    config: LoraConfig,
) -> List[str]:
    """Freeze ``model`` and inject Q/V LoRA into explicit final backbone blocks.

    The caller supplies the ordered block sequence, keeping prior-specific
    architecture discovery outside the canonical core.
    """
    config.validate()
    if not backbone_blocks:
        raise ValueError("backbone_blocks must not be empty")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    start = max(0, len(backbone_blocks) - config.last_blocks)
    injected: List[str] = []
    for block_index in range(start, len(backbone_blocks)):
        block = backbone_blocks[block_index]
        for relative_name, module in list(block.named_modules()):
            if not relative_name or not isinstance(module, nn.Linear):
                continue
            leaf = relative_name.lower().split(".")[-1]
            parent = block
            parts = relative_name.split(".")
            for part in parts[:-1]:
                parent = parent[int(part)] if part.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)) else getattr(parent, part)
            if leaf == "qkv":
                replacement: nn.Module = LoraFusedQkvLinear(module, config.rank, config.alpha, config.dropout)
            else:
                continue
            if parts[-1].isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
                parent[int(parts[-1])] = replacement
            else:
                setattr(parent, parts[-1], replacement)
            injected.append(f"blocks.{block_index}.{relative_name}")
    if not injected:
        raise ValueError("no Q/V projection was found in the supplied final blocks")
    return injected
