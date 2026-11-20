"""Linked view selection for RGB, calibration, affines and frozen prior outputs."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import torch
from torch import Tensor
from mvhand.models.interfaces import Form2FrameBatch, HandPriorOutput


@dataclass(frozen=True)
class ViewSelection:
    indices: Tensor
    available: Tensor


def sample_training_views(available: Tensor, *, generator: torch.Generator,
                          distribution: str = 'uniform') -> ViewSelection:
    """Uniformly sample 2..N available views, retaining reference index zero.

    If its image is unavailable, reference calibration is still kept as a masked
    extra slot. Such groups remain usable when two other images are available.
    The sampler uses a CPU generator so its sequence is independent of device.
    """
    if available.ndim != 2 or available.dtype != torch.bool:
        raise ValueError("availability must be Boolean [B,V]")
    if distribution not in ('uniform','d9_small_view_emphasis'):
        raise ValueError('unknown view-count distribution')
    rows = []
    masks = []
    for mask in available.detach().cpu():
        active = torch.nonzero(mask).flatten()
        if len(active) < 2:
            raise ValueError("training geometry needs two available images")
        if distribution == 'uniform':
            count = int(torch.randint(2, len(active) + 1, (), generator=generator))
        else:
            counts=torch.arange(2,len(active)+1)
            small=counts<=4
            probabilities=torch.where(small,.6/small.sum().clamp_min(1),.4/(~small).sum().clamp_min(1))
            count=int(counts[torch.multinomial(probabilities,1,generator=generator)[0]])
        nonreference = active[active != 0]
        shuffled = nonreference[torch.randperm(len(nonreference), generator=generator)]
        take = count - int(mask[0])
        row = torch.cat((torch.zeros(1, dtype=torch.long), shuffled[:take]))
        rows.append(row)
        masks.append(mask[row])
    width = max(len(row) for row in rows)
    indices = torch.zeros((len(rows), width), dtype=torch.long, device=available.device)
    selected = torch.zeros((len(rows), width), dtype=torch.bool, device=available.device)
    for i, (row, mask) in enumerate(zip(rows, masks)):
        indices[i, :len(row)] = row.to(available.device)
        selected[i, :len(row)] = mask.to(available.device)
    return ViewSelection(indices, selected)


def select_view_tensor(value: Tensor, selection: ViewSelection) -> Tensor:
    if value.ndim < 2 or value.shape[0] != selection.indices.shape[0]:
        raise ValueError("view tensor must start with matching batch/view dimensions")
    index = selection.indices.to(value.device)
    index = index.reshape(*index.shape, *([1] * (value.ndim - 2))).expand(
        *selection.indices.shape, *value.shape[2:])
    return torch.gather(value, 1, index)


def select_batch_views(batch: Form2FrameBatch, selection: ViewSelection) -> Form2FrameBatch:
    values = {f.name: (None if getattr(batch, f.name) is None else
                       select_view_tensor(getattr(batch, f.name), selection)) for f in fields(batch)}
    values['runtime_view_mask'] = selection.available.to(batch.crops.device)
    return replace(batch, **values)


def select_prior_views(prior: HandPriorOutput, selection: ViewSelection) -> HandPriorOutput:
    return replace(prior, **{f.name: select_view_tensor(getattr(prior, f.name), selection)
                             for f in fields(prior)})
