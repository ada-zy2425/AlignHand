"""Shared anchor-centered regional fusion, identical for every dataset."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import torch
from torch import Tensor
from mvhand.regional.candidates import CANDIDATES, REGIONS, VIEWS, VISUAL_REGION_DIM, CandidateEvidenceHead, source_tags
from mvhand.runtime.precision import READER_VERSION
BASE_SLOT=0
H_SLICE=slice(1,9)
SECONDARY_SLICE=slice(9,17)
FAMILIES=('HH','HW')
OUTPUT_MODES=('anchor_centered',)
CACHE_SCHEMA='form2frame.unified-route-cache.v1'
VALID_SPLITS=frozenset(('fit','screen','development'))
PARENT_POLICIES=frozenset(('unified_anchor_v1',))
@dataclass
class AnchorOutput:
    vertices: Tensor
    q_new: Tensor
    q_init: Tensor
    delta: Tensor

class TransferCache:
    """Memory-mapped reader for the registered target-cache contract."""

    def __init__(self, manifest_path: Path):
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        if self.manifest.get("schema") != CACHE_SCHEMA:
            raise ValueError(
                f"formal transfer cache requires {CACHE_SCHEMA}, "
                f"got {self.manifest.get('schema')!r}"
            )
        identity = self.manifest.get("identity")
        if not isinstance(identity, dict):
            raise ValueError("cache manifest has no bound identity")
        if identity.get("runtime", {}).get("reader") != READER_VERSION:
            raise ValueError("cache reader identity mismatch")
        if identity.get("parent_policy") not in PARENT_POLICIES:
            raise ValueError("cache parent policy is not registered")
        self.rows = {str(row["key"]): row for row in self.manifest["rows"]}
        if len(self.rows) != len(self.manifest["rows"]):
            raise ValueError("duplicate cache key")
        self.directories = []
        for item in self.manifest["shards"]:
            path = Path(item["path"])
            if not path.is_absolute():
                path = self.manifest_path.parent / path
            self.directories.append(path.resolve())
        self.arrays: dict[tuple[int, str], np.ndarray] = {}

    def keys(self, split: str) -> list[str]:
        if split not in VALID_SPLITS:
            raise ValueError(f"unknown split {split!r}")
        return sorted(
            key for key, row in self.rows.items() if str(row["split"]) == split
        )

    def _array(self, shard: int, name: str) -> np.ndarray:
        token = (shard, name)
        if token not in self.arrays:
            self.arrays[token] = np.load(
                self.directories[shard] / f"{name}.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
        return self.arrays[token]

    def tensor(
        self,
        name: str,
        keys: list[str],
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        if not keys:
            raise ValueError("empty cache request")
        first = self.rows[keys[0]]
        template = self._array(int(first["shard"]), name)
        output = np.empty((len(keys), *template.shape[1:]), dtype=template.dtype)
        grouped: dict[int, list[tuple[int, int]]] = {}
        for position, key in enumerate(keys):
            row = self.rows[key]
            grouped.setdefault(int(row["shard"]), []).append(
                (position, int(row["ordinal"]))
            )
        for shard, pairs in grouped.items():
            positions = [pair[0] for pair in pairs]
            ordinals = [pair[1] for pair in pairs]
            output[positions] = self._array(shard, name)[ordinals]
        result = torch.from_numpy(output).to(device)
        return result if dtype is None else result.to(dtype=dtype)

def candidate_validity(
    view_valid: Tensor,
    secondary_valid: Tensor,
    *,
    family: str,
) -> Tensor:
    """Return strict validity for [baseline, H1..H8, Z1..Z8]."""
    if family not in FAMILIES:
        raise ValueError(family)
    if view_valid.ndim != 2 or view_valid.shape[1] != VIEWS:
        raise ValueError(f"view_valid must be [B,{VIEWS}]")
    if secondary_valid.shape != view_valid.shape:
        raise ValueError("secondary_valid must match view_valid")
    view_valid = view_valid.bool()
    secondary_valid = secondary_valid.bool() & view_valid
    result = torch.zeros(
        (view_valid.shape[0], CANDIDATES),
        dtype=torch.bool,
        device=view_valid.device,
    )
    result[:, BASE_SLOT] = True
    result[:, H_SLICE] = view_valid
    result[:, SECONDARY_SLICE] = view_valid if family == "HH" else secondary_valid
    return result

def linked_slot_permutation(view_order: Tensor) -> Tensor:
    """Candidate permutation bound to the same physical-view permutation."""
    view_order = view_order.to(dtype=torch.long)
    if view_order.shape != (VIEWS,):
        raise ValueError(f"view_order must have {VIEWS} entries")
    if not torch.equal(
        view_order.sort().values,
        torch.arange(VIEWS, device=view_order.device),
    ):
        raise ValueError("view_order must be a bijection")
    return torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=view_order.device),
            1 + view_order,
            9 + view_order,
        )
    )

def masked_mean(value: Tensor, valid: Tensor, *, dim: int) -> Tensor:
    """Finite masked mean; invalid NaNs are neutral and never counted."""
    valid = valid.bool()
    clean = torch.nan_to_num(value)
    while valid.ndim < clean.ndim:
        valid = valid.unsqueeze(-1)
    return torch.where(valid, clean, torch.zeros_like(clean)).sum(dim) / valid.sum(dim).clamp_min(1)

def masked_max(value: Tensor, valid: Tensor, *, dim: int) -> Tensor:
    """Finite masked maximum with zero output when no entry is valid."""
    valid = valid.bool()
    clean = torch.nan_to_num(value)
    while valid.ndim < clean.ndim:
        valid = valid.unsqueeze(-1)
    maximum = clean.masked_fill(~valid, float("-inf")).amax(dim)
    return torch.where(valid.any(dim), maximum, torch.zeros_like(maximum))

def sanitize_padded_inputs(
    *,
    candidates: Tensor,
    anchor: Tensor,
    geometry: Tensor,
    visual_descriptor: Tensor | None,
    candidate_valid: Tensor,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Neutralize invalid candidates after clearing arbitrary NaN/Inf bytes."""
    mask_mesh = candidate_valid[..., None, None]
    clean_candidates = torch.nan_to_num(candidates)
    clean_candidates = torch.where(mask_mesh, clean_candidates, anchor[:, None])
    mask_geometry = candidate_valid[:, None, :, None]
    clean_geometry = torch.where(
        mask_geometry,
        torch.nan_to_num(geometry),
        torch.zeros_like(geometry),
    )
    clean_visual = None
    if visual_descriptor is not None:
        clean_visual = torch.where(
            mask_geometry,
            torch.nan_to_num(visual_descriptor),
            torch.zeros_like(visual_descriptor),
        )
    return clean_candidates, clean_geometry, clean_visual

def initial_probabilities(
    view_valid: Tensor,
    secondary_valid: Tensor,
    *,
    family: str,
    dtype: torch.dtype = torch.float64,
    gate: float = 0.10,
    secondary_share: float = 0.05,
) -> tuple[Tensor, Tensor]:
    """Fixed initial candidate mixture; no output-reference subtraction."""
    if not 0 <= gate < 1 or not 0 <= secondary_share <= 1:
        raise ValueError("invalid gate/share")
    candidate_valid = candidate_validity(view_valid, secondary_valid, family=family)
    weights = view_valid.to(dtype)
    weights = weights / weights.sum(1, keepdim=True).clamp_min(1)
    base_initial = torch.zeros(
        (view_valid.shape[0], REGIONS, CANDIDATES),
        dtype=dtype,
        device=view_valid.device,
    )
    base_initial[..., BASE_SLOT] = 1.0 - gate
    base_initial[..., H_SLICE] = gate * weights[:, None]
    q_init = base_initial.clone()
    moved = secondary_share * base_initial[..., H_SLICE]
    if family == "HH":
        q_init[..., H_SLICE] -= moved
        q_init[..., SECONDARY_SLICE] += moved
    else:
        allowed = (secondary_valid.bool() & view_valid.bool())[:, None]
        actual = torch.where(allowed, moved, torch.zeros_like(moved))
        q_init[..., H_SLICE] -= actual
        q_init[..., SECONDARY_SLICE] += actual
    return q_init, candidate_valid

def fuse_anchor_candidates(
    *,
    head: CandidateEvidenceHead,
    anchor: Tensor,
    candidates: Tensor,
    geometry: Tensor,
    basis: Tensor,
    q_init: Tensor,
    candidate_valid: Tensor,
    family: str,
    secondary_valid: Tensor,
    visual_enabled: bool,
    visual_descriptor: Tensor | None,
    enforce_secondary_fallback: bool = True,
    linked_permutation: Tensor | None = None,
    output_mode: str = "anchor_centered",
    candidate_source_tags: Tensor | None = None,
) -> AnchorOutput:
    """Return anchor + regional weighted candidate offsets, using only anchor-centered residuals."""
    if family not in (*FAMILIES, "P"):
        raise ValueError(family)
    if output_mode not in OUTPUT_MODES:
        raise ValueError(f"unsupported output_mode: {output_mode!r}")
    candidates, geometry, visual_descriptor = sanitize_padded_inputs(
        candidates=candidates,
        anchor=anchor,
        geometry=geometry,
        visual_descriptor=visual_descriptor,
        candidate_valid=candidate_valid,
    )
    count = candidates.shape[1]
    if candidate_source_tags is None:
        if count != CANDIDATES or family == "P":
            raise ValueError("dynamic single-prior candidates require explicit source tags")
        source_values = source_tags(geometry.device, geometry.dtype)
    else:
        if candidate_source_tags.shape != (count, 3):
            raise ValueError("candidate source tags must have shape [C,3]")
        source_values = candidate_source_tags.to(device=geometry.device, dtype=geometry.dtype)
    origin_values = torch.ones(
        (count,), device=geometry.device, dtype=geometry.dtype
    )
    origin_values[BASE_SLOT] = 0
    if linked_permutation is not None:
        order = linked_permutation.to(device=geometry.device, dtype=torch.long)
        if order.shape != (count,) or not torch.equal(
            order.sort().values,
            torch.arange(count, device=geometry.device),
        ):
            raise ValueError("linked permutation must be a candidate-slot bijection")
        candidates = candidates.index_select(1, order)
        geometry = geometry.index_select(2, order)
        q_init = q_init.index_select(-1, order)
        candidate_valid = candidate_valid.index_select(1, order)
        source_values = source_values.index_select(0, order)
        origin_values = origin_values.index_select(0, order)
        if visual_descriptor is not None:
            visual_descriptor = visual_descriptor.index_select(2, order)

    batch = geometry.shape[0]
    if visual_enabled and visual_descriptor is not None:
        visual = visual_descriptor.to(geometry.dtype)
    else:
        visual = torch.zeros(
            (batch, REGIONS, count, VISUAL_REGION_DIM),
            device=geometry.device,
            dtype=geometry.dtype,
        )
    region = torch.eye(
        REGIONS, device=geometry.device, dtype=geometry.dtype
    )[None, :, None].expand(batch, -1, count, -1)
    source = source_values[None, None].expand(batch, REGIONS, -1, -1)
    origin = origin_values[None, None, :, None].expand(
        batch, REGIONS, -1, -1
    )
    valid_scalar = candidate_valid[:, None, :, None].expand(
        -1, REGIONS, -1, -1
    ).to(geometry.dtype)
    embedding = head.phi(
        torch.cat((geometry, visual, region, source, origin, valid_scalar), -1)
    )
    peer_means = []
    peer_maxima = []
    for candidate in range(count):
        peer = candidate_valid.clone()
        peer[:, candidate] = False
        peer_mask = peer[:, None, :, None]
        peer_means.append(
            (embedding * peer_mask).sum(2) / peer_mask.sum(2).clamp_min(1)
        )
        maximum = embedding.masked_fill(~peer_mask, float("-inf")).amax(2)
        peer_maxima.append(
            torch.where(peer_mask.any(2), maximum, torch.zeros_like(maximum))
        )
    context_mean = torch.stack(peer_means, 2)
    context_max = torch.stack(peer_maxima, 2)
    delta = head.score(
        torch.cat(
            (
                embedding,
                context_mean,
                context_max,
                region,
                source,
                origin,
                valid_scalar,
            ),
            -1,
        )
    ).squeeze(-1)
    logits = q_init.clamp_min(1e-300).log() + delta
    logits = logits.masked_fill(~candidate_valid[:, None], float("-inf"))
    q_new = torch.softmax(logits, -1)

    offset = candidates.double() - anchor.double()[:, None]
    mixture_weights = q_new
    output_reference = anchor.double()
    correction = torch.einsum("brc,bcmk->brmk", mixture_weights, offset)
    vertices = output_reference + torch.einsum(
        "bmr,brmk->bmk",
        basis.double()[None].expand(batch, -1, -1),
        correction,
    )
    if family == "HW" and enforce_secondary_fallback:
        all_secondary_missing = ~secondary_valid.bool().any(1)
        vertices = torch.where(
            all_secondary_missing[:, None, None],
            anchor.double(),
            vertices,
        )
        q_new = torch.where(
            all_secondary_missing[:, None, None],
            torch.nn.functional.one_hot(torch.zeros_like(q_new[..., 0], dtype=torch.long), count).to(q_new.dtype),
            q_new,
        )
        delta = torch.where(
            all_secondary_missing[:, None, None],
            torch.zeros_like(delta),
            delta,
        )
    return AnchorOutput(vertices, q_new, q_init, delta)
