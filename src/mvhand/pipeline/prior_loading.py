"""Load the official frozen prior and adapt its outputs to the pipeline."""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

# Preserve the frozen runtime's headless import backend. No renderer is used.
os.environ.setdefault("PYOPENGL_PLATFORM", "glx")
import numpy as np
import torch
from mvhand.data.ho3d import decode_ho3d_group, load_localizer_ledger
from mvhand.data.ho3d_poem_reader import decode_ho3d_wds_group
from mvhand.data.ho3d_poem_reader import ADAPTER_KIND as HO3D_WDS_ADAPTER_KIND
from mvhand.data.reference_targets import reference_target
from mvhand.interfaces import Form2FrameBatch
from mvhand.training.index import IndexedWDS
from mvhand.training.validation import load_mano_base_regressor
from mvhand.io import sha256
from mvhand.prior.feature_layout import canonical_storage_feature_maps

def load_assignments(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    result = {str(row["key"]): row for row in rows}
    if not rows or len(result) != len(rows):
        raise ValueError("split assignments must be nonempty and key-unique")
    return result

def load_wilor(source: Path, checkpoint: Path, device: str):
    """Construct the frozen official WiLoR without importing HanCo capture code."""
    source_text = str(Path(source).resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    from wilor.configs import get_config
    from wilor.models import WiLoR

    cfg = get_config(Path(source) / "pretrained_models/model_config.yaml", update_cachedir=True)
    cfg.defrost()
    cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS", None)
    cfg.MODEL.BBOX_SHAPE = [192, 256]
    cfg.MANO.DATA_DIR = str(Path(source) / "mano_data")
    cfg.MANO.MODEL_PATH = str(Path(source) / "mano_data")
    cfg.MANO.MEAN_PARAMS = str(Path(source) / "mano_data/mano_mean_params.npz")
    cfg.MANO.GENDER = "right"
    cfg.freeze()
    model = WiLoR.load_from_checkpoint(
        str(checkpoint), strict=False, cfg=cfg, init_renderer=False, map_location="cpu"
    ).to(device).eval()
    model.requires_grad_(False)
    return model, cfg

def _ho3d_data_key(name: str, *, terminal: bool) -> str:
    return f"terminal_{name}" if terminal else name

def dense_tensor(value: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        result = value
    elif isinstance(value, (tuple, list)):
        choices = [item for item in value if isinstance(item, torch.Tensor) and item.ndim == 4]
        if not choices:
            raise RuntimeError("HaMeR backbone hook exposed no dense tensor")
        result = choices[-1]
    elif isinstance(value, dict):
        choices = [item for item in value.values() if isinstance(item, torch.Tensor) and item.ndim == 4]
        if not choices:
            raise RuntimeError("HaMeR backbone mapping exposed no dense tensor")
        result = choices[-1]
    else:
        raise TypeError(f"unsupported backbone output {type(value)}")
    return canonical_storage_feature_maps(result)

def open_array(root: Path, name: str, shape: tuple[int, ...], dtype) -> np.memmap:
    return np.lib.format.open_memmap(
        root / f"{name}.partial.npy", mode="w+", dtype=dtype, shape=shape
    )

def _dex_groups(
    config: dict,
    assignments: dict[str, dict],
    selected_keys: list[str],
    *,
    terminal: bool = False,
    terminal_audit: dict | None = None,
):
    index_key = "terminal_dexycb_index" if terminal else "dexycb_index"
    identity_key = (
        "terminal_dexycb_dataset_sha256" if terminal else "dexycb_dataset_sha256"
    )
    reader = IndexedWDS(
        Path(config[index_key]),
        Path(config["dataset_root"]),
        str(config[identity_key]),
    )
    hamer_source = Path(config["assets"]["hamer_source"])
    try:
        for key in selected_keys:
            source_split = str(assignments[key]["source_split"]).lower()
            if source_split in {"validation", "development", "dev"}:
                source_split = "val"
            if terminal and source_split in {"terminal", "s0_test"}:
                source_split = "test"
            allowed = {"test"} if terminal else {"train", "val"}
            if source_split not in allowed:
                raise ValueError(f"capture refuses DexYCB source split {source_split!r}")
            if terminal:
                if terminal_audit is None:
                    raise PermissionError("terminal reader has no preterminal audit")
                yield reader.get_terminal_group(key, hamer_source, terminal_audit)
            else:
                yield reader.get_group(source_split, key, hamer_source)
    finally:
        reader.db.close()

def _ho3d_rows(path: Path) -> list[dict]:
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]
    value = json.loads(path.read_text())
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and isinstance(value.get("groups"), list):
        return value["groups"]
    raise ValueError("HO3D group manifest must be JSONL, a JSON list, or {groups:[...]}")

def _ho3d_groups(config: dict, selected_keys: list[str], *, terminal: bool = False):
    rows = _ho3d_rows(Path(config[_ho3d_data_key(
        "ho3d_group_manifest", terminal=terminal
    )]))
    by_key = {str(row["key"]): row for row in rows}
    if len(by_key) != len(rows):
        raise ValueError("duplicate key in HO3D group manifest")
    missing = sorted(set(selected_keys) - set(by_key))
    if missing:
        raise ValueError(f"HO3D source loader misses assigned keys: {missing[:8]}")
    mano_path = Path(config["assets"]["mano_right"])
    regressor = load_mano_base_regressor(mano_path)
    localizer = load_localizer_ledger(Path(config[_ho3d_data_key(
        "localizer_ledger", terminal=terminal
    )]))
    if config.get("ho3d_adapter_kind") is not None:
        if config["ho3d_adapter_kind"] != HO3D_WDS_ADAPTER_KIND:
            raise ValueError("unregistered HO3D adapter kind")
        for key in selected_keys:
            yield decode_ho3d_wds_group(
                by_key[key], wds_root=Path(config[_ho3d_data_key(
                    "ho3d_wds_root", terminal=terminal
                )]),
                localizer=localizer, mano_base_regressor=regressor,
            )
        return
    try:
        import smplx
    except ImportError as error:
        raise RuntimeError("HO3D raw capture requires the pinned smplx runtime") from error
    mano = smplx.MANO(
        model_path=str(mano_path), is_rhand=True, use_pca=False, flat_hand_mean=True
    )
    for key in selected_keys:
        yield decode_ho3d_group(
            by_key[key],
            rgb_root=Path(config["ho3d_rgb_root"]),
            metadata_root=Path(config["ho3d_metadata_root"]),
            localizer=localizer,
            mano_layer=mano,
            mano_base_regressor=regressor,
        )

def _batch(group, device: str) -> tuple[Form2FrameBatch, torch.Tensor, torch.Tensor]:
    if len(group.camera_ids) != 8:
        raise ValueError("runtime group must contain eight ordered camera slots")
    runtime_mask = group.runtime_view_mask.bool()
    valid_ids = [group.camera_ids[index] for index in runtime_mask.nonzero().flatten().tolist()]
    if len(set(valid_ids)) != len(valid_ids):
        raise ValueError("valid runtime slots must contain unique physical cameras")
    if runtime_mask.all() and "cam_serial" in group.label and tuple(map(str, group.label["cam_serial"])) != tuple(group.camera_ids):
        raise ValueError("publisher camera order and runtime camera order differ")
    raw_to_crop = group.raw_to_crop[None].to(device).float()
    model = Form2FrameBatch(
        crops=group.rgb_crops[None].to(device).float(),
        intrinsics=group.intrinsics[None].to(device).float(),
        camera_to_world=group.camera_to_world[None].to(device).float(),
        crop_to_raw=group.crop_to_raw[None].to(device).float(),
        runtime_view_mask=group.runtime_view_mask[None].to(device).bool(),
        raw_to_crop=raw_to_crop,
    )
    # Validate both supervised endpoints even though this transfer cache stores
    # the mesh target.  POEM-v2 stores one reference-frame GT copy per runtime
    # view; the maintained validation path and collator use frozen slot 0.
    reference_target(group.label, "joints_3d", 21)
    target_value = reference_target(group.label, "verts_3d", 778)
    target = torch.as_tensor(target_value, device=device).double()[None]
    return model, raw_to_crop.double(), target
