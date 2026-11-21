from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib
from pathlib import Path
import sys
from types import ModuleType
from typing import Iterator, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mvhand.models.interfaces import HandPriorOutput
from mvhand.models._qkv_compat import LoraConfig, inject_lora_qv, trainable_parameter_names


HAMER_COMMIT = "3a01849f4148352e9260b69bf28b65d1671a4905"
HAMER_TREE = "293b0dc7238ae8e39a8d2dfd5fe34e787570b787"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class RuntimeAsset:
    path: Path
    sha256: str

    def validate(self) -> Path:
        path = Path(self.path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
            raise ValueError("expected SHA-256 must be 64 lowercase hexadecimal characters")
        actual = sha256_file(path)
        if actual != self.sha256:
            raise ValueError(f"asset SHA-256 mismatch for {path}")
        return path


@dataclass(frozen=True)
class Form2FrameRuntimeAssets:
    hamer_source: Path
    hamer_checkpoint: RuntimeAsset
    hamer_config: RuntimeAsset
    mano_right: RuntimeAsset
    mano_mean_params: RuntimeAsset
    dataset_contract: RuntimeAsset

    def validate(self) -> None:
        source = Path(self.hamer_source).resolve()
        if not (source / "hamer/models/hamer.py").is_file():
            raise FileNotFoundError("HaMeR source tree is incomplete")
        for asset in (
            self.hamer_checkpoint, self.hamer_config, self.mano_right,
            self.mano_mean_params, self.dataset_contract,
        ):
            asset.validate()


class RendererDisabled(nn.Module):
    def __init__(self, *_: object, **__: object) -> None:
        raise RuntimeError("renderer construction is disabled in the Form2Frame runtime")


def install_renderer_free_hamer_namespace(source_root: Path) -> None:
    """Expose HaMeR utility submodules without importing optional renderers."""
    source_root = Path(source_root).resolve()
    if not (source_root / "hamer" / "utils" / "geometry.py").is_file():
        raise FileNotFoundError("HaMeR utility source is missing")
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    importlib.import_module("hamer")
    utils = ModuleType("hamer.utils")
    utils.__path__ = [str(source_root / "hamer" / "utils")]
    utils.Renderer = RendererDisabled
    utils.MeshRenderer = RendererDisabled
    utils.SkeletonRenderer = RendererDisabled
    sys.modules["hamer.utils"] = utils


def load_hamer_renderer_free(source_root: Path, checkpoint: Path) -> tuple[nn.Module, object]:
    install_renderer_free_hamer_namespace(source_root)
    models = importlib.import_module("hamer.models")
    configs = importlib.import_module("hamer.configs")
    checkpoint = Path(checkpoint).resolve()
    config_path = checkpoint.parent.parent / "model_config.yaml"
    if not checkpoint.is_file() or not config_path.is_file():
        raise FileNotFoundError("checkpoint and adjacent model_config.yaml are required")
    cfg = configs.get_config(str(config_path), update_cachedir=True)
    if cfg.MODEL.BACKBONE.TYPE == "vit" and "BBOX_SHAPE" not in cfg.MODEL:
        cfg.defrost(); cfg.MODEL.BBOX_SHAPE = [192, 256]; cfg.freeze()
    if "PRETRAINED_WEIGHTS" in cfg.MODEL.BACKBONE:
        cfg.defrost(); cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS"); cfg.freeze()
    model = models.HAMER.load_from_checkpoint(
        str(checkpoint), strict=False, cfg=cfg, init_renderer=False, map_location="cpu"
    )
    if model.renderer is not None or model.mesh_renderer is not None:
        raise RuntimeError("renderer-free HaMeR load initialized a renderer")
    return model, cfg


def load_canonical_hamer(assets: Form2FrameRuntimeAssets) -> tuple[nn.Module, object]:
    """Validate the runtime bundle once and construct renderer-free HaMeR."""
    from mvhand.numpy_compat import install_numpy_chumpy_compat
    # smplx unpickles the licensed MANO file through chumpy, whose last
    # release still imports NumPy aliases removed in NumPy 1.24; restore
    # them at this licensed-asset boundary (no numerics change).
    install_numpy_chumpy_compat()
    assets.validate()
    source_root = Path(assets.hamer_source).resolve()
    install_renderer_free_hamer_namespace(source_root)
    models = importlib.import_module("hamer.models")
    configs = importlib.import_module("hamer.configs")
    cfg = configs.get_config(str(assets.hamer_config.path), update_cachedir=False)
    cfg.defrost()
    cfg.MANO.MODEL_PATH = str(Path(assets.mano_right.path).resolve().parent)
    cfg.MANO.MEAN_PARAMS = str(Path(assets.mano_mean_params.path).resolve())
    if cfg.MODEL.BACKBONE.TYPE == "vit" and "BBOX_SHAPE" not in cfg.MODEL:
        cfg.MODEL.BBOX_SHAPE = [192, 256]
    if "PRETRAINED_WEIGHTS" in cfg.MODEL.BACKBONE:
        cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
    cfg.freeze()
    model = models.HAMER.load_from_checkpoint(
        str(Path(assets.hamer_checkpoint.path).resolve()),
        strict=False, cfg=cfg, init_renderer=False, map_location="cpu",
    )
    if model.renderer is not None or model.mesh_renderer is not None:
        raise RuntimeError("renderer-free HaMeR load initialized a renderer")
    return model, cfg


def inject_canonical_hamer_lora(model: nn.Module) -> list[str]:
    blocks = getattr(getattr(model, "backbone", None), "blocks", None)
    if not isinstance(blocks, nn.ModuleList) or len(blocks) != 32:
        raise ValueError("canonical HaMeR backbone must expose exactly 32 blocks")
    names = inject_lora_qv(model, blocks, LoraConfig(last_blocks=4, rank=8, alpha=16, dropout=0))
    expected = [f"blocks.{index}.attn.qkv" for index in range(28, 32)]
    if names != expected:
        raise ValueError(f"HaMeR LoRA target mismatch: {names} != {expected}")
    return names


class CanonicalHamerProvider(nn.Module):
    """Concrete renderer-free HaMeR prior for the maintained Form2Frame core."""
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, normalized_crops: Tensor) -> HandPriorOutput:
        if normalized_crops.ndim != 5 or normalized_crops.shape[2:] != (3, 256, 256):
            raise ValueError("normalized crops must have shape [B,V,3,256,256]")
        b, v = normalized_crops.shape[:2]
        flat = normalized_crops.reshape(b * v, 3, 256, 256)
        captured: list[Tensor] = []
        hook = self.model.backbone.register_forward_hook(lambda _m, _i, output: captured.append(output))
        try:
            raw = self.model({"img": flat})
        finally:
            hook.remove()
        if not isinstance(raw, Mapping) or len(captured) != 1:
            raise RuntimeError("HaMeR forward did not expose its backbone feature map")
        features = captured[0]
        joints_2d = raw["pred_keypoints_2d"]
        if features.ndim != 4 or features.shape[1] != 1280:
            raise ValueError("HaMeR backbone feature dimension must be 1280")
        grid = (2.0 * joints_2d).reshape(b * v, -1, 1, 2)
        visual = F.grid_sample(features, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        visual = visual.squeeze(-1).transpose(1, 2)
        joints_3d = raw["pred_keypoints_3d"]
        vertices = raw["pred_vertices"]
        if vertices.shape[-2:] != (778, 3):
            raise ValueError("HaMeR must emit 778 MANO vertices")
        tensors = (joints_2d, visual, joints_3d, vertices)
        if any(not torch.isfinite(value).all().item() for value in tensors):
            raise ValueError("HaMeR provider emitted non-finite values")
        return HandPriorOutput(
            joints_2d_normalized=joints_2d.reshape(b, v, *joints_2d.shape[1:]),
            visual_features=visual.reshape(b, v, *visual.shape[1:]),
            local_joints_3d=joints_3d.reshape(b, v, *joints_3d.shape[1:]),
            local_vertices_3d=vertices.reshape(b, v, 778, 3),
        )
