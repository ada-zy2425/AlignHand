"""Small utilities: atomic file writes, hashing, seeding and environment snapshots."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json_atomic(path: str | os.PathLike[str], payload: Any) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=target.parent, delete=False, prefix=f".{target.name}."
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
        temp_name = handle.name
    os.replace(temp_name, target)
    return target


def write_text_atomic(path: str | os.PathLike[str], text: str) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=target.parent, delete=False, prefix=f".{target.name}."
    ) as handle:
        handle.write(text)
        if text and not text.endswith("\n"):
            handle.write("\n")
        temp_name = handle.name
    os.replace(temp_name, target)
    return target


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in YAML file: {path}")
    return payload


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(path: str | os.PathLike[str]) -> dict[str, Any]:
    repo = Path(path)
    result: dict[str, Any] = {"path": str(repo), "is_git": False}
    if not repo.exists():
        result["error"] = "missing"
        return result
    try:
        top = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        head = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain"], text=True
        ).splitlines()
        result.update(
            {
                "is_git": True,
                "toplevel": top,
                "head": head,
                "dirty": bool(status),
                "status_porcelain": status[:200],
            }
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        result["error"] = str(exc)
    return result


def environment_snapshot() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    devices = []
    if cuda_available:
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            devices.append(
                {
                    "index": idx,
                    "name": props.name,
                    "total_memory_bytes": int(props.total_memory),
                    "capability": [int(props.major), int(props.minor)],
                }
            )
    return {
        "created_at_utc": utc_now_iso(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_executable": os.sys.executable,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "cuda_devices": devices,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }


def set_deterministic_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def device_from_arg(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return device


def format_markdown_table(rows: Iterable[dict[str, Any]], columns: list[str]) -> str:
    rows = list(rows)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                value = f"{value:.8g}"
            values.append(str(value).replace("|", "\\|"))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *body])
