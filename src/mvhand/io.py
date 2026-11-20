"""Identity checks and atomic result files shared by all datasets."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import torch
from torch import Tensor

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def module_source_hashes(*module_names: str) -> dict[str, str]:
    """Bind installed source files without relying on a scripts directory."""
    result = {}
    for name in module_names:
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None or not Path(spec.origin).is_file():
            raise ValueError(f"Cannot locate source module: {name}")
        result[name] = sha256(Path(spec.origin))
    return result

def tensor_sha256(value: Tensor) -> str:
    value = value.detach().contiguous().cpu().numpy()
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode() + b"\0")
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode() + b"\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()

def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)

def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)

def atomic_torch(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)
