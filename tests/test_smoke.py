import hashlib
import importlib
import json
import pkgutil
import py_compile
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OPTIONAL = {"cv2", "webdataset", "smplx", "scipy", "chumpy", "hamer", "pytorch_lightning", "yacs", "timm"}


def _modules():
    import mvhand
    yield "mvhand"
    for m in pkgutil.walk_packages(mvhand.__path__, "mvhand."):
        yield m.name


@pytest.mark.parametrize("name", list(_modules()))
def test_import(name):
    try:
        importlib.import_module(name)
    except ModuleNotFoundError as e:
        if e.name and e.name.split(".")[0] in OPTIONAL:
            pytest.skip(f"optional dependency {e.name}")
        raise


@pytest.mark.parametrize("path", sorted((ROOT / "scripts").rglob("*.py")), ids=lambda p: str(p.relative_to(ROOT)))
def test_script_compiles(path):
    py_compile.compile(str(path), doraise=True)


def test_manifest_matches_paper_checkpoints():
    man = json.loads((ROOT / "checkpoints" / "MANIFEST.json").read_text())
    sha = {f["name"]: f["sha256"] for f in man["files"]}
    assert sha["R_FINAL.pt"] == "c9bf7ffd640f728b1d6589c15fb999424e8b69efe6880ff48c531fce62e1c426"
    assert sha["REF_FINAL.pt"] == "0104af18ea26cc32d979a57c497016fbc22388b4928b6fc85fc56c2ecc420f75"
    assert sum(f["bytes"] for f in man["files"]) < 10_000_000


def test_download_verifies_sha256(tmp_path):
    src = tmp_path / "src"; src.mkdir()
    files = []
    for i, dest in enumerate(["results/u2/a.pt", "handoff/b.npy"]):
        data = f"payload{i}".encode()
        (src / f"f{i}").write_bytes(data)
        files.append({"name": f"f{i}", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "dest": dest})
    man = tmp_path / "m.json"
    man.write_text(json.dumps({"base_url": "http://invalid", "files": files}))
    cmd = [sys.executable, str(ROOT / "checkpoints" / "download.py"), "--manifest", str(man), "--root", str(tmp_path / "root"), "--from-dir", str(src)]
    assert subprocess.run(cmd, capture_output=True).returncode == 0
    assert (tmp_path / "root" / "results/u2/a.pt").read_bytes() == b"payload0"
    (tmp_path / "root" / "handoff/b.npy").write_bytes(b"corrupt")
    (src / "f1").write_bytes(b"also corrupt")
    assert subprocess.run(cmd, capture_output=True).returncode != 0


@pytest.mark.parametrize("path", sorted((ROOT / "scripts").rglob("*.py")), ids=lambda p: str(p.relative_to(ROOT)))
def test_script_path_setup_resolves(path):
    """Every relative lib/ or src/ path a script puts on sys.path must exist."""
    import re
    here = path.parent
    for parts in re.findall(r"os\.path\.join\((?:os\.path\.dirname\(os\.path\.abspath\(__file__\)\)|_HERE),((?:\s*'[^']+',?)+)\)", path.read_text()):
        rel = [p for p in re.findall(r"'([^']+)'", parts)]
        assert (here.joinpath(*rel)).resolve().is_dir(), (path.name, rel)
