"""Download the AlignHand weights and verify their sha256 (configs/weights.json).

    python scripts/download_weights.py                    # into $ALIGNHAND_DATA (default: data/)
    python scripts/download_weights.py --root /path/data  # explicit data root
    python scripts/download_weights.py --from-dir DIR     # files already downloaded by hand

Files are placed in the layout the evaluation and demo scripts expect.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(os.environ.get("ALIGNHAND_DATA", "data")))
    ap.add_argument("--from-dir", type=Path, default=None)
    ap.add_argument("--manifest", type=Path, default=ROOT / "configs" / "weights.json")
    a = ap.parse_args(argv)
    man = json.loads(a.manifest.read_text())
    bad = 0
    for f in man["files"]:
        dst = a.root / f["dest"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not (dst.exists() and sha256(dst) == f["sha256"]):
            if a.from_dir is not None:
                shutil.copyfile(a.from_dir / f["name"], dst)
            else:
                url = f'{man["base_url"]}/{f["name"]}'
                print(f"downloading {url}")
                urllib.request.urlretrieve(url, dst)
        ok = sha256(dst) == f["sha256"]
        bad += not ok
        print(f'{"OK " if ok else "BAD"} {f["dest"]}  ({f["bytes"]:,} B)')
    if bad:
        sys.exit(f"{bad} file(s) failed sha256 verification")
    print(f"all {len(man['files'])} files verified under {a.root}")


if __name__ == "__main__":
    main()
