#!/usr/bin/env python3
"""Create deterministic, sequence-atomic transfer splits from metadata only.

The input is JSONL and must contain ``key``, ``sequence_id`` and
``source_split``. It may contain a more conservative ``leakage_unit`` (for
example subject/action/sequence); when present, that is the atomic split unit.
This program reads metadata only. It never opens RGB, labels, or terminal
prediction bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


def _bucket(dataset: str, unit: str, seed: int) -> float:
    payload = f"{dataset}|{seed}|{unit}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def assign(dataset: str, row: dict, seed: int) -> str:
    source = str(row["source_split"]).lower()
    sequence = str(row["sequence_id"])
    unit = str(row.get("leakage_unit", sequence))
    if dataset == "dexycb":
        if source in {"val", "validation", "dev", "development"}:
            return "development"
        if source in {"test", "s0_test"}:
            return "terminal"
        if source not in {"train", "training", "s0_train"}:
            raise ValueError(f"unknown DexYCB source split: {source}")
        return "fit" if _bucket(dataset, unit, seed) < 0.80 else "screen"
    if dataset == "ho3d":
        if sequence in {"GPMF1", "SB1"} or source in {"terminal", "heldout"}:
            return "terminal"
        if source not in {"train", "training"}:
            raise ValueError(f"unknown HO3D source split: {source}")
        value = _bucket(dataset, unit, seed)
        if value < 0.70:
            return "fit"
        return "screen" if value < 0.85 else "development"
    raise ValueError(dataset)


def _ho3d_small_population_unit_map(rows: list[dict], seed: int) -> tuple[dict[str, str], list[dict]]:
    """Freeze the preregistered five-sequence HO3D fallback before model results.

    The ordinary 70/15/15 bucket rule can leave a split empty when the source
    population has only five sequences.  This opt-in rule ranks the same
    SHA256(dataset|seed|unit) values and assigns whole units 2/1/2 to
    Fit/Screen/Development.  Terminal sequences never enter the ranking.
    """
    nonterminal: dict[str, str] = {}
    terminal: set[str] = set()
    for row in rows:
        source = str(row["source_split"]).lower()
        sequence = str(row["sequence_id"])
        unit = str(row.get("leakage_unit", sequence))
        is_terminal = sequence in {"GPMF1", "SB1"} or source in {"terminal", "heldout"}
        if is_terminal:
            terminal.add(unit)
            continue
        if source not in {"train", "training"}:
            raise ValueError(f"unknown HO3D source split: {source}")
        previous = nonterminal.setdefault(unit, sequence)
        if previous != sequence:
            raise ValueError(f"HO3D leakage unit {unit!r} contains multiple sequences")
    overlap = terminal & set(nonterminal)
    if overlap:
        raise ValueError(f"HO3D terminal/nonterminal leakage-unit overlap: {sorted(overlap)}")
    if len(nonterminal) != 5:
        raise ValueError(
            "--ho3d-small-population-rebalance requires exactly five "
            f"nonterminal leakage units, found {len(nonterminal)}"
        )

    ranked = sorted(
        ((_bucket("ho3d", unit, seed), unit, sequence) for unit, sequence in nonterminal.items()),
        key=lambda item: (item[0], item[1]),
    )
    split_order = ("fit", "fit", "screen", "development", "development")
    unit_map = {unit: split_order[index] for index, (_, unit, _) in enumerate(ranked)}
    ranking = [
        {
            "rank": index + 1,
            "leakage_unit": unit,
            "sequence_id": sequence,
            "sha256_bucket": f"{value:.17g}",
            "split": split_order[index],
        }
        for index, (value, unit, sequence) in enumerate(ranked)
    ]
    return unit_map, ranking


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("dexycb", "ho3d"), required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument(
        "--ho3d-small-population-rebalance",
        action="store_true",
        help=(
            "for exactly five nonterminal HO3D leakage units, rank the frozen "
            "SHA256 buckets and assign 2 Fit / 1 Screen / 2 Development"
        ),
    )
    args = parser.parse_args()
    if args.ho3d_small_population_rebalance and args.dataset != "ho3d":
        parser.error("--ho3d-small-population-rebalance is valid only for --dataset ho3d")

    rows = []
    with args.metadata.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = {"key", "sequence_id", "source_split"} - set(row)
            if missing:
                raise ValueError(f"line {line_number} missing {sorted(missing)}")
            rows.append(row)
    if not rows:
        raise ValueError("empty metadata")
    keys = [str(row["key"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate group key")

    small_population_map: dict[str, str] | None = None
    ranking: list[dict] = []
    if args.ho3d_small_population_rebalance:
        small_population_map, ranking = _ho3d_small_population_unit_map(rows, args.seed)

    unit_to_split: dict[str, str] = {}
    output_rows = []
    for row in rows:
        unit = str(row.get("leakage_unit", row["sequence_id"]))
        if small_population_map is not None:
            ordinary = assign(args.dataset, row, args.seed)
            split = "terminal" if ordinary == "terminal" else small_population_map[unit]
        else:
            split = assign(args.dataset, row, args.seed)
        previous = unit_to_split.setdefault(unit, split)
        if previous != split:
            raise ValueError(f"leakage unit {unit!r} crosses {previous}/{split}")
        output_rows.append({
            "key": str(row["key"]),
            "sequence_id": str(row["sequence_id"]),
            "leakage_unit": unit,
            "source_split": str(row["source_split"]),
            "split": split,
        })
    output_rows.sort(key=lambda row: row["key"])
    args.output.mkdir(parents=True, exist_ok=False)
    assignments = args.output / "SPLIT_ASSIGNMENTS.jsonl"
    _write_jsonl(assignments, output_rows)
    counts: dict[str, int] = {}
    sequences: dict[str, set[str]] = {}
    for row in output_rows:
        counts[row["split"]] = counts.get(row["split"], 0) + 1
        sequences.setdefault(row["split"], set()).add(row["sequence_id"])
    digest = hashlib.sha256(assignments.read_bytes()).hexdigest()
    metadata_digest = hashlib.sha256(args.metadata.read_bytes()).hexdigest()
    schema = (
        "form2frame.three-dataset-r-splits.v2"
        if args.ho3d_small_population_rebalance
        else "form2frame.three-dataset-r-splits.v1"
    )
    algorithm = (
        "sha256(dataset|seed|leakage_unit) ranked over exactly five nonterminal "
        "HO3D units: lowest 2 Fit, middle 1 Screen, highest 2 Development; "
        "whole-unit assignment; GPMF1/SB1 terminal isolation"
        if args.ho3d_small_population_rebalance
        else "sha256(dataset|seed|leakage_unit), whole-unit assignment"
    )
    report = {
        "schema": schema,
        "dataset": args.dataset,
        "seed": args.seed,
        "algorithm": algorithm,
        "mode": (
            "ho3d_five_sequence_preregistered_rebalance_v1"
            if args.ho3d_small_population_rebalance
            else "default_bucket_v1"
        ),
        "metadata_sha256": metadata_digest,
        "counts": counts,
        "sequence_counts": {key: len(value) for key, value in sequences.items()},
        "assignment_sha256": digest,
        "terminal_bytes_opened": 0,
    }
    if ranking:
        report["nonterminal_unit_ranking"] = ranking
        report["selection_evidence"] = "metadata identities and frozen SHA256 seed only; no model outputs"
    _write_json(args.output / "SPLIT_REPORT.json", report)
    if args.ho3d_small_population_rebalance:
        contract = {
            "schema": "form2frame.ho3d-small-population-split-contract.v1",
            "dataset": "ho3d",
            "seed": args.seed,
            "mode": report["mode"],
            "algorithm": algorithm,
            "metadata_sha256": metadata_digest,
            "assignment_sha256": digest,
            "whole_leakage_unit_atomic": True,
            "terminal_sequences": ["GPMF1", "SB1"],
            "terminal_bytes_opened": 0,
            "nonterminal_unit_ranking": ranking,
            "performance_observations_used": 0,
        }
        _write_json(args.output / "SPLIT_CONTRACT.json", contract)


if __name__ == "__main__":
    main()
