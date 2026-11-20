#!/usr/bin/env python3
"""Hash-gate and index the fixed POEM-v2 HO3D_mv WDS without GT crops."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile

import numpy as np

from mvhand.data.ho3d_wds_adapter import (
    ADAPTER_KIND, ADAPTER_VERSION, CROSS_VIEW_ALIGNMENT_TOLERANCE_M,
    PHYSICAL_TO_SEQUENCE_SUFFIX, SOURCE_REVISION, SUPERVISION_TARGET_POLICY,
    decode_label_bytes, validate_wds_label,
)
from mvhand.data.ho3d_mv_runtime import (
    REFERENCE_PHYSICAL_CAMERA,
    REFERENCE_RUNTIME_SLOT,
    RUNTIME_SLOT_TO_PHYSICAL,
)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def jpeg_size(payload: bytes) -> tuple[int, int]:
    if payload[:2] != b"\xff\xd8":
        raise ValueError("WDS image is not JPEG")
    offset = 2
    while offset + 9 <= len(payload):
        if payload[offset] != 0xFF:
            offset += 1; continue
        marker = payload[offset + 1]
        offset += 2
        if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
            continue
        if offset + 2 > len(payload):
            break
        length = int.from_bytes(payload[offset:offset + 2], "big")
        if length < 2 or offset + length > len(payload):
            break
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            return int.from_bytes(payload[offset + 5:offset + 7], "big"), int.from_bytes(payload[offset + 3:offset + 5], "big")
        offset += length
    raise ValueError("JPEG has no valid size marker")


def alignment_summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "groups": int(array.size),
        "median_m": float(np.median(array)),
        "p90_m": float(np.quantile(array, 0.90)),
        "p95_m": float(np.quantile(array, 0.95)),
        "p99_m": float(np.quantile(array, 0.99)),
        "max_m": float(array.max()),
        "groups_above_5e_5_m": int((array > CROSS_VIEW_ALIGNMENT_TOLERANCE_M).sum()),
        "groups_above_1e_3_m": int((array > 1e-3).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wds-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--population", choices=("nonterminal-train", "terminal-test"),
        default="nonterminal-train",
    )
    parser.add_argument("--terminal-preaudit", type=Path)
    parser.add_argument("--terminal-run-id")
    parser.add_argument("--limit-groups", type=int, default=0)
    parser.add_argument("--materialize-rgb", action="store_true")
    args = parser.parse_args()
    source = json.loads(args.source_manifest.read_text())
    terminal = args.population == "terminal-test"
    expected_count = 3 if terminal else 9
    basename = "HO3D_mv_test" if terminal else "HO3D_mv_train"
    if source.get("revision") != SOURCE_REVISION or len(source.get("files", [])) != expected_count:
        parser.error("source manifest is not the registered fixed-revision population")
    expected_names = {f"{basename}-{index:06d}.tar" for index in range(expected_count)}
    if {Path(row["path"]).name for row in source["files"]} != expected_names:
        parser.error("unexpected HO3D WDS shard population")
    if terminal:
        if args.terminal_preaudit is None or not args.terminal_run_id:
            parser.error("terminal test indexing requires a preaudit and run ID")
        audit_path = args.terminal_preaudit.resolve()
        audit = json.loads(audit_path.read_text())
        if (
            audit.get("schema") != "form2frame.three-dataset-r-preterminal-gate-audit.v1"
            or audit.get("dataset") != "ho3d"
            or not audit.get("passed")
            or audit.get("selection") != "seed1 epoch10 HW_GV"
            or int(audit.get("terminal_model_evaluations_before_audit", -1)) != 0
        ):
            raise PermissionError("HO3D terminal preaudit does not authorize indexing")
    elif args.terminal_preaudit is not None or args.terminal_run_id:
        parser.error("terminal authorization supplied to the nonterminal population")
    if args.output.exists() or args.limit_groups < 0:
        parser.error("output exists or limit is invalid")

    verified = []
    for item in sorted(source["files"], key=lambda value: value["path"]):
        path = args.wds_root / Path(item["path"]).name
        if not path.is_file() or path.stat().st_size != int(item["size_bytes"]):
            raise ValueError(f"missing/wrong-size WDS shard: {path}")
        digest = sha256(path)
        if digest != item["sha256"]:
            raise ValueError(f"WDS shard SHA256 mismatch: {path}")
        verified.append({"name": path.name, "bytes": path.stat().st_size, "sha256": digest})

    partial = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    rows, alignment_values, reprojection_values, terminal_rows = [], [], [], []
    alignment_by_sequence: dict[str, list[float]] = {}
    try:
        if args.materialize_rgb:
            (partial / "rgb").mkdir()
        stop = False
        for shard_row in verified:
            shard_path = args.wds_root / shard_row["name"]
            with tarfile.open(shard_path, "r:") as archive:
                groups: dict[str, dict[str, tarfile.TarInfo]] = {}
                for member in archive:
                    name = member.name
                    if "." not in name or not member.isfile():
                        raise ValueError(f"unexpected WDS member: {name}")
                    key, suffix = name.split(".", 1)
                    if not key.isdigit() or len(key) != 9:
                        raise ValueError(f"invalid WDS sample key: {key}")
                    if suffix not in {"label.pyd", *(f"image_{i}.jpg" for i in range(5))}:
                        raise ValueError(f"unexpected WDS member suffix: {suffix}")
                    if suffix in groups.setdefault(key, {}):
                        raise ValueError(f"duplicate WDS member: {name}")
                    groups[key][suffix] = member
                for wds_key, members in sorted(groups.items()):
                    if set(members) != {"label.pyd", *(f"image_{i}.jpg" for i in range(5))}:
                        raise ValueError(f"incomplete WDS group: {wds_key}")
                    label_info = members["label.pyd"]
                    label_stream = archive.extractfile(label_info)
                    if label_stream is None:
                        raise OSError("failed to open WDS label member")
                    label = decode_label_bytes(label_stream.read())
                    identity = validate_wds_label(label)
                    alignment_values.append(float(identity["mesh_alignment_rms_m"]))
                    reprojection_values.append(float(identity["reprojection_max_px"]))
                    sequence_id = str(identity["sequence_id"])
                    alignment_by_sequence.setdefault(sequence_id, []).append(
                        float(identity["mesh_alignment_rms_m"])
                    )
                    physical_to_index = tuple(int(value) for value in identity["physical_to_wds_index"])
                    views = []
                    group_rgb = partial / "rgb" / wds_key
                    if args.materialize_rgb:
                        group_rgb.mkdir()
                    for physical_id, wds_index in enumerate(physical_to_index):
                        info = members[f"image_{wds_index}.jpg"]
                        image_stream = archive.extractfile(info)
                        if image_stream is None:
                            raise OSError("failed to open WDS RGB member")
                        encoded = image_stream.read()
                        if jpeg_size(encoded) != (640, 480):
                            raise ValueError("HO3D WDS RGB is not raw 640x480")
                        relative_rgb = f"rgb/{wds_key}/physical_{physical_id}.jpg"
                        if args.materialize_rgb:
                            (partial / relative_rgb).write_bytes(encoded)
                        views.append({
                            "physical_camera_id": physical_id,
                            "sequence_suffix": PHYSICAL_TO_SEQUENCE_SUFFIX[physical_id],
                            "camera_serial": label["cam_serial"][wds_index],
                            "wds_image_index": wds_index,
                            "wds_member": info.name,
                            "offset": info.offset_data,
                            "size": info.size,
                            "rgb_member": relative_rgb,
                        })
                    row = {
                        "schema": "form2frame.ho3d-wds-group.v1",
                        "adapter_kind": ADAPTER_KIND,
                        "adapter_version": ADAPTER_VERSION,
                        "key": (
                            f"ho3d_wds_test_d6f16d9c:{wds_key}"
                            if terminal else f"ho3d_wds_d6f16d9c:{wds_key}"
                        ),
                        "wds_key": wds_key,
                        "wds_shard": shard_path.name,
                        "label_member": {"wds_member": label_info.name, "offset": label_info.offset_data, "size": label_info.size},
                        "sequence_id": sequence_id,
                        "leakage_unit": sequence_id,
                        "frame_id": str(identity["frame_id"]),
                        "cross_view_mesh_alignment_rms_m": float(
                            identity["mesh_alignment_rms_m"]
                        ),
                        "cross_view_mesh_consistent": bool(
                            identity["cross_view_mesh_consistent"]
                        ),
                        "supervision_target_policy": SUPERVISION_TARGET_POLICY,
                        "source_split": "terminal" if terminal else "train",
                        "views": views,
                    }
                    if terminal:
                        if sequence_id not in {"GPMF1", "SB1"}:
                            raise ValueError(
                                f"terminal shard contains unregistered sequence {sequence_id!r}"
                            )
                        row["split"] = "terminal"
                        rows.append(row)
                    elif sequence_id in {"GPMF1", "SB1"}:
                        terminal_rows.append(row)
                    else:
                        rows.append(row)
                    if args.limit_groups and len(rows) >= args.limit_groups:
                        stop = True; break
            if stop:
                break
        if not rows:
            raise ValueError("no HO3D WDS groups in registered population")
        if terminal and len(rows) != 2706:
            raise ValueError(f"HO3D terminal population must contain 2706 groups, found {len(rows)}")
        if args.limit_groups and len(rows) != args.limit_groups:
            raise ValueError("source contains fewer groups than requested")
        keys = [row["key"] for row in rows]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate qualified HO3D WDS key")
        write_jsonl(partial / "HO3D_GROUPS.jsonl", rows)
        (partial / "LOCALIZER_MANIFEST.json").write_text(json.dumps({"groups": rows}, separators=(",", ":")) + "\n")
        if terminal_rows:
            write_jsonl(partial / "TERMINAL_QUARANTINE_METADATA_ONLY.jsonl", terminal_rows)
        group_hash = sha256(partial / "HO3D_GROUPS.jsonl")
        source_hash = sha256(args.source_manifest)
        alignment_audit = {
            "schema": "form2frame.ho3d-wds-label-alignment-audit.v1",
            "adapter_version": ADAPTER_VERSION,
            "supervision_target_policy": SUPERVISION_TARGET_POLICY,
            "interpretation": (
                "Cross-view residual is a diagnostic only. Supervision is the "
                "publisher label from physical camera 0 in the frozen reference frame; "
                "labels from other views are never averaged into the target."
            ),
            "all": alignment_summary(alignment_values),
            "by_sequence": {
                key: alignment_summary(values)
                for key, values in sorted(alignment_by_sequence.items())
            },
        }
        alignment_audit_path = partial / "HO3D_LABEL_ALIGNMENT_AUDIT.json"
        alignment_audit_path.write_text(
            json.dumps(alignment_audit, indent=2, sort_keys=True) + "\n"
        )
        contract = {
            "schema": "form2frame.ho3d-wds-adapter-contract.v1",
            "adapter_kind": ADAPTER_KIND,
            "adapter_version": ADAPTER_VERSION,
            "source_repository": source["repository"],
            "source_revision": SOURCE_REVISION,
            "source_manifest_sha256": source_hash,
            "source_shards": verified,
            "groups": len(rows),
            "terminal_groups_opened": len(rows) if terminal else len(terminal_rows),
            "group_manifest_sha256": group_hash,
            "rgb_materialized": bool(args.materialize_rgb),
            "scope": (
                "terminal_test" if terminal
                else ("smoke" if args.limit_groups else "full_nonterminal_train")
            ),
            "terminal_authorization": ({
                "run_id": args.terminal_run_id,
                "preterminal_audit": str(audit_path),
                "preterminal_audit_sha256": sha256(audit_path),
            } if terminal else None),
            "crop_source": "FROZEN_HAMER_OFFICIAL_LOCALIZER_V1_ONLY",
            "gt_bbox_used": False,
            "supervision_target_policy": SUPERVISION_TARGET_POLICY,
            "cross_view_label_policy": (
                "diagnostic_only_never_average_into_supervision"
            ),
            "label_alignment_audit": alignment_audit_path.name,
            "label_alignment_audit_sha256": sha256(alignment_audit_path),
            "camera_contract": {
                "label_cam_extr": "camera_to_common_world",
                "runtime_world": "physical_camera_0_reference",
                "physical_to_sequence_suffix": list(PHYSICAL_TO_SEQUENCE_SUFFIX),
                "runtime_slot_to_physical": list(RUNTIME_SLOT_TO_PHYSICAL),
                "reference_physical_camera": REFERENCE_PHYSICAL_CAMERA,
                "reference_runtime_slot": REFERENCE_RUNTIME_SLOT,
                "raw_size": [640, 480],
                "units_3d": "metres",
            },
            "joints_2d_space": "crop",
            "joints_2d_transform_count": 1,
            "mesh_alignment_rms_m_max": max(alignment_values),
            "mesh_alignment_groups_above_5e_5_m": int(sum(
                value > CROSS_VIEW_ALIGNMENT_TOLERANCE_M
                for value in alignment_values
            )),
            "joint_reprojection_max_px": max(reprojection_values),
        }
        (partial / "HO3D_WDS_ADAPTER_CONTRACT.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
        report = {
            "status": "PASS", "groups": len(rows),
            "terminal_groups_quarantined": len(terminal_rows),
            "terminal_groups_opened": len(rows) if terminal else 0,
            "source_bytes": sum(row["bytes"] for row in verified), "source_shards_verified": len(verified),
            "materialized_rgb_files": len(rows) * 5 if args.materialize_rgb else 0,
            "contract_sha256": sha256(partial / "HO3D_WDS_ADAPTER_CONTRACT.json"),
            "label_alignment_audit_sha256": sha256(alignment_audit_path),
            "group_manifest_sha256": group_hash,
        }
        (partial / "BUILD_REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        os.replace(partial, args.output)
        print(json.dumps(report, sort_keys=True))
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
