"""Population manifests: ordered sample IDs per split, their hashes and consistency checks."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Tuple, Union
import unicodedata

import numpy as np

from mvhand.contracts import ArrayLike, CoordinateFrame, LengthUnit


PUBLISHED_RIGHT_HAND_S0_COUNT = 4951
LEGACY_EVALUATOR_BRIDGE_COUNT = 4950
LEGACY_BRIDGE_EXCLUDED_ID = "DexYCB_mv_test-000003.tar::000004950"


class PopulationName(str, Enum):
    PUBLISHED_RIGHT_HAND_S0_4951 = "published_right_hand_s0_4951"
    LEGACY_EVALUATOR_BRIDGE_4950 = "legacy_evaluator_bridge_4950"


@dataclass(frozen=True, order=True)
class CanonicalSampleId:
    shard: str
    local_key: str

    def __post_init__(self) -> None:
        for name, value in (("shard", self.shard), ("local_key", self.local_key)):
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string")
            if not value or value != value.strip():
                raise ValueError(f"{name} must be a non-empty, trimmed string")
            if "::" in value:
                raise ValueError(f"{name} must not contain '::'")
            if any(
                unicodedata.category(character).startswith("C")
                or unicodedata.category(character) in {"Zl", "Zp"}
                for character in value
            ):
                raise ValueError(f"{name} must not contain control or record-separator characters")

    @classmethod
    def parse(cls, value: str) -> "CanonicalSampleId":
        if not isinstance(value, str) or value.count("::") != 1:
            raise ValueError("canonical ID must be '<shard>::<local_key>'")
        return cls(*value.split("::", 1))

    def __str__(self) -> str:
        return f"{self.shard}::{self.local_key}"


@dataclass(frozen=True)
class PopulationManifest:
    population_name: PopulationName
    declared_count: int
    sample_ids: Tuple[CanonicalSampleId, ...]
    ordered_ids_sha256: Optional[str] = None


@dataclass(frozen=True)
class MultiViewGroupRecord:
    """Metadata-only calibrated multiview group.

    View-indexed fields have leading shape ``[V,...]``. Intrinsics are
    ``[V,3,3]`` in raw pixels; reference-to-camera transforms are ``[V,4,4]``;
    affine transforms are ``[V,2,3]``; visibility is ``[V,J]``. Stored joints
    and vertices end in ``[J,3]`` and ``[M,3]`` in ``length_unit`` and
    ``coordinate_frame``. Images remain references and are never decoded here.
    View zero is the canonical reference unless explicitly represented by the
    validated ``reference_view_index``. NumPy fields are owned read-only
    copies, and metadata is recursively frozen.
    """

    sample_id: CanonicalSampleId
    camera_ids: Tuple[str, ...]
    reference_view_index: int
    image_references: Tuple[str, ...]
    raw_image_sizes: Tuple[Tuple[int, int], ...]
    crop_image_sizes: Tuple[Tuple[int, int], ...]
    raw_to_crop: ArrayLike
    crop_to_raw: ArrayLike
    intrinsics: ArrayLike
    reference_to_camera: ArrayLike
    visibility: ArrayLike
    handedness: str
    coordinate_frame: CoordinateFrame
    length_unit: LengthUnit
    subject: Optional[str] = None
    sequence: Optional[str] = None
    frame: Optional[str] = None
    stored_joints_3d: Optional[ArrayLike] = None
    stored_vertices_3d: Optional[ArrayLike] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "raw_to_crop",
            "crop_to_raw",
            "intrinsics",
            "reference_to_camera",
            "visibility",
            "stored_joints_3d",
            "stored_vertices_3d",
        ):
            value = getattr(self, name)
            if isinstance(value, np.ndarray):
                object.__setattr__(self, name, _owned_readonly_array(value))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


def ordered_id_sha256(sample_ids: Sequence[CanonicalSampleId]) -> str:
    digest = hashlib.sha256()
    values = tuple(sample_ids)
    digest.update(b"FORM2FRAME_ORDERED_IDS_V1\x00")
    digest.update(len(values).to_bytes(8, byteorder="big", signed=False))
    for sample_id in values:
        if not isinstance(sample_id, CanonicalSampleId):
            raise TypeError("sample_ids must contain CanonicalSampleId values")
        encoded = str(sample_id).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def validate_population_manifest(
    manifest: PopulationManifest,
) -> None:
    if not isinstance(manifest, PopulationManifest):
        raise TypeError("manifest must be a PopulationManifest")
    if not isinstance(manifest.population_name, PopulationName):
        raise TypeError("population_name must be a PopulationName")
    if type(manifest.declared_count) is not int or manifest.declared_count <= 0:
        raise ValueError("declared_count must be a positive Python integer")
    if not manifest.sample_ids:
        raise ValueError("sample_ids must not be empty")
    if len(manifest.sample_ids) != manifest.declared_count:
        raise ValueError("declared_count does not match sample_ids")
    if any(not isinstance(value, CanonicalSampleId) for value in manifest.sample_ids):
        raise TypeError("sample_ids must contain CanonicalSampleId values")
    if len(set(manifest.sample_ids)) != len(manifest.sample_ids):
        raise ValueError("canonical sample IDs must be unique")
    expected = {
        PopulationName.PUBLISHED_RIGHT_HAND_S0_4951: PUBLISHED_RIGHT_HAND_S0_COUNT,
        PopulationName.LEGACY_EVALUATOR_BRIDGE_4950: LEGACY_EVALUATOR_BRIDGE_COUNT,
    }[manifest.population_name]
    if manifest.declared_count != expected:
        raise ValueError(f"{manifest.population_name.value} requires declared_count={expected}")
    if manifest.ordered_ids_sha256 is not None:
        if not _is_sha256(manifest.ordered_ids_sha256):
            raise ValueError("ordered_ids_sha256 must be 64 hexadecimal characters")
        if ordered_id_sha256(manifest.sample_ids) != manifest.ordered_ids_sha256.lower():
            raise ValueError("ordered sample ID hash mismatch")


def validate_expected_ids(
    manifest: PopulationManifest,
    expected: Sequence[CanonicalSampleId],
) -> None:
    validate_population_manifest(manifest)
    expected_tuple = tuple(expected)
    if len(set(expected_tuple)) != len(expected_tuple):
        raise ValueError("expected IDs must be unique")
    actual_set = set(manifest.sample_ids)
    expected_set = set(expected_tuple)
    missing = sorted(str(value) for value in expected_set - actual_set)
    unexpected = sorted(str(value) for value in actual_set - expected_set)
    if missing or unexpected:
        raise ValueError(f"population ID mismatch: missing={missing}, unexpected={unexpected}")
    for index, (actual, wanted) in enumerate(zip(manifest.sample_ids, expected_tuple)):
        if actual != wanted:
            raise ValueError(f"population ID order mismatch at index {index}")


def validate_population_relationship(
    published: PopulationManifest,
    legacy: PopulationManifest,
) -> None:
    validate_population_manifest(published)
    validate_population_manifest(legacy)
    if published.population_name is not PopulationName.PUBLISHED_RIGHT_HAND_S0_4951:
        raise ValueError("published manifest has the wrong population name")
    if legacy.population_name is not PopulationName.LEGACY_EVALUATOR_BRIDGE_4950:
        raise ValueError("legacy manifest has the wrong population name")
    validate_single_extra_relationship(
        published.sample_ids,
        legacy.sample_ids,
        CanonicalSampleId.parse(LEGACY_BRIDGE_EXCLUDED_ID),
    )


def validate_single_extra_relationship(
    published_ids: Sequence[CanonicalSampleId],
    legacy_ids: Sequence[CanonicalSampleId],
    expected_extra: CanonicalSampleId,
    *,
    published_ordered_sha256: Optional[str] = None,
    legacy_ordered_sha256: Optional[str] = None,
) -> None:
    """Validate a generic one-extra set relation without naming a protocol."""
    published = _validated_id_sequence(published_ids, "published_ids")
    legacy = _validated_id_sequence(legacy_ids, "legacy_ids")
    if not isinstance(expected_extra, CanonicalSampleId):
        raise TypeError("expected_extra must be a CanonicalSampleId")
    for name, values, expected_hash in (
        ("published", published, published_ordered_sha256),
        ("legacy", legacy, legacy_ordered_sha256),
    ):
        if expected_hash is not None:
            if not _is_sha256(expected_hash):
                raise ValueError(f"{name}_ordered_sha256 must be 64 hexadecimal characters")
            if ordered_id_sha256(values) != expected_hash.lower():
                raise ValueError(f"{name} ordered sample ID hash mismatch")
    published_set = set(published)
    legacy_set = set(legacy)
    if published_set - legacy_set != {expected_extra} or legacy_set - published_set:
        raise ValueError("population relationship does not contain exactly the expected extra ID")


def population_manifest_to_json(manifest: PopulationManifest) -> str:
    validate_population_manifest(manifest)
    payload = {
        "population_name": manifest.population_name.value,
        "declared_count": manifest.declared_count,
        "ordered_ids_sha256": manifest.ordered_ids_sha256,
        "samples": [
            {"shard": value.shard, "local_key": value.local_key}
            for value in manifest.sample_ids
        ],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def read_population_manifest_json(
    path: Union[str, Path],
    *,
    strict: bool = True,
) -> PopulationManifest:
    supplied = Path(path)
    if not supplied.is_file():
        raise FileNotFoundError(f"population manifest is not a regular file: {supplied}")
    try:
        payload = json.loads(supplied.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not parse population manifest JSON: {exc}") from exc
    return _population_from_mapping(payload, strict=strict)


def read_population_manifest_csv(
    path: Union[str, Path],
    *,
    strict: bool = True,
) -> PopulationManifest:
    supplied = Path(path)
    if not supplied.is_file():
        raise FileNotFoundError(f"population manifest is not a regular file: {supplied}")
    try:
        with supplied.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            columns = set(reader.fieldnames or ())
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"could not parse population manifest CSV: {exc}") from exc
    required = {"population_name", "declared_count", "shard", "local_key"}
    allowed = required | {"ordered_ids_sha256"}
    if not required <= columns or strict and not columns <= allowed:
        raise ValueError("population CSV columns do not match the configured schema")
    if not rows:
        raise ValueError("population CSV must not be empty")
    names = {row["population_name"] for row in rows}
    counts = {row["declared_count"] for row in rows}
    hashes = {row.get("ordered_ids_sha256", "") for row in rows}
    if len(names) != 1 or len(counts) != 1 or len(hashes) != 1:
        raise ValueError("population CSV metadata must be identical on every row")
    try:
        declared_count = int(next(iter(counts)))
    except (TypeError, ValueError) as exc:
        raise ValueError("declared_count must be an integer") from exc
    manifest = PopulationManifest(
        PopulationName(next(iter(names))),
        declared_count,
        tuple(CanonicalSampleId(row["shard"], row["local_key"]) for row in rows),
        next(iter(hashes)) or None,
    )
    validate_population_manifest(manifest)
    return manifest


def validate_multiview_group_record(record: MultiViewGroupRecord) -> None:
    if not isinstance(record, MultiViewGroupRecord):
        raise TypeError("record must be a MultiViewGroupRecord")
    views = len(record.camera_ids)
    if views <= 0 or len(set(record.camera_ids)) != views:
        raise ValueError("camera_ids must be non-empty and unique")
    if any(not isinstance(value, str) or not value for value in record.camera_ids):
        raise ValueError("camera IDs must be non-empty strings")
    if type(record.reference_view_index) is not int or not 0 <= record.reference_view_index < views:
        raise ValueError("reference_view_index is out of range")
    for name, value in (
        ("image_references", record.image_references),
        ("raw_image_sizes", record.raw_image_sizes),
        ("crop_image_sizes", record.crop_image_sizes),
    ):
        if len(value) != views:
            raise ValueError(f"{name} must contain one value per view")
    for sizes in (record.raw_image_sizes, record.crop_image_sizes):
        if any(len(size) != 2 or any(type(x) is not int or x <= 0 for x in size) for size in sizes):
            raise ValueError("image sizes must be positive integer (width, height) pairs")
    expected_shapes = {
        "raw_to_crop": (views, 2, 3),
        "crop_to_raw": (views, 2, 3),
        "intrinsics": (views, 3, 3),
        "reference_to_camera": (views, 4, 4),
    }
    for name, expected in expected_shapes.items():
        _require_shape_and_finite(getattr(record, name), expected, name)
    visibility_shape = _shape(record.visibility)
    if visibility_shape is None or len(visibility_shape) != 2 or visibility_shape[0] != views:
        raise ValueError("visibility must have shape [V,J]")
    _require_finite(record.visibility, "visibility")
    if record.handedness not in {"right", "left"}:
        raise ValueError("handedness must be 'right' or 'left'")
    if not isinstance(record.coordinate_frame, CoordinateFrame) or not isinstance(record.length_unit, LengthUnit):
        raise TypeError("coordinate_frame and length_unit must use contract enums")
    joints = _shape(record.stored_joints_3d) if record.stored_joints_3d is not None else None
    if joints is not None and (len(joints) != 2 or joints[-1] != 3 or joints[0] != visibility_shape[1]):
        raise ValueError("stored_joints_3d must have shape [J,3]")
    vertices = _shape(record.stored_vertices_3d) if record.stored_vertices_3d is not None else None
    if vertices is not None and (len(vertices) != 2 or vertices[-1] != 3):
        raise ValueError("stored_vertices_3d must have shape [M,3]")
    if record.stored_joints_3d is not None:
        _require_finite(record.stored_joints_3d, "stored_joints_3d")
    if record.stored_vertices_3d is not None:
        _require_finite(record.stored_vertices_3d, "stored_vertices_3d")


def _population_from_mapping(payload: Any, *, strict: bool) -> PopulationManifest:
    if not isinstance(payload, Mapping):
        raise ValueError("population JSON top level must be an object")
    required = {"population_name", "declared_count", "samples"}
    allowed = required | {"ordered_ids_sha256"}
    keys = set(payload)
    if not required <= keys or strict and not keys <= allowed:
        raise ValueError("population JSON fields do not match the configured schema")
    samples = payload["samples"]
    if not isinstance(samples, list):
        raise ValueError("samples must be a list")
    parsed = []
    for sample in samples:
        if not isinstance(sample, Mapping) or set(sample) != {"shard", "local_key"}:
            raise ValueError("each sample must contain exactly shard and local_key")
        parsed.append(CanonicalSampleId(sample["shard"], sample["local_key"]))
    manifest = PopulationManifest(
        PopulationName(payload["population_name"]),
        payload["declared_count"],
        tuple(parsed),
        payload.get("ordered_ids_sha256"),
    )
    validate_population_manifest(manifest)
    return manifest


def _validated_id_sequence(
    values: Sequence[CanonicalSampleId],
    name: str,
) -> Tuple[CanonicalSampleId, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of CanonicalSampleId values")
    result = tuple(values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    if any(not isinstance(value, CanonicalSampleId) for value in result):
        raise TypeError(f"{name} must contain CanonicalSampleId values")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique IDs")
    return result


def _owned_readonly_array(value: np.ndarray) -> np.ndarray:
    result = np.array(value, copy=True, order="K")
    result.setflags(write=False)
    return result


def _freeze_metadata(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("metadata floats must be finite")
        return value
    if isinstance(value, np.ndarray):
        return _owned_readonly_array(value)
    if isinstance(value, Mapping):
        frozen = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError("metadata mapping keys must be strings")
            frozen[key] = _freeze_metadata(child)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata(child) for child in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_metadata(child) for child in value)
    raise TypeError(f"unsupported mutable metadata value: {type(value).__name__}")


def _shape(value: Any) -> Optional[Tuple[int, ...]]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(int(item) for item in shape)
    return None


def _require_shape_and_finite(value: Any, expected: Tuple[int, ...], name: str) -> None:
    if _shape(value) != expected:
        raise ValueError(f"{name} must have shape {expected}")
    _require_finite(value, name)


def _require_finite(value: Any, name: str) -> None:
    if hasattr(value, "tolist"):
        value = value.tolist()

    def visit(item: Any) -> None:
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        elif isinstance(item, bool):
            return
        elif not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            raise ValueError(f"{name} must contain only finite numeric values")

    visit(value)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdefABCDEF" for character in value)
