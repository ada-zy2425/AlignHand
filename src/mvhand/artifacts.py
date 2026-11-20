from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
import json
import math
import re
from typing import Any, Dict, Mapping, Optional, Tuple

from mvhand.contracts import (
    AlignmentPolicy,
    ArrayLike,
    CanonicalEvaluationProtocol,
    CoordinateFrame,
    EvaluationEndpoint,
    InvalidValuePolicy,
    LengthUnit,
    MetricName,
)


_AMBIGUOUS_ARRAY_KEYS = {
    "pred_joints",
    "gt_joints",
    "mesh_mpjpe",
    "official_mpjpe",
}


@dataclass
class PredictionBundle:
    schema_version: str
    sample_keys: Tuple[str, ...]
    population_masks: Dict[str, ArrayLike]
    arrays: Dict[str, ArrayLike]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Legacy generic summary; not a canonical reproduction result manifest."""
    schema_version: str
    protocol_name: str
    population_scope: str
    sample_count: int
    metrics: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArtifactRecord:
    logical_name: str
    role: str
    sha256: str
    size_bytes: int
    relative_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReproducibilityManifest:
    schema_version: str
    project_name: str
    method_name: str
    dataset: str
    split: str
    prior: Optional[str]
    code_commit: Optional[str]
    coordinate_frame: CoordinateFrame
    length_unit: LengthUnit
    endpoint_definitions: Dict[str, str]
    joint_conventions: Dict[str, str]
    population: Dict[str, Any]
    artifacts: Tuple[ArtifactRecord, ...]
    metrics: Dict[str, Any]
    unresolved_historical_claims: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FrozenEvaluationArtifact:
    schema_version: str
    protocol: CanonicalEvaluationProtocol
    ids_key: str
    prediction_key: str
    reference_key: str
    validity_key: Optional[str]
    prediction_shape: Tuple[int, ...]
    reference_shape: Tuple[int, ...]
    prediction_dtype: str
    reference_dtype: str
    prediction_sha256: str
    reference_sha256: str
    container_sha256: Optional[str]
    software_commit: str
    checkpoint_manifest_id: Optional[str] = None
    asset_manifest_id: Optional[str] = None


@dataclass(frozen=True)
class CanonicalEvaluationResult:
    schema_version: str
    protocol_id: str
    population_name: str
    population_manifest_sha256: str
    ordered_ids_sha256: str
    sample_count: int
    prediction_endpoint: EvaluationEndpoint
    reference_endpoint: EvaluationEndpoint
    prediction_convention: str
    reference_convention: str
    coordinate_frame: CoordinateFrame
    input_length_unit: LengthUnit
    metric_output_unit: LengthUnit
    alignment_policy: AlignmentPolicy
    invalid_value_policy: InvalidValuePolicy
    metric_name: MetricName
    metric_version: str
    input_artifact_sha256: str
    prediction_array_sha256: str
    reference_array_sha256: str
    per_sample_metrics_sha256: str
    aggregate_result_sha256: str
    count_total: int
    count_valid: int
    count_invalid: int
    mean_mm: float
    software_commit: str
    checkpoint_manifest_id: Optional[str] = None
    asset_manifest_id: Optional[str] = None


def _require_non_empty_string(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError(f"{name} must be exactly 64 hexadecimal characters")


def validate_canonical_evaluation_protocol(
    protocol: CanonicalEvaluationProtocol,
) -> None:
    if not isinstance(protocol, CanonicalEvaluationProtocol):
        raise TypeError("protocol must be a CanonicalEvaluationProtocol")
    for name in (
        "schema_version", "protocol_id", "population_name",
        "prediction_convention", "reference_convention", "metric_version",
    ):
        _require_non_empty_string(getattr(protocol, name), name)
    _require_sha256(protocol.population_manifest_sha256, "population_manifest_sha256")
    _require_sha256(protocol.ordered_ids_sha256, "ordered_ids_sha256")
    if type(protocol.sample_count) is not int or protocol.sample_count <= 0:
        raise ValueError("sample_count must be a positive Python int")
    reserved_counts = {
        "published_right_hand_s0_4951": 4951,
        "legacy_evaluator_bridge_4950": 4950,
    }
    expected_count = reserved_counts.get(protocol.population_name)
    if expected_count is not None and protocol.sample_count != expected_count:
        raise ValueError(
            f"{protocol.population_name} requires sample_count={expected_count}"
        )
    if protocol.population_name in {"official", "full", "test", "default"}:
        raise ValueError("population_name must be an explicit canonical population identifier")
    for value, expected, name in (
        (protocol.prediction_endpoint, EvaluationEndpoint, "prediction_endpoint"),
        (protocol.reference_endpoint, EvaluationEndpoint, "reference_endpoint"),
        (protocol.coordinate_frame, CoordinateFrame, "coordinate_frame"),
        (protocol.length_unit, LengthUnit, "length_unit"),
        (protocol.metric_output_unit, LengthUnit, "metric_output_unit"),
        (protocol.alignment_policy, AlignmentPolicy, "alignment_policy"),
        (protocol.metric_name, MetricName, "metric_name"),
        (protocol.invalid_value_policy, InvalidValuePolicy, "invalid_value_policy"),
    ):
        if not isinstance(value, expected):
            raise TypeError(f"{name} has the wrong enum type")
    prediction_vertex = protocol.prediction_endpoint is EvaluationEndpoint.FINAL_VERTICES
    reference_vertex = protocol.reference_endpoint is EvaluationEndpoint.FINAL_VERTICES
    if prediction_vertex != reference_vertex:
        raise ValueError("joint and vertex endpoints cannot be mixed")
    if (protocol.metric_name is MetricName.MPVPE) != prediction_vertex:
        raise ValueError("MPVPE requires two final-vertex endpoints; MPJPE requires joints")
    if protocol.prediction_convention != protocol.reference_convention:
        raise ValueError("endpoint conventions must match; mapping must occur explicitly upstream")
    if protocol.length_unit is not LengthUnit.METER:
        raise ValueError("canonical metric inputs must be expressed in metres")
    if protocol.metric_output_unit is not LengthUnit.MILLIMETER:
        raise ValueError("canonical metric outputs must be expressed in millimetres")
    if protocol.alignment_policy is not AlignmentPolicy.NONE:
        raise ValueError("only unaligned absolute camera-space evaluation is supported")


def validate_frozen_evaluation_artifact(value: FrozenEvaluationArtifact) -> None:
    if not isinstance(value, FrozenEvaluationArtifact):
        raise TypeError("value must be a FrozenEvaluationArtifact")
    validate_canonical_evaluation_protocol(value.protocol)
    for name in (
        "schema_version", "ids_key", "prediction_key", "reference_key",
        "prediction_dtype", "reference_dtype", "software_commit",
    ):
        _require_non_empty_string(getattr(value, name), name)
    keys = [value.ids_key, value.prediction_key, value.reference_key]
    if value.validity_key is not None:
        _require_non_empty_string(value.validity_key, "validity_key")
        keys.append(value.validity_key)
    if len(keys) != len(set(keys)):
        raise ValueError("artifact array keys must be distinct")
    for name in ("prediction_shape", "reference_shape"):
        shape = getattr(value, name)
        if not isinstance(shape, tuple) or len(shape) != 3 or any(type(x) is not int or x <= 0 for x in shape):
            raise ValueError(f"{name} must be a positive (N, P, 3) tuple")
        if shape[-1] != 3 or shape[0] != value.protocol.sample_count:
            raise ValueError(f"{name} does not match the protocol")
    if value.prediction_shape != value.reference_shape:
        raise ValueError("prediction and reference shapes must match")
    allowed_float_dtypes = {"float16", "float32", "float64"}
    if value.prediction_dtype not in allowed_float_dtypes or value.reference_dtype not in allowed_float_dtypes:
        raise ValueError("prediction and reference dtypes must be explicit floating types")
    if value.prediction_dtype != value.reference_dtype:
        raise ValueError("prediction and reference dtypes must match")
    _require_sha256(value.prediction_sha256, "prediction_sha256")
    _require_sha256(value.reference_sha256, "reference_sha256")
    if value.container_sha256 is not None:
        _require_sha256(value.container_sha256, "container_sha256")
    for name in ("checkpoint_manifest_id", "asset_manifest_id"):
        optional_value = getattr(value, name)
        if optional_value is not None:
            _require_non_empty_string(optional_value, name)


def validate_canonical_evaluation_result(value: CanonicalEvaluationResult) -> None:
    if not isinstance(value, CanonicalEvaluationResult):
        raise TypeError("value must be a CanonicalEvaluationResult")
    for name in (
        "schema_version", "protocol_id", "population_name",
        "prediction_convention", "reference_convention", "metric_version",
        "software_commit",
    ):
        _require_non_empty_string(getattr(value, name), name)
    for name in (
        "population_manifest_sha256", "ordered_ids_sha256", "input_artifact_sha256",
        "prediction_array_sha256", "reference_array_sha256",
        "per_sample_metrics_sha256", "aggregate_result_sha256",
    ):
        _require_sha256(getattr(value, name), name)
    for name in ("sample_count", "count_total", "count_valid", "count_invalid"):
        number = getattr(value, name)
        if type(number) is not int or number < 0:
            raise ValueError(f"{name} must be a non-negative Python int")
    if value.sample_count <= 0 or value.count_total != value.sample_count:
        raise ValueError("result count_total must equal positive sample_count")
    if value.count_valid <= 0 or value.count_valid + value.count_invalid != value.count_total:
        raise ValueError("result valid/invalid counts are inconsistent")
    if not isinstance(value.mean_mm, float) or not math.isfinite(value.mean_mm):
        raise ValueError("mean_mm must be a finite Python float")
    for name in ("checkpoint_manifest_id", "asset_manifest_id"):
        optional_value = getattr(value, name)
        if optional_value is not None:
            _require_non_empty_string(optional_value, name)
    validate_canonical_evaluation_protocol(CanonicalEvaluationProtocol(
        schema_version=value.schema_version,
        protocol_id=value.protocol_id,
        population_name=value.population_name,
        population_manifest_sha256=value.population_manifest_sha256,
        ordered_ids_sha256=value.ordered_ids_sha256,
        sample_count=value.sample_count,
        prediction_endpoint=value.prediction_endpoint,
        reference_endpoint=value.reference_endpoint,
        prediction_convention=value.prediction_convention,
        reference_convention=value.reference_convention,
        coordinate_frame=value.coordinate_frame,
        length_unit=value.input_length_unit,
        metric_output_unit=value.metric_output_unit,
        alignment_policy=value.alignment_policy,
        metric_name=value.metric_name,
        metric_version=value.metric_version,
        invalid_value_policy=value.invalid_value_policy,
    ))


def _reject_invalid_array_payload(value: object, name: str) -> None:
    if isinstance(value, (str, bytes, Mapping)):
        raise TypeError(f"{name} must be an array-like payload")


def _validate_population_mask_payload(mask: object, name: str) -> None:
    _reject_invalid_array_payload(mask, name)
    if isinstance(mask, (list, tuple)) and any(
        type(value) is not bool for value in mask
    ):
        raise TypeError(f"{name} list or tuple values must all be bool")


def _first_dimension_from_shape(value: object, name: str) -> Optional[int]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        if len(shape) == 0:
            raise ValueError(f"{name} must have a first dimension")
        return int(shape[0])
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"{name} has an invalid shape") from exc


def validate_prediction_bundle(
    bundle: PredictionBundle,
) -> None:
    if not isinstance(bundle, PredictionBundle):
        raise TypeError("bundle must be a PredictionBundle")
    _require_non_empty_string(bundle.schema_version, "schema_version")
    if not isinstance(bundle.sample_keys, tuple) or not bundle.sample_keys:
        raise ValueError("sample_keys must be a non-empty tuple")
    seen_sample_keys = set()
    for index, sample_key in enumerate(bundle.sample_keys):
        _require_non_empty_string(sample_key, f"sample_keys[{index}]")
        if sample_key in seen_sample_keys:
            raise ValueError(f"sample_keys contains a duplicate at index {index}")
        seen_sample_keys.add(sample_key)
    sample_count = len(bundle.sample_keys)

    if not isinstance(bundle.population_masks, Mapping):
        raise TypeError("population_masks must be a mapping")
    for name, mask in bundle.population_masks.items():
        _require_non_empty_string(name, "population mask name")
        if mask is None:
            raise ValueError(f"population mask {name!r} must not be None")
        _validate_population_mask_payload(mask, f"population mask {name!r}")
        first_dimension = _first_dimension_from_shape(mask, f"population mask {name!r}")
        if first_dimension is None:
            try:
                first_dimension = len(mask)
            except TypeError:
                first_dimension = None
        if first_dimension is not None and first_dimension != sample_count:
            raise ValueError(
                f"population mask {name!r} first dimension must equal sample count"
            )

    if not isinstance(bundle.arrays, Mapping) or not bundle.arrays:
        raise ValueError("arrays must be a non-empty mapping")
    for name, array in bundle.arrays.items():
        _require_non_empty_string(name, "array name")
        if name in _AMBIGUOUS_ARRAY_KEYS:
            raise ValueError(f"ambiguous array key is forbidden: {name!r}")
        if array is None:
            raise ValueError(f"array {name!r} must not be None")
        _reject_invalid_array_payload(array, f"array {name!r}")
        first_dimension = _first_dimension_from_shape(array, f"array {name!r}")
        if first_dimension is not None and first_dimension != sample_count:
            raise ValueError(f"array {name!r} first dimension must equal sample count")
    to_jsonable(bundle.metadata)


def validate_evaluation_result(
    result: EvaluationResult,
) -> None:
    if not isinstance(result, EvaluationResult):
        raise TypeError("result must be an EvaluationResult")
    _require_non_empty_string(result.schema_version, "schema_version")
    _require_non_empty_string(result.protocol_name, "protocol_name")
    _require_non_empty_string(result.population_scope, "population_scope")
    if type(result.sample_count) is not int or result.sample_count <= 0:
        raise ValueError("sample_count must be a positive Python integer")
    if not isinstance(result.metrics, Mapping) or not result.metrics:
        raise ValueError("metrics must be a non-empty mapping")
    for metric_name in result.metrics:
        _require_non_empty_string(metric_name, "metric name")
    to_jsonable(result.metrics)
    to_jsonable(result.metadata)


def _validate_relative_artifact_path(relative_path: str) -> None:
    if relative_path == ".":
        raise ValueError(
            "relative_path must identify a file or directory below the root"
        )
    posix_path = PurePosixPath(relative_path)
    windows_path = PureWindowsPath(relative_path)
    if posix_path.is_absolute():
        raise ValueError("relative_path must not be an absolute POSIX path")
    if windows_path.drive or windows_path.is_absolute():
        raise ValueError("relative_path must not contain a Windows drive or UNC path")
    path_parts = tuple(posix_path.parts) + tuple(windows_path.parts)
    if ".." in path_parts:
        raise ValueError("relative_path must not contain parent traversal")
    if any(part.startswith("~") for part in path_parts):
        raise ValueError("relative_path must not contain a home-directory path")


def validate_artifact_record(
    record: ArtifactRecord,
) -> None:
    if not isinstance(record, ArtifactRecord):
        raise TypeError("record must be an ArtifactRecord")
    _require_non_empty_string(record.logical_name, "logical_name")
    _require_non_empty_string(record.role, "role")
    if not isinstance(record.sha256, str) or re.fullmatch(
        r"[0-9a-fA-F]{64}", record.sha256
    ) is None:
        raise ValueError("sha256 must contain exactly 64 hexadecimal characters")
    if type(record.size_bytes) is not int or record.size_bytes < 0:
        raise ValueError("size_bytes must be a non-negative Python integer")
    if record.relative_path is not None:
        _require_non_empty_string(record.relative_path, "relative_path")
        _validate_relative_artifact_path(record.relative_path)
    to_jsonable(record.metadata)


def _to_jsonable(value: Any, active_ids: set) -> Any:
    if isinstance(value, Enum):
        return _to_jsonable(value.value, active_ids)
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite floating-point values are not JSON-safe")
        return value
    if isinstance(value, Path):
        return str(value)

    value_id = id(value)
    if value_id in active_ids:
        raise TypeError("recursive values are not supported")

    if is_dataclass(value) and not isinstance(value, type):
        active_ids.add(value_id)
        try:
            normalized = {
                dataclass_field.name: _to_jsonable(
                    getattr(value, dataclass_field.name),
                    active_ids,
                )
                for dataclass_field in fields(value)
            }
        finally:
            active_ids.remove(value_id)
        if isinstance(value, ArtifactRecord):
            normalized["sha256"] = value.sha256.lower()
        return normalized

    if isinstance(value, Mapping):
        active_ids.add(value_id)
        try:
            normalized_mapping = {}
            for key, item in value.items():
                normalized_key = str(key)
                if normalized_key in normalized_mapping:
                    raise ValueError(
                        "mapping keys collide after string conversion: "
                        f"{normalized_key!r}"
                    )
                normalized_mapping[normalized_key] = _to_jsonable(item, active_ids)
            return normalized_mapping
        finally:
            active_ids.remove(value_id)

    if isinstance(value, (list, tuple)):
        active_ids.add(value_id)
        try:
            return [_to_jsonable(item, active_ids) for item in value]
        finally:
            active_ids.remove(value_id)

    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            item_value = item_method()
        except (TypeError, ValueError, RuntimeError):
            item_value = value
        if item_value is not value:
            return _to_jsonable(item_value, active_ids)

    tolist_method = getattr(value, "tolist", None)
    if callable(tolist_method):
        try:
            list_value = tolist_method()
        except (TypeError, ValueError, RuntimeError):
            list_value = value
        if list_value is not value:
            return _to_jsonable(list_value, active_ids)

    raise TypeError(f"unsupported JSON value type: {type(value).__name__}")


def to_jsonable(value: Any) -> Any:
    normalized = _to_jsonable(value, set())
    json.dumps(normalized, allow_nan=False)
    return normalized


def to_json_dict(value: Any) -> Dict[str, Any]:
    normalized = to_jsonable(value)
    if not isinstance(normalized, dict):
        raise TypeError("normalized top-level value must be a dictionary")
    return normalized


def validate_reproducibility_manifest(
    manifest: ReproducibilityManifest,
) -> None:
    if not isinstance(manifest, ReproducibilityManifest):
        raise TypeError("manifest must be a ReproducibilityManifest")
    for field_name in (
        "schema_version",
        "project_name",
        "method_name",
        "dataset",
        "split",
    ):
        _require_non_empty_string(getattr(manifest, field_name), field_name)
    for field_name in ("prior", "code_commit"):
        value = getattr(manifest, field_name)
        if value is not None:
            _require_non_empty_string(value, field_name)
    if not isinstance(manifest.coordinate_frame, CoordinateFrame):
        raise TypeError("coordinate_frame must be a CoordinateFrame")
    if not isinstance(manifest.length_unit, LengthUnit):
        raise TypeError("length_unit must be a LengthUnit")
    for mapping_name, definitions in (
        ("endpoint_definitions", manifest.endpoint_definitions),
        ("joint_conventions", manifest.joint_conventions),
    ):
        if not isinstance(definitions, Mapping):
            raise TypeError(f"{mapping_name} must be a mapping")
        for key, value in definitions.items():
            _require_non_empty_string(key, f"{mapping_name} key")
            _require_non_empty_string(value, f"{mapping_name} value")
    if not isinstance(manifest.population, Mapping) or not manifest.population:
        raise ValueError("population must be a non-empty mapping")
    if not isinstance(manifest.artifacts, tuple):
        raise TypeError("artifacts must be a tuple")
    logical_names = set()
    for artifact in manifest.artifacts:
        validate_artifact_record(artifact)
        if artifact.logical_name in logical_names:
            raise ValueError(
                f"duplicate artifact logical name: {artifact.logical_name!r}"
            )
        logical_names.add(artifact.logical_name)
    if not isinstance(manifest.metrics, Mapping) or not manifest.metrics:
        raise ValueError("metrics must be a non-empty mapping")
    to_jsonable(manifest.population)
    to_jsonable(manifest.metrics)
    to_jsonable(manifest.unresolved_historical_claims)
    to_jsonable(manifest.metadata)
    for artifact in manifest.artifacts:
        to_jsonable(artifact.metadata)
