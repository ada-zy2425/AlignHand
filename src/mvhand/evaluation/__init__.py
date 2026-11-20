from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import io
import json
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

import torch
from torch import Tensor

from mvhand.contracts import EndpointName, Form2FrameOutput, MultiViewHandSample
from mvhand.artifacts import (
    CanonicalEvaluationResult,
    FrozenEvaluationArtifact,
    validate_canonical_evaluation_result,
    validate_frozen_evaluation_artifact,
)
from mvhand.contracts import InvalidValuePolicy
from mvhand.data.manifest import (
    CanonicalSampleId,
    PopulationName,
    ordered_id_sha256,
    read_population_manifest_json,
)
from mvhand.common import sha256_file


@dataclass
class MetricSummary:
    population_scope: str
    sample_count: int
    mean_mm: float
    median_mm: float
    p90_mm: float
    p95_mm: float
    max_mm: float
    sum_mm: float


@dataclass
class LedgerSpec:
    sample_key_column: str
    metric_columns: Dict[str, str]
    population_column: Optional[str] = None


@dataclass
class LedgerData:
    sample_keys: Tuple[str, ...]
    metrics_mm: Dict[str, Tuple[float, ...]]
    population_mask: Optional[Tuple[bool, ...]] = None


@dataclass
class LedgerParityResult:
    row_count: int
    sample_key_sequence_valid: bool
    population_mask_equal: Optional[bool]
    max_abs_diff_mm: Dict[str, float]
    parity_valid: Dict[str, bool]


@dataclass
class LeaveOneOutIntegrity:
    population: int
    included_count: int
    excluded_count: int
    excluded_index: int
    full_mean_mm: float
    included_mean_mm: float
    excluded_error_mm: float
    full_sum_mm: float
    included_sum_mm: float
    identity_residual_mm: float
    valid: bool


@dataclass(frozen=True)
class PerSampleMetric:
    sample_id: str
    value_mm: Optional[float]
    valid: bool
    prediction_endpoint: str
    reference_endpoint: str
    prediction_convention: str
    reference_convention: str
    coordinate_frame: str
    input_length_unit: str
    metric_output_unit: str
    alignment_policy: str
    invalid_value_policy: str
    population_name: str
    metric_name: str


@dataclass(frozen=True)
class FrozenEvaluationData:
    artifact: FrozenEvaluationArtifact
    sample_ids: Tuple[CanonicalSampleId, ...]
    prediction: np.ndarray
    reference: np.ndarray
    validity: np.ndarray


@dataclass(frozen=True)
class CanonicalEvaluationRun:
    per_sample: Tuple[PerSampleMetric, ...]
    result: CanonicalEvaluationResult


PER_SAMPLE_COLUMNS = (
    "sample_id", "value_mm", "valid", "prediction_endpoint",
    "reference_endpoint", "prediction_convention", "reference_convention",
    "coordinate_frame", "input_length_unit", "metric_output_unit",
    "alignment_policy", "invalid_value_policy", "population_name", "metric_name",
)


def per_sample_csv_text(rows: Sequence[PerSampleMetric]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=PER_SAMPLE_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row.__dict__)
    return stream.getvalue()


def canonical_json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def aggregate_result_dict(result: CanonicalEvaluationResult) -> Dict[str, Any]:
    return {
        "schema_version": result.schema_version,
        "protocol_id": result.protocol_id,
        "population_name": result.population_name,
        "population_manifest_sha256": result.population_manifest_sha256,
        "ordered_ids_sha256": result.ordered_ids_sha256,
        "sample_count": result.sample_count,
        "prediction_endpoint": result.prediction_endpoint.value,
        "reference_endpoint": result.reference_endpoint.value,
        "prediction_convention": result.prediction_convention,
        "reference_convention": result.reference_convention,
        "coordinate_frame": result.coordinate_frame.value,
        "input_length_unit": result.input_length_unit.value,
        "metric_output_unit": result.metric_output_unit.value,
        "alignment_policy": result.alignment_policy.value,
        "invalid_value_policy": result.invalid_value_policy.value,
        "metric_name": result.metric_name.value,
        "metric_version": result.metric_version,
        "count_total": result.count_total,
        "count_valid": result.count_valid,
        "count_invalid": result.count_invalid,
        "mean_mm": result.mean_mm,
    }


def _require_floating_finite_tensor(value: object, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must contain only finite values")
    return value


def extract_prediction_endpoint(
    output: Form2FrameOutput,
    endpoint: EndpointName,
) -> Tensor:
    if not isinstance(output, Form2FrameOutput):
        raise TypeError("output must be a Form2FrameOutput")
    if not isinstance(endpoint, EndpointName):
        raise TypeError("endpoint must be an EndpointName")

    if endpoint is EndpointName.DIRECT_JOINTS_3D:
        value = output.direct_joints_3d
    elif endpoint is EndpointName.CALIBRATED_SCAFFOLD_3D:
        value = output.calibrated_scaffold_3d
    elif endpoint is EndpointName.FINAL_VERTICES_3D:
        value = output.final_vertices_3d
    elif endpoint is EndpointName.FINAL_MESH_JOINTS_3D:
        value = output.final_mesh_joints_3d
    elif endpoint is EndpointName.XGEO_JOINTS_3D:
        value = output.xgeo_joints_3d
    elif endpoint is EndpointName.POSITIONED_PRIOR_JOINTS_3D:
        value = output.positioned_prior_joints_3d
    elif endpoint is EndpointName.RAW_MANO_MESH_JOINTS_3D:
        value = output.mesh_joints_mano_3d
    elif endpoint is EndpointName.DATASET_MAPPED_MESH_JOINTS_3D:
        value = output.mesh_joints_dataset_3d
    elif endpoint is EndpointName.DATASET_STORED_JOINTS_3D:
        raise ValueError("dataset-stored joints are not a prediction endpoint")
    else:
        raise ValueError(f"unsupported prediction endpoint: {endpoint}")

    return _require_floating_finite_tensor(value, f"prediction endpoint {endpoint.value}")


def extract_reference_endpoint(
    sample: MultiViewHandSample,
    endpoint: EndpointName,
) -> Tensor:
    if not isinstance(sample, MultiViewHandSample):
        raise TypeError("sample must be a MultiViewHandSample")
    if not isinstance(endpoint, EndpointName):
        raise TypeError("endpoint must be an EndpointName")

    if endpoint is EndpointName.DATASET_STORED_JOINTS_3D:
        value = sample.dataset_stored_joints_3d
    elif endpoint is EndpointName.FINAL_VERTICES_3D:
        value = sample.ground_truth_vertices_3d
    else:
        raise ValueError(f"reference endpoint is unavailable from the sample: {endpoint}")

    return _require_floating_finite_tensor(value, f"reference endpoint {endpoint.value}")


def per_sample_point_error_mm(
    prediction: Tensor,
    reference: Tensor,
) -> Tensor:
    prediction = _require_floating_finite_tensor(prediction, "prediction")
    reference = _require_floating_finite_tensor(reference, "reference")
    if prediction.shape != reference.shape:
        raise ValueError("prediction and reference must have exactly equal shapes")
    if prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError("prediction and reference must have shape (N, P, 3)")
    if prediction.shape[0] <= 0 or prediction.shape[1] <= 0:
        raise ValueError("N and P must both be positive")
    if prediction.dtype != reference.dtype:
        raise TypeError("prediction and reference must have the same dtype")
    if prediction.device != reference.device:
        raise ValueError("prediction and reference must be on the same device")

    return torch.linalg.vector_norm(prediction - reference, dim=-1).mean(dim=-1) * 1000.0


def mpjpe_per_sample_mm(
    prediction: Tensor,
    reference: Tensor,
) -> Tensor:
    return per_sample_point_error_mm(prediction, reference)


def mpvpe_per_sample_mm(
    prediction: Tensor,
    reference: Tensor,
) -> Tensor:
    return per_sample_point_error_mm(prediction, reference)


def array_sha256(value: np.ndarray) -> str:
    """Hash an array's dtype, shape, and canonical C-order bytes."""
    if not isinstance(value, np.ndarray):
        raise TypeError("value must be a numpy.ndarray")
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(b"FORM2FRAME_ARRAY_V1\x00")
    dtype = contiguous.dtype.str.encode("ascii")
    digest.update(len(dtype).to_bytes(4, "big"))
    digest.update(dtype)
    digest.update(contiguous.ndim.to_bytes(4, "big"))
    for dimension in contiguous.shape:
        digest.update(int(dimension).to_bytes(8, "big"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _require_exact_mapping(payload: object, required: set, optional: set, name: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{name} must be an object")
    keys = set(payload)
    missing = sorted(required - keys)
    unknown = sorted(keys - required - optional)
    if missing or unknown:
        raise ValueError(f"{name} fields mismatch: missing={missing}, unknown={unknown}")
    return payload


def frozen_artifact_from_json(path: Union[str, Path]) -> FrozenEvaluationArtifact:
    """Read strict metadata only; array payload bytes are loaded separately."""
    from mvhand.contracts import (
        AlignmentPolicy, CanonicalEvaluationProtocol, CoordinateFrame,
        EvaluationEndpoint, LengthUnit, MetricName,
    )

    supplied = Path(path)
    if not supplied.is_file():
        raise FileNotFoundError(f"evaluation manifest is not a regular file: {supplied}")
    try:
        payload = json.loads(supplied.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not parse evaluation manifest: {exc}") from exc
    top_required = {
        "schema_version", "protocol", "ids_key", "prediction_key", "reference_key",
        "validity_key", "prediction_shape", "reference_shape", "prediction_dtype",
        "reference_dtype", "prediction_sha256", "reference_sha256",
        "container_sha256", "software_commit", "checkpoint_manifest_id",
        "asset_manifest_id",
    }
    data = _require_exact_mapping(payload, top_required, set(), "evaluation manifest")
    protocol_required = {
        "schema_version", "protocol_id", "population_name",
        "population_manifest_sha256", "ordered_ids_sha256",
        "sample_count", "prediction_endpoint", "reference_endpoint",
        "prediction_convention", "reference_convention", "coordinate_frame",
        "length_unit", "metric_output_unit", "alignment_policy",
        "metric_name", "metric_version",
        "invalid_value_policy",
    }
    p = _require_exact_mapping(data["protocol"], protocol_required, set(), "protocol")
    try:
        protocol = CanonicalEvaluationProtocol(
            schema_version=p["schema_version"], protocol_id=p["protocol_id"],
            population_name=p["population_name"],
            population_manifest_sha256=p["population_manifest_sha256"],
            ordered_ids_sha256=p["ordered_ids_sha256"],
            sample_count=p["sample_count"], prediction_endpoint=EvaluationEndpoint(p["prediction_endpoint"]),
            reference_endpoint=EvaluationEndpoint(p["reference_endpoint"]),
            prediction_convention=p["prediction_convention"], reference_convention=p["reference_convention"],
            coordinate_frame=CoordinateFrame(p["coordinate_frame"]), length_unit=LengthUnit(p["length_unit"]),
            metric_output_unit=LengthUnit(p["metric_output_unit"]),
            alignment_policy=AlignmentPolicy(p["alignment_policy"]), metric_name=MetricName(p["metric_name"]),
            metric_version=p["metric_version"], invalid_value_policy=InvalidValuePolicy(p["invalid_value_policy"]),
        )
        artifact = FrozenEvaluationArtifact(
            schema_version=data["schema_version"], protocol=protocol,
            ids_key=data["ids_key"], prediction_key=data["prediction_key"],
            reference_key=data["reference_key"], validity_key=data["validity_key"],
            prediction_shape=tuple(data["prediction_shape"]), reference_shape=tuple(data["reference_shape"]),
            prediction_dtype=data["prediction_dtype"], reference_dtype=data["reference_dtype"],
            prediction_sha256=data["prediction_sha256"], reference_sha256=data["reference_sha256"],
            container_sha256=data["container_sha256"], software_commit=data["software_commit"],
            checkpoint_manifest_id=data["checkpoint_manifest_id"], asset_manifest_id=data["asset_manifest_id"],
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError(f"invalid evaluation manifest value: {exc}") from exc
    validate_frozen_evaluation_artifact(artifact)
    return artifact


def load_frozen_evaluation_artifact(
    manifest_path: Union[str, Path],
    artifact_path: Union[str, Path],
    population_manifest_path: Union[str, Path],
) -> FrozenEvaluationData:
    artifact = frozen_artifact_from_json(manifest_path)
    population_path = Path(population_manifest_path)
    if not population_path.is_file():
        raise FileNotFoundError(
            f"population manifest is not a regular file: {population_path}"
        )
    if sha256_file(population_path) != artifact.protocol.population_manifest_sha256.lower():
        raise ValueError("population manifest SHA-256 mismatch")
    try:
        population_payload = json.loads(population_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not parse population manifest: {exc}") from exc
    population_data = _require_exact_mapping(
        population_payload,
        {"population_name", "declared_count", "ordered_ids_sha256", "samples"},
        set(),
        "population manifest",
    )
    if not isinstance(population_data["samples"], list):
        raise TypeError("population manifest samples must be a list")
    population_ids = []
    for index, sample in enumerate(population_data["samples"]):
        item = _require_exact_mapping(sample, {"shard", "local_key"}, set(), f"population sample {index}")
        population_ids.append(CanonicalSampleId(item["shard"], item["local_key"]))
    population_ids = tuple(population_ids)
    if len(population_ids) != population_data["declared_count"]:
        raise ValueError("population manifest declared_count mismatch")
    if len(set(population_ids)) != len(population_ids):
        raise ValueError("population manifest IDs must be unique")
    population_hash = ordered_id_sha256(population_ids)
    if population_hash != population_data["ordered_ids_sha256"]:
        raise ValueError("population manifest ordered-ID hash mismatch")
    if population_data["population_name"] != artifact.protocol.population_name:
        raise ValueError("artifact and population manifest names differ")
    if len(population_ids) != artifact.protocol.sample_count:
        raise ValueError("artifact and population manifest counts differ")
    if population_hash != artifact.protocol.ordered_ids_sha256:
        raise ValueError("artifact and population manifest ordered-ID hashes differ")
    if population_data["population_name"] in {member.value for member in PopulationName}:
        phase4_manifest = read_population_manifest_json(population_path)
        if phase4_manifest.sample_ids != population_ids:
            raise ValueError("Phase-4 population manifest IDs differ")
    supplied = Path(artifact_path)
    if not supplied.is_file():
        raise FileNotFoundError(f"artifact is not a regular file: {supplied}")
    if artifact.container_sha256 is not None and sha256_file(supplied) != artifact.container_sha256.lower():
        raise ValueError("artifact container SHA-256 mismatch")
    expected_keys = {artifact.ids_key, artifact.prediction_key, artifact.reference_key}
    if artifact.validity_key is not None:
        expected_keys.add(artifact.validity_key)
    try:
        with np.load(supplied, allow_pickle=False) as arrays:
            if set(arrays.files) != expected_keys:
                raise ValueError("artifact array keys do not exactly match the manifest")
            ids_array = np.array(arrays[artifact.ids_key], copy=True)
            prediction = np.array(arrays[artifact.prediction_key], copy=True)
            reference = np.array(arrays[artifact.reference_key], copy=True)
            validity = (
                np.array(arrays[artifact.validity_key], copy=True)
                if artifact.validity_key is not None
                else np.ones(artifact.protocol.sample_count, dtype=np.bool_)
            )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"could not load non-pickle NPZ artifact: {exc}") from exc
    if ids_array.ndim != 1 or ids_array.dtype.kind not in {"U", "S"}:
        raise ValueError("ID array must be a one-dimensional string array")
    raw_ids = ids_array.tolist()
    if ids_array.dtype.kind == "S":
        try:
            raw_ids = [value.decode("utf-8") for value in raw_ids]
        except UnicodeDecodeError as exc:
            raise ValueError("ID array contains invalid UTF-8") from exc
    sample_ids = tuple(CanonicalSampleId.parse(value) for value in raw_ids)
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("artifact canonical IDs must be unique")
    if len(sample_ids) != artifact.protocol.sample_count:
        raise ValueError("artifact ID count does not match protocol sample_count")
    if ordered_id_sha256(sample_ids) != artifact.protocol.ordered_ids_sha256.lower():
        raise ValueError("artifact ordered canonical-ID hash mismatch")
    if sample_ids != population_ids:
        missing = sorted(str(value) for value in set(population_ids) - set(sample_ids))
        unexpected = sorted(str(value) for value in set(sample_ids) - set(population_ids))
        if missing or unexpected:
            raise ValueError(
                f"artifact population IDs mismatch: missing={missing}, unexpected={unexpected}"
            )
        raise ValueError("artifact population ID order mismatch")
    for value, expected_shape, expected_dtype, expected_hash, name in (
        (prediction, artifact.prediction_shape, artifact.prediction_dtype, artifact.prediction_sha256, "prediction"),
        (reference, artifact.reference_shape, artifact.reference_dtype, artifact.reference_sha256, "reference"),
    ):
        if value.shape != expected_shape or value.dtype.name != expected_dtype:
            raise ValueError(f"{name} shape or dtype does not match manifest")
        if array_sha256(value) != expected_hash.lower():
            raise ValueError(f"{name} array SHA-256 mismatch")
    if validity.shape != (artifact.protocol.sample_count,) or validity.dtype != np.bool_:
        raise ValueError("validity array must have Boolean shape (N,)")
    finite = np.isfinite(prediction).all(axis=(1, 2)) & np.isfinite(reference).all(axis=(1, 2))
    if artifact.protocol.invalid_value_policy is InvalidValuePolicy.FAIL and not finite.all():
        raise ValueError("artifact inputs contain non-finite values")
    if artifact.protocol.invalid_value_policy is InvalidValuePolicy.FAIL and not validity.all():
        raise ValueError("FAIL invalid-value policy requires every row to be valid")
    if artifact.protocol.invalid_value_policy is InvalidValuePolicy.EXPLICIT_VALIDITY_MASK:
        if artifact.validity_key is None:
            raise ValueError("explicit validity policy requires validity_key")
        if np.any(validity & ~finite):
            raise ValueError("valid artifact rows must contain finite arrays")
    for value in (prediction, reference, validity):
        value.setflags(write=False)
    return FrozenEvaluationData(artifact, sample_ids, prediction, reference, validity)


def evaluate_frozen_artifact(data: FrozenEvaluationData, container_sha256: str) -> CanonicalEvaluationRun:
    if not isinstance(data, FrozenEvaluationData):
        raise TypeError("data must be FrozenEvaluationData")
    artifact = data.artifact
    valid = data.validity
    if not valid.any():
        raise ValueError("evaluation requires at least one valid sample")
    prediction = torch.tensor(data.prediction, dtype=torch.float64)
    reference = torch.tensor(data.reference, dtype=torch.float64)
    errors = per_sample_point_error_mm(prediction, reference).cpu()
    rows = tuple(
        PerSampleMetric(
            sample_id=str(sample_id), value_mm=float(errors[index].item()) if valid[index] else None,
            valid=bool(valid[index]),
            prediction_endpoint=artifact.protocol.prediction_endpoint.value,
            reference_endpoint=artifact.protocol.reference_endpoint.value,
            prediction_convention=artifact.protocol.prediction_convention,
            reference_convention=artifact.protocol.reference_convention,
            coordinate_frame=artifact.protocol.coordinate_frame.value,
            input_length_unit=artifact.protocol.length_unit.value,
            metric_output_unit=artifact.protocol.metric_output_unit.value,
            alignment_policy=artifact.protocol.alignment_policy.value,
            invalid_value_policy=artifact.protocol.invalid_value_policy.value,
            population_name=artifact.protocol.population_name,
            metric_name=artifact.protocol.metric_name.value,
        )
        for index, sample_id in enumerate(data.sample_ids)
    )
    per_sample_text = per_sample_csv_text(rows)
    count_valid = int(valid.sum())
    count_invalid = int((~valid).sum())
    valid_errors = errors[torch.tensor(valid.tolist(), dtype=torch.bool)]
    mean_mm = float(valid_errors.mean().item())
    aggregate_payload = {
        "schema_version": "1", "protocol_id": artifact.protocol.protocol_id,
        "population_name": artifact.protocol.population_name,
        "population_manifest_sha256": artifact.protocol.population_manifest_sha256,
        "ordered_ids_sha256": artifact.protocol.ordered_ids_sha256,
        "sample_count": artifact.protocol.sample_count,
        "prediction_endpoint": artifact.protocol.prediction_endpoint.value,
        "reference_endpoint": artifact.protocol.reference_endpoint.value,
        "prediction_convention": artifact.protocol.prediction_convention,
        "reference_convention": artifact.protocol.reference_convention,
        "coordinate_frame": artifact.protocol.coordinate_frame.value,
        "input_length_unit": artifact.protocol.length_unit.value,
        "metric_output_unit": artifact.protocol.metric_output_unit.value,
        "alignment_policy": artifact.protocol.alignment_policy.value,
        "invalid_value_policy": artifact.protocol.invalid_value_policy.value,
        "metric_name": artifact.protocol.metric_name.value,
        "metric_version": artifact.protocol.metric_version,
        "count_total": len(rows), "count_valid": count_valid,
        "count_invalid": count_invalid, "mean_mm": mean_mm,
    }
    aggregate_text = canonical_json_text(aggregate_payload)
    result = CanonicalEvaluationResult(
        schema_version="1", protocol_id=artifact.protocol.protocol_id,
        population_name=artifact.protocol.population_name,
        population_manifest_sha256=artifact.protocol.population_manifest_sha256,
        ordered_ids_sha256=artifact.protocol.ordered_ids_sha256,
        sample_count=artifact.protocol.sample_count,
        prediction_endpoint=artifact.protocol.prediction_endpoint,
        reference_endpoint=artifact.protocol.reference_endpoint,
        prediction_convention=artifact.protocol.prediction_convention,
        reference_convention=artifact.protocol.reference_convention,
        coordinate_frame=artifact.protocol.coordinate_frame,
        input_length_unit=artifact.protocol.length_unit,
        metric_output_unit=artifact.protocol.metric_output_unit,
        alignment_policy=artifact.protocol.alignment_policy,
        invalid_value_policy=artifact.protocol.invalid_value_policy,
        metric_name=artifact.protocol.metric_name, metric_version=artifact.protocol.metric_version,
        input_artifact_sha256=container_sha256,
        prediction_array_sha256=artifact.prediction_sha256,
        reference_array_sha256=artifact.reference_sha256,
        per_sample_metrics_sha256=hashlib.sha256(per_sample_text.encode("utf-8")).hexdigest(),
        aggregate_result_sha256=hashlib.sha256(aggregate_text.encode("utf-8")).hexdigest(),
        count_total=len(rows), count_valid=count_valid, count_invalid=count_invalid,
        mean_mm=mean_mm, software_commit=artifact.software_commit,
        checkpoint_manifest_id=artifact.checkpoint_manifest_id,
        asset_manifest_id=artifact.asset_manifest_id,
    )
    validate_canonical_evaluation_result(result)
    return CanonicalEvaluationRun(rows, result)


def canonical_result_from_json(path: Union[str, Path]) -> CanonicalEvaluationResult:
    from mvhand.contracts import AlignmentPolicy, CoordinateFrame, EvaluationEndpoint, LengthUnit, MetricName

    supplied = Path(path)
    try:
        payload = json.loads(supplied.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not parse canonical result manifest: {exc}") from exc
    required = {
        "schema_version", "protocol_id", "population_name",
        "population_manifest_sha256", "ordered_ids_sha256", "sample_count",
        "prediction_endpoint", "reference_endpoint", "prediction_convention",
        "reference_convention", "coordinate_frame", "input_length_unit",
        "metric_output_unit", "alignment_policy", "invalid_value_policy",
        "metric_name", "metric_version", "input_artifact_sha256",
        "prediction_array_sha256", "reference_array_sha256",
        "per_sample_metrics_sha256", "aggregate_result_sha256", "count_total",
        "count_valid", "count_invalid", "mean_mm", "software_commit",
        "checkpoint_manifest_id", "asset_manifest_id",
    }
    data = _require_exact_mapping(payload, required, set(), "canonical result manifest")
    try:
        result = CanonicalEvaluationResult(
            schema_version=data["schema_version"], protocol_id=data["protocol_id"],
            population_name=data["population_name"],
            population_manifest_sha256=data["population_manifest_sha256"],
            ordered_ids_sha256=data["ordered_ids_sha256"], sample_count=data["sample_count"],
            prediction_endpoint=EvaluationEndpoint(data["prediction_endpoint"]),
            reference_endpoint=EvaluationEndpoint(data["reference_endpoint"]),
            prediction_convention=data["prediction_convention"],
            reference_convention=data["reference_convention"],
            coordinate_frame=CoordinateFrame(data["coordinate_frame"]),
            input_length_unit=LengthUnit(data["input_length_unit"]),
            metric_output_unit=LengthUnit(data["metric_output_unit"]),
            alignment_policy=AlignmentPolicy(data["alignment_policy"]),
            invalid_value_policy=InvalidValuePolicy(data["invalid_value_policy"]),
            metric_name=MetricName(data["metric_name"]), metric_version=data["metric_version"],
            input_artifact_sha256=data["input_artifact_sha256"],
            prediction_array_sha256=data["prediction_array_sha256"],
            reference_array_sha256=data["reference_array_sha256"],
            per_sample_metrics_sha256=data["per_sample_metrics_sha256"],
            aggregate_result_sha256=data["aggregate_result_sha256"],
            count_total=data["count_total"], count_valid=data["count_valid"],
            count_invalid=data["count_invalid"], mean_mm=data["mean_mm"],
            software_commit=data["software_commit"],
            checkpoint_manifest_id=data["checkpoint_manifest_id"],
            asset_manifest_id=data["asset_manifest_id"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid canonical result value: {exc}") from exc
    validate_canonical_evaluation_result(result)
    return result


def verify_result_bundle(
    bundle_dir: Union[str, Path],
    evaluation_manifest_path: Union[str, Path],
    artifact_path: Union[str, Path],
    population_manifest_path: Union[str, Path],
) -> CanonicalEvaluationResult:
    bundle = Path(bundle_dir)
    expected_names = {"per_sample_metrics.csv", "aggregate_result.json", "result_manifest.json"}
    if not bundle.is_dir() or {value.name for value in bundle.iterdir()} != expected_names:
        raise ValueError("result bundle must contain exactly the three canonical files")
    per_sample_path = bundle / "per_sample_metrics.csv"
    aggregate_path = bundle / "aggregate_result.json"
    result_path = bundle / "result_manifest.json"
    result = canonical_result_from_json(result_path)
    if sha256_file(per_sample_path) != result.per_sample_metrics_sha256:
        raise ValueError("per-sample metrics SHA-256 mismatch")
    if sha256_file(aggregate_path) != result.aggregate_result_sha256:
        raise ValueError("aggregate result SHA-256 mismatch")
    if sha256_file(artifact_path) != result.input_artifact_sha256:
        raise ValueError("input artifact SHA-256 mismatch")
    if sha256_file(population_manifest_path) != result.population_manifest_sha256:
        raise ValueError("population manifest SHA-256 mismatch")
    loaded = load_frozen_evaluation_artifact(
        evaluation_manifest_path, artifact_path, population_manifest_path
    )
    protocol = loaded.artifact.protocol
    for actual, expected, name in (
        (result.protocol_id, protocol.protocol_id, "protocol_id"),
        (result.population_name, protocol.population_name, "population_name"),
        (result.ordered_ids_sha256, protocol.ordered_ids_sha256, "ordered_ids_sha256"),
        (result.prediction_endpoint, protocol.prediction_endpoint, "prediction_endpoint"),
        (result.reference_endpoint, protocol.reference_endpoint, "reference_endpoint"),
        (result.prediction_convention, protocol.prediction_convention, "prediction_convention"),
        (result.reference_convention, protocol.reference_convention, "reference_convention"),
        (result.coordinate_frame, protocol.coordinate_frame, "coordinate_frame"),
        (result.input_length_unit, protocol.length_unit, "input_length_unit"),
        (result.metric_output_unit, protocol.metric_output_unit, "metric_output_unit"),
        (result.alignment_policy, protocol.alignment_policy, "alignment_policy"),
        (result.invalid_value_policy, protocol.invalid_value_policy, "invalid_value_policy"),
        (result.metric_name, protocol.metric_name, "metric_name"),
    ):
        if actual != expected:
            raise ValueError(f"result {name} differs from evaluation protocol")
    if result.prediction_array_sha256 != loaded.artifact.prediction_sha256:
        raise ValueError("result prediction array hash differs from artifact manifest")
    if result.reference_array_sha256 != loaded.artifact.reference_sha256:
        raise ValueError("result reference array hash differs from artifact manifest")
    try:
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not parse aggregate result: {exc}") from exc
    if aggregate != aggregate_result_dict(result):
        raise ValueError("aggregate result content differs from result manifest")
    with per_sample_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PER_SAMPLE_COLUMNS:
            raise ValueError("per-sample CSV columns do not match canonical contract")
        rows = list(reader)
    if len(rows) != result.count_total:
        raise ValueError("per-sample row count differs from result manifest")
    values = []
    valid_count = 0
    for index, (row, expected_id) in enumerate(zip(rows, loaded.sample_ids)):
        if row["sample_id"] != str(expected_id):
            raise ValueError(f"per-sample ID mismatch at row {index}")
        valid = row["valid"] == "True"
        if row["valid"] not in {"True", "False"}:
            raise ValueError(f"invalid validity token at row {index}")
        expected_fields = {
            "prediction_endpoint": result.prediction_endpoint.value,
            "reference_endpoint": result.reference_endpoint.value,
            "prediction_convention": result.prediction_convention,
            "reference_convention": result.reference_convention,
            "coordinate_frame": result.coordinate_frame.value,
            "input_length_unit": result.input_length_unit.value,
            "metric_output_unit": result.metric_output_unit.value,
            "alignment_policy": result.alignment_policy.value,
            "invalid_value_policy": result.invalid_value_policy.value,
            "population_name": result.population_name,
            "metric_name": result.metric_name.value,
        }
        if any(row[name] != expected for name, expected in expected_fields.items()):
            raise ValueError(f"per-sample provenance mismatch at row {index}")
        if valid:
            try:
                value = float(row["value_mm"])
            except ValueError as exc:
                raise ValueError(f"invalid metric value at row {index}") from exc
            if not math.isfinite(value):
                raise ValueError(f"non-finite metric value at row {index}")
            values.append(value)
            valid_count += 1
        elif row["value_mm"] not in {"", "None"}:
            raise ValueError(f"invalid row {index} must not contain a metric value")
    if valid_count != result.count_valid or len(rows) - valid_count != result.count_invalid:
        raise ValueError("per-sample validity counts differ from result manifest")
    if not values or abs(sum(values) / len(values) - result.mean_mm) > 1e-12:
        raise ValueError("aggregate mean differs from per-sample metrics")
    return result


def apply_population_mask(
    per_sample_errors_mm: Tensor,
    population_mask: Optional[Tensor],
) -> Tensor:
    errors = _require_floating_finite_tensor(
        per_sample_errors_mm,
        "per_sample_errors_mm",
    )
    if errors.ndim != 1 or errors.shape[0] <= 0:
        raise ValueError("per_sample_errors_mm must have shape (N,) with N positive")
    if population_mask is None:
        return errors
    if not isinstance(population_mask, Tensor):
        raise TypeError("population_mask must be a torch.Tensor")
    if population_mask.ndim != 1 or population_mask.shape != errors.shape:
        raise ValueError("population_mask must have shape (N,)")
    if population_mask.dtype != torch.bool:
        raise TypeError("population_mask must have dtype torch.bool")
    if population_mask.device != errors.device:
        raise ValueError("population_mask must be on the same device as errors")
    if not population_mask.any().item():
        raise ValueError("population_mask must select at least one sample")
    return errors[population_mask]


def summarize_errors_mm(
    per_sample_errors_mm: Tensor,
    population_scope: str,
    population_mask: Optional[Tensor] = None,
) -> MetricSummary:
    if not isinstance(population_scope, str) or not population_scope.strip():
        raise ValueError("population_scope must be a non-empty string")
    selected = apply_population_mask(per_sample_errors_mm, population_mask)
    return MetricSummary(
        population_scope=population_scope,
        sample_count=int(selected.shape[0]),
        mean_mm=float(selected.mean().item()),
        median_mm=float(torch.quantile(selected, 0.50).item()),
        p90_mm=float(torch.quantile(selected, 0.90).item()),
        p95_mm=float(torch.quantile(selected, 0.95).item()),
        max_mm=float(selected.max().item()),
        sum_mm=float(selected.sum().item()),
    )


def _validated_sample_keys(keys: Sequence[str], name: str) -> Tuple[str, ...]:
    if not isinstance(keys, Sequence):
        raise TypeError(f"{name} must be a sequence")
    validated = tuple(keys)
    seen = set()
    for index, key in enumerate(validated):
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{name} contains an invalid key at index {index}")
        if key in seen:
            raise ValueError(f"{name} contains a duplicate key at index {index}: {key!r}")
        seen.add(key)
    return validated


def validate_sample_key_sequence(
    expected: Sequence[str],
    observed: Sequence[str],
) -> None:
    expected_keys = _validated_sample_keys(expected, "expected")
    observed_keys = _validated_sample_keys(observed, "observed")
    if len(expected_keys) != len(observed_keys):
        mismatch_index = min(len(expected_keys), len(observed_keys))
        raise ValueError(
            "sample-key sequence length mismatch at index "
            f"{mismatch_index}: expected {len(expected_keys)} rows, "
            f"observed {len(observed_keys)}"
        )
    for index, (expected_key, observed_key) in enumerate(
        zip(expected_keys, observed_keys)
    ):
        if expected_key != observed_key:
            raise ValueError(
                f"sample-key mismatch at index {index}: "
                f"expected {expected_key!r}, observed {observed_key!r}"
            )


def _validate_ledger_spec(spec: LedgerSpec) -> None:
    if not isinstance(spec, LedgerSpec):
        raise TypeError("spec must be a LedgerSpec")
    if not isinstance(spec.sample_key_column, str) or not spec.sample_key_column.strip():
        raise ValueError("sample_key_column must be a non-empty string")
    if not isinstance(spec.metric_columns, dict) or not spec.metric_columns:
        raise ValueError("metric_columns must be a non-empty dictionary")
    configured_columns = [spec.sample_key_column]
    for metric_name, column_name in spec.metric_columns.items():
        if not isinstance(metric_name, str) or not metric_name.strip():
            raise ValueError("metric names must be non-empty strings")
        if not isinstance(column_name, str) or not column_name.strip():
            raise ValueError("metric column names must be non-empty strings")
        configured_columns.append(column_name)
    if spec.population_column is not None:
        if not isinstance(spec.population_column, str) or not spec.population_column.strip():
            raise ValueError("population_column must be a non-empty string when provided")
        configured_columns.append(spec.population_column)
    if len(configured_columns) != len(set(configured_columns)):
        raise ValueError("configured ledger columns must be distinct")


def read_metric_ledger(
    path: Union[str, Path],
    spec: LedgerSpec,
) -> LedgerData:
    _validate_ledger_spec(spec)
    if not isinstance(path, (str, Path)):
        raise TypeError("path must be a string or pathlib.Path")
    ledger_path = Path(path)
    if not ledger_path.is_file():
        raise ValueError(f"ledger path must be a regular file: {ledger_path}")

    sample_keys = []
    metric_values = {name: [] for name in spec.metric_columns}
    population_values = [] if spec.population_column is not None else None
    population_tokens = {
        "0": False,
        "1": True,
        "true": True,
        "false": False,
        "True": True,
        "False": False,
    }

    with ledger_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("ledger is missing a header")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("ledger header contains duplicate columns")
        required_columns = {spec.sample_key_column, *spec.metric_columns.values()}
        if spec.population_column is not None:
            required_columns.add(spec.population_column)
        missing_columns = sorted(required_columns.difference(reader.fieldnames))
        if missing_columns:
            raise ValueError(f"ledger is missing required columns: {missing_columns}")

        for row_index, row in enumerate(reader):
            key = row[spec.sample_key_column]
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"ledger has an invalid sample key at row {row_index}")
            sample_keys.append(key)
            for metric_name, column_name in spec.metric_columns.items():
                raw_value = row[column_name]
                try:
                    value = float(raw_value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"ledger metric {metric_name!r} is invalid at row {row_index}"
                    ) from exc
                if not math.isfinite(value):
                    raise ValueError(
                        f"ledger metric {metric_name!r} is non-finite at row {row_index}"
                    )
                metric_values[metric_name].append(value)
            if spec.population_column is not None:
                raw_population = row[spec.population_column]
                if raw_population not in population_tokens:
                    raise ValueError(
                        f"ledger population value is invalid at row {row_index}"
                    )
                population_values.append(population_tokens[raw_population])

    if not sample_keys:
        raise ValueError("ledger must contain at least one row")
    validated_keys = _validated_sample_keys(sample_keys, "ledger sample keys")
    return LedgerData(
        sample_keys=validated_keys,
        metrics_mm={
            name: tuple(values) for name, values in metric_values.items()
        },
        population_mask=(
            tuple(population_values) if population_values is not None else None
        ),
    )


def _validate_tolerance(tolerance_mm: float) -> float:
    if isinstance(tolerance_mm, bool) or not isinstance(tolerance_mm, (int, float)):
        raise TypeError("tolerance_mm must be a real number")
    tolerance = float(tolerance_mm)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance_mm must be finite and non-negative")
    return tolerance


def validate_ledger_parity(
    expected_sample_keys: Sequence[str],
    computed_metrics_mm: Mapping[str, Tensor],
    ledger: LedgerData,
    tolerance_mm: float,
    expected_population_mask: Optional[Tensor] = None,
) -> LedgerParityResult:
    tolerance = _validate_tolerance(tolerance_mm)
    if not isinstance(ledger, LedgerData):
        raise TypeError("ledger must be LedgerData")
    validate_sample_key_sequence(expected_sample_keys, ledger.sample_keys)
    row_count = len(ledger.sample_keys)
    sample_key_sequence_valid = True

    if not isinstance(computed_metrics_mm, Mapping):
        raise TypeError("computed_metrics_mm must be a mapping")

    population_mask_equal = None
    if expected_population_mask is not None:
        if not isinstance(expected_population_mask, Tensor):
            raise TypeError("expected_population_mask must be a torch.Tensor")
        if expected_population_mask.ndim != 1 or expected_population_mask.shape[0] != row_count:
            raise ValueError("expected_population_mask must have shape (N,)")
        if expected_population_mask.dtype != torch.bool:
            raise TypeError("expected_population_mask must have dtype torch.bool")
        if ledger.population_mask is not None:
            population_mask_equal = tuple(
                bool(value) for value in expected_population_mask.detach().cpu().tolist()
            ) == ledger.population_mask

    max_abs_diff_mm = {}
    parity_valid = {}
    population_valid = population_mask_equal is not False
    for metric_name, computed in computed_metrics_mm.items():
        if not isinstance(metric_name, str) or not metric_name:
            raise ValueError("computed metric names must be non-empty strings")
        if metric_name not in ledger.metrics_mm:
            raise ValueError(f"ledger is missing computed metric {metric_name!r}")
        computed = _require_floating_finite_tensor(
            computed,
            f"computed metric {metric_name}",
        )
        if computed.ndim != 1 or computed.shape[0] != row_count:
            raise ValueError(f"computed metric {metric_name!r} must have shape (N,)")
        ledger_values = ledger.metrics_mm[metric_name]
        if len(ledger_values) != row_count:
            raise ValueError(f"ledger metric {metric_name!r} must have N values")
        if any(not math.isfinite(value) for value in ledger_values):
            raise ValueError(f"ledger metric {metric_name!r} contains non-finite values")
        computed_work = computed.detach().to(
            device="cpu",
            dtype=torch.float64,
        )
        ledger_work = torch.tensor(
            ledger_values,
            dtype=torch.float64,
        )
        max_difference = float(
            torch.max(torch.abs(computed_work - ledger_work)).item()
        )
        max_abs_diff_mm[metric_name] = max_difference
        parity_valid[metric_name] = population_valid and max_difference <= tolerance

    return LedgerParityResult(
        row_count=row_count,
        sample_key_sequence_valid=sample_key_sequence_valid,
        population_mask_equal=population_mask_equal,
        max_abs_diff_mm=max_abs_diff_mm,
        parity_valid=parity_valid,
    )


def check_leave_one_out_integrity(
    per_sample_errors_mm: Tensor,
    included_mask: Tensor,
    tolerance_mm: float = 1e-9,
) -> LeaveOneOutIntegrity:
    tolerance_mm = _validate_tolerance(tolerance_mm)
    errors = _require_floating_finite_tensor(
        per_sample_errors_mm,
        "per_sample_errors_mm",
    )
    if errors.ndim != 1:
        raise ValueError("per_sample_errors_mm must have shape (N,)")
    population = int(errors.shape[0])
    if population < 2:
        raise ValueError("leave-one-out evaluation requires at least two errors")
    if not isinstance(included_mask, Tensor):
        raise TypeError("included_mask must be a torch.Tensor")
    if included_mask.ndim != 1 or included_mask.shape != errors.shape:
        raise ValueError("included_mask must have shape (N,)")
    if included_mask.dtype != torch.bool:
        raise TypeError("included_mask must have dtype torch.bool")
    if included_mask.device != errors.device:
        raise ValueError("included_mask must be on the same device as errors")

    excluded_mask = ~included_mask
    excluded_count = int(excluded_mask.sum().item())
    if excluded_count != 1:
        raise ValueError("included_mask must exclude exactly one sample")
    included_count = int(included_mask.sum().item())
    if included_count < 1:
        raise ValueError("included_mask must include at least one sample")
    excluded_index = int(torch.nonzero(excluded_mask, as_tuple=False).item())
    work_errors = errors.to(dtype=torch.float64)
    full_sum = work_errors.sum()
    included_sum = work_errors[included_mask].sum()
    excluded_error = work_errors[excluded_index]
    residual = full_sum - (included_sum + excluded_error)

    return LeaveOneOutIntegrity(
        population=population,
        included_count=included_count,
        excluded_count=excluded_count,
        excluded_index=excluded_index,
        full_mean_mm=float(work_errors.mean().item()),
        included_mean_mm=float(work_errors[included_mask].mean().item()),
        excluded_error_mm=float(excluded_error.item()),
        full_sum_mm=float(full_sum.item()),
        included_sum_mm=float(included_sum.item()),
        identity_residual_mm=float(residual.item()),
        valid=abs(float(residual.item())) <= tolerance_mm,
    )
