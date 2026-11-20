from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


ArrayLike = Any


class CoordinateFrame(str, Enum):
    WORLD = "world"
    CAMERA_REFERENCE = "camera_reference"
    CAMERA_LOCAL = "camera_local"
    CROP_LOCAL = "crop_local"


class LengthUnit(str, Enum):
    METER = "meter"
    MILLIMETER = "millimeter"


class JointConvention(str, Enum):
    MANO_16 = "mano_16"
    MANO_21_AUGMENTED = "mano_21_augmented"
    OPENPOSE_21 = "openpose_21"
    DEXYCB_21 = "dexycb_21"


class EndpointName(str, Enum):
    """Runtime model/sample tensor slots; not frozen-artifact semantics."""
    DIRECT_JOINTS_3D = "direct_joints_3d"
    CALIBRATED_SCAFFOLD_3D = "calibrated_scaffold_3d"
    FINAL_VERTICES_3D = "final_vertices_3d"
    FINAL_MESH_JOINTS_3D = "final_mesh_joints_3d"
    XGEO_JOINTS_3D = "xgeo_joints_3d"
    POSITIONED_PRIOR_JOINTS_3D = "positioned_prior_joints_3d"
    RAW_MANO_MESH_JOINTS_3D = "raw_mano_mesh_joints_3d"
    DATASET_MAPPED_MESH_JOINTS_3D = "dataset_mapped_mesh_joints_3d"
    DATASET_STORED_JOINTS_3D = "dataset_stored_joints_3d"


class PopulationScope(str, Enum):
    FULL = "full"
    OFFICIAL = "official"
    CUSTOM = "custom"


class EvaluationEndpoint(str, Enum):
    """Canonical semantic endpoints for evaluation artifacts and results."""
    STORED_PREDICTION_JOINTS = "stored_prediction_joints"
    STORED_DATASET_GT_JOINTS = "stored_dataset_gt_joints"
    MAPPED_PREDICTION_JOINTS = "mapped_prediction_joints"
    MAPPED_GT_JOINTS = "mapped_gt_joints"
    POEM_MESH_DERIVED_JOINTS = "poem_mesh_derived_joints"
    GENERIC_MANO_AUGMENTED_JOINTS = "generic_mano_augmented_joints"
    POSITIONED_PRIOR_JOINTS = "positioned_prior_joints"
    BLENDED_FINAL_JOINTS = "blended_final_joints"
    FINAL_VERTICES = "final_vertices"


class MetricName(str, Enum):
    MPJPE = "mpjpe"
    MPVPE = "mpvpe"


class AlignmentPolicy(str, Enum):
    NONE = "none"
    ROOT_RELATIVE = "root_relative"
    PROCRUSTES = "procrustes"


class InvalidValuePolicy(str, Enum):
    FAIL = "fail"
    EXPLICIT_VALIDITY_MASK = "explicit_validity_mask"


@dataclass(frozen=True)
class CanonicalEvaluationProtocol:
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
    length_unit: LengthUnit
    metric_output_unit: LengthUnit
    alignment_policy: AlignmentPolicy
    metric_name: MetricName
    metric_version: str
    invalid_value_policy: InvalidValuePolicy


@dataclass
class CameraContract:
    camera_id: str
    intrinsics: ArrayLike
    reference_to_camera: ArrayLike
    image_size: Tuple[int, int]

    def __post_init__(self) -> None:
        intrinsics_shape = getattr(self.intrinsics, "shape", None)
        if intrinsics_shape is not None and tuple(intrinsics_shape[-2:]) != (3, 3):
            raise ValueError("intrinsics must end in shape (3, 3)")

        transform_shape = getattr(self.reference_to_camera, "shape", None)
        if transform_shape is not None and tuple(transform_shape[-2:]) != (4, 4):
            raise ValueError("reference_to_camera must end in shape (4, 4)")


@dataclass
class DatasetContract:
    dataset_name: str
    split: str
    coordinate_frame: CoordinateFrame
    length_unit: LengthUnit
    joint_convention: JointConvention
    expected_num_views: Optional[int] = None
    vertex_topology: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiViewHandSample:
    sample_key: str
    view_ids: List[str]
    images: Optional[ArrayLike] = None
    cameras: List[CameraContract] = field(default_factory=list)
    runtime_view_mask: Optional[ArrayLike] = None
    gt_joint_visibility: Optional[ArrayLike] = None
    dataset_stored_joints_3d: Optional[ArrayLike] = None
    ground_truth_vertices_3d: Optional[ArrayLike] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Form2FrameOutput:
    final_vertices_3d: Optional[ArrayLike] = None
    final_mesh_joints_3d: Optional[ArrayLike] = None
    xgeo_joints_3d: Optional[ArrayLike] = None
    positioned_prior_joints_3d: Optional[ArrayLike] = None
    mesh_joints_mano_3d: Optional[ArrayLike] = None
    mesh_joints_dataset_3d: Optional[ArrayLike] = None
    direct_joints_3d: Optional[ArrayLike] = None
    calibrated_scaffold_3d: Optional[ArrayLike] = None
    relative_view_weights: Optional[ArrayLike] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationProtocol:
    """Legacy runtime endpoint-selection contract.

    Canonical frozen reproduction uses :class:`CanonicalEvaluationProtocol`.
    This distinct runtime contract is retained for Phase-1 extraction only.
    """
    protocol_name: str
    coordinate_frame: CoordinateFrame
    length_unit: LengthUnit
    population_scope: PopulationScope
    target_endpoint: EndpointName
    reference_endpoint: EndpointName
    target_joint_convention: Optional[JointConvention] = None
    reference_joint_convention: Optional[JointConvention] = None
    align_root: bool = False
    align_scale: bool = False
    align_rigid: bool = False
    align_procrustes: bool = False
    max_parity_diff_mm: float = 1e-3
