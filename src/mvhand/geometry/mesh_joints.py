"""MANO joint regressors and the 21-joint layouts used for evaluation (POEM and DexYCB orders)."""
from typing import Sequence, Tuple

import torch
from torch import Tensor


MANO_VERTEX_COUNT = 778
MANO_BASE_JOINT_COUNT = 16
MANO_AUGMENTED_JOINT_COUNT = 21

MANO_FINGERTIP_VERTEX_IDS: Tuple[int, ...] = (
    745,
    317,
    444,
    556,
    673,
)

MANO_21_TO_DEXYCB_OPENPOSE_21: Tuple[int, ...] = (
    0, 13, 14, 15, 16,
    1, 2, 3, 17,
    4, 5, 6, 18,
    10, 11, 12, 19,
    7, 8, 9, 20,
)

POEM_PUBLISHED_FINGERTIP_VERTEX_IDS: Tuple[int, ...] = (
    744,
    320,
    443,
    555,
    672,
)

POEM_PUBLISHED_MANO_21_REORDER: Tuple[int, ...] = (
    0, 13, 14, 15, 16,
    1, 2, 3, 17,
    4, 5, 6, 18,
    10, 11, 12, 19,
    7, 8, 9, 20,
)


def validate_mano_base_regressor(
    regressor: Tensor,
) -> None:
    if not isinstance(regressor, Tensor):
        raise TypeError("regressor must be a torch.Tensor")
    if tuple(regressor.shape) != (MANO_BASE_JOINT_COUNT, MANO_VERTEX_COUNT):
        raise ValueError("regressor must have shape (16, 778)")
    if not regressor.is_floating_point():
        raise TypeError("regressor must be floating point")
    if not torch.isfinite(regressor).all().item():
        raise ValueError("regressor must contain only finite values")


def build_mano_21_regressor(
    base_regressor: Tensor,
) -> Tensor:
    validate_mano_base_regressor(base_regressor)

    fingertip_rows = base_regressor.new_zeros(
        (len(MANO_FINGERTIP_VERTEX_IDS), MANO_VERTEX_COUNT)
    )
    for row_index, vertex_index in enumerate(MANO_FINGERTIP_VERTEX_IDS):
        fingertip_rows[row_index, vertex_index] = 1.0

    regressor = torch.cat((base_regressor, fingertip_rows), dim=0)
    if tuple(regressor.shape) != (MANO_AUGMENTED_JOINT_COUNT, MANO_VERTEX_COUNT):
        raise ValueError("augmented regressor must have shape (21, 778)")
    if not torch.isfinite(regressor).all().item():
        raise ValueError("augmented regressor must contain only finite values")
    return regressor


def build_poem_published_21_regressor(
    base_regressor: Tensor,
) -> Tensor:
    """Build the separately named POEM published mesh-joint regressor.

    This convention uses POEM's verified fingertip vertices and does not alter
    the existing generic MANO-21 augmentation contract.
    """
    validate_mano_base_regressor(base_regressor)
    fingertip_rows = base_regressor.new_zeros(
        (len(POEM_PUBLISHED_FINGERTIP_VERTEX_IDS), MANO_VERTEX_COUNT)
    )
    for row_index, vertex_index in enumerate(POEM_PUBLISHED_FINGERTIP_VERTEX_IDS):
        fingertip_rows[row_index, vertex_index] = 1.0
    return torch.cat((base_regressor, fingertip_rows), dim=0)


def validate_joint_mapping(
    mapping: Sequence[int],
    source_joint_count: int,
) -> Tuple[int, ...]:
    if type(source_joint_count) is not int:
        raise TypeError("source_joint_count must be a Python integer")
    if source_joint_count <= 0:
        raise ValueError("source_joint_count must be positive")
    if not isinstance(mapping, Sequence):
        raise TypeError("mapping must be a sequence")

    validated_mapping = tuple(mapping)
    if len(validated_mapping) != source_joint_count:
        raise ValueError("mapping length must equal source_joint_count")
    if any(type(index) is not int for index in validated_mapping):
        raise TypeError("mapping values must be Python integers")
    if sorted(validated_mapping) != list(range(source_joint_count)):
        raise ValueError("mapping must be a bijection over the source joint range")
    return tuple(mapping)


def regress_mesh_joints(
    vertices: Tensor,
    joint_regressor: Tensor,
) -> Tensor:
    if not isinstance(vertices, Tensor):
        raise TypeError("vertices must be a torch.Tensor")
    if not isinstance(joint_regressor, Tensor):
        raise TypeError("joint_regressor must be a torch.Tensor")
    if vertices.ndim < 2 or tuple(vertices.shape[-2:]) != (MANO_VERTEX_COUNT, 3):
        raise ValueError("vertices must end in shape (778, 3)")
    if joint_regressor.ndim != 2 or joint_regressor.shape[1] != MANO_VERTEX_COUNT:
        raise ValueError("joint_regressor must have shape (J, 778)")
    if not vertices.is_floating_point():
        raise TypeError("vertices must be floating point")
    if not joint_regressor.is_floating_point():
        raise TypeError("joint_regressor must be floating point")
    if not torch.isfinite(vertices).all().item():
        raise ValueError("vertices must contain only finite values")
    if not torch.isfinite(joint_regressor).all().item():
        raise ValueError("joint_regressor must contain only finite values")
    if vertices.dtype != joint_regressor.dtype:
        raise TypeError("vertices and joint_regressor must have the same dtype")
    if vertices.device != joint_regressor.device:
        raise ValueError("vertices and joint_regressor must be on the same device")

    return torch.einsum("jv,...vc->...jc", joint_regressor, vertices)


def apply_joint_mapping(
    joints: Tensor,
    mapping: Sequence[int],
) -> Tensor:
    if not isinstance(joints, Tensor):
        raise TypeError("joints must be a torch.Tensor")
    if joints.ndim < 2 or joints.shape[-1] != 3:
        raise ValueError("joints must end in shape (J, 3)")

    validated_mapping = validate_joint_mapping(mapping, joints.shape[-2])
    index = torch.tensor(
        validated_mapping,
        dtype=torch.long,
        device=joints.device,
    )
    return torch.index_select(joints, dim=-2, index=index)


def mano_21_to_dexycb_openpose_21(
    joints_mano_21: Tensor,
) -> Tensor:
    if not isinstance(joints_mano_21, Tensor):
        raise TypeError("joints_mano_21 must be a torch.Tensor")
    if joints_mano_21.ndim < 2 or tuple(joints_mano_21.shape[-2:]) != (
        MANO_AUGMENTED_JOINT_COUNT,
        3,
    ):
        raise ValueError("joints_mano_21 must end in shape (21, 3)")
    return apply_joint_mapping(
        joints_mano_21,
        MANO_21_TO_DEXYCB_OPENPOSE_21,
    )


def poem_published_mesh_joints_21(
    vertices: Tensor,
    mano_base_regressor: Tensor,
) -> Tensor:
    """Regress and directly gather POEM's published 21 mesh-derived joints."""
    augmented = build_poem_published_21_regressor(mano_base_regressor)
    joints = regress_mesh_joints(vertices, augmented)
    return apply_joint_mapping(joints, POEM_PUBLISHED_MANO_21_REORDER)


def poem_published_mesh_joints_21_ieee_fp32(
    vertices: Tensor,
    mano_base_regressor: Tensor,
) -> Tensor:
    """Compute POEM joints with the frozen explicit IEEE-FP32 contract."""
    if not isinstance(vertices, Tensor):
        raise TypeError("vertices must be a torch.Tensor")
    if vertices.ndim < 2 or tuple(vertices.shape[-2:]) != (MANO_VERTEX_COUNT, 3):
        raise ValueError("vertices must end in shape (778, 3)")
    if not vertices.is_floating_point() or not torch.isfinite(vertices).all().item():
        raise ValueError("vertices must be finite floating point")
    validate_mano_base_regressor(mano_base_regressor)
    if vertices.device != mano_base_regressor.device:
        raise ValueError("vertices and mano_base_regressor must share a device")

    vertices_fp32 = vertices.to(dtype=torch.float32)
    regressor_fp32 = mano_base_regressor.to(dtype=torch.float32)
    interior = (
        regressor_fp32[..., None] * vertices_fp32.unsqueeze(-3)
    ).sum(dim=-2, dtype=torch.float32)
    fingertip_index = torch.tensor(
        POEM_PUBLISHED_FINGERTIP_VERTEX_IDS,
        dtype=torch.long,
        device=vertices.device,
    )
    fingertips = torch.index_select(vertices_fp32, dim=-2, index=fingertip_index)
    augmented = torch.cat((interior, fingertips), dim=-2)
    return apply_joint_mapping(augmented, POEM_PUBLISHED_MANO_21_REORDER)
