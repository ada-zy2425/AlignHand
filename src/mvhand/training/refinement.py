"""Ref supervision, kept separate from the target-free refinement forward."""
import torch
from mvhand.regional.candidates import metric_joints,project_points


def refinement_loss(result,targets,*,regressor,projection_matrices,raw_to_crop,
                    supervision_mask,view_mask,laplacian):
    """Per-example MPVPE + .5 MPJPE + .1 2D + .01 residual Laplacian.

    3D distances and residual smoothing are millimetres; reprojection is pixels
    in the same 256-square crop coordinates as the Stage I loss. This function
    is used after prediction and must not supply anything to the forward API.
    """
    vertices=result.vertices
    truth=targets['vertices_reference_m'].to(vertices)
    joints=metric_joints(vertices,regressor)
    target_joints=metric_joints(truth,regressor)
    vertex=torch.linalg.vector_norm(vertices-truth,dim=-1).mean(-1)*1000
    joint=torch.linalg.vector_norm(joints-target_joints,dim=-1).mean(-1)*1000
    raw,depth=project_points(projection_matrices,joints[:,None]);raw=raw[:,0];depth=depth[:,0]
    target_raw=targets['joints_2d_raw'].to(raw)
    def crop(x):
        return torch.einsum('bvij,bvqj->bvqi',raw_to_crop.double(),torch.cat((x,torch.ones_like(x[...,:1])),-1))
    crop_target=crop(target_raw);crop_prediction=crop(raw)
    valid=supervision_mask.bool() & view_mask[...,None] & torch.isfinite(crop_target).all(-1)
    valid &= ((crop_target>=0)&(crop_target<=256)).all(-1)
    if not valid.any((1,2)).all():raise ValueError('Ref training group has no supported 2D supervision')
    # Invalid model depth is not removed from supervision; it remains a failure.
    if not ((depth>0)|~valid).all():raise FloatingPointError('Ref projected joints behind a supervised camera')
    difference=torch.where(valid[...,None],crop_prediction-crop_target,0)
    pixel=torch.linalg.vector_norm(difference,dim=-1).sum((1,2))/valid.sum((1,2))
    smooth=torch.linalg.vector_norm(torch.einsum('mn,bnc->bmc',laplacian.to(vertices),result.displacement),dim=-1).mean(-1)*1000
    total=vertex+.5*joint+.1*pixel+.01*smooth
    return total,dict(MPVPE_mm=vertex,mesh_MPJPE_mm=joint,reprojection_crop_px=pixel,laplacian_displacement_mm=smooth)
