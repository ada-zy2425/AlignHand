"""Zero-initialized, pixel-aligned multi-view residual refinement (Ref trial)."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from mvhand.geometry.mesh_depth import mesh_depth_at_pixels
from mvhand.geometry.plucker import reference_plucker_features


@dataclass
class RefinementOutput:
    vertices: torch.Tensor
    displacement: torch.Tensor
    coarse_displacement: torch.Tensor
    view_attention: torch.Tensor
    sampled_view_valid: torch.Tensor


class MultiViewRefinement(nn.Module):
    """No dataset/physical-camera IDs and no supervision in the forward API."""
    def __init__(self,operators,*,feature_dim=1280,dim=128,layers=2,
                 displacement_bound_m=.008,include_correction_magnitude=True):
        super().__init__()
        if displacement_bound_m not in (.005,.008) or layers not in (2,4):
            raise ValueError('unregistered Ref capacity/bound')
        down=torch.as_tensor(operators['downsample']).float()
        up=torch.as_tensor(operators['upsample']).float()
        if down.ndim!=2 or up.shape!=down.T.shape or (up<0).any():raise ValueError('invalid mesh operators')
        if not torch.allclose(up.sum(1),torch.ones(up.shape[0]),atol=1e-6,rtol=0):raise ValueError('upsampling must be convex')
        self.register_buffer('downsample',down);self.register_buffer('upsample',up)
        self.register_buffer('faces',torch.as_tensor(operators['faces']).long())
        self.register_buffer('laplacian',torch.as_tensor(operators['laplacian']).float())
        self.bound=displacement_bound_m;self.include_correction=include_correction_magnitude
        self.feature_projection=nn.Linear(feature_dim,dim)
        # Plucker ray (6), depth difference (1), hit availability (1), optional
        # Stage I observation correction magnitude in crop pixels (1).
        self.geometry_projection=nn.Linear(8+int(include_correction_magnitude),dim)
        self.view_score=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim),nn.GELU(),nn.Linear(dim,1))
        layer=nn.TransformerEncoderLayer(dim,4,dim*4,dropout=0.,batch_first=True,norm_first=True)
        self.vertices_attention=nn.TransformerEncoder(layer,layers,enable_nested_tensor=False)
        self.offset=nn.Linear(dim,3)
        nn.init.zeros_(self.offset.weight);nn.init.zeros_(self.offset.bias)

    def forward(self,vertices,feature_maps,intrinsics,reference_to_camera,raw_to_crop,
                view_mask,support_weights,scaffold_joints,placed_prior_vertices,
                *,feature_crop_bounds=(32,0,192,256),correction_magnitude=None):
        b,views=view_mask.shape;count=self.downsample.shape[0]
        if vertices.shape!=(b,self.downsample.shape[1],3) or support_weights.shape!=(b,views,21):
            raise ValueError('Ref mesh/support shape mismatch')
        if not view_mask.any(1).all():raise ValueError('Ref cannot create an output for an all-missing group')
        if self.include_correction and (correction_magnitude is None or correction_magnitude.shape!=(b,views,21)):
            raise ValueError('D3-a Ref requires the observation-correction magnitude')
        with torch.no_grad():
            intrinsics=torch.where(view_mask[...,None,None],intrinsics,torch.eye(3,device=vertices.device,dtype=intrinsics.dtype))
            reference_to_camera=torch.where(view_mask[...,None,None],reference_to_camera,torch.eye(4,device=vertices.device,dtype=reference_to_camera.dtype))
            raw_to_crop=torch.where(view_mask[...,None,None],raw_to_crop,torch.zeros_like(raw_to_crop))
            placed_prior_vertices=torch.where(view_mask[...,None,None],placed_prior_vertices,torch.zeros_like(placed_prior_vertices))
            coarse=torch.einsum('qm,bmc->bqc',self.downsample,vertices.float())
            rotation=reference_to_camera[...,:3,:3].float();translation=reference_to_camera[...,:3,3].float()
            camera=torch.einsum('bvij,bqj->bvqi',rotation,coarse)+translation[:,:,None]
            projected=torch.einsum('bvij,bvqj->bvqi',intrinsics.float(),camera)
            pixels=projected[...,:2]/projected[...,2:].clamp_min(1e-8)
            homogeneous=torch.cat((pixels,torch.ones_like(pixels[...,:1])),-1)
            crop=torch.einsum('bvij,bvqj->bvqi',raw_to_crop.float(),homogeneous)
            left,top,width,height=feature_crop_bounds
            grid=2*(crop-crop.new_tensor((left,top)))/crop.new_tensor((width,height))-1
            usable=view_mask[:,:,None] & (camera[...,2]>0) & torch.isfinite(grid).all(-1) & (grid.abs()<=1).all(-1)
            grid=torch.where(usable[...,None],grid,torch.zeros_like(grid))
            prior_camera=torch.einsum('bvij,bvmj->bvmi',rotation,placed_prior_vertices.float())+translation[:,:,None]
            depth,hit=mesh_depth_at_pixels(prior_camera.flatten(0,1),self.faces,pixels.flatten(0,1),intrinsics.float().flatten(0,1))
            depth=depth.reshape(b,views,count);hit=hit.reshape(b,views,count) & usable
            depth_difference=torch.where(hit,camera[...,2]-depth,0)/.01
            direction,moment=reference_plucker_features(pixels,intrinsics.float(),reference_to_camera.float())
            geometry=[direction,moment,depth_difference[...,None],hit.float()[...,None]]
            nearest=torch.cdist(coarse,scaffold_joints.float()).argmin(-1)
            nearest=nearest[:,None].expand(-1,views,-1)
            support=support_weights.float().gather(-1,nearest)
            if self.include_correction:geometry.append(correction_magnitude.float().gather(-1,nearest)[...,None]/256.)
            geometry=torch.cat(geometry,-1)
            geometry=torch.where(usable[...,None],torch.nan_to_num(geometry),0)
        maps=torch.where(view_mask[:,:,None,None,None],feature_maps.float(),0)
        sampled=F.grid_sample(maps.flatten(0,1),grid.flatten(0,1)[:,:,None],align_corners=False,padding_mode='zeros')
        sampled=sampled[:,:,:,0].transpose(1,2).reshape(b,views,count,-1)
        tokens=self.feature_projection(sampled)+self.geometry_projection(geometry)
        scores=self.view_score(tokens).squeeze(-1)+support.clamp_min(1e-8).log()
        # Vertex-specific all-missing views get zero evidence, never NaN weights.
        scores=scores.masked_fill(~usable,-torch.finfo(scores.dtype).max)
        attention=scores.softmax(1)*usable
        attention=attention/attention.sum(1,keepdim=True).clamp_min(1e-8)
        aggregated=(tokens*attention[...,None]).sum(1)
        refined=self.vertices_attention(aggregated)
        delta=self.bound*self.offset(refined).tanh()
        displacement=torch.einsum('mq,bqc->bmc',self.upsample,delta).to(vertices.dtype)
        return RefinementOutput(vertices+displacement,displacement,delta,attention,usable)

    def laplacian_penalty(self,displacement):
        return torch.linalg.vector_norm(torch.einsum('mn,bnc->bmc',self.laplacian,displacement.float()),dim=-1).mean()
