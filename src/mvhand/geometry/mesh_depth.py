"""Query exact visible prior-mesh depth at projected vertices, without GT."""
import torch


@torch.no_grad()
def mesh_depth_at_pixels(vertices_camera, faces, pixels, intrinsics, *, chunk=32):
    """Nearest positive ray/triangle hit. Rays have z=1, hence t is depth.

    Inputs are flattened camera batches [N,M,3], [F,3], [N,Q,2], [N,3,3].
    Both triangle orientations are accepted; misses return depth zero + false.
    """
    if vertices_camera.ndim!=3 or pixels.ndim!=3 or vertices_camera.shape[0]!=pixels.shape[0]:
        raise ValueError('camera/query batch mismatch')
    triangles=vertices_camera[:,faces.long()]
    first=triangles[:,:,0];e1=triangles[:,:,1]-first;e2=triangles[:,:,2]-first
    source=-first
    qvec=torch.cross(source,e1,dim=-1)
    numerator=(e2*qvec).sum(-1)
    homogeneous=torch.cat((pixels,torch.ones_like(pixels[...,:1])),-1)
    rays=torch.einsum('nij,nqj->nqi',torch.linalg.inv(intrinsics),homogeneous)
    rays=rays/rays[...,2:].clamp_min(1e-12)
    depths=[];valid=[]
    for start in range(0,pixels.shape[1],chunk):
        direction=rays[:,start:start+chunk,None]
        pvec=torch.cross(direction.expand(-1,-1,e2.shape[1],-1),e2[:,None].expand(-1,direction.shape[1],-1,-1),dim=-1)
        determinant=(e1[:,None]*pvec).sum(-1)
        nonparallel=determinant.abs()>1e-10
        inverse=torch.where(nonparallel,determinant,torch.ones_like(determinant)).reciprocal()
        u=(source[:,None]*pvec).sum(-1)*inverse
        v=(direction*qvec[:,None]).sum(-1)*inverse
        t=numerator[:,None]*inverse
        hit=nonparallel & (u>=-1e-6) & (v>=-1e-6) & (u+v<=1+1e-6) & (t>1e-8) & torch.isfinite(t)
        nearest=t.masked_fill(~hit,float('inf')).amin(-1)
        available=torch.isfinite(nearest)
        depths.append(torch.where(available,nearest,torch.zeros_like(nearest)));valid.append(available)
    return torch.cat(depths,1),torch.cat(valid,1)
