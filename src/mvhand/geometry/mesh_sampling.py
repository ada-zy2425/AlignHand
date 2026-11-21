"""Deterministic coarse-mesh operators (195 sampled vertices) used by pixel-aligned refinement."""
import numpy as np


def coarse_mesh_operators(template_vertices, faces, count=195):
    """Farthest-point vertices and nonnegative three-neighbour interpolation.

    This function accepts a model template, never a data-population statistic.
    The saved matrices contain weights/indices, not template coordinates.
    Nonnegative interpolation preserves the componentwise displacement bound.
    """
    points=np.asarray(template_vertices,dtype=np.float64)
    faces=np.asarray(faces,dtype=np.int64)
    n=len(points)
    if points.shape!=(n,3) or not np.isfinite(points).all() or not 3<=count<=n:
        raise ValueError('invalid template or coarse count')
    if faces.ndim!=2 or faces.shape[1]!=3 or faces.min()<0 or faces.max()>=n:
        raise ValueError('invalid template topology')
    selected=[0];distance=np.full(n,np.inf)
    for _ in range(1,count):
        distance=np.minimum(distance,((points-points[selected[-1]])**2).sum(1))
        distance[selected]=-1
        selected.append(int(distance.argmax()))
    selected=np.asarray(selected,dtype=np.int64)
    down=np.eye(n,dtype=np.float32)[selected]
    squared=((points[:,None]-points[selected][None])**2).sum(-1)
    neighbors=np.argsort(squared,axis=1,kind='stable')[:,:3]
    weights=1/np.maximum(np.sqrt(np.take_along_axis(squared,neighbors,1)),1e-12)
    weights/=weights.sum(1,keepdims=True)
    up=np.zeros((n,count),dtype=np.float32)
    np.put_along_axis(up,neighbors,weights.astype(np.float32),axis=1)
    up[selected]=np.eye(count,dtype=np.float32)
    adjacency=np.zeros((n,n),dtype=np.float32)
    for a,b in ((0,1),(1,2),(2,0)):
        adjacency[faces[:,a],faces[:,b]]=1;adjacency[faces[:,b],faces[:,a]]=1
    degree=adjacency.sum(1)
    if (degree==0).any():raise ValueError('isolated template vertex')
    laplacian=np.eye(n,dtype=np.float32)-adjacency/degree[:,None]
    return dict(downsample=down,upsample=up,coarse_indices=selected,
                laplacian=laplacian,faces=faces)
