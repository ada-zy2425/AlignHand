"""Geometry feature standardization shared by all datasets."""
import torch
from torch import Tensor

def safe_standardize(x:Tensor,mean:Tensor,scale:Tensor)->Tensor:
 safe=torch.where(scale>0,scale,torch.ones_like(scale));value=(x-mean)/safe
 return torch.where(scale>0,value,torch.zeros_like(value))
