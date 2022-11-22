from typing import Tuple, Optional
import time
import itertools
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn

from src.icosahedron import Icosahedron


class FeatureSphereLibrary(nn.Module):
    def __init__(self,
                 n_objects: int,
                 input_dim: int,
                 output_dim: int,
                 num_vertices: int=12):
        super().__init__()
        self.weight = nn.Parameter(data= torch.zeros((n_objects, num_vertices, input_dim), dtype=torch.float32),
                                   requires_grad= True)
        self.bias = nn.Parameter(data= torch.zeros((n_objects, output_dim), dtype=torch.float32),
                                   requires_grad= True)

        torch.nn.init.xavier_uniform_(self.weight.data)
        torch.nn.init.xavier_uniform_(self.bias.data)

    def get_spheres(self, obj_ids: Tensor):
        weight = torch.index_select(self.weight, 0, obj_ids.squeeze())
        bias = torch.index_select(self.bias, 0, obj_ids.squeeze())
        return weight, bias


class DynamicIcoConv(nn.Module):
    def __init__(self, ico_level: int) -> None:
        super().__init__()
        self.icosahedron = Icosahedron(ico_level)
        self.num_vertices = self.icosahedron.num_vertices
        self.order = 60

        rotmats = torch.tensor(self.icosahedron.rotation_matrices, dtype=torch.float32)
        self.rotmats = nn.Parameter(rotmats, requires_grad=False)

        half_quotient_reps = torch.tensor(self.icosahedron.quotient_reps[:,:,self.icosahedron.halfmesh_mask],
                                          dtype=torch.float32)
        self.half_quotient_reps = nn.Parameter(half_quotient_reps, requires_grad=False)

    def forward(self, x: Tensor, weight: Tensor, bias: Optional[Tensor]=None):
        '''Correlation between two meshes, where weight is half a mesh
        x = input tensor of shape (B, V, N)
        weight = filter tensor of shape (B, V//2, N, M)
        bias = bias tensor of shape (B, G, M)
        '''
        shifted_x = torch.matmul(torch.transpose(self.half_quotient_reps, 1, 2),
                                 x.unsqueeze(1),
                                 )
        out = ( weight.unsqueeze(1) * shifted_x.unsqueeze(-1) ).sum((-2,-3))

        if bias is not None:
            out = out + bias.unsqueeze(-2)

        return out
