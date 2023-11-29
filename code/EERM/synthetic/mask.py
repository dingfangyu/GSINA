import torch.nn as nn
from torch import Tensor

from torch_geometric.typing import OptTensor
from torch_geometric.nn.conv import MessagePassing

import torch_geometric as pyg

def set_masks(mask: Tensor, model: nn.Module, apply_sigmoid=True):
    for module in model.modules():
        if isinstance(module, MessagePassing):
            if pyg.__version__ < '2':
                module.__explain__ = True
                module.__edge_mask__ = mask
            else:
                module.explain  = True
                module._edge_mask = mask
                module._apply_sigmoid = apply_sigmoid

def clear_masks(model: nn.Module):
    for module in model.modules():
        if isinstance(module, MessagePassing):
            if pyg.__version__ < '2':
                module.__explain__ = False
                module.__edge_mask__ = None
            else:
                module.explain = False
                module._edge_mask = None
