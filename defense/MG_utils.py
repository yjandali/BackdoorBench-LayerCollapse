import copy
import time
import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from torchvision.datasets import *
from torchvision.transforms import *
import torchvision.models as models
import timm

import numpy as np

# # import Subset function of torchvision
# from torchvision.datasets.utils import 
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import torch.multiprocessing as mp

from ..defense.collapsible_MLP import CollapsibleMlp

import LiveTune as lt


MLPBLOCK_INSTANCE = models.vision_transformer.MLPBlock



def get_sparsity(tensor: torch.Tensor) -> float:
    """
    calculate the sparsity of the given tensor
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    """
    return 1 - float(tensor.count_nonzero()) / tensor.numel()


def get_model_sparsity(model: nn.Module) -> float:
    """
    calculate the sparsity of the given model
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    """
    num_nonzeros, num_elements = 0, 0
    for param in model.parameters():
        num_nonzeros += param.count_nonzero()
        num_elements += param.numel()
    return 1 - float(num_nonzeros) / num_elements


def get_num_parameters(model: nn.Module, count_nonzero_only=False) -> int:
    """
    calculate the total number of parameters of model
    :param count_nonzero_only: only count nonzero weights
    """
    num_counted_elements = 0
    for param in model.parameters():
        if count_nonzero_only:
            num_counted_elements += param.count_nonzero()
        else:
            num_counted_elements += param.numel()
    return num_counted_elements


def get_model_size(model: nn.Module, data_width=32, count_nonzero_only=False) -> int:
    """
    calculate the model size in bits
    :param data_width: #bits per element
    :param count_nonzero_only: only count nonzero weights
    """
    return get_num_parameters(model, count_nonzero_only) * data_width


def get_model_linear_loss(model, fraction=1.0):
    linear_loss = 0
    num_mlp_layers = len(list(model.named_modules()))
    for name, module in list(model.named_modules())[::-1][:int(num_mlp_layers * fraction)]:
        if isinstance(module, CollapsibleMlp):
            linear_loss += module.linear_loss()
    return linear_loss

def get_model_collapsible_slopes(model, fraction=1.0):
    num_mlp_layers = len(list(model.named_modules()))
    for name, module in list(model.named_modules())[::-1][:int(num_mlp_layers * fraction)]:
        if isinstance(module, CollapsibleMlp):
            if isinstance(module.act, nn.Identity):
                print(name, 1)
            else:
                print(name, module.act.weight.item())

def collapse_model(model, fraction=1.0, threshold=0.05, device=None):
    num_mlp_layers = len(list(model.named_modules()))
    for name, module in list(model.named_modules())[::-1][:int(num_mlp_layers * fraction)]:
        if isinstance(module, CollapsibleMlp):
            print("Collapsing layer {}".format(name))
            module.collapse(threshold=threshold)
            

def change_module(model, name, module):
    name_list = name.split(".")
    if len(name_list) == 1:
        model._modules[name_list[0]] = module
    else:
        change_module(model._modules[name_list[0]], ".".join(name_list[1:]), module)


def get_collapsible_model(model, fraction=1.0, device=None):

    num_mlp_layers = len(list(model.named_modules()))
    copy_model = copy.deepcopy(model).to(device)
    for name, module in list(copy_model.named_modules())[::-1][:int(num_mlp_layers * fraction)]:
        if isinstance(module, MLPBLOCK_INSTANCE):
            print("Collapsing layer {}".format(name))
            in_features = module.linear_1.in_features
            hidden_features = module.linear_2.in_features
            out_features = module.linear_2.out_features
            bias = module.linear_2.bias
            collapsibleMLP = CollapsibleMlp(in_features=in_features, hidden_features=hidden_features, out_features=out_features, batch_norm=False, bias=bias, drop=0)
            collapsibleMLP.load_from_Mlp(module)
            if device is not None:
                collapsibleMLP.to(device)
            change_module(copy_model, name, collapsibleMLP)
    return copy_model.to(device)


def criterion_function(output, target, model=None, reg_strength=0.1,fraction=1.0):
    # get_model_collapsible_slopes(model, fraction=fraction)
    return nn.CrossEntropyLoss()(output, target) + get_model_linear_loss(model, fraction=fraction) * reg_strength
