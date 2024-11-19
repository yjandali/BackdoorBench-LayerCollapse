""" MLP module w/ dropout layer

Hacked together by / Copyright 2020 Ross Wightman
Changed by Zibakhsh Shabgahi to add the collapsible MLP
"""
from functools import partial

import torch
from torch import nn as nn

from itertools import repeat
import collections.abc

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)


class CollapsibleMlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            batch_norm=False,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = nn.PReLU(num_parameters=1, init=0.01)
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = nn.BatchNorm1d(hidden_features) if batch_norm else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])
        self.batch_norm = batch_norm


    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    
    def linear_loss(self):
        if isinstance(self.act, nn.Identity):
            return 0
        return (self.act.weight - 1)**2
    
    def collapse(self, threshold=0.05):
        if isinstance(self.act, nn.Identity):
            return
        if (self.act.weight - 1).abs() < threshold:
            if self.batch_norm:
                W1 = self.fc1.weight.data
                B1 = self.fc1.bias.data
                gamma = self.norm.weight.data
                beta = self.norm.bias.data
                mean = self.norm.running_mean
                var = self.norm.running_var
                eps = self.norm.eps
                W2 = self.fc2.weight.data
                B2 = self.fc2.bias.data

                new_W = W2 @ torch.diag(gamma / torch.sqrt(var + eps)) @ W1
                new_B = W2 @ (gamma * (B1 - mean) / torch.sqrt(var + eps) + beta) + B2

                self.fc1 = nn.Linear(self.fc1.in_features, self.fc2.out_features)
                self.fc1.weight.data = new_W
                self.fc1.bias.data = new_B
                self.fc2 = nn.Identity()
                self.norm = nn.Identity()
                self.act = nn.Identity()
                self.drop1 = nn.Identity()
            else:
                W1 = self.fc1.weight.data
                B1 = self.fc1.bias.data
                W2 = self.fc2.weight.data
                B2 = self.fc2.bias.data

                new_W = W2 @ W1
                new_B = W2 @ B1 + B2

                self.fc1 = nn.Linear(self.fc1.in_features, self.fc2.out_features)
                self.fc1.weight.data = new_W
                self.fc1.bias.data = new_B
                self.fc2 = nn.Identity()
                self.act = nn.Identity()
                self.drop1 = nn.Identity()
        else:
            print("Not collapsible")

    def load_from_Mlp(self, module):
        self.fc1.weight.data = module.linear_1.weight.data
        self.fc1.bias.data = module.linear_1.bias.data
        self.fc2.weight.data = module.linear_2.weight.data
        self.fc2.bias.data = module.linear_2.bias.data
    

class CollapsiblePreActBlock(nn.Module):
    """Pre-activation version of the BasicBlock."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(CollapsiblePreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.act1 = nn.ReLU()
        self.act2 = nn.PReLU(num_parameters=1, init=0.0)

        self.ind = None

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = self.act1(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(self.act2(self.bn2(out)))
        if self.ind is not None:
            out += shortcut[:, self.ind, :, :]
        else:
            out += shortcut
        return out
    
    def load_from_preactblock(self, module):
        self.bn1.weight.data = module.bn1.weight.data
        self.bn1.bias.data = module.bn1.bias.data
        self.bn1.running_mean = module.bn1.running_mean
        self.bn1.running_var = module.bn1.running_var
        self.conv1.weight.data = module.conv1.weight.data
        self.bn2.weight.data = module.bn2.weight.data
        self.bn2.bias.data = module.bn2.bias.data
        self.bn2.running_mean = module.bn2.running_mean
        self.bn2.running_var = module.bn2.running_var
        self.conv2.weight.data = module.conv2.weight.data

    def linear_loss(self):
        return (self.act2.weight - 1)**2