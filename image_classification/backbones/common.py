import re
import json

import torch
import torch.nn as nn

from .mobilenetv2 import *
from .mobilenetv3 import *
from .resnet import *
from .resnext import *
from .vision_transformer import *
from .swin_mlp import *
from .swin_transformer_v2 import *
from .torch_reference import *

from utils import load_pretrained_weight

class Network(nn.Module):
    def __init__(self, arch, num_classes, im_size=None, pth_file=None, **kwargs):
        arch_dict = {
            'mobilenetv2': mobilenetv2,
            'mobilenetv3_small': mobilenetv3_small,
            'mobilenetv3_large': mobilenetv3_large,
            'resnet18': resnet18,
            'resnet34': resnet34,
            'resnet50': resnet50,
            'resnet101': resnet101,
            'resnet152': resnet152,
            'resnext18': resnext18,
            'resnext34': resnext34,
            'resnext50': resnext50,
            'resnext101': resnext101,
            'resnext152': resnext152,
            'vit_b_16': vit_b_16,
            'vit_b_32': vit_b_32,
            'vit_l_16': vit_l_16,
            'vit_l_32': vit_l_32,
            'vit_h_14': vit_h_14,
            'swin_mlp': SwinMLP,
            'swin_transformer_v2': SwinTransformerV2,
            'torchref_mobilenet_v2': torchref_mobilenet_v2,
            'torchref_mobilenet_v3_small': torchref_mobilenet_v3_small,
            'torchref_mobilenet_v3_large': torchref_mobilenet_v3_large,
            'torchref_resnext50_32x4d': torchref_resnext50_32x4d,
            'torchref_resnext101_32x8d': torchref_resnext101_32x8d,
            'torchref_resnext101_64x4d': torchref_resnext101_64x4d,
        }
        super(Network, self).__init__()
        # ========== backbone =================
        if arch in arch_dict.keys():
            if arch in ['swin_mlp', 'swin_transformer_v2']:
                win_base = 32
                assert im_size is not None and im_size % win_base == 0
                self.net = arch_dict[arch](num_classes=num_classes, img_size=im_size, window_size=im_size//win_base, **kwargs)
            elif arch in ['vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32', 'vit_h_14']:
                assert im_size is not None
                self.net = arch_dict[arch](num_classes=num_classes, image_size=im_size, **kwargs)
            else:
                self.net = arch_dict[arch](num_classes=num_classes, **kwargs)
        else:
            raise Exception("Undefined Backbone Type!")

        if pth_file is not None:
            load_pretrained_weight(self.net, pth_file)

    def forward(self, x):
        return self.net(x)
