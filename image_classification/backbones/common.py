import re
import json

import torch
import torch.nn as nn

from .mobilenetv2 import *
from .mobilenetv3 import *
from .resnet import *
from .resnext import *

class Network(nn.Module):
    def __init__(self, arch, pth_file=None, **kwargs):
        super(Network, self).__init__()
        # ========== backbone =================
        if arch == 'mobilenetv2':
            self.net = mobilenetv2(**kwargs)
        else:
            raise Exception("Undefined Backbone Type!")

    def forward(self, x):
        return self.net(x)
