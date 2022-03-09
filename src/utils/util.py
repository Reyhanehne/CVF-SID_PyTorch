import json
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import numpy as np
import torch
from time import time
import re

import numpy.core.numeric as NX
from numpy.core import (atleast_2d, hstack, dot, array, ones)
from numpy.core import overrides
from numpy.lib.twodim_base import diag
from numpy.linalg import eigvals

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def apply_funcs(tensor, option):
    numbers = re.findall(r'\d+', option)
    scale0 = int(numbers[0]) if option[0].isdecimal() else 1
    if option.find("tanh")>=0:
        xscale = int(numbers[1]) if option[-1].isdecimal() else 1
        return scale0 * torch.tanh(tensor/xscale)
    elif option.find("relu")>=0:
        upperbound = int(numbers[-1]) if option[-1].isdecimal() else float('inf')
        return scale0 * torch.min(torch.max(tensor,torch.zeros(1,device='cuda')), upperbound*torch.ones(1,device='cuda'))
    elif option.find("sig")>=0:
        offset = int(numbers[-1]) if option[-1].isdecimal() else 0
        return scale0 * torch.sigmoid(tensor - offset)
    elif option.find("None")>=0:
        return tensor
    else:
        raise
            
def apply_funcs_diff(tensor, tensor_diff, option):
    """
    tensor:      (PI_funcs) - (batch, num_points, num_functions)
    tensor_diff: (PI_diff)  - (batch, num_points, num_functions, 3)
    """
    numbers = re.findall(r'\d+', option)
    scale0 = int(numbers[0]) if option[0].isdecimal() else 1
    if option.find("tanh")>=0:
        xscale = int(numbers[1]) if option[-1].isdecimal() else 1
        tanh_diff = (torch.ones(1,device='cuda') - torch.tanh(tensor)**2) # tanh -'> (1-tanh^2)
        return scale0 * tensor_diff * tanh_diff.unsqueeze(dim=3)
    elif option.find("relu")>=0: # not differentiable but approximatly...
        upperbound = int(numbers[-1]) if option[-1].isdecimal() else float('inf')
        relu_diff = ((tensor>=0)*(tensor<=upperbound)).float()
        return scale0 * tensor_diff * relu_diff.unsqueeze(dim=3)
    elif option.find("sig")>=0:
        offset = int(numbers[-1]) if option[-1].isdecimal() else 0
        sig_diff = torch.sigmoid(tensor - offset) * (1 - torch.sigmoid(tensor - offset))
        return scale0 * tensor_diff * sig_diff.unsqueeze(dim=3)
    elif option.find("None")>=0:
        return tensor_diff # * torch.ones_like(tensor, device='cuda').unsqueeze(dim=3)
    else:
        raise

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()
        
    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
            # self.writer.close()
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]
    
    def result(self):
        return dict(self._data.average)
