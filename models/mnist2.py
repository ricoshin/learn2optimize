import itertools
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from models.model_helpers import ParamsFlattener
from models.meprop import SBLinear, UnifiedSBLinear
from torch.utils import data
from torchvision import datasets
from utils import utils

C = utils.getCudaManager('default')


class MNISTModel2(nn.Module):
  def __init__(self, input_sz=28 * 28, hidden_sz=500, n_layers=3,
    n_classes=10, params=None, type='cnn'):
    super().__init__()
    assert type in ['linear', 'cnn']
    self.input_sz = input_sz
    self.hidden_sz = hidden_sz
    self.n_layers = n_layers
    self.n_classes = n_classes
    self.type = type
    self.layers = nn.ModuleList()
    if self.type == 'linear':
      for i in range(n_layers + 1):
        input_sz = self.input_sz if i == 0 else self.hidden_sz
        output_sz = self.n_classes if i == self.n_layers else self.hidden_sz
        self.layers.append(nn.Linear(input_sz, output_sz, bias=True))
        if not i == n_layers:
          self.layers.append(nn.ReLU())
    elif self.type == 'cnn':
      n_channel = 32
      # 1, 32, 64, 128, ... , 10
      for i in range(n_layers + 1):
        in_ = n_channel * i if i > 0 else 1
        out_ = n_channel * (i + 1) if i < n_layers else 10
        self.layers.append(nn.Conv2d(in_, out_, 3, 1, 1))
        if i % 2 == 0 and not i == n_layers:
          self.layers.append(nn.MaxPool2d(2, 2))
        if not i == n_layers:
          self.layers.append(nn.ReLU(True))
        else:
          self.layers.append(nn.AvgPool2d(6, 6))
    # self.reset_variables()
    self.layers = self.layers.cuda()
    if params is not None:
      if not isinstance(params, ParamsFlattener):
        raise TypeError("params argumennts has to be "
                        "an instance of ParamsFlattener!")
      self.params = params
      self.params2layers(self.params, self.layers)
    else:
      self.params = self.layers2params(self.layers)

    self.sb_linear = {}
    self._activations = {}
    self.nonlinear = nn.Sigmoid()
    self.loss = nn.NLLLoss()


  def init_parameters(self, mat, bias, type='simple'):
    assert type in ['kaiming_uniform', 'simple']
    if type == 'kaiming_uniform':
      nn.init.kaiming_uniform_(mat, a=math.sqrt(5))
      fan_in, _ = nn.init._calculate_fan_in_and_fan_out(mat)
      bound = 1 / math.sqrt(fan_in)
      nn.init.uniform_(bias, -bound, bound)
    elif type == 'simple':
      mat.normal_(0., 0.001)
      bias.fill_(0.)
    else:
      raise Exception(f'Unknown type: {type}')


  def params2layers(self, params, layers):
    j = 0
    for i, layer in enumerate(layers):
      if layer.__class__.__name__ in ['Linear', 'Conv2d']:
        """FIX LATER:
        1. there is no need to use diffrent names for the same matrices.
        2. t() operation is unneccessary."""
        layer.weight.requires_grad_(False).copy_(
          params.unflat['mat_' + str(j)])
        if layer.bias is not None:
          layer.bias.requires_grad_(False).copy_(
            params.unflat['bias_' + str(j)])
        j += 1
    return layers

  def layers2params(self, layers):
    j = 0
    params = {}
    for i, layer in enumerate(layers):
      if layer.__class__.__name__ in ['Linear', 'Conv2d']:
        """FIX LATER: there is no need to use diffrent names for the same
        matrices. (weight, mat)"""
        params['mat_' + str(j)] = layer.weight.data.clone()
        if layer.bias is not None:
          params['bias_' + str(j)] = layer.bias.data.clone()
        j += 1
    return ParamsFlattener(params)


  def forward(self, inp, out=None):
    inp = C(inp)
    if self.type == 'linear':
      inp = inp.view(inp.size(0), self.input_sz)

    #import pdb; pdb.set_trace()
    """
    activation_index = 0
    for layer_index in range(self.n_layers):
      inp = self.layer[layer_index](inp)
      if isinstance(self.layer[layer_index], nn.Linear) or isinstance(self.layer[layer_index], nn.Conv2d):
        self._activations[f'layer_{activation_index}'] = inp
        activation_index += 1
    """
    #import pdb; pdb.set_trace()
    """
    for layer_index in range(self.n_layers):
      if layer_index == self.n_layers - 1:
        inp = inp.view(inp.size(0), -1)
        #import pdb; pdb.set_trace()
      #print(layer_index, inp.size())
      inp = self.layer[layer_index](inp)
    """
    for layer in self.layers:
      inp = layer(inp)
    inp = F.log_softmax(inp, dim=1).squeeze()
    if out is not None:
      out = C(out)
      loss = self.loss(inp, out)
      acc = (inp.argmax(dim=1) == out).float().mean()
    else:
      loss = None
      acc = None
    return loss, acc


if __name__ == "__main__":
  model = MNISTModel2()
  print(model)
  import pdb; pdb.set_trace()
