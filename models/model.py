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


class Model(nn.Module):
  """Target model(optimizee) class."""
  def __init__(self, params=None, dataset='imagenet', type='linear'):
    super().__init__()
    assert dataset in ['mnist', 'imagenet']
    assert type in ['linear', 'conv']
    self.type = type
    if params is not None:
      if not isinstance(params, ParamsFlattener):
        raise TypeError("params argumennts has to be "
                        "an instance of ParamsFlattener!")
      self.params = params
      self.layers = C(self.stack_layers(type, dataset, params.channels()))
      self.params2layers(self.params, self.layers)
    else:
      self.layers = C(self.stack_layers(type, dataset))
      self.params = self.layers2params(self.layers)

    # self.reset_variables()
    # self.layers = self.layers.cuda()
    self.nonlinear = nn.Sigmoid()
    self.loss = nn.NLLLoss()

  def _adapt_dataset(self, dataset):
    if dataset == 'mnist':
      in_c = 1
      in_hw = 28
    elif dataset == 'imagenet':
      in_c = 3
      in_hw = 32
    else:
      raise Exception(f'Unknown dataset: {dataset}')
    return in_c, in_hw

  def stack_layers(self, type, dataset, channels=None):
    """Layer stacking function (just for MNIST).
    FIX LATER: full argument accessibility."""
    assert channels is None or isinstance(channels, (list, tuple))
    in_c, in_hw = self._adapt_dataset(dataset)
    if type == 'linear':
      if channels is None:
        channels = [500]
      layers = self._stack_linears(in_hw*in_hw*in_c, 10, channels)
    elif type == 'conv':
      if channels is None:
        channels = [32, 64, 128]
      layers = self._stack_convs(in_hw, in_c, 10, channels)
    else:
      raise Exception(f'Unknown type: {type}')
    return layers

  def _stack_linears(self, input_sz, output_sz, channels):
    """Function that returns automatically stacked linear layers.

    Args:
      input_sz (int): # of input features of the first layer.
      output_sz (int): # of output features of the last layer(n_classes).
      channels (list): a list of integers including # of features for
        intermediate hidden layers in sequence(len(channels) == n_layers).

    Returns:
      layers (ModuleList): PyTorch ModuleList instance.
    """
    assert isinstance(channels, (list, tuple))
    layers = nn.ModuleList()
    n_layers = len(channels) + 1
    for i in range(n_layers):
      in_sz = input_sz if i == 0 else channels[i - 1]
      out_sz = output_sz if i == n_layers - 1 else channels[i]
      layers.append(nn.Linear(in_sz, out_sz, bias=True))
      if not i == n_layers - 1:
        layers.append(nn.ReLU())
    return layers

  def _stack_convs(self, input_hw, input_ch, output_ch, channels, kernels=3,
    padding=True, downsize_after=1):
    """Function that returns automatically stacked conv layers.

    Args:
      input_sz (int): # of input channels of the first layer.
      output_sz (int): # output channels of the last layer(n_classes).
      channels (list): a list of integers including # of channels for
        intermediate hidden layers in sequence(len(channels) == n_layers).
      kernels (list or int): Kernel size for corresponding channels. If integer
        is passed, it will be broadcasted to agree with # of channels.
      padding (bool): If set true, pad zeros to preserve width and height.
      downsize_after (int): determine how many conv layers will come before
        subsampling, namely, max pooling with kernel size 2x2.

    Returns:
      layers (ModuleList): PyTorch ModuleList instance.

    Example:
      >> stack_convs(28, 1, 10, [32, 64, 128], 3, True, 2)
      ModuleList(
        (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace)
        (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ...)
        (3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): ReLU(inplace)
        (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ...)
        (6): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (7): ReLU(inplace)
        (8): AvgPool2d(kernel_size=7, stride=7, padding=0)
        (9): Conv2d(128, 10, kernel_size=(1, 1), stride=(1, 1))
      )
    """
    assert isinstance(channels, (list, tuple))
    assert isinstance(kernels, (list, tuple, int))
    if isinstance(kernels, int):
      kernels = [kernels] * len(channels)
    elif isinstance(kernels, (list, tuple)):
      assert len(channels) == len(kernels)
    # channels = [32, 64, 128, 256]
    hw_sz = input_hw
    ch = [input_ch] + channels + [output_ch]
    ker = kernels + [1]  # last conv layer with kernel size 1
    n_layers = len(channels) + 1
    layers = nn.ModuleList()
    for i in range(n_layers):
      layers.append(nn.Conv2d(
        in_channels=(ch[i]),
        out_channels=ch[i+1],
        kernel_size=ker[i],
        padding=(ker[i] // 2 if padding else 0),
        stride=1,
        bias=True,
        ))
      if not padding:
        hw_sz -= (ker[i] // 2) * 2
      if i < (n_layers - 1):
        layers.append(nn.ReLU(True))
      if i < (n_layers - 2) and (i + 1) % downsize_after == 0:
        layers.append(nn.MaxPool2d(2, 2))
        hw_sz //= 2
      elif i == (n_layers - 2):
        layers.append(nn.AvgPool2d(hw_sz, hw_sz))
      # do nothing after the last conv layer (i == n_layers)
    return layers

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
        there is no need to use diffrent names for the same matrices."""
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
        params['mat_' + str(j)] = C(layer.weight.data.clone())
        if layer.bias is not None:
          params['bias_' + str(j)] = C(layer.bias.data.clone())
        j += 1
    return ParamsFlattener(params)

  def forward(self, inp, out=None):
    inp = C(inp)
    if self.type == 'linear':
      inp = inp.view(inp.size(0), -1)

    for layer in self.layers:
      inp = layer(inp)
    inp = F.log_softmax(inp, dim=1).squeeze()

    if out is not None:
      out = C(out)
      import pdb; pdb.set_trace()
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
