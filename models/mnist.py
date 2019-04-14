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


class DataloaderWrapper(object):
  """Just a simple wrapper for dataloader to load data whenever neccessary
  without forcibley using iterative loops.
  """
  def __init__(self, dataloader):
    self.dataloader = dataloader
    self.dataloader_iter = iter(dataloader)

  def __len__(self):
    return len(self.dataloader)

  def load(self):
    try:
      return next(self.dataloader_iter)
    except StopIteration:
      self.dataloader_iter = iter(self.dataloader)
      return next(self.dataloader_iter)
    except AttributeError:
      import pdb; pdb.set_trace()

  @property
  def batch_size(self):
    return self.dataloader.batch_size

  @property
  def full_size(self):
    return self.batch_size * len(self)


class MNISTData:
  """Current data scheme is as follows:
    - train data
      - outer-train data(70%): shuffled at every meta-iteration
        - inner-train data: for model objective w.r.t. theta. (50%)
        - inner-valid data: for meta-optimizer objective w.r.t phi. (50%)
      - outer-valid data(30%)
        - this will be used to determine when to do early-stopping
    - test data: held-out for meta-test
  """

  def __init__(self, outer_train_ratio=0.7, inner_train_ratio=0.5,
               batch_size=128):
    self.batch_size = batch_size
    self.outer_train_ratio = outer_train_ratio
    self.inner_train_ratio = inner_train_ratio

    train_data = datasets.MNIST('./mnist', train=True, download=True,
                                transform=torchvision.transforms.ToTensor())
    test_data = datasets.MNIST('./mnist', train=False, download=True,
                               transform=torchvision.transforms.ToTensor())
    self.train_data, self.valid_data = self._random_split(
        train_data, outer_train_ratio)
    self.test_data = test_data
    self.loaders = {}
    self.iters = {}
    self._initialize()

  # def __get__(self, instance, owner):
  #   return getattr(instance, 'loaders')

  def _initialize(self):
    attrs = ['train_data', 'valid_data', 'test_data']
    assert all(hasattr(self, attr) for attr in attrs)
    self.new_train_data()
    self.new_valid_data()
    self.new_test_data()

  def new_train_data(self):
    inner_train, inner_valid = self._random_split(
        self.train_data, self.inner_train_ratio)
    self.loaders['train'] = self._get_wrapped_dataloader(
      self.train_data, self.batch_size)
    self.loaders['inner_train'] = self._get_wrapped_dataloader(
      inner_train, self.batch_size)
    self.loaders['inner_valid'] = self._get_wrapped_dataloader(
      inner_valid, self.batch_size)

  def new_valid_data(self):
    self.loaders['valid'] = self._get_wrapped_dataloader(
      self.valid_data, self.batch_size)

  def new_test_data(self):
    self.loaders['test'] = self._get_wrapped_dataloader(
      self.test_data, self.batch_size)

  def _random_split(self, dataset, ratio):
    n_total = len(dataset)
    n_a = int(n_total * ratio)
    n_b = n_total - n_a
    data_a, data_b = data.random_split(dataset, [n_a, n_b])
    return data_a, data_b

  # def _fixed_split(self, dataset, ratio):
  #   import pdb; pdb.set_trace()
  #   data_a = dataset[int(len(dataset) * ratio):]
  #   data_b = dataset[:int(len(dataset) * ratio)]
  #   return data_a, data_b

  def _get_wrapped_dataloader(self, dataset, batch_size):
    sampler = data.sampler.RandomSampler(dataset, replacement=True)
    loader = data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return DataloaderWrapper(loader)

  def pseudo_sample(self):
    """return pseudo sample for checking activation size."""
    return (torch.zeros(1, 1, 28, 28), None)


class MNISTModel(nn.Module):
  def __init__(self, input_size=28 * 28, layer_size=500, n_layers=1,
               sb_mode='none', params=None):
    super().__init__()
    # Sadly this network needs to be implemented without using the convenient pytorch
    # abstractions such as nn.Linear, because afaik there is no way to load parameters
    # in those in a way that preserves gradients.
    self.input_size = input_size
    self.layer_size = layer_size
    self.n_layers = n_layers

    assert sb_mode in ['none', 'normal', 'unified']
    if sb_mode == 'normal':
      self.sb_linear_cls = SBLinear
    elif sb_mode == 'unified':
      self.sb_linear_cls = UnifiedSBLinear
    self.sb_mode = sb_mode

    if params is not None:
      if not isinstance(params, ParamsFlattener):
        raise TypeError("params argumennts has to be "
                        "an instance of ParamsFlattener!")
      self.params = params
    else:
      inp_size = 28 * 28
      params = {}
      for i in range(n_layers):
        params[f'mat_{i}'] = torch.randn(inp_size, layer_size) * 0.001
        # TODO: use a better initialization
        params[f'bias_{i}'] = torch.zeros(layer_size)
        self._reset_parameters(params[f'mat_{i}'], params[f'bias_{i}'])
        inp_size = layer_size

      params[f'mat_{n_layers}'] = torch.randn(
          inp_size, 10) * 0.001  # TODO:init
      params[f'bias_{n_layers}'] = torch.zeros(10)
      self._reset_parameters(
          params[f'mat_{n_layers}'], params[f'bias_{n_layers}'])
      self.params = ParamsFlattener(params)
      self.params.register_parameter_to(self)

    self.sb_linear = {}
    self._activations = {}
    self.nonlinear = nn.Sigmoid()
    self.loss = nn.NLLLoss()

  # def all_named_parameters(self):
  #   return [(k, v) for k, v in self.params.dict.items()]

  # def cuda(self):
  #   self.params = self.params.cuda()
  #   return self
  #
  # def parameters(self):
  #   for param in self.params.unflat.values():
  #     yield param

  def apply_backprop_mask(self, mask, drop_mode='no_drop'):
    if self.sb_linear is None:
      raise RuntimeError("To apply sparse backprop mak, model has to be "
                         "feed-forwared first having sparse mode normal or unified!")
    assert isinstance(mask, dict)
    assert drop_mode in ['no_drop', 'soft_drop', 'hard_drop']
    for k, v in self.sb_linear.items():
      #act_k = "_".join(['act', k.split('_')[1]])
      v.mask_backprop(mask[k], drop_mode)

  @property
  def activations(self):
    if len(self._activations) == 0:
      raise RuntimeError("model has to be feed-forwarded first!")
    else:
      return ParamsFlattener(self._activations)

  @property
  def activation_mean(self):
    mean = {k: v.mean(0) for k, v in self._activations.items()}
    return ParamsFlattener(mean)

  @property
  def activation_std(self):
    std = {k: v.std(0) for k, v in self._activations.items()}
    return ParamsFlattener(std)

  def _reset_parameters(self, mat, bias):
    nn.init.kaiming_uniform_(mat, a=math.sqrt(5))
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(mat)
    bound = 1 / math.sqrt(fan_in)
    nn.init.uniform_(bias, -bound, bound)

  def forward(self, inp, out=None):
    inp = C(inp.view(inp.size(0), self.input_size))
    i = 0
    params = self.params.unflat
    linears = {}
    while f'mat_{i}' in params:
      if self.sb_mode == 'none':
        inp = torch.matmul(inp, params[f'mat_{i}'])
        inp = inp + params[f'bias_{i}']

      elif self.sb_mode in ['normal', 'unified']:
        if i == len(params) - 1:
          self.sb_linear[f'layer_{i}'] = self.sb_linear_cls(f'layer_{i}')#, k=-1)
        else:
          self.sb_linear[f'layer_{i}'] = self.sb_linear_cls(f'layer_{i}')
        inp = self.sb_linear[f'layer_{i}'](
            inp, params[f'mat_{i}'], params[f'bias_{i}'])

      self._activations[f'layer_{i}'] = inp
      inp.retain_grad()

      if not i == self.n_layers:
        inp = self.nonlinear(inp)
      # if self.sb_mode in ['normal', 'unified']:

      i += 1

    if out is not None:
      inp = F.log_softmax(inp, dim=1)
      loss = self.loss(inp, C(out))
    else:
      loss = None
    return loss
