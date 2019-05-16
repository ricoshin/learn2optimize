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
from models.datasets import ImageNet
from utils import utils

C = utils.getCudaManager('default')


class IterDataLoader(object):
  """Just a simple custom dataloader to load data whenever neccessary
  without forcibley using iterative loops.
  """
  def __init__(self, dataset, batch_size, sampler=None):
    self.dataset = dataset
    self.dataloader = data.DataLoader(dataset, batch_size, sampler)
    self.iterator = iter(self.dataloader)

  def __len__(self):
    return len(self.dataloader)

  def load(self, eternal=True):
    if eternal:
      try:
        return next(self.iterator)
      except StopIteration:
        self.iterator = iter(self.dataloader)
        return next(self.iterator)
      except AttributeError:
        import pdb; pdb.set_trace()
    else:
      return next(self.iterator)

  @property
  def batch_size(self):
    return self.dataloader.batch_size

  @property
  def full_size(self):
    return self.batch_size * len(self)

  def new_batchsize(self):
    self.dataloader

  @classmethod
  def from_dataset(cls, dataset, batch_size, rand_with_replace=True):
    if rand_with_replace:
      sampler = data.sampler.RandomSampler(dataset, replacement=True)
    else:
      sampler = None
    # loader = data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return cls(dataset, batch_size, sampler)


class ImageNetData:
  """Current data scheme is as follows:
    - meta-train data(30K)
      - inner-train data(15K)
      - inner-test data(15K)
    - meta-valid data(20K)
      - inner-train data(15K)
      - inner-test data(5K)
    - meta-test data(20K)
      - inner-train data(15K)
      - inner-test data(5K)
  """

  def __init__(self, batch_size=128, fixed=False):
    self.fixed = fixed
    self.batch_size = batch_size
    path='/v9/whshin/imagenet'
    # path ='./imagenet'
    train_data = ImageNet(path, split='train', download=True,
                                transform=torchvision.transforms.ToTensor())
    test_data = ImageNet(path, split='val', download=True,
                               transform=torchvision.transforms.ToTensor())
    import pdb; pdb.set_trace()
    self.m_train_d, self.m_valid_d, self.m_test_d = self._meta_data_split(
      train_data, test_data)
    self.meta_train = self.meta_valid = self.meta_test = {}

  def sample_meta_train(self, ratio=0.5, fixed=None):
    fixed = fixed if fixed is not None else self.fixed
    if fixed and self.meta_train:
      return self.meta_train
    inner_train, inner_test = self._random_split(self.m_train_d, ratio)
    self.meta_train['in_train'] = IterDataLoader.from_dataset(
      inner_train, self.batch_size)
    self.meta_train['in_test'] = IterDataLoader.from_dataset(
      inner_test, self.batch_size)
    return self.meta_train

  def sample_meta_valid(self, ratio=0.75, fixed=None):
    fixed = fixed if fixed is not None else self.fixed
    if fixed and self.meta_valid:
      return self.meta_valid
    inner_train, inner_test = self._random_split(self.m_valid_d, ratio)
    self.meta_valid['in_train'] = IterDataLoader.from_dataset(
      inner_train, self.batch_size)
    self.meta_valid['in_test'] = IterDataLoader.from_dataset(
      inner_test, self.batch_size)
    return self.meta_valid

  def sample_meta_test(self, ratio=0.75, fixed=None):
    fixed = fixed if fixed is not None else self.fixed
    if fixed and self.meta_test:
      return self.meta_test
    inner_train, inner_test = self._random_split(self.m_test_d, ratio)
    self.meta_test['in_train'] = IterDataLoader.from_dataset(
      inner_train, self.batch_size)
    self.meta_test['in_test'] = IterDataLoader.from_dataset(
      inner_test, self.batch_size)
    return self.meta_test

  def _meta_data_split(self, train_data, test_data):
    data_ = data.dataset.ConcatDataset([train_data, test_data])
    meta_train, meta_valid_test = self._fixed_split(data_, 30/70)
    meta_valid, meta_test = self._fixed_split(meta_valid_test, 20/40)
    return meta_train, meta_valid, meta_test

  def _random_split(self, dataset, ratio=0.5):
    assert isinstance(dataset, data.dataset.Dataset)
    n_total = len(dataset)
    n_a = int(n_total * ratio)
    n_b = n_total - n_a
    data_a, data_b = data.random_split(dataset, [n_a, n_b])
    return data_a, data_b

  def _fixed_split(self, dataset, ratio=0.5):
    assert isinstance(dataset, data.dataset.Dataset)
    n_total = len(dataset)
    thres = int(n_total * ratio)
    id_a = range(len(dataset))[:thres]
    id_b = range(len(dataset))[thres:]
    data_a = data.Subset(dataset, id_a)
    data_b = data.Subset(dataset, id_b)
    return data_a, data_b

  # def _fixed_split(self, dataset, ratio):
  #   import pdb; pdb.set_trace()
  #   data_a = dataset[int(len(dataset) * ratio):]
  #   data_b = dataset[:int(len(dataset) * ratio)]
  #   return data_a, data_b

  def _get_wrapped_dataloader(self, dataset, batch_size):
    sampler = data.sampler.RandomSampler(dataset, replacement=True)
    loader = data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return IterDataLoader(loader)

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
        self.init_parameters(params[f'mat_{i}'], params[f'bias_{i}'])
        inp_size = layer_size

      params[f'mat_{n_layers}'] = torch.randn(
          inp_size, 10) * 0.001  # TODO:init
      params[f'bias_{n_layers}'] = torch.zeros(10)
      self.init_parameters(
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

  def init_parameters(self, mat, bias, type='he_uniform'):
    assert type in ['he_uniform', 'simple']
    if type == 'he_uniform':
      nn.init.kaiming_uniform_(mat, a=math.sqrt(5))
      fan_in, _ = nn.init._calculate_fan_in_and_fan_out(mat)
      bound = 1 / math.sqrt(fan_in)
      nn.init.uniform_(bias, -bound, bound)
    elif type == 'simple':
      mat.normal_(0., 0.001)
      bias.fill_(0.)
    else:
      raise Exception(f'Unknown type: {type}')


  def forward(self, inp, out=None):
    inp = C(inp.view(inp.size(0), self.input_size))
    i = 0
    params = self.params.unflat
    linears = {}
    while f'mat_{i}' in params:
      if self.sb_mode == 'none':
        try:
          inp = torch.matmul(inp, params[f'mat_{i}'])
          inp = inp + params[f'bias_{i}']
        except:
          import pdb; pdb.set_trace()

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
      out = C(out)
      inp = F.log_softmax(inp, dim=1)
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
