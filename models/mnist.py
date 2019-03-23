import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets
from utils import utils
from models.model_helpers import ParamsFlattener
from models.sparse_func import SBLinear, UnifiedSBLinear

C = utils.getCudaManager('default')

class MNISTData:
  def __init__(self, mode):
    assert mode in ['train', 'valid', 'test']
    train = False if mode == 'test' else True
    dataset = datasets.MNIST(
        './mnist', train=train, download=True,
        transform=torchvision.transforms.ToTensor()
    )
    indices = list(range(len(dataset)))
    np.random.RandomState(10).shuffle(indices)
    if mode == 'train':
      indices = indices[:(len(indices) * 0.7)]
    elif mode == 'valid':
      indices = indices[len(indices) * 0.7:]1
    import pdb; pdb.set_trace()

    self.loader = torch.utils.data.DataLoader(
        dataset, batch_size=128,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))

    self.batches = []
    self.cur_batch = 0

  def pseudo_sample(self):
    """return pseudo sample for checking activation size."""
    return (torch.zeros(1, 1, 28, 28), None)

  def sample(self):
    if self.cur_batch >= len(self.batches):
      self.batches = []
      self.cur_batch = 0
      for b in self.loader:
        self.batches.append(b)
    batch = self.batches[self.cur_batch]
    self.cur_batch += 1
    return (batch[0], batch[1])


class MNISTModel(nn.Module):
  def __init__(self, input_size=28*28, layer_size=20, n_layers=1,
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

      params[f'mat_{n_layers}'] = torch.randn(inp_size, 10) * 0.001  # TODO:init
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
