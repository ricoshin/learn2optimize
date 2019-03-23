import copy
import os
import math
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple
from functools import partial
from utils import utils

C = utils.getCudaManager('default')


class GradientPreprocessor(nn.Module):
  def __init__(self, p=10.0, eps=1e-6):
    super(GradientPreprocessor, self).__init__()
    self.eps = eps
    self.p = p

  def forward(self, x):
    indicator = (x.abs() > math.exp(-self.p)).float()
    x1 = (x.abs() + self.eps).log() / self.p * indicator - (1 - indicator)
    x2 = x.sign() * indicator + math.exp(self.p) * x * (1 - indicator)

    return torch.cat((x1, x2), 1)


class LSTMStates(object):
  def __init__(self, h, c):
    if not isinstance(h, torch.Tensor):
      raise TypeError("Expected torch.Tensor for h, c arguments!")
    assert h.size() == c.size()
    self.h = h
    self.c = c

  def __repr__(self):
    return f"LSTMStates(h={self.h}, c={self.c})"

  def __iter__(self):
    return iter([self.h, self.c])

  @classmethod
  def zero_states(cls, size, device=None):
    if not (isinstance(size, list) or isinstance(size, tuple)):
      raise TypeError("Expected list or tuple for size arugment!")
    return cls(h=C(torch.zeros(size, device=device)),
               c=C(torch.zeros(size, device=device)))

  def size(self, dim):
    assert self.h.size() == self.c.size()
    return self.h.size(dim)

  def apply_func(self, func, other=None):
    if other is None:
      return LSTMStates(h=func(self.h), c=func(self.c))
    else:
      return LSTMStates(h=func(self.h, other), c=func(self.c, other))

  @property
  def device(self):
    assert self.h.device == self.c.device
    return self.h.device

  def cuda(self):
    return LSTMStates(h=self.h.cuda(), c=self.c.cuda())

  def clone(self):
    return LSTMStates(h=self.h.clone(), c=self.c.clone())

  def detach(self):
    return LSTMStates(h=self.h.detach(), c=self.c.detach())

  def detach_(self):
    self.h.detach_()
    self.c.detach_()
    return self

  def update(self, new):
    if not isinstance(item, LSTMStates):
      raise TypeError(f"{item}is not an instance of LSTMStates!")
    self.h = new.h
    self.c = new.c

  def __getitem__(self, key):
    return LSTMStates(self.h[key], self.c[key])

  def __setitem__(self, key, item):
    if not isinstance(item, LSTMStates):
      raise TypeError(f"{item}is not an instance of LSTMStates!")
    self.h[key] = item.h
    self.c[key] = item.c


class OptimizerStates(list):
  def __init__(self, states, rnn_cell='gru'):
    assert rnn_cell in ['lstm', 'gru']
    self.rnn_cell = rnn_cell
    if not isinstance(states, (list, tuple)):
      raise TypeError(f"Expected list but invalid data type({type(states)})!")
    if not isinstance(states[0], (LSTMStates, torch.Tensor)):
      raise TypeError(f"Expected list of LSTMStates or normal tensorbut wrong "
                      "type of element({type(states[0])}) in the list!")
    self.n_layers = len(states)
    self.n_params = states[0].size(0)
    self.hidden_sz = states[0].size(1)
    super().__init__(states)

  def __repr__(self):
    return f"OptimizerStates({super().__repr__()})"

  def size(self):
    return torch.Size((self.n_layers, self.n_params, self.hidden_sz))

  def __neg__(self):
    return self.mul(-1)

  def __add__(self, other):
    return self.add(other)

  def __sub__(self, other):
    return self.add(-other)

  def __mul__(self, other):
    return self.mul(other)

  def __truediv__(self, other):
    return self.div(other)

  def add(self, other):
    return self.apply_op_with_other('add', other)

  def mul(self, other):
    return self.apply_op_with_other('mul', other)

  def div(self, other):
    return self.apply_op_with_other('div', other)

  def apply_op_with_other(self, op_name, other):
    assert isinstance(other, (OptimizerStates, torch.Tensor, int, float))
    func = lambda t1, t2: getattr(torch, op_name)(t1, t2)
    return self.apply_func(func, other)

  def apply_func(self, func, other=None):
    assert isinstance(other, (OptimizerStates, torch.Tensor, int, float))
    if other is None:
      out = [func(state) for state in self]
    elif isinstance(other, (int, float, torch.Tensor)):
      out = [func(state, other) for state in self]
    elif isinstance(other, OptimizerStates):
      out = [func(s, s_o) for s, s_o in zip(self, other)]
    return OptimizerStates(out)

  @classmethod
  def initial_zeros(cls, size, rnn_cell='gru', device=None):
    assert rnn_cell in ['lstm', 'gru']
    if not len(size) == 3:
      raise Exception('OptimizerStates should have 3-dim torch.Size.')
    if rnn_cell == 'lstm':
      return cls([LSTMStates.zero_states(
        [size[1], size[2]], device=device) for _ in range(size[0])])
    elif rnn_cell == 'gru':
      return cls([torch.zeros(
        [size[1], size[2]], device=device) for _ in range(size[0])])

  def new_zeros(self, size):
    device = [states for states in self][0].device
    assert all([states.device == device for states in self])
    return OptimizerStates.initial_zeros(
      size=size, rnn_cell=self.rnn_cell, device=device)

  def detach(self):
    return OptimizerStates([state.detach() for state in self])

  def detach_(self):
    for states in self:
      states.detach_()
    return self

  def clone(self):
    return OptimizerStates([state.clone() for state in self])

  def cuda(self):
    return OptimizerStates([state.cuda() for state in self])


class StatesSlicingWrapper(OptimizerStates):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def __repr__(self):
    return f"SlicingWrapper({super().__repr__()})"

  def __getitem__(self, key):
    return OptimizerStates([state[key] for state in self])

  def __setitem__(self, key, item):
    if not isinstance(item, OptimizerStates):
      raise TypeError(f"{item}is not an instance of LSTMStates!")
    assert len(self) == len(item)
    for state, state_new in zip(self, item):
      state[key] = state_new

  def new_zeros(self, size):
    return StatesSlicingWrapper(super().new_zeros(size))


class OptimizerParams(object):
  _save_ext = '.params'
  def __init__(self, state_dict):
    self.state_dict = state_dict

  @classmethod
  def from_optimizer(cls, optimizer):
    return cls(optimizer.state_dict())

  def to_optimizer(self, optimizer):
    optimizer.load_state_dict(self.state_dict)

  @staticmethod
  def is_loadable(name, load_dir):
    filepath = os.path.join(load_dir, name + OptimizerParams._save_ext)
    return True if load_dir and os.path.exists(filepath) else False

  @classmethod
  def load(cls, name, load_dir):
    filepath = os.path.join(load_dir, name + OptimizerParams._save_ext)
    with open(filepath, 'rb') as f:
      params = cls(torch.load(f))
    print(f'\nLoaded learned params from: {filepath}')
    return params

  def save(self, name, save_dir):
    if save_dir is None:
      return self
    filepath = os.path.join(save_dir, name + OptimizerParams._save_ext)
    with open(filepath, 'wb') as f:
      torch.save(self.state_dict, f)
    print(f'\n\n\n\nSaved learned params as: {filepath}')
    return self


class BatchManagerArgument(object):
  def __init__(self, immutable=False, volatile=False, **kwargs):
    """immutable_names: a name list of args that only need getters.
    Setters will not be provided by BatchManager and memory will be saved.
    """
    names = [name for name in kwargs.keys()]
    self.get_names = set(names)
    self.set_names = set(names)
    self.out_names = set(names)
    self._asdict = self._set_as_attribute(kwargs)
    if immutable:
      self.immutable()
    else:
      self.volatile()

  def __repr__(self):
    return f"BatchManagerArgument(attr={self._asdict})"

  def _set_as_attribute(self, dict_):
    for k, v in dict_.items():
      setattr(self, k, v)
    return dict_

  def names(self):
    return [name for name in self._asdict.keys()]

  def named_args(self):
    return [(name, arg) for name, arg in self._asdict.items()]

  def update_by_args(self, **kwargs):
    self._asdict.update(self._set_as_attribute(kwargs))
    return self

  def update(self, other):
    self.get_names |= other.get_names
    self.set_names |= other.set_names
    self.out_names |= other.out_names
    return self.update_by_args(**other._asdict)

  def immutable(self):
    self.set_names = set()
    return self

  def volatile(self):
    self.out_names = set()
    return self

  def set_immutable_by_name(self, names):
    assert isinstance(names, (list, tuple))
    if not all([n in self._asdict for n in names]):
      raise Exception("Unknown name!")
    self.set_names -= set(names)
    return self

  def set_volatile_by_name(self, names):
    assert isinstance(names, (list, tuple))
    if not all([n in self._asdict for n in names]):
      raise Exception("Unknown name!")
    self.out_names -= set(names)
    return self

  def detach_(self):
    for v in self._asdict.values():
      v.detach_()


class BatchIndexer(nn.Module):
  @property
  def index(self):
    raise NotImplementedError()

  def __len__(self):
    return int(np.prod(self.index.size()))

  def _shuffle_1d(self, tensor):
    assert len(tensor.size()) == 1
    return tensor[torch.randperm(tensor.size(0))]

  def batch(self, size, shuffle):
    # w_shuffle = utils.StopWatch('shuffle')
    # w_chunk = utils.StopWatch('chunk')

    # w_shuffle.touch()
    index = self._shuffle_1d(self.index) if shuffle else self.index
    # w_shuffle.touch('interval', True)
    # w_chunk.touch()
    num_chunk = math.ceil(len(self) / size)
    out =  index.chunk(num_chunk, dim=0)
    # w_chunk.touch('interval', True)
    return out


class DefaultIndexer(BatchIndexer):
  def __init__(self, input, index=None, shuffle=False):
    super().__init__()
    assert len(input.squeeze().size()) == 1 # has to be 1-D tensor after squeeze
    self._id = index
    self._shuffle = shuffle
    self._num = input.size(0)

  def __len__(self):
    if self._id is None:
      return self._num
    else:
      return len(self._id)

  @property
  def index(self):
    if self._id is None:
      return list(range(0, self._num))
    else:
      return self._id

  def batch(self, size):
    # if not self._shuffle and self._id is None:
    #   return [slice(None)] # NOTE indexing [:] causes inplace operation
    index = self._shuffle_1d(self.index) if self._shuffle else self.index
    return [index]


class TopKIndexer(BatchIndexer):
  def __init__(
    self, input, k_ratio, bound=None, random_k_mode=False, shuffle=False):
    super().__init__()
    assert k_ratio >= .0 and k_ratio <= 1. # [0, 1]
    assert len(input.squeeze().size()) == 1 # has to be 1-D tensor after squeeze
    self._k_ratio = k_ratio
    self._shuffle = shuffle
    input = input.squeeze()
    # sort = utils.StopWatch('sort')
    # sort.touch()
    if bound is not None:
      assert isinstance(bound, (list, tuple))
      k_id = []
      b_prev = 0
      for b_next in bound:
        chunk = input[b_prev:b_next]
        k = int(np.prod(chunk.size()) * k_ratio)
        if random_k_mode:
          k_id_chunk = torch.tensor(np.random.randint(0, chunk.size(0), k))
        else:
          k_id_chunk = chunk.abs().topk(k, 0, sorted=False)[1]
        k_id_chunk = k_id_chunk + b_prev
        k_id.append(k_id_chunk)
        b_prev = b_next
      self._k_id = torch.cat(k_id, dim=0)
    else:
      k = int(np.prod(input.size()) * k_ratio)
      if random_k_mode:
        self._k_id = torch.tensor(np.random.randint(0, input.size(0), k))
      else:
        self._k_id = input.abs().topk(k, 0, sorted=False)[1]
    # sort.touch('interval', True)
    assert len(self._k_id.size()) == 1 # 1-D tensor

  @property
  def index(self):
    return self._k_id

  def batch(self, size):
    return super().batch(size, self._shuffle)


class OptimizerBatchManager(object):
  def __init__(self, batch_arg, indexer, batch_size=30000):
    if not isinstance(batch_arg, BatchManagerArgument):
      raise TypeError("Expected optim_helpers.BatchManagerArgument "
                      "as a second to_gett batch_arg, but got "
                      f"{type(batch_arg)}!")
    batch_arg_d = batch_arg._asdict # convert to dict
    to_get = {k: v for k, v in batch_arg_d.items() if k in batch_arg.get_names}
    to_set = {k: v for k, v in batch_arg_d.items() if k in batch_arg.set_names}
    self.to_get = to_get
    self.to_set = to_get #self._new_zeros(to_set)
    required_batch_arg = ['params'] #, 'states']
    if any([arg not in batch_arg_d for arg in required_batch_arg]):
      raise Exception("Unable to find expected arguments: "
                      f"{required_batch_arg}")
    self._Getters = namedtuple("BatchGetters", batch_arg.get_names)
    self._Setters = namedtuple("BatchSetters", batch_arg.get_names)#batch_arg.set_names)
    # batch planning
    n_params = len(batch_arg_d['params'])
    self.batch_size = n_params if batch_size > n_params else batch_size
    self.indexer = indexer


  def __len__(self):
    return len(self.batch_sizes)

  def _set_as_attribute(self, dict_):
    for k, v in dict_.items():
      setattr(self, k, v)
    return dict_

  def _new_zeros(self, dict_):
    return {k: v.new_zeros(v.size()).detach_() for k, v in dict_.items()}

  @staticmethod
  def placeholder(size):
    assert isinstance(size, torch.Size)
    return C(torch.zeros(size))

  @property
  def batch_arg_to_get(self):
    return BatchManagerArgument(**self.to_get)

  @property
  def batch_arg_to_set(self):
    return BatchManagerArgument(**self.to_set)

  def iterator(self):
    start = 0
    # w = utils.StopWatch('iterator')
    # w.touch()
    for index in self.indexer.batch(self.batch_size):
      # w.touch('interval', True)
      getters, setters = {}, {}
      for k, v in self.to_get.items():
        def _get_func(src):
          return src[index]
        getters[k] = partial(_get_func, src=v)
      # w.touch('interval', True)
      for k, v in self.to_set.items():
        def _set_func(src, tar):
          tar[index] = src
        setters[k] = partial(_set_func, tar=v)
      # w.touch('interval', True)
      yield self._Getters(**getters), self._Setters(**setters)

  # def iterator2(self):
  #   start = 0
  #   for i in range(len(self.batch_sizes)):
  #     end = start + self.batch_sizes[i]
  #     getters, setters = {}, {}
  #     for k, v in self.to_get.items():
  #       getters[k] = v[start:end]
  #     for k, v in self.to_set.items():
  #       def _set_func(src, tar):
  #         tar[start:end] = src
  #       setters[k] = partial(_set_func, tar=v)
  #     yield self._Getters(**getters), self._Setters(**setters)
  #     start = end
