from collections import OrderedDict
import numpy as np
from operator import itemgetter
import functools

import torch
import torch.nn as nn
from utils import utils


C = utils.getCudaManager('default')


class GradientPreprocessor(nn.Module):
  def __init__(self, p=10.0, eps=1e-6):
    super(GradientPreprocessor, self).__init__()
    self.eps = eps
    self.p = p

  def forward(self, input):
    indicator = (x.abs() > math.exp(-self.p)).float()
    x1 = (x.abs() + self.eps).log() / self.p * indicator - (1 - indicator)
    x2 = x.sign() * indicator + math.exp(self.p) * input * (1 - indicator)

    return torch.cat((x1, x2), 1)


class LSTMStateTuple(tuple):
  __slots__ = {}
  _fields = ('h', 'c')

  def __new__(cls, h=None, c=None):
    'Create new instance of LSTMStateTuple(h, c)'
    init = lambda state: torch.Tensor() if state is None else state
    return tuple.__new__(cls, (init(h), init(c)))

  def apply_slice(self, key):
    return LSTMStateTuple(h=self.h[key], c=self.c[key])

  def assign(self, tuple_):
    assert isinstance(tuple_, tuple) and len(tuple_) == 2
    assert np.shape(tuple_) == np.shape(self.tuple)
    return LSTMStateTuple(h=tuple_[0], c=tuple_[1])

  @classmethod
  def _make(cls, iterable, new=tuple.__new__, len=len):
    'Make a new LSTMStateTuple object from a sequence or iterable'
    result = new(cls, iterable)
    if len(result) != 2:
      raise TypeError('Expected 2 arguments, got %d' % len(result))
    return result

  def _replace(_self, **kwds):
    'Return a new LSTMStateTuple object replacing specified fields '
    'with new values'
    result = _self._make(map(kwds.pop, ('h', 'c'), _self))
    if kwds:
      raise ValueError('Got unexpected field names: %r' % list(kwds))
    return result

  def __repr__(self):
    'Return a nicely formatted representation string'
    return self.__class__.__name__ + '(h=%r, c=%r)' % self

  def _asdict(self):
    'Return a new OrderedDict which maps field names to their values.'
    return OrderedDict(zip(self._fields, self))

  def __getnewargs__(self):
    'Return self as a plain tuple.  Used by copy and pickle.'
    return tuple(self)

  h = property(itemgetter(0), doc='Alias for field number 0')
  c = property(itemgetter(1), doc='Alias for field number 1')


class CoordinatewiseLSTMStates(object):
  def __init__(self, n_layers, n_params, hidden_sz):
    super(CoordinatewiseLSTMStates, self).__init__()
    self.n_layers = n_layers
    self.n_params = n_params
    self.hidden_sz = hidden_sz
    # self._tuple =
    # self._state_tuple = (self.hx, self.cx)
    self.subset_idx = 0
    self._state_tuple = None
    self.reset()

  def __repr__(self):
    return f"CoordinatewiseLSTMStates.tuple({self.tuple})"

  def __len__(self):
    return self.n_layers

  def size(self):
    return (self.n_layers, self.n_params, self.hidden_sz)

  def __getitem__(self, key):
    # if not isinstance(key, int) or key >= self.n_layers:
    #   raise KeyError("CoordinatewiseLSTMStates.__getitem__ key "
    #                  f"has to be integer larger than n_layers({self.n_layers})")
    #return (self._h[key], self._c[key])
    tuple = self.tuple[key]
    return CoordinatewiseLSTMStates(
      len(tuple), self.n_params, self.hidden_sz).assign(tuple)

    # n_params = len(range(*key.indices(self.n_layers)))
    # states = CoordinatewiseLSTMStates(n_params, self.n_layers, self.hidden_sz)
    # slice_fn = lambda tuple_: LSTMStateTuple(*[state[key] for state in tuple_])
    # sliced = self._nested_map(slice_fn, self.tuple)
    # import pdb; pdb.set_trace()
    # return states.assign(sliced)

  # def __setitem__(self, key, value):
  #   if not (isinstance(value, tuple) or isinstance(value, list)):
  #     raise ValueError("CoordinatewiseLSTMStates.__setitem__ value "
  #                      "has to be list(hx, cx) or tuple(hx, cx)!")
  #   self._h[key], self._c[key] = value

  # @property
  # def h(self):
  #   return self._h
  #
  # @property
  # def c(self):
  #   return self._c

  @property
  def tuple(self):
    if self._state_tuple is None:
      init = lambda : torch.Tensor()
      self._state_tuple = [LSTMStateTuple(
        h=init(), c=init()) for _ in range(self.n_layers)]
    #return (self.h, self.c)
    return self._state_tuple

  # @tuple.setter
  # def tuple(self, new_state_tuple):
  #   if (not isinstance(new_state_tuple, tuple) or
  #       not len(new_state_tuple) != 2 or
  #       not all(map(lambda t: isinstance(t, torch.Tensor), new_state_tuple)):
  #     raise ValueError("CoordinatewiseLSTMStates.tuple has to be set "
  #                      "by the data type tuple(tensor, tensor)")
  #   self._h, self._c = new_state_tuple[0], new_state_tuple[1]

  def _nested_map(self, func, target):
    if isinstance(target, LSTMStateTuple):
      return func(target)
    else:
      result = [self._nested_map(func, tar) for tar in target]
      return tuple(result) if isinstance(target, tuple) else result
    #self.tuple = (list(map(func, state)) for state in self.tuple)

  def _from_zeros(self, _):
    zeros = lambda : C(torch.zeros(self.n_params, self.hidden_sz))
    return LSTMStateTuple(h=zeros(), c=zeros())

  def _from_ones(self, _):
    zeros = lambda : C(torch.ones(self.n_params, self.hidden_sz))
    return LSTMStateTuple(h=zeros(), c=zeros())

  def _from_detached(self, tuple_):
    return LSTMStateTuple(*list(map(torch.tensor, tuple_)))

  def _set_state_tuple(self, process_fn, tuple_=None):
    tuple_ = self.tuple if tuple_ is None else tuple_
    self._state_tuple = self._nested_map(process_fn, tuple_)
    return self
      # self.hx.append(C(torch.zeros(self.n_params, self.hidden_sz)))
      # self.cx.append(C(torch.zeros(self.n_params, self.hidden_sz)))

  # def _from_sliced(self, start, end):
  #   return LSTMStateTuple(*slice(state), start=start, end=end)

  def detach(self):
    return self._set_state_tuple(self._from_detached)

  def reset(self):
    return self._set_state_tuple(self._from_zeros)

  def reset_one(self):
    return self._set_state_tuple(self._from_ones)

  def assign(self, new_tuple):
    assert len(new_tuple) == self.n_layers
    assert np.shape(self.tuple[0].h) == np.shape(new_tuple[0].h)
    #self._set_state_tuple(self._from_tuple, new_tuple)
    self._set_state_tuple(lambda x: x, new_tuple)  # pass through indentity func
    return self

  def sequential_subset(self, n_sub_params):
    start_idx = self.subset_idx
    end_idx = start_idx + n_sub_params
    self.check_subset_idx(end_idx)
    return self[start_idx:end_idx]
    # return CoordinatewiseLSTMStates(
    #   n_sub_params, self.n_layers, self.hidden_sz).assign(sliced)
    #sliced_state =
    # for i in range(self.n_layers):
    #   hx.append(self.hx[i][start_idx:end_idx])
    #   cx.append(self.cx[i][start_idx:end_idx])
    # self.subset_idx += n_sub_params
    #return CoordinatewiseLSTMStates(
    #  n_sub_params, self.n_layers, self.hidden_sz).update(self[start_idx:end_idx])

  def reset_subset(self):
    if check_idx != self.n_params:
      n_left_out = self.n_params - check_idx
      raise Warning(f"{n_left_out} params has been left out "
                    "for subsetting. They're expected to be fully consumed.")
    self.subset_idx = 0

  def check_subset_idx(self, check_idx, reset=False):
    if check_idx > self.n_params:
      raise AssertionError("No more elements left for subsetting. "
                           "Can't get subset before .reset_subset() is called!")

  # @subset.setter
  # def subset(self, new_subset):
  #   start_idx = new_subset.sub_set_idx
  #   end_idx = start_idx + new_subset.n_params
  #   for i in range(self.n_layers):
  #     self.hx[i][start_idx:end_idx] = new_subset.hx[i]
  #     self.cx[i][start_idx:end_idx] = new_subset.cx[i]


class NeuralOptimizer(nn.Module):
  def __init__(self, optimizee, n_layers=2, hidden_sz=20, preproc=None):
    super(NeuralOptimizer, self).__init__()
    self.optimizee = optimizee
    self.preproc = preproc
    self.hidden_sz = hidden_sz

    self.lstms = []
    input_size = 1 if preproc is None else preproc.output_size
    layer_size = [input_size] + [hidden_sz] * n_layers
    for i in range(n_layers):
      lstm = nn.LSTMCell(layer_size[i], layer_size[i+1])
      self.lstms.append(lstm)
      self.add_module(f'LSTM_{i}', lstm)
    self.linear = nn.Linear(layer_size[-1], 1)

  def forward(self, x, state):
    assert x.size(0) == state



opt = NeuralOptimizer(None, 2, 20, None)
C.set_cuda(True)
ss = CoordinatewiseLSTMStates(5, 2, 10)
aa = ss[2:5]
#bb = ss[:3]
# aaa = ss.subset(2)
# bbb = ss.subset(2)
import pdb; pdb.set_trace()
print('end of program')
