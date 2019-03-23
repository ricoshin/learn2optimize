import copy
import math
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
from utils import utils
from models.model_helpers import ParamsIndexTracker
from optimizers.optim_helpers import OptimizerBatchManager
from tqdm import tqdm
from utils.result import ResultDict

C = utils.getCudaManager('default')


class LSTMStates(object):
  def __init__(self, h, c):
    if not isinstance(h, torch.Tensor):
      raise TypeError("Expected torch.Tensor for h, c arguments!")
    assert h.size() == c.size()
    self.h = h
    self.c = c

  def __repr__(self):
    return f"LSTMStates(h={self.h}, c={self.c})"

  @classmethod
  def zero_states(cls, size):
    if not (isinstance(size, list) or isinstance(size, tuple)):
      raise TypeError("Expected list or tuple for size arugment!")
    return cls(h=C(torch.zeros(*size)), c=C(torch.zeros(*size)))

  def size(self, dim):
    assert self.h.size() == self.c.size()
    return self.h.size(dim)

  def clone(self):
    return LSTMStates(h=self.h.clone(), c=self.c.clone())

  def detach(self):
    return LSTMStates(h=self.h.detach(), c=self.c.detach())

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


class OptimizerStatesSlicer(object):
  def __init__(self, states):
    self.states = states

  def __getitem__(self, key):
    return OptimizerStates([state[key] for state in self.states])

  def __setitem__(self, key, item):
    if not isinstance(item, OptimizerStates):
      raise TypeError(f"{item}is not an instance of LSTMStates!")
    assert len(self.states) == len(item)
    for state, state_new in zip(self.states, item):
      state[key] = state_new


class OptimizerStates(list):
  def __init__(self, states):
    if not isinstance(states, list):
      raise TypeError(f"Expected list but invalid data type({type(states)})!")
    if not isinstance(states[0], LSTMStates):
      raise TypeError(f"Expected list of LSTMStates but wrong type of element"
                      "({type(states[0])}) in the list!")
    self.n_layers = len(states)
    self.n_params = states[0].size(0)
    self.hidden_sz = states[0].size(1)
    self.slicer = OptimizerStatesSlicer(self)
    super().__init__(states)

  @classmethod
  def zero_states(cls, n_layers, n_params, hidden_sz):
    size = [n_params, hidden_sz]
    return cls([LSTMStates.zero_states(size) for _ in range(n_layers)])

  def detach(self):
    return OptimizerStates([state.detach() for state in self])

  def detach_(self):
    self = OptimizerStates([state.detach() for state in self])

  def clone(self):
    return OptimizerStates([state.clone() for state in self])


class Optimizer(nn.Module):
  def __init__(self, preproc=False, hidden_sz=20, preproc_factor=10.0):
    super().__init__()
    self.hidden_sz = hidden_sz
    self.n_layers = 2  # NOTE: fix later!
    if preproc:
      self.recurs = nn.LSTMCell(2, hidden_sz)
    else:
      self.recurs = nn.LSTMCell(1, hidden_sz)
    self.recurs2 = nn.LSTMCell(hidden_sz, hidden_sz)
    self.output = nn.Linear(hidden_sz, 1)
    self.preproc = preproc
    self.preproc_factor = preproc_factor
    self.preproc_threshold = np.exp(-preproc_factor)

  def forward(self, inp, states):
    if self.preproc:
      # Implement preproc described in Appendix A

      # Note: we do all this work on tensors, which means
      # the gradients won't propagate through inp. This
      # should be ok because the algorithm involves
      # making sure that inp is already detached.
      inp = inp.data.squeeze()
      inp2 = C(torch.zeros(inp.size()[0], 2))
      keep_grads = (torch.abs(inp) >= self.preproc_threshold).squeeze()
      inp2[:, 0][keep_grads] = torch.log(
          torch.abs(inp[keep_grads]) + 1e-8) / self.preproc_factor
      inp2[:, 1][keep_grads] = torch.sign(inp[keep_grads])

      inp2[:, 0][~keep_grads] = -1
      inp2[:, 1][~keep_grads] = float(
          np.exp(self.preproc_factor)) * inp[~keep_grads]
      inp = C(inp2)
    s_0 = LSTMStates(*self.recurs(inp, (states[0].h, states[0].c)))
    s_1 = LSTMStates(*self.recurs2(s_0.h, (states[1].h, states[1].c)))
    return self.output(s_1.h), OptimizerStates([s_0, s_1])

  def meta_optimize(self, meta_optimizer, data_cls, model_cls, optim_it, unroll,
                    out_mul, should_train=True):
    if should_train:
      self.train()
    else:
      self.eval()
      unroll = 1

    target = data_cls(training=should_train)
    optimizee = C(model_cls())
    optimizer_states = OptimizerStates.zero_states(
        self.n_layers, len(optimizee.params), self.hidden_sz)
    result_dict = ResultDict()
    # n_params = 0
    # named_shape = {}
    # for name, p in optimizee.all_named_parameters():
    #   n_params += int(np.prod(p.size()))
    #   named_shape[name] = p.size()

    # batch_size = 30000
    # batch_sizes = []
    # if batch_size > n_params:
    #   batch_sizes = [n_params]
    # else:
    #   batch_sizes = [batch_size for _ in range(n_params // batch_size)]
    #   batch_sizes.append(n_params % batch_size)

    coordinate_tracker = ParamsIndexTracker(
        n_params=len(optimizee.params), n_tracks=30)
    # update_track_num = 10
    # n_params = len(optimizee.params)
    # if update_track_num > n_params:
    #   update_track_num = n_params
    # update_track_id = np.random.randint(0, n_params, update_track_num)
    # updates_tracks = []

    # hidden_states = [
    #     C(torch.zeros(n_params, self.hidden_sz)) for _ in range(2)]
    # cell_states = [
    #     C(torch.zeros(n_params, self.hidden_sz)) for _ in range(2)]
    all_losses_ever = []
    timestamps = []

    if should_train:
      meta_optimizer.zero_grad()
    unroll_losses = None

    ############################################################################
    iter_pbar = tqdm(range(1, optim_it + 1), 'optim_iteration')
    iter_watch = utils.StopWatch('optim_iteration')
    for iteration in iter_pbar:
      loss = optimizee(target)
      iter_pbar.set_description(f'optim_iteration[loss:{loss}]')

      if unroll_losses is None:
        unroll_losses = loss
      else:
        unroll_losses += loss
      result_dict.add('losses', loss)
      try:
        loss.backward(retain_graph=should_train)
      except Exception as ex:
        print(ex)
        import pdb
        pdb.set_trace()

      # all_losses_ever.append(loss.data.cpu().numpy())

      # hidden_states2 = [
      #     C(torch.zeros(n_params, self.hidden_sz)) for _ in range(2)]
      # cell_states2 = [
      #     C(torch.zeros(n_params, self.hidden_sz)) for _ in range(2)]
      ##########################################################################
      # params_flat = []
      # grads_flat = []
      #
      # for name, p in optimizee.all_named_parameters():
      #   named_shape[name] = p.size()
      #   params_flat.append(p.view(-1, 1))
      #   grads_flat.append(C.detach(p.grad.view(-1, 1)))
      # params_flat = torch.cat(params_flat, 0)
      # grads_flat = torch.cat(grads_flat, 0)
      #
      # offset = 0
      # params_batches = []
      # grads_batches = []
      # for i in range(len(batch_sizes)):
      #   params_batches.append(params_flat[offset:offset + batch_sizes[i], :])
      #   grads_batches.append(grads_flat[offset:offset + batch_sizes[i], :])
      #   offset += batch_sizes[i]
      # assert(offset == params_flat.size(0))
      # batches = zip(batch_sizes, params_batches, grads_batches)
      ##########################################################################
      offset = 0
      result_params_flat = []
      result_params = {}
      updates_track = []
      batch_manager = OptimizerBatchManager(
          params=optimizee.params, states=optimizer_states,
          getters=['params', 'states'],
          setters=['params', 'states', 'updates'],
          batch_size=30000)

      # for batch_sz, params, grads in batches:
      for get, set in batch_manager.iterator():
        # We do this so the gradients are disconnected from the graph
        #  but we still get gradients from the rest
        updates, new_states = self(
            get.params.grad.detach(), get.states.clone())
        set.states(new_states.clone())
        set.params(get.params.clone() + updates * out_mul)
        set.updates(updates)

        #     grads,
        #     [h[offset:offset + batch_sz] for h in hidden_states],
        #     [c[offset:offset + batch_sz] for c in cell_states]
        # )
        # for i in range(len(new_hidden)):
        #   hidden_states2[i][offset:offset + batch_sz] = new_hidden[i]
        #   cell_states2[i][offset:offset + batch_sz] = new_cell[i]

        # result_params_flat.append(params + updates * out_mul)
        # updates_track.append(updates)
        # offset += batch_sz
      ##########################################################################
      # result_params_flat = torch.cat(result_params_flat, 0)
      # updates_track = torch.cat(updates_track, 0)
      # updates_track = updates_track.data.cpu().numpy()
      # updates_track = np.take(updates_track, update_track_id)
      # updates_tracks.append(updates_track)

      # offset = 0
      # for name, shape in named_shape.items():
      #   n_unit_params = int(np.prod(shape))
      #   result_params[name] = result_params_flat[
      #     offset:offset + n_unit_params, :].view(shape)
      #   result_params[name].retain_grad()
      #   offset += n_unit_params
      ##########################################################################
      # NOTE: optimizee.states.states <-- looks a bit weird
      if iteration % unroll == 0:

        if should_train:
          meta_optimizer.zero_grad()
          unroll_losses.backward()
          meta_optimizer.step()
        unroll_losses = None

        optimizee.params.detach_()
        optimizer_states = optimizer_states.detach()

      #   detached_params = {k: C.detach(v) for k, v in result_params.items()}
      #   optimizee = C(model_cls(**detached_params))
      #   hidden_states = [C.detach(s) for s in hidden_states2]
      #   cell_states = [C.detach(s) for s in cell_states2]
      #
      # else:
      #   optimizee = C(model_cls(**result_params))
      #   assert len(list(optimizee.all_named_parameters()))
      #   hidden_states = hidden_states2
      #   cell_states = cell_states2
      optimizee = C(model_cls(optimizee.params))
      # timestamps.append(iter_watch.touch())
      result_dict.add('updates', coordinate_tracker(batch_manager.updates))
      result_dict.add('timestamps', iter_watch.touch())
    return all_losses_ever, updates_tracks, timestamps
