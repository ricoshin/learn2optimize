"""Learning to learn by gradient descent by gradient descent.
  Andrychowicz et al.
  + GRU cell (optional)
  + inner validation(test) training

"""
import pdb

import numpy as np
import torch
import torch.nn as nn
from models.model_helpers import ParamsIndexTracker
from optimize import log_pbar, log_tf_event
from optimizers.optim_helpers import (BatchManagerArgument, DefaultIndexer,
                                      GradientPreprocessor, LSTMStates,
                                      OptimizerBase, OptimizerBatchManager,
                                      OptimizerParams, OptimizerStates,
                                      StatesSlicingWrapper, OptimizerBase)
from tqdm import tqdm
from utils import utils
from utils.result import ResultDict
from utils.timer import Walltime, WalltimeChecker
from utils.torchviz import make_dot

C = utils.getCudaManager('default')


class Optimizer(OptimizerBase):
  def __init__(self, preproc=True, hidden_sz=20, preproc_factor=10.0,
               rnn_cell='lstm'):
    super().__init__()
    assert rnn_cell in ['lstm', 'gru']
    self.rnn_cell = rnn_cell
    rnn_cell = nn.LSTMCell if rnn_cell == 'lstm' else nn.GRUCell
    self.hidden_sz = hidden_sz
    self.n_layers = 2  # TODO: fix later!
    if preproc:
      self.preproc = GradientPreprocessor()
      self.recurs = rnn_cell(2, hidden_sz)
    else:
      self.recurs = rnn_cell(1, hidden_sz)
    self.recurs2 = rnn_cell(hidden_sz, hidden_sz)
    self.output = nn.Linear(hidden_sz, 1)
    self.params_tracker = ParamsIndexTracker(n_tracks=100)

  def forward(self, inp, states):
    if self.preproc:
      inp = C(self.preproc(inp))

    if self.rnn_cell == 'lstm':
      s_0 = LSTMStates(*self.recurs(inp, (states[0].h, states[0].c)))
      s_1 = LSTMStates(*self.recurs2(s_0.h, (states[1].h, states[1].c)))
      out = s_1.h
    elif self.rnn_cell == 'gru':
      s_0 = self.recurs(inp, states[0])
      s_1 = self.recurs2(s_0, states[1])
      out = s_1
    else:
      raise RuntimeError(f'Unknown rnn_cell type: {self.rnn_cell}')
    return self.output(out), OptimizerStates([s_0, s_1])

  def meta_optimize(self, args, meta_optimizer, data, model_cls, optim_it, unroll,
                    out_mul, writer, mode='train'):
    assert mode in ['train', 'valid', 'test']
    self.set_mode(mode)
    use_indexer = False

    params = C(model_cls()).params
    states = C(OptimizerStates.initial_zeros(
        size=(self.n_layers, len(params), self.hidden_sz),
        rnn_cell=self.rnn_cell))

    result_dict = ResultDict()
    unroll_losses = 0
    walltime = Walltime()
    update = C(torch.zeros(params.size().flat()))

    if use_indexer:
      batch_arg = BatchManagerArgument(
          params=params,
          states=StatesSlicingWrapper(states),
          updates=update,
      )

    iter_pbar = tqdm(range(1, optim_it + 1), 'inner_train')
    for iter in iter_pbar:

      with WalltimeChecker(walltime):
        model_train = C(model_cls(params=params.detach()))
        train_nll, train_acc = model_train(*data['in_train'].load())
        train_nll.backward()
        if model_train.params.flat.grad is not None:
          g = model_train.params.flat.grad
        else:
          g = model_train._grad2params(model_train.params)
        assert g is not None

        if use_indexer:
          # This indexer was originally for outer-level batch split
          #  in case that the whold parameters do not fit in the VRAM.
          # Which is not good, unless unavoidable, since you cannot avoid
          #  additional time cost induced by the serial computation.
          batch_arg.update(BatchManagerArgument(
              grad=g).immutable().volatile())
          indexer = DefaultIndexer(input=g, shuffle=False)
          batch_manager = OptimizerBatchManager(
              batch_arg=batch_arg, indexer=indexer)
          ######################################################################
          for get, set in batch_manager.iterator():
            updates, new_states = self(get.grad().detach(), get.states())
            set.states(new_states)
            set.params(get.params() + updates * out_mul)
            if not mode == 'train':
              set.updates(updates)
          ######################################################################
          batch_arg = batch_manager.batch_arg_to_set
          params = batch_arg.params
        else:
          updates, states = self(g.detach(), states)
          updates = params.new_from_flat(updates)
          params = params + updates * out_mul
          #params = params - params.new_from_flat(g) * 0.1

      with WalltimeChecker(walltime if mode == 'train' else None):
        model_test = C(model_cls(params=params))
        test_nll, test_acc = model_test(*data['in_test'].load())
        if mode == 'train':
          unroll_losses += test_nll
          if iter % unroll == 0:
            meta_optimizer.zero_grad()
            unroll_losses.backward()
            nn.utils.clip_grad_value_(self.parameters(), 0.01)
            meta_optimizer.step()
            unroll_losses = 0

      with WalltimeChecker(walltime):
        if not mode == 'train' or iter % unroll == 0:
          if use_indexer:
            batch_arg.detach_()
          else:
            params.detach_()
            states.detach_()
        # model.params = model.params.detach()
        # states = states.detach()
      result = dict(
        train_nll=train_nll.tolist(),
        test_nll=test_nll.tolist(),
        train_acc=train_acc.tolist(),
        test_acc=test_acc.tolist(),
        walltime=walltime.time,
      )
      result_dict.append(result)
      log_pbar(result, iter_pbar)

    return result_dict, params
