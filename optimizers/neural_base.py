import numpy as np
import torch
import torch.nn as nn
from utils import utils
from models.model_helpers import ParamsIndexTracker
from optimizers.optim_helpers import (GradientPreprocessor, LSTMStates,
                                      OptimizerStates, StatesSlicingWrapper,
                                      OptimizerParams, BatchManagerArgument,
                                      OptimizerBatchManager, DefaultIndexer)
from tqdm import tqdm
from utils.result import ResultDict
from utils.torchviz import make_dot


C = utils.getCudaManager('default')


class Optimizer(nn.Module):
  def __init__(
    self, preproc=False, hidden_sz=20, preproc_factor=10.0, rnn_cell='lstm'):
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

  @property
  def params(self):
    return OptimizerParams.from_optimizer(self)

  @params.setter
  def params(self, optimizer_params):
    optimizer_params.to_optimizer(self)

  def meta_optimize(self, meta_optimizer, data, model_cls, optim_it, unroll,
                    out_mul, mode):
    assert mode in ['train', 'valid', 'test']
    if mode == 'train':
      self.train()
      inner_data = data.loaders['inner_train']
      outer_data = data.loaders['inner_valid']
    elif mode == 'valid':
      self.eval()
      inner_data = data.loaders['valid']
      outer_data = None
    elif mode == 'eval':
      self.eval()
      inner_data = data.loaders['test']
      outer_data = None

    model = C(model_cls())
    optimizer_states = C(OptimizerStates.initial_zeros(
        size=(self.n_layers, len(model.params), self.hidden_sz),
        rnn_cell=self.rnn_cell))

    result_dict = ResultDict()
    unroll_losses = 0
    walltime = 0
    update = C(torch.zeros(model.params.size().flat()))

    batch_arg = BatchManagerArgument(
      params=model.params,
      states=StatesSlicingWrapper(optimizer_states),
      updates=update,
    )

    iter_pbar = tqdm(range(1, optim_it + 1), 'optim_iteration')
    iter_watch = utils.StopWatch('optim_iteration')
    for iteration in iter_pbar:
      iter_watch.touch()
      if mode == 'train':
        model = C(model_cls(params=batch_arg.params))
        loss = model(*outer_data.load())
        unroll_losses += loss

      model_detached = C(model_cls(params=batch_arg.params.detach()))
      loss_detached = model_detached(*inner_data.load())
      # if mode == 'train':
      #   assert loss == loss_detached
      loss_detached.backward()
      assert model_detached.params.flat.grad is not None
      grad = model_detached.params.flat.grad
      batch_arg.update(BatchManagerArgument(grad=grad).immutable().volatile())
      iter_pbar.set_description(f'optim_iteration[loss:{loss_detached}]')

      ##########################################################################
      indexer = DefaultIndexer(input=grad, shuffle=False)
      batch_manager = OptimizerBatchManager(
        batch_arg=batch_arg, indexer=indexer)

      for get, set in batch_manager.iterator():
        updates, new_states = self(get.grad().detach(), get.states())
        set.states(new_states)
        set.params(get.params() + updates * 0.1)
        if not mode == 'train':
          set.updates(updates)
      ##########################################################################
      batch_arg = batch_manager.batch_arg_to_set
      # import pdb; pdb.set_trace()
      if mode == 'train' and iteration % unroll == 0:
        meta_optimizer.zero_grad()
        unroll_losses.backward()
        # nn.utils.clip_grad_norm_(self.parameters(), 10)
        meta_optimizer.step()
        unroll_losses = 0

      if not mode == 'train' or iteration % unroll == 0:
        batch_arg.detach_()
        # model.params = model.params.detach()
        # optimizer_states = optimizer_states.detach()
      walltime += iter_watch.touch('interval')
      result_dict.append(loss=loss_detached)
      if not mode == 'train':
        result_dict.append(
          walltime=walltime,
          **self.params_tracker(
            grad=grad,
            update=batch_arg.updates,
          )
        )
    return result_dict
