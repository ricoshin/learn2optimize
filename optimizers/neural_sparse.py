import numpy as np
import torch
import torch.nn as nn
from utils import utils
from models.model_helpers import ParamsIndexTracker
from optimizers.optim_helpers import (GradientPreprocessor, LSTMStates,
                                      OptimizerStates, StatesSlicingWrapper,
                                      OptimizerParams, BatchManagerArgument,
                                      OptimizerBatchManager, TopKIndexer,
                                      DefaultIndexer)
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
    self.params_tracker = ParamsIndexTracker(n_tracks=10)

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

  def meta_optimize(self, meta_optimizer, data_cls, model_cls, optim_it, unroll,
                    out_mul, should_train=True):
    data = data_cls(training=should_train)
    model = C(model_cls())
    optimizer_states = C(OptimizerStates.initial_zeros(
        size=(self.n_layers, len(model.params), self.hidden_sz),
        rnn_cell=self.rnn_cell))

    if should_train:
      self.train()
    else:
      self.eval()

    result_dict = ResultDict()
    walltime = 0
    unroll_losses = 0
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
      # entire_loop.touch()
      # interval.touch()
      data_ = data.sample()
      if should_train:
        model = C(model_cls(params=batch_arg.params))
        loss = model(*data_)
        unroll_losses += loss
      # print("\n[1] : "+str(interval.touch('interval', False)))
      # model_ff.touch()
      model_detached = C(model_cls(params=batch_arg.params.detach()))
      loss_detached = model_detached(*data_)
      # model_ff.touch('interval', True)
      if should_train:
        assert loss == loss_detached
      # model_bp.touch()
      loss_detached.backward()
      # model_bp.touch('interval', True)
      # interval.touch()
      assert model_detached.params.flat.grad is not None
      grad = model_detached.params.flat.grad
      batch_arg.update(BatchManagerArgument(grad=grad).immutable().volatile())
      iter_pbar.set_description(f'optim_iteration[loss:{loss_detached}]')
      # print("\n[2-1] : "+str(interval.touch('interval', False)))
      ##########################################################################
      # indexer = DefaultIndexer(input=grad, shuffle=False)
      indexer = TopKIndexer(input=grad, k_ratio=0.1, random_k_mode=False)# bound=model_dt.params.size().bound_layerwise())
      #indexer = DefaultIndexer(input=grad)
      batch_manager = OptimizerBatchManager(
        batch_arg=batch_arg, indexer=indexer)
      # print("\n[2-2] : "+str(interval.touch('interval', False)))
      # optim_ff.touch()
      # optim_ff_1.touch()
      for get, set in batch_manager.iterator():

        # optim_ff_1.touch('interval', True)
        # optim_ff_2.touch()

        # optim_ff_2.touch('interval', True)
        # optim_ff_3.touch()
        updates, new_states = self(get.grad().detach(),  get.states())
        # optim_ff_3.touch('interval', True)
        # optim_ff_4.touch()
        updates = updates * out_mul
        set.states(new_states)
        set.params(get.params() + updates)
        if not should_train:
          set.updates(updates)
        # optim_ff_4.touch('interval', True)
      # optim_ff.touch('interval', True)
      ##########################################################################
      # interval.touch()
      batch_arg = batch_manager.batch_arg_to_set
      # print("\n[3] : "+str(interval.touch('interval', False)))
      if should_train and iteration % unroll == 0:

        # optim_bp.touch()
        meta_optimizer.zero_grad()
        unroll_losses.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 10)
        meta_optimizer.step()
        unroll_losses = 0
        # optim_bp.touch('interval', True)
      # interval.touch()
      if not should_train or iteration % unroll == 0:
        batch_arg.detach_()
      # print("\n[4] : "+str(interval.touch('interval', False)))

      # interval.touch()
        # model.params = model.params.detach()
        # optimizer_states = optimizer_states.detach()
      walltime += iter_watch.touch('interval')
      result_dict.append(loss=loss_detached)
      # print("\n[5] : "+str(interval.touch('interval', False)))
      if not should_train:
        result_dict.append(
          walltime=iter_watch.touch(),
          **self.params_tracker(
            grad=grad,
            update=batch_arg.updates,
          )
        )
      # entire_loop.touch('interval', True)
    return result_dict
