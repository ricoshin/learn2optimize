import numpy as np
import torch
import torch.nn as nn
from utils import utils
from models.model_helpers import ModelParamsIndexTracker
from optimizers.optim_helpers import (GradientPreprocessor, LSTMStates,
                                      OptimizerStates, StatesSlicingWrapper,
                                      OptimizerParams, BatchManagerArgument,
                                      OptimizerBatchManager)
from tqdm import tqdm
from utils.result import ResultDict
from utils.torchviz import make_dot


C = utils.getCudaManager('default')


class Optimizer(nn.Module):
  def __init__(self, preproc=False, hidden_sz=20, preproc_factor=10.0):
    super().__init__()
    self.hidden_sz = hidden_sz
    self.n_layers = 2  # TODO: fix later!
    if preproc:
      self.preproc = GradientPreprocessor()
      self.recurs = nn.LSTMCell(2, hidden_sz)
    else:
      self.recurs = nn.LSTMCell(1, hidden_sz)
    self.recurs2 = nn.LSTMCell(hidden_sz, hidden_sz)
    self.output = nn.Linear(hidden_sz, 1)
    self.params_tracker = ModelParamsIndexTracker(n_tracks=100)

  def forward(self, inp, states):
    if self.preproc:
      inp = C(self.preproc(inp))
    s_0 = LSTMStates(*self.recurs(inp, (states[0].h, states[0].c)))
    s_1 = LSTMStates(*self.recurs2(s_0.h, (states[1].h, states[1].c)))
    return self.output(s_1.h), OptimizerStates([s_0, s_1])

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
        (self.n_layers, len(model.params), self.hidden_sz)))

    if should_train:
      self.train()
    else:
      self.eval()

    result_dict = ResultDict()
    unroll_losses = 0
    update = C(torch.zeros(model.params.size().flat))
    iter_pbar = tqdm(range(1, optim_it + 1), 'optim_iteration')
    iter_watch = utils.StopWatch('optim_iteration')

    batch_arg = BatchManagerArgument(
      params=model.params,
      states=StatesSlicingWrapper(optimizer_states),
      updates=update,
    )

    for iteration in iter_pbar:
      data_ = data.sample()
      loss = model(*data_)
      unroll_losses += loss
      model_dt = C(model_cls(params=batch_arg.params.detach()))
      model_dt_loss = model_dt(*data_)
      assert loss == model_dt_loss
      model_dt_loss.backward()
      assert model_dt.params.flat.grad is not None
      grad = model_dt.params.flat.grad
      batch_arg.update(BatchManagerArgument(grad=grad).immutable().volatile())
      iter_pbar.set_description(f'optim_iteration[loss:{loss}]')

      ##########################################################################
      batch_manager = OptimizerBatchManager(batch_arg=batch_arg)

      for get, set in batch_manager.iterator():

        updates, new_states = self(get.grad.detach(), get.states)
        set.states(new_states)
        set.params(get.params + updates * out_mul)
        if not should_train:
          set.updates(updates)
      ##########################################################################
      batch_arg = batch_manager.batch_arg_to_set
      if should_train and iteration % unroll == 0:
        meta_optimizer.zero_grad()
        unroll_losses.backward()
        meta_optimizer.step()

      if not should_train or iteration % unroll == 0:
        unroll_losses = 0
        batch_arg.detach_()
        # model.params = model.params.detach()
        # optimizer_states = optimizer_states.detach()
      model = C(model_cls(params=batch_arg.params))
      result_dict.append(loss=loss)
      if not should_train:
        result_dict.append(
          walltime=iter_watch.touch(),
          **self.params_tracker(
            grad=grad,
            update=batch_arg.updates,
          )
        )
    return result_dict
