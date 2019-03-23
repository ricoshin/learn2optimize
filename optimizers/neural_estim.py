import numpy as np
import torch
import torch.nn as nn
from utils import utils
from models.model_helpers import ParamsIndexTracker
from optimizers.optim_helpers import (GradientPreprocessor, LSTMStates,
                                      OptimizerStates, StatesSlicingWrapper,
                                      OptimizerParams, BatchManagerArgument,
                                      OptimizerBatchManager)
from tqdm import tqdm
from utils.result import ResultDict

C = utils.getCudaManager('default')


class Optimizer(nn.Module):
  def __init__(self, preproc=False, hidden_sz=20, preproc_factor=10.0):
    super().__init__()
    self.hidden_sz = hidden_sz
    self.n_layers = 2  # TODO: fix later!
    self.k = 1
    if preproc:
      self.preproc = GradientPreprocessor()
      self.g_rnn = nn.LSTMCell(4, hidden_sz)
      self.u_rnn = nn.LSTMCell(2, hidden_sz)
    else:
      self.g_rnn = nn.LSTMCell(2, hidden_sz)
      self.u_rnn = nn.LSTMCell(1, hidden_sz)
    self.g_rnn2 = nn.LSTMCell(hidden_sz, hidden_sz)
    self.u_rnn2 = nn.LSTMCell(hidden_sz, hidden_sz)
    self.g_output = nn.Linear(hidden_sz, 1)
    self.u_output = nn.Linear(hidden_sz, 1)
    self.mse_loss = nn.MSELoss(reduction='sum')


  def forward(self, g_prev, u_prev, g_states, u_states):
    if self.preproc:
      g_prev = C(self.preproc(g_prev))
      u_prev = C(self.preproc(u_prev))

    g_u_cat = torch.cat([g_prev, u_prev], dim=1) # [n_param, 4 or 2 (g+u)]
    g_s_0 = LSTMStates(*self.g_rnn(g_u_cat, (g_states[0].h, g_states[0].c)))
    g_s_1 = LSTMStates(*self.g_rnn2(g_s_0.h, (g_states[1].h, g_states[1].c)))
    g_cur = self.g_output(g_s_1.h) * 0.1
    g_states_ = OptimizerStates([g_s_0, g_s_1])

    if self.preproc:
      g_cur_p = C(self.preproc(g_cur.detach())) # NOTE: Detach!
    u_s_0 = LSTMStates(*self.u_rnn(g_cur_p, (u_states[0].h, u_states[0].c)))
    u_s_1 = LSTMStates(*self.u_rnn2(u_s_0.h, (u_states[1].h, u_states[1].c)))
    u_cur = self.u_output(u_s_1.h)
    u_states_ = OptimizerStates([u_s_0, u_s_1])
    return g_cur, u_cur, g_states_, u_states_

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
    g_states = C(OptimizerStates.initial_zeros(
        (self.n_layers, len(model.params), self.hidden_sz)))
    u_states = C(OptimizerStates.initial_zeros(
        (self.n_layers, len(model.params), self.hidden_sz)))

    if should_train:
      self.train()
    else:
      self.eval()
      grad_pred_tracker = ParamsIndexTracker(
        name='grad_pred', n_tracks=10, n_params=len(model.params))
      grad_real_tracker = grad_pred_tracker.clone_new_name('grad_real')
      update_tracker = grad_pred_tracker.clone_new_name('update')

    result_dict = ResultDict()
    model_dt = None
    model_losses = 0
    grad_losses = 0
    train_with_lazy_tf = False
    eval_test_grad_loss = True
    grad_prev = C(torch.zeros(model.params.size().flat()))
    update_prev = C(torch.zeros(model.params.size().flat()))
    #grad_real_prev = None
    grad_real_cur = None
    grad_real_prev = C(torch.zeros(model.params.size().flat()))
    iter_pbar = tqdm(range(1, optim_it + 1), 'optim_iteration')
    iter_watch = utils.StopWatch('optim_iteration')

    """grad_real_prev:
        teacher-forcing current input of RNN.
        required during all of training steps + every 'k' test steps.
       grad_real_cur:
        MSE loss target for current output of RNN.
        required during all the training steps only.
    """
    batch_arg = BatchManagerArgument(
      params=model.params,
      grad_prev=grad_prev,
      update_prev=update_prev,
      g_states=StatesSlicingWrapper(g_states),
      u_states=StatesSlicingWrapper(u_states),
      mse_loss=C(torch.zeros(1)),
      #updates=OptimizerBatchManager.placeholder(model.params.flat.size()),
    )

    for iteration in iter_pbar:
      inp, out = data.sample()
      model_loss = model(inp, out)
      model_losses += model_loss
      # if grad_real_cur is not None: # if iteration % self.k == 0
      #   grad_cur_prev = grad_real_cur # back up previous grad
      if (iteration + 1) % self.k == 0 or should_train or eval_test_grad_loss:
        # get grad every step while training
        # get grad one step earlier than teacher-forcing steps while testing
        model_dt = C(model_cls(params=model.params.detach()))
        model_loss_dt = model_dt(inp, out)
        # model.params.register_grad_cut_hook_()
        model_loss_dt.backward()
        # model.params.remove_grad_cut_hook_()
        grad_real_cur = model_dt.params.flat.grad
        assert grad_real_cur is not None
        del model_dt, model_loss_dt

      if should_train or eval_test_grad_loss:
        # mse loss target
        # grad_real_cur is not used as a mse target while testing
        assert grad_real_cur is not None
        batch_arg.update(BatchManagerArgument(
          grad_real_cur=grad_real_cur).immutable().volatile())

      if iteration % self.k == 0 or (should_train and not train_with_lazy_tf):
        # teacher-forcing
        assert grad_real_prev is not None
        batch_arg.update(BatchManagerArgument(
          grad_real_prev=grad_real_prev).immutable().volatile())
        grad_real_prev = None

      ########################################################################
      batch_manager = OptimizerBatchManager(batch_arg=batch_arg)

      for get, set in batch_manager.iterator():
        if iteration % self.k == 0 or (should_train and not train_with_lazy_tf):
          grad_prev = get.grad_real_prev.detach() # teacher-forcing
        else:
          grad_prev = get.grad_prev.detach() # free-running
        update_prev = get.update_prev.detach()
        grad_cur, update_cur, g_states, u_states = self(
          grad_prev, update_prev, get.g_states, get.u_states)
        set.grad_prev(grad_cur)
        set.update_prev(update_cur)
        set.g_states(g_states)
        set.u_states(u_states)
        set.params(get.params + update_cur * out_mul)
        if should_train or eval_test_grad_loss:
          set.mse_loss(self.mse_loss(grad_cur, get.grad_real_cur.detach()))
      ##########################################################################

      batch_arg = batch_manager.batch_arg_to_set
      grad_loss = batch_arg.mse_loss * 100
      grad_losses += grad_loss
      #to_float = lambda x: float(x.data.cpu().numpy())
      iter_pbar.set_description(
        'optim_iteration[loss(model):{:10.6f}/(grad):{:10.6f}]'.format(
          model_loss.tolist(), grad_loss.tolist()[0]))

      if should_train and iteration % unroll == 0:
        meta_optimizer.zero_grad()
        losses = model_losses + grad_losses
        losses.backward()
        meta_optimizer.step()

      if not should_train or iteration % unroll == 0:
        model_losses = 0
        grad_losses = 0
        batch_arg.detach_()
        # batch_arg.params.detach_()
        # batch_arg.g_states.detach_()
        # batch_arg.u_states.detach_()

      model = C(model_cls(params=batch_arg.params))
      result_dict.append(
        step_num=iteration,
        loss=model_loss,
        grad_loss=grad_loss,
      )

      if not should_train:
        result_dict.append(
          walltime=iter_watch.touch(),
          **grad_pred_tracker.track_num,
          **grad_pred_tracker(batch_arg.grad_prev),
          **grad_real_tracker(grad_real_cur),
          **update_tracker(batch_arg.update_prev),
        )

      if (iteration + 1) % self.k == 0 or should_train or eval_test_grad_loss:
        grad_real_prev = grad_real_cur
        grad_real_cur = None

    return result_dict
