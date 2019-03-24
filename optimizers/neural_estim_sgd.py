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
from utils.torchviz import make_dot

C = utils.getCudaManager('default')


class FixedDropout(nn.Module):
  def __init__(self, p=0.5):
    super().__init__()
    if p < 0. or p > 1.:
      raise ValueError("dropout probability has to be between 0 and 1, "
                       "but got {}".format(p))
    self.p = p
    self.mask = None

  def reset_mask(self):
    self.mask = None

  def forward(self, input):
    def mask(tensor):
      if self.training:
        if self.mask is not None:
          assert self.mask.size() == tensor.size()
        else:
          self.mask = torch.bernoulli(
            tensor.data.new(tensor.data.size()).fill_(self.p))
        return tensor * self.mask / (1 - self.p + 1e-16)
      else:
        return tensor
    if isinstance(input, OptimizerStates):
      if self.mask is None:
        self.mask = []
        mask_gen = lambda : torch.bernoulli(
          input[0].h.data.new(input[0].h.data.size()).fill_(self.p))
        self.mask = [mask_gen() for _ in range(4)]
      else:
        assert len(self.mask) == 4
      scaler = 1 - self.p + 1e-16
      input[0].h = input[0].h * self.mask[0] * scaler
      input[0].c = input[0].c * self.mask[1] * scaler
      input[1].h = input[1].h * self.mask[2] * scaler
      input[1].c = input[1].c * self.mask[3] * scaler
      return input

    elif isinstance(input, torch.Tensor):
      return mask(input)
    else:
      raise RuntimeError("Unknown input type for FixedDropout()!")


class GradSelector(nn.Module):
  def __init__(self):
    pass

  def forward(self, pred_grad, real_grad):
    # grads size : [n_params, 1]
    pred_grad


class Optimizer(nn.Module):
  def __init__(self, preproc=False, hidden_sz=20, preproc_factor=10.0):
    super().__init__()
    self.hidden_sz = hidden_sz
    self.n_layers = 2  # TODO: fix later!
    self.k = 10
    if preproc:
      self.preproc = GradientPreprocessor()
      self.g_rnn = nn.LSTMCell(6, hidden_sz)
    else:
      self.g_rnn = nn.LSTMCell(3, hidden_sz)
    self.g_rnn2 = nn.LSTMCell(hidden_sz, hidden_sz)
    self.g_output = nn.Linear(hidden_sz, 2)
    self.mse_loss = nn.MSELoss(reduction='sum')
    self.params_tracker = ParamsIndexTracker(n_tracks=100)
    drop_prob = 0.0
    self.ig_drop = FixedDropout(drop_prob)
    self.iu_drop = FixedDropout(drop_prob)
    self.ip_drop = FixedDropout(drop_prob)
    self.i2_drop = FixedDropout(drop_prob)
    self.state_drop = FixedDropout(drop_prob)
    self.output_drop = FixedDropout(drop_prob)
    self.softplus = nn.Softplus()

  @property
  def mu(self):
    if self._mu is None:
      raise RuntimeError("mu can be accessed after .forward()!")
    return self._mu

  @property
  def sigma(self):
    if self._sigma is None:
      raise RuntimeError("sigma can be accessed after .forward()!")
    return self._sigma

  def forward(self, g_prev, u_prev, params, g_states):
    if self.preproc:
      g_prev = C(self.preproc(self.ig_drop(g_prev)))
      u_prev = C(self.preproc(self.iu_drop(u_prev)))
      params = C(self.preproc(self.ip_drop(params)))

    g_u_cat = torch.cat([g_prev, u_prev, params], dim=1) # [n_param, 4 or 2 (g+u)]
    g_s_0 = LSTMStates(*self.g_rnn(g_u_cat, (g_states[0].h, g_states[0].c)))
    g_s_0_h = self.i2_drop(g_s_0.h)
    g_s_1 = LSTMStates(*self.g_rnn2(g_s_0_h, (g_states[1].h, g_states[1].c)))
    mu_logvar = self.g_output(g_s_1.h) * 0.1
    self._mu = mu_logvar[:, 0].unsqueeze(1)
    self._sigma = self.softplus(mu_logvar[:, 1]).unsqueeze(1)
    eps = torch.tensor(self._sigma.data.new(size=self._sigma.size()).normal_())
    g_cur = (self._mu + self._sigma * eps)
    g_states_ = OptimizerStates([g_s_0, g_s_1])
    return self.output_drop(g_cur), self.state_drop(g_states_)

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

    if should_train:
      self.train()
    else:
      self.eval()

    result_dict = ResultDict()
    model_dt = None
    grad_losses = 0
    train_with_lazy_tf = True
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
      mse_loss=C(torch.zeros(1)),
      sigma=C(torch.zeros(model.params.size().flat())),
      #updates=OptimizerBatchManager.placeholder(model.params.flat.size()),
    )

    for iteration in iter_pbar:
      data_ = data.sample()
      # if grad_real_cur is not None: # if iteration % self.k == 0
      #   grad_cur_prev = grad_real_cur # back up previous grad
      if (iteration + 1) % self.k == 0 or should_train or eval_test_grad_loss:
        # get grad every step while training
        # get grad one step earlier than teacher-forcing steps while testing
        model_dt = C(model_cls(params=model.params.detach()))
        model_loss_dt = model_dt(*data_)
        # model.params.register_grad_cut_hook_()
        model_loss_dt.backward()
        # model.params.remove_grad_cut_hook_()
        grad_real_cur = model_dt.params.flat.grad
        assert grad_real_cur is not None

      if should_train or eval_test_grad_loss:
        # mse loss target
        # grad_real_cur is not used as a mse target while testing
        assert grad_real_cur is not None
        batch_arg.update(BatchManagerArgument(
          grad_real_cur=grad_real_cur).immutable())

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
          grad_prev = get.grad_real_prev # teacher-forcing
        else:
          grad_prev = get.grad_prev # free-running
        grad_cur, g_states = self(
          grad_prev.detach(), get.update_prev.detach(),
          get.params.detach(), get.g_states)
        set.grad_prev(grad_cur)
        set.g_states(g_states)
        set.params(get.params - 1.0 * grad_cur.detach())
        set.sigma(self.sigma)
        if should_train or eval_test_grad_loss:
          set.mse_loss(self.mse_loss(grad_cur, get.grad_real_cur.detach()))
      ##########################################################################

      batch_arg = batch_manager.batch_arg_to_set
      grad_real = batch_manager.batch_arg_to_get.grad_real_cur.detach()
      cosim_loss = nn.functional.cosine_similarity(
        batch_arg.grad_prev, grad_real, dim=0)
      grad_loss = batch_arg.mse_loss * 100
      cosim_loss = -torch.log((cosim_loss + 1) * 0.5) * 100
      grad_losses += (grad_loss + cosim_loss)
      #to_float = lambda x: float(x.data.cpu().numpy())
      sigma = batch_arg.sigma.detach().mean()
      iter_pbar.set_description(
        'optim_iteration[loss(model):{:10.6f}/(mse):{:10.6f}/(cosim):{:10.6f}/(sigma):{:10.6f}]'
        ''.format(model_loss_dt.tolist(), grad_loss.tolist()[0], cosim_loss.tolist()[0], sigma.tolist()))

      if should_train and iteration % unroll == 0:
        w_decay_loss = 0
        for p in self.parameters():
          w_decay_loss += torch.sum(p * p)
        w_decay_loss = (w_decay_loss * 0.5) * 1e-4
        total_loss = grad_losses + w_decay_loss
        meta_optimizer.zero_grad()
        total_loss.backward()
        meta_optimizer.step()

      if not should_train or iteration % unroll == 0:
        model_losses = 0
        grad_losses = 0
        batch_arg.detach_()
        # batch_arg.params.detach_()
        # batch_arg.g_states.detach_()
        # batch_arg.u_states.detach_()

      result_dict.append(
        step_num=iteration,
        loss=model_loss_dt,
        grad_loss=grad_loss,
      )

      if not should_train:
        result_dict.append(
          walltime=iter_watch.touch(),
          **self.params_tracker(
            grad_real=grad_real_cur,
            grad_pred=batch_arg.grad_prev,
            update=batch_arg.update_prev,
            sigma=batch_arg.sigma,
          )
        )

      if (iteration + 1) % self.k == 0 or should_train or eval_test_grad_loss:
        grad_real_prev = grad_real_cur
        grad_real_cur = None

    return result_dict
