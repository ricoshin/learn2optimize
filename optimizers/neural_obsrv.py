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


class SearchPointSampler(nn.Module):
  def __init__(self, hidden_sz=20, n_layers=2, preproc=False):
    super().__init__()
    self.hidden_sz = hidden_sz
    self.n_layers = n_layers
    if preproc:
      self.preproc = GradientPreprocessor()
      self.recurs = nn.LSTMCell(2, hidden_sz)
    else:
      self.recurs = nn.LSTMCell(2, hidden_sz)
    self.recurs2 = nn.LSTMCell(hidden_sz, hidden_sz)
    self.mu_logvar = nn.Linear(hidden_sz, 2)
    self._mu = None
    self._sigma = None
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

  def forward(self, g_cur, u_prev, states, n_points):
    if self.preproc:
      #g_cur = C(self.preproc(g_cur))
      u_prev = C(self.preproc(u_prev))

    #inp = torch.cat([g_cur, u_prev], dim=1) # [n_params, 4 or 2]
    inp = u_prev
    s_0 = LSTMStates(*self.recurs(inp, (states[0].h, states[0].c)))
    s_1 = LSTMStates(*self.recurs2(s_0.h, (states[1].h, states[1].c)))
    mu_logvar = self.mu_logvar(s_1.h) # [n_params, 2]
    self._mu = mu_logvar[:, 0].unsqueeze(1)
    self._sigma = self.softplus(mu_logvar[:, 1]).unsqueeze(1) #.mul(0.5).exp_().unsqueeze(1)
    #self._mu, self._sigma = mu_logvar[0], mu_logvar[1].mul(0.5).exp_()
    # mu: [n_params, 1] / sigma: [n_params, 1]
    points_size = [n_points, *self._sigma.size()]
    eps = torch.tensor(self._sigma.data.new(size=points_size).normal_())
    points_rel = (self._mu + self._sigma * eps)
    # points: [n_points, n_params, 1]
    return points_rel, OptimizerStates([s_0, s_1])


class LossLandscapeObserver(nn.Module):
  def __init__(self, eps=1e-6):
    super().__init__()
    self.eps = eps

  def forward(self, points_abs, model_cls, data):
    # points_abs: [n_points, n_params, 1]
    losses = []
    for i in range(points_abs.size(0)):
      model = C(model_cls())
      model.params.flat = points_abs[i] # [n_params, 1]
      losses.append(model(*data))
    losses = torch.stack(losses) # [n_points]
    # normalize losses to the range of [0, 1]
    min, max = torch.min(losses), torch.max(losses)
    losses_normal = (losses - min) / (max - min + self.eps)
    return losses_normal


class MinimumPointSelector(nn.Module):
  def __init__(self, temp=100):
    super().__init__()
    self.softmax = nn.Softmax(dim=0)
    self.temp = temp
    self.softmin = lambda x: self.softmax(-x*temp)

  def forward(self, points_rel, losses):
    # points_rel: [n_points, n_params, 1] / losses: [n_points]
    attention = self.softmin(losses).view(-1, 1, 1)
    # attention:  [n_points, n_params, 1]
    points_attn = points_rel.mul(attention.expand_as(points_rel))
    # points_attn: [n_points, n_params, 1]
    point = points_attn.sum(0)
    return point # point: [n_params, 1]



class CoordVarAwareStepPredictor(nn.Module):
  """NOTE: This class expects single-batch input but this also could be done by
  using randomly sampled multi-batch with index permutation and average all the
  steps that has been predicted from each minibatches."""
  def __init__(self, hidden_sz=20, n_coord=10, top_ratio=0.5):
    super().__init__()
    n_top = math.ceil(n_coord * top_ratio)
    n_bottom = n_coord - n_top
    self.n_top = n_top
    self.n_bottom = n_bottom
    self.hidden_sz = hidden_sz
    self.linear_1 = nn.Linear(n_top + n_bottom + 1, hidden_sz)
    self.linear_2 = nn.Linear(hidden_sz, n_top + n_bottom)

  def forward(self, points_rel, losses):
    # points_rel: [n_params, n_coord] / losses: [n_coord]
    # sample n_top & n_bottom coordinates in order of variance
    _, update_id = coord_var = points_rel.var(1).sort(descending=True)
    coord_top = points_rel.index_select(0, update_id[self.n_top:])
    coord_bottom = points_rel.index_select(0, update_id[:-self.n_bottom])
    coord_sampled = torch.cat([coord_top, coord_bottom], 0)
    # normalize grid with the average of l-2 norm
    # coord_sampled: [n_top+n_bottom, n_coord]
    coord_norm_sum = coord_sampled.norm(p=2, dim=0).sum().detach()
    coord_normal = coord_sampled.div(coord_norm_sum.expand_as(coord_sampled))
    coordwise_inp = torch.cat([coord_normal, losses.view(1, -1)], 0).t()
    # coordwise_inp: [n_coord, n_top + n_bottom + 1(loss)]
    coordwise_h = torch.tanh(self.linear_1(coordwise_inp)) # [n_coord, hidden_sz]
    update_selected = self.linear_2(coordwise_h.sum(0)) # [1, n_top + n_bottom]
    update_all = C(torch.zeros([points_rel.size(0), 1]))
    update_all.index_add_(0, update_id, update_selected.unsqueeze(1))
    update_all = update_all.mul(coord_norm_sum.expand_as(update_all))
    return update_all


class Optimizer(nn.Module):
  def __init__(self, preproc=False, hidden_sz=20, n_points=10):
    super().__init__()
    self.hidden_sz = hidden_sz
    self.n_points = n_points
    self.n_layers = 2 # TODO: fix later!
    self.point_sampler = SearchPointSampler(hidden_sz, 2, preproc)
    self.loss_observer = LossLandscapeObserver()
    self.point_selector = MinimumPointSelector()
    self.params_tracker = ParamsIndexTracker(n_tracks=100)

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
    states = C(OptimizerStates.initial_zeros(
      (self.n_layers, len(model.params), self.hidden_sz)))

    if should_train:
      self.train()
    else:
      self.eval()

    result_dict = ResultDict()
    model_dt = None
    model_losses = 0
    #grad_real_prev = None
    iter_pbar = tqdm(range(1, optim_it + 1), 'optim_iteration')
    iter_watch = utils.StopWatch('optim_iteration')

    batch_arg = BatchManagerArgument(
      params=model.params,
      states=StatesSlicingWrapper(states),
      update_prev=C(torch.zeros(model.params.size().flat())),
      mu=C(torch.zeros(model.params.size().flat())),
      sigma=C(torch.zeros(model.params.size().flat())),
    )

    for iteration in iter_pbar:
      data_ = data.sample()
      model_loss = model(*data_)
      model_losses += model_loss

      # if iteration == 21:
      #   import pdb; pdb.set_trace()
      try:
        make_dot(model_losses)
      except:
        import pdb; pdb.set_trace()
      # if iteration == 1 or iteration == 21:
      #   import pdb; pdb.set_trace()


      model_dt = C(model_cls(params=model.params.detach()))
      model_loss_dt = model_dt(*data_)
      # model.params.register_grad_cut_hook_()
      model_loss_dt.backward()
      assert model_dt.params.flat.grad is not None
      grad_cur = model_dt.params.flat.grad
      # model.params.remove_grad_cut_hook_()
      batch_arg.update(BatchManagerArgument(grad_cur=grad_cur))
      # NOTE: read_only, write_only

      ########################################################################
      batch_manager = OptimizerBatchManager(batch_arg=batch_arg)

      for get, set in batch_manager.iterator():
        grad_cur = get.grad_cur.detach() # free-running
        update_prev = get.update_prev.detach()
        points_rel, states = self.point_sampler(
          grad_cur, update_prev, get.states, self.n_points)
        points_rel = points_rel * out_mul
        points_abs = get.params.detach() + points_rel # NOTE: check detach
        losses = self.loss_observer(points_abs, model_cls, data_)
        update = self.point_selector(points_rel, losses)
        set.params(get.params + update)
        set.states(states)
        set.update_prev(update)
        set.mu(self.point_sampler.mu)
        set.sigma(self.point_sampler.sigma)
      ##########################################################################

      batch_arg = batch_manager.batch_arg_to_set
      iter_pbar.set_description(
        'optim_iteration[loss:{:10.6f}]'.format(model_loss.tolist()))

      if should_train and iteration % unroll == 0:
        meta_optimizer.zero_grad()
        try:
          model_losses.backward()
        except:
          import pdb; pdb.set_trace()
        meta_optimizer.step()

      if not should_train or iteration % unroll == 0:
        model_losses = 0
        batch_arg.detach_()
        # batch_arg.params.detach_()
        # batch_arg.states.detach_()
        # NOTE: just do batch_arg.detach_()

      model = C(model_cls(params=batch_arg.params))
      result_dict.append(
        step_num=iteration,
        loss=model_loss,
      )

      if not should_train:
        result_dict.append(
          walltime=iter_watch.touch(),
          **self.params_tracker(
            grad=grad_cur,
            update=batch_arg.update_prev,
            mu=batch_arg.mu,
            sigma=batch_arg.sigma,
          )
        )
    return result_dict
