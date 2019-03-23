import math
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from utils import utils
from models.model_helpers import ParamsIndexTracker
from optimizers.optim_helpers import  (GradientPreprocessor, LSTMStates,
                                      OptimizerStates, StatesSlicingWrapper,
                                      OptimizerParams, OptimizerBatchManager)
from utils.result import ResultDict

C = utils.getCudaManager('default')


class GridUnitGenerator(nn.Module):
  def __init__(self, hidden_sz=20, n_layers=2, preproc=False):
    super().__init__()
    self.hidden_sz = hidden_sz
    self.n_layers = n_layers
    if preproc:
      self.preproc = GradientPreprocessor()
      self.recurs = nn.LSTMCell(2, hidden_sz)
    else:
      self.recurs = nn.LSTMCell(1, hidden_sz)
    self.recurs2 = nn.LSTMCell(hidden_sz, hidden_sz)
    self.output = nn.Linear(hidden_sz, 1)

  def forward(self, inp, states):
    if self.preproc:
      inp = C(self.preproc(inp))
    s_0 = LSTMStates(*self.recurs(inp, (states[0].h, states[0].c)))
    s_1 = LSTMStates(*self.recurs2(s_0.h, (states[1].h, states[1].c)))
    return self.output(s_1.h), OptimizerStates([s_0, s_1])


class EvenLinearGridMaker(nn.Module):
  def __init__(self, n_grid):
    super().__init__()
    self.n_grid = n_grid

  def forward(self, unit, params):
    even_scale = C(torch.FloatTensor([i for i in range(1, self.n_grid + 1)]))
    even_scale = even_scale.unsqueeze(0).expand(-1, self.n_grid).detach()
    # [n_param, n_grid]
    unit = unit.expand(-1, self.n_grid)
    params = params.expand(-1, self.n_grid)
    grid_rel = unit * even_scale
    grid_abs = params.detach() + grid_rel
    return grid_abs, grid_rel # [n_param, n_grid]


class LossLandscapeObserver(nn.Module):
  def __init__(self, eps=1e-6):
    super().__init__()
    self.eps = eps

  def forward(self, coord_abs, model_cls, target):
    _, n_coord = coord_abs.size() # [n_params, n_coord]
    losses = []
    for i in range(n_coord):
      optimizee = C(model_cls())
      optimizee.params.flat = coord_abs[:, i].view(-1, 1)
      losses.append(optimizee(target))
    losses = torch.stack(losses) # [n_coord]
    # normalize losses to the range of [0, 1]
    min, max = torch.min(losses), torch.max(losses)
    losses_normal = (losses - min) / (max - min + self.eps)
    return losses_normal


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
    # self.attn = nn.Sequential(
    #   nn.Linear(n_top + n_bottom + 1, hidden_sz),
    #   nn.LeakyReLU(),
    #   nn.Linear(hidden_sz, 1),
    #   nn.Softmax(0), # [n_coord, 1]
    # )
    self.linear_1 = nn.Linear(n_top + n_bottom + 1, hidden_sz)
    self.linear_2 = nn.Linear(hidden_sz, n_top + n_bottom)

  def forward(self, coord_rel, losses):
    # coord_rel: [n_params, n_coord] / losses: [n_coord]
    # sample n_top & n_bottom coordinates in order of variance
    _, update_id = coord_var = coord_rel.var(1).sort(descending=True)
    coord_top = coord_rel.index_select(0, update_id[self.n_top:])
    coord_bottom = coord_rel.index_select(0, update_id[:-self.n_bottom])
    coord_sampled = torch.cat([coord_top, coord_bottom], 0)
    # normalize grid with the average of l-2 norm
    # coord_sampled: [n_top+n_bottom, n_coord]
    coord_norm_sum = coord_sampled.norm(p=2, dim=0).sum().detach()
    coord_normal = coord_sampled.div(coord_norm_sum.expand_as(coord_sampled))
    coordwise_inp = torch.cat([coord_normal, losses.view(1, -1)], 0).t()
    # coordwise_inp: [n_coord, n_top + n_bottom + 1(loss)]
    # attn = self.attn(coordwise_inp) # [n_coord, 1]

    # deep set
    coordwise_h = torch.tanh(self.linear_1(coordwise_inp)) # [n_coord, hidden_sz]
    update_selected = self.linear_2(coordwise_h.sum(0)) # [1, n_top + n_bottom]
    update_all = C(torch.zeros([coord_rel.size(0), 1]))
    update_all.index_add_(0, update_id, update_selected.unsqueeze(1))
    update_all = update_all.mul(coord_norm_sum.expand_as(update_all))
    return update_all


class Optimizer(nn.Module):
  def __init__(self, preproc=False, hidden_sz=20, n_coord=5):
    super().__init__()
    self.hidden_sz = hidden_sz
    self.n_coord = n_coord
    self.n_layers = 2 # TODO: fix later!
    self.unit_maker = GridUnitGenerator(hidden_sz, 2, preproc) # fix n_layers
    self.grid_maker = EvenLinearGridMaker(n_coord)
    self.loss_observer = LossLandscapeObserver()
    self.step_maker = CoordVarAwareStepPredictor(hidden_sz, 10, 0.5)

  @property
  def params(self):
    return OptimizerParams.from_optimizer(self)

  @params.setter
  def params(self, optimizer_params):
    optimizer_params.to_optimizer(self)

  def meta_optimize(self, meta_optimizer, loss_cls, model_cls, optim_it, unroll,
                    out_mul, should_train=True):
    target = loss_cls(training=should_train)
    optimizee = C(model_cls())
    optimizer_states = OptimizerStates.zero_states(
        self.n_layers, len(optimizee.params), self.hidden_sz)

    if should_train:
      self.train()
    else:
      self.eval()
      coordinate_tracker = ParamsIndexTracker(
        n_params=len(optimizee.params), n_tracks=30)

    result_dict = ResultDict()
    unroll_losses = 0
    iter_pbar = tqdm(range(1, optim_it + 1), 'optim_iteration')
    iter_watch = utils.StopWatch('optim_iteration')

    for iteration in iter_pbar:
      loss = optimizee(target)
      unroll_losses += loss
      loss.backward(retain_graph=should_train)
      iter_pbar.set_description(f'optim_iteration[loss:{loss}]')
      unit_size = torch.Size([len(optimizee.params), 1])
      grid_size = torch.Size([len(optimizee.params), self.n_coord])
      batch_manager = OptimizerBatchManager(
        params=optimizee.params,
        states=StatesSlicingWrapper(optimizer_states),
        units=OptimizerBatchManager.placeholder(unit_size),
        grid_abs=OptimizerBatchManager.placeholder(grid_size),
        grid_rel=OptimizerBatchManager.placeholder(grid_size),
        batch_size=30000,
      )
      ##########################################################################
      for get, set in batch_manager.iterator():
        units, new_states = self.unit_maker(
          C.detach(get.params.grad), get.states.clone())
        units = units * out_mul / self.n_coord / 2
        grid_abs, grid_rel = self.grid_maker(units, get.params)
        set.states(new_states)
        set.units(units)
        set.grid_abs(grid_abs)
        set.grid_rel(grid_rel)
      ##########################################################################

      landscape = self.loss_observer(batch_manager.grid_abs, model_cls, target)
      updates = self.step_maker(batch_manager.grid_rel, landscape)
      optimizee.params.add_flat_(updates)

      if should_train and iteration % unroll == 0:
        meta_optimizer.zero_grad()
        unroll_losses.backward()
        meta_optimizer.step()

      if not should_train or iteration % unroll == 0:
        unroll_losses = 0
        optimizee.params = optimizee.params.detach()
        optimizer_states = optimizer_states.detach()

      optimizee = C(model_cls(optimizee.params))
      result_dict.append(loss=loss)
      if not should_train:
        result_dict.append(
            grid=coordinate_tracker(batch_manager.units),
            update=coordinate_tracker(updates),
            walltime=iter_watch.touch(),
        )
    return result_dict
