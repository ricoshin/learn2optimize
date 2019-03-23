import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from utils import utils
from optimizers.optim_helpers import (LSTMStates, OptimizerStates,
                                OptimizerParams, OptimizerBatchManager)
from utils.result import ResultDict

C = utils.getCudaManager('default')


class LinearNavigator(nn.Module):
  def __init__(self, preproc=False, n_navi_step=5, hidden_sz=20,
               preproc_factor=10.0):
    super().__init__()
    self.hidden_sz = hidden_sz
    self.n_layers = 2 # TODO: fix later!
    if preproc:
      self.recurs = nn.LSTMCell(2, hidden_sz)
    else:
      self.recurs = nn.LSTMCell(1, hidden_sz)
    self.recurs2 = nn.LSTMCell(hidden_sz, hidden_sz)
    self.output = nn.Linear(hidden_sz, 1)
    self.preproc = preproc
    self.preproc_factor = preproc_factor
    self.preproc_threshold = np.exp(-preproc_factor)
    self.n_navi_step = n_navi_step

  def forward(self, inp, hidden, cell):
    if self.preproc:
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
    hidden0, cell0 = self.recurs(inp, (hidden[0], cell[0]))
    hidden1, cell1 = self.recurs2(hidden0, (hidden[1], cell[1]))
    output = self.output(hidden1)
    step_scale = C(torch.FloatTensor(
      [i for i in range(1, self.n_navi_step + 1)]))
    output = output.expand(-1, self.n_navi_step)
    output = output * step_scale.expand_as(output)
    return output, (hidden0, hidden1), (cell0, cell1)


class StepPredictor(nn.Module):
  def __init__(self, hidden_size=20, n_navi_step=5):
    super().__init__()
    self.linear_1 = nn.Linear(n_navi_step, hidden_size)
    self.linear_2 = nn.Linear(hidden_size, n_navi_step)

  def forward(self, navi_step_loss):
    attn_w = F.softmax(self.linear_2(self.linear_1(navi_step_loss)), dim=0)
    step_scale = C(torch.FloatTensor(
      [i for i in range(1, navi_step_loss.size(0) + 1)]))
    return torch.sum(attn_w * step_scale)


class Optimizer(nn.Module):
  def __init__(self, preproc=False, n_navi_step=5, hidden_sz=20,
               preproc_factor=10.0):
    self.hidden_sz = hidden_sz
    self.n_navi_step = n_navi_step
    self.navi = LinearNavigator(preproc, n_navi_step, hidden_sz, preproc_factor)
    self.step = StepPredictor(hidden_sz, n_navi_step)

  @property
  def params(self):
    return OptimizerParams.from_optimizer(self)

  @params.setter
  def params(self, optimizer_params):
    optimizer_params.to_optimizer(self)

  def parameters(self):
    return chain(self.navi.parameters(), self.step.parameters())

  def forward(self, inp, states):
    pass


  def meta_optimize(self, meta_optimizer, data_cls, model_cls, optim_it, unroll,
                    out_mul, should_train=True):
    if should_train:
      self.train()
    else:
      self.eval()
      unroll = 1

    target = data_cls(training=should_train)
    optimizee = C(model_cls())
    n_params = 0

    for p in optimizee.parameters():
      n_params += int(np.prod(p.size()))

    update_track_num = 10
    if update_track_num > n_params:
      update_track_num = n_params
    update_track_id = np.random.randint(0, n_params, update_track_num)
    updates_tracks = []
    grids_tracks = []

    hidden_states = [
      C(torch.zeros(n_params, self.hidden_sz)) for _ in range(2)]
    cell_states = [
      C(torch.zeros(n_params, self.hidden_sz)) for _ in range(2)]
    all_losses_ever = []
    timestamps = []

    if should_train:
      meta_optimizer.zero_grad()
    all_losses = None

    ############################################################################
    iter_pbar = tqdm(range(1, optim_it + 1), 'optim_iteration')
    iter_watch = utils.StopWatch('optim_iteration')
    for iteration in iter_pbar:
      #watch.go('forward')
      loss = optimizee(target)
      #forward_time.append(watch.stop('forward'))
      iter_pbar.set_description(f'optim_iteration[loss:{loss}]')

      if all_losses is None:
        all_losses = loss
      else:
        all_losses += loss

      all_losses_ever.append(loss.data.cpu().numpy())
      #watch.go('backward')
      loss.backward(retain_graph=should_train)
      #backward_time.append(watch.stop('backward'))

      offset = 0
      navi_step_params = [dict() for _ in range(self.n_navi_step)]
      navi_step_optimizee = []
      navi_step_loss = []
      navi_step_delta = []
      result_params = {}
      hidden_states2 = [
        C(torch.zeros(n_params, self.hidden_sz)) for _ in range(2)]
      cell_states2 = [
        C(torch.zeros(n_params, self.hidden_sz)) for _ in range(2)]

      ##########################################################################
      # torch.Parameter-wise linear navigation: get linear step points.
      for name, p in optimizee.all_named_parameters():
        cur_sz = int(np.prod(p.size()))
        # We do this so the gradients are disconnected from the graph
        #  but we still get gradients from the rest
        gradients = C.detach(p.grad.view(cur_sz, 1))
        navi_step, new_hidden, new_cell = self.navi(
          gradients,
          [h[offset:offset+cur_sz] for h in hidden_states],
          [c[offset:offset+cur_sz] for c in cell_states],
        )
        for i in range(len(new_hidden)):
          hidden_states2[i][offset:offset+cur_sz] = new_hidden[i]
          cell_states2[i][offset:offset+cur_sz] = new_cell[i]
        offset += cur_sz

        for i in range(self.n_navi_step):
          navi_delta = navi_step[:, i].view(*p.size()) * out_mul / self.n_navi_step / 2
          if i == 0:
            navi_step_delta.append(navi_delta)
          navi_step_params[i][name] = p + navi_delta
          navi_step_params[i][name].retain_grad() # non-leaf tensor

      ##########################################################################
      # Evalute loss at every linear navigation points
      #   and choose a single point(proper step size in linear grid) between them.

      for i in range(self.n_navi_step):
        optimizee = C(model_cls(**navi_step_params[i]))
        assert len(list(optimizee.all_named_parameters()))
        navi_step_loss.append(optimizee(target))

      navi_step_loss = torch.stack(navi_step_loss)
      min = torch.min(navi_step_loss)
      max = torch.max(navi_step_loss)
      navi_step_loss = (navi_step_loss - min) / (max - min + 1e-10)
      step_size = self.step(navi_step_loss)

      ##########################################################################

      result_params = {}
      updates_track = []
      grids_track = []
      pairs = zip(optimizee.all_named_parameters(), navi_step_delta)
      for (name, param), delta in pairs:
        update = delta * step_size
        grid = delta * self.n_navi_step
        if step_size > self.n_navi_step:
          import pdb; pdb.set_trace()
        if step_size < 0:
          import pdb; pdb.set_trace()
        result_params[name] = param + update #* out_mul
        result_params[name].retain_grad()
        updates_track.append(update)
        grids_track.append(grid)
      updates_track = torch.cat(updates_track, 0).data.cpu().numpy()
      updates_track = np.take(updates_track, update_track_id)
      updates_tracks.append(updates_track)
      grids_track = torch.cat(grids_track, 0).data.cpu().numpy()
      grids_tracks.append(grids_track)


      if iteration % unroll == 0:
        if should_train:
          meta_optimizer.zero_grad()
          all_losses.backward()
          meta_optimizer.step()

        all_losses = None

        detached_params = {k: C.detach(v) for k, v in result_params.items()}
        optimizee = C(model_cls(**detached_params))
        hidden_states = [C.detach(s) for s in hidden_states2]
        cell_states = [C.detach(s) for s in cell_states2]

      else:
        optimizee = C(model_cls(**result_params))
        assert len(list(optimizee.all_named_parameters()))
        hidden_states = hidden_states2
        cell_states = cell_states2

      timestamps.append(iter_watch.touch())
    return all_losses_ever, updates_tracks, grids_tracks, timestamps
