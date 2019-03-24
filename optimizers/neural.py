import copy
import math

import numpy as np
import torch
import torch.nn as nn
from utils import utils
from tqdm import tqdm

C = utils.getCudaManager('default')


class Optimizer(nn.Module):
  def __init__(self, preproc=False, hidden_sz=20, preproc_factor=10.0):
    super().__init__()
    self.hidden_sz = hidden_sz
    if preproc:
      self.recurs = nn.LSTMCell(2, hidden_sz)
    else:
      self.recurs = nn.LSTMCell(1, hidden_sz)
    self.recurs2 = nn.LSTMCell(hidden_sz, hidden_sz)
    self.output = nn.Linear(hidden_sz, 1)
    self.preproc = preproc
    self.preproc_factor = preproc_factor
    self.preproc_threshold = np.exp(-preproc_factor)

  def forward(self, inp, hidden, cell):
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
    hidden0, cell0 = self.recurs(inp, (hidden[0], cell[0]))
    hidden1, cell1 = self.recurs2(hidden0, (hidden[1], cell[1]))
    return self.output(hidden1), (hidden0, hidden1), (cell0, cell1)

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
      loss = optimizee(target)
      iter_pbar.set_description(f'optim_iteration[loss:{loss}]')

      if all_losses is None:
        all_losses = loss
      else:
        all_losses += loss

      all_losses_ever.append(loss.data.cpu().numpy())
      loss.backward(retain_graph=should_train)

      import pdb; pdb.set_trace()

      offset = 0
      result_params = {}
      updates_track = []
      hidden_states2 = [
          C(torch.zeros(n_params, self.hidden_sz)) for _ in range(2)]
      cell_states2 = [
          C(torch.zeros(n_params, self.hidden_sz)) for _ in range(2)]

      ##########################################################################
      # named_shape = dict()
      # params_flat = []
      # grads_flat = []
      # for name, p in optimizee.all_named_parameters():
      #   named_shape[name] = p.size()
      #   params_flat.append(p.view(-1, 1))
      #   grads_flat.append(C.detach(p.grad.view(-1, 1)))
      # params_flat = torch.cat(params_flat, 0)
      # grads_flat = torch.cat(grads_flat, 0)
      #
      # batch_size = 10000
      # batch_sizes = []
      # if batch_size > n_params:
      #   batch_size = [n_params]
      # else:
      #   batch_size = [batch_size for _ in range(n_params // batch_size)]
      #   batch_sizes.append(n_params % batch_size)
      #
      # batch_sizes = deque(batch_sizes)
      #
      # offset = 0
      # params_batches = []
      # grads_batches = []
      # for _ in range(len(batch_sz)):
      #   b_size = batch_sz.popleft()
      #   params_batches.append(params_flat[offset:offset+b_size, :])
      #   grads_batches.append(grads_flat[offset:offset+b_size, :])
      #   offset += b_size
      # assert(offset == params_flat.size(0))
      # batches = zip(batch_sz, params_batches, grads_batches)
      ##########################################################################

      for name, p in optimizee.all_named_parameters():
        cur_sz = int(np.prod(p.size()))
        # We do this so the gradients are disconnected from the graph
        #  but we still get gradients from the rest
        gradients = C.detach(p.grad.view(cur_sz, 1))
        updates, new_hidden, new_cell = self(
            gradients,
            [h[offset:offset + cur_sz] for h in hidden_states],
            [c[offset:offset + cur_sz] for c in cell_states]
        )
        for i in range(len(new_hidden)):
          hidden_states2[i][offset:offset + cur_sz] = new_hidden[i]
          cell_states2[i][offset:offset + cur_sz] = new_cell[i]
        result_params[name] = p + updates.view(*p.size()) * out_mul
        result_params[name].retain_grad()
        offset += cur_sz
        updates_track.append(updates)
      ##########################################################################

      updates_track = torch.cat(updates_track, 0)
      updates_track = updates_track.data.cpu().numpy()
      updates_track = np.take(updates_track, update_track_id)
      updates_tracks.append(updates_track)

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
    return all_losses_ever, updates_tracks, timestamps
