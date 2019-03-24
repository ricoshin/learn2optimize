import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
debug_sigint = utils.getDebugger('SIGINT')
debug_sigstp = utils.getDebugger('SIGTSTP')


class LSTMCell(nn.Module):

    """
    An implementation of Hochreiter & Schmidhuber:
    'Long-Short Term Memory' cell.
    http://www.bioinf.jku.at/publications/older/2604.pdf

    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):

        hx, cx = hidden

        x = x.view(-1, x.size(1))

        gates = self.x2h(x) + self.h2h(hx)

        gates = gates.squeeze()

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)


        cy = torch.mul(cx, forgetgate) +  torch.mul(ingate, cellgate)

        hy = torch.mul(outgate, F.tanh(cy))

        return (hy, cy)


class GRUCell(nn.Module):

    """
    An implementation of GRUCell.

    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, h_old, mask=None):

        x = x.view(-1, x.size(1))
        gate_x = self.x2h(x)
        gate_h = self.h2h(h_old)

        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()

        x_r, x_z, x_h = gate_x.chunk(3, 1)
        h_r, h_z, h_h = gate_h.chunk(3, 1)


        r = torch.sigmoid(x_r + h_r) # r
        z = torch.sigmoid(x_z + h_z) # z
        h_new = torch.tanh(x_h + (r * h_h)) # h tilde

        if mask is None:
          hy = z * h_old + (1 - z) * h_new
        else:
          hy = (1 - mask*(1 - z)) * h_old + mask*(1 - z) * h_new
        #hy = newgate + inputgate * (hidden - newgate)
        return hy


class RNNBase(nn.Module):
  def __init__(self, input_sz=1, out_sz=1, hidden_sz=20, preproc_factor=10.0,
               preproc=True, n_layers=1, rnn_cell='gru'):
    super().__init__()
    assert rnn_cell in ['lstm', 'gru']
    assert isinstance(preproc, (list, tuple, bool))
    self.rnn_cell = rnn_cell
    rnn_cell = LSTMCell if rnn_cell == 'lstm' else GRUCell
    self.hidden_sz = hidden_sz
    self.n_layers = n_layers
    self.rnn = []
    if isinstance(preproc, (list, tuple)) and any(preproc):
      assert len(preproc) == input_sz
      self.process = GradientPreprocessor()
      self.preproc = preproc
      input_sz += sum(preproc)
    elif isinstance(preproc, bool) and preproc:
      self.process = GradientPreprocessor()
      self.preproc = [preproc] * input_sz
      if preproc:
        input_sz *= 2
    module = rnn_cell(input_sz, hidden_sz) # mean, std
    self.add_module('rnn_0', module)
    self.rnn.append(module)
    for i in range(self.n_layers - 1):
      module = rnn_cell(hidden_sz, hidden_sz)
      self.add_module(f'rnn_{i+1}', module)
      self.rnn.append(module)
    self.output = nn.Linear(hidden_sz, out_sz)

  def forward(self, x, states, mask=None):
    assert isinstance(x, (tuple, list, torch.Tensor))
    if isinstance(x, (list, tuple)):
      assert len(x) == len(self.preproc)
      x_p = []
      for bool, inp in zip(self.preproc, x):
        inp = C(self.process(inp)) if bool else C(inp)
        x_p.append(inp)
      x = torch.cat(x_p, dim=1)
    elif isinstance(x, torch.Tensor) and self.preproc:
      assert len(self.preproc) == 1
      x = C(self.process(x))

    out_states = []
    if self.rnn_cell == 'lstm':
      for i, rnn in enumerate(self.rnn):
        out_states.append(LSTMStates(*rnn(x, (states[i].h, states[i].c))))
        x = out_states[-1].h
    elif self.rnn_cell == 'gru':
      for i, rnn in enumerate(self.rnn):
        out_states.append(rnn(x, states[i], mask))
        x = out_states[-1]
    else:
      raise RuntimeError(f'Unknown rnn_cell type: {self.rnn_cell}')
    return self.output(x), OptimizerStates(out_states)


class MaskGenerator(RNNBase):
  def __init__(
    self, input_sz=1, hidden_sz=20, preproc_factor=10.0, preproc=True,
    n_layers=1, rnn_cell='gru', mask_type='hybrid'):
    assert mask_type in ['normal', 'unified', 'hybrid']
    if mask_type in ['normal', 'hybrid']:
      out_sz = 2  # for beta & gamma
    elif mask_type in ['unified']:
      out_sz = 1  # for directly computed p
    super().__init__(
      input_sz=input_sz, out_sz=out_sz, hidden_sz=hidden_sz,
      preproc_factor=preproc_factor, preproc=preproc,
      n_layers=n_layers, rnn_cell=rnn_cell)
    self.mask_type = mask_type
    self.sigmoid = nn.Sigmoid()
    self.relu = nn.ReLU()
    self.sparsity_loss = nn.L1Loss(reduction='sum')

  def forward(self, activations, states, debug=False):
    mean = activations.abs().mean(0)
    std =  activations.std(0)
    # local (layerwise) mean & std
    mean_l = activations.abs().mean().expand_as(mean)
    std_l = activations.std().expand_as(std)
    # global mean & std
    mean_g = mean.new_ones(mean.size()) * activations.flat.abs().mean()
    std_g = std.new_ones(std.size()) * activations.flat.std()

    x = [mean.flat.detach(), mean_l.flat.detach(), mean_g.flat.detach(),
         std.flat.detach(), std_l.flat.detach(), std_g.flat.detach()]

    # x = [x_ * 10 for x_ in x]

    output, states = super().forward(x, states)
    # mask = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
    #   temperature=0.1, probs=probs).rsample()
    # mask = mean.new_from_flat(mask)
    if self.mask_type in ['unified']:
      p_logit = output[:, 0]
      # l_logit = output[:, 1]

    elif self.mask_type in ['normal', 'hybrid']: # NOTE: include l_logit!
      gamma = mean.new_from_flat(output[:, 0])
      beta = mean.new_from_flat(output[:, 1])
      mean = activations.mean(0)
      p_logit = gamma * ((activations - mean) / (std + 1e-12)) + beta
      if self.mask_type == 'hybrid':
        p_logit = p_logit.mean(0)
      p_logit = p_logit.flat

    probs = self.sigmoid(p_logit)
    # probs = torch.clamp(p_logit, min=0, max=1)
    # lambda_ = self.sigmoid(l_logit).mean()
    # mask = probs
    mask = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
      temperature=1, probs=probs).rsample()
    # sparsity_loss = mask.norm(p=2) + probs.norm(p=1)
    sparsity_loss = probs.norm(p=1)
    if debug:
      print(probs)
      print(sparsity_loss)
      # min = 0.001
      # max = 0.01
      # print('out_lambda', lambda_)
      # print('applied_lambda', ((max - min) * lambda_ + min))
      import pdb; pdb.set_trace()


    if self.mask_type in ['normal']:
      mask = activations.new_from_flat(mask)
    elif self.mask_type in ['unified', 'hybrid']:
      mask = mean.new_from_flat(mask)

    return mask, states, sparsity_loss#, lambda_


class Optimizer(nn.Module):
  def __init__(self, hidden_sz=20, preproc_factor=10.0, preproc=False,
               n_layers=1, rnn_cell='gru', sb_mode='none'):
    super().__init__()
    assert rnn_cell in ['lstm', 'gru']
    assert sb_mode in ['none', 'normal', 'unified']
    self.rnn_cell = rnn_cell
    self.n_layers = n_layers
    self.hidden_sz = hidden_sz
    self.sb_mode = sb_mode
    self.mask_generator = MaskGenerator(
      input_sz=6,
      mask_type='unified',  # NOTE
      hidden_sz=hidden_sz,
      preproc_factor=preproc_factor,
      preproc=preproc,
      n_layers=n_layers,
      rnn_cell=rnn_cell,
    )
    self.updater = RNNBase(
      input_sz=1,
      out_sz=1,
      preproc_factor=preproc_factor,
      preproc=preproc,
      n_layers=n_layers,
      rnn_cell=rnn_cell,
    )
    self.params_tracker = ParamsIndexTracker(n_tracks=10)

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
      # data.new_train_data()
      inner_data = data.loaders['inner_train']
      outer_data = data.loaders['inner_valid']
      # inner_data = data.loaders['train']
      # outer_data = inner_data
      drop_mode = 'soft_drop'
    elif mode == 'valid':
      self.eval()
      inner_data = data.loaders['valid']
      outer_data = None
      drop_mode = 'hard_drop'
    elif mode == 'eval':
      self.eval()
      inner_data = data.loaders['test']
      outer_data = None
      drop_mode = 'hard_drop'

    # data = dataset(mode=mode)
    model = C(model_cls(sb_mode=self.sb_mode))
    model(*data.pseudo_sample())
    # layer_sz = sum([v for v in model.params.size().unflat(1, 'mat').values()])
    mask_states = C(OptimizerStates.initial_zeros(
        (self.n_layers, len(model.activations.mean(0)), self.hidden_sz)))
    update_states = C(OptimizerStates.initial_zeros(
        (self.n_layers, len(model.params), self.hidden_sz)))

    result_dict = ResultDict()
    unroll_losses = 0
    walltime = 0
    update = C(torch.zeros(model.params.size().flat()))

    batch_arg = BatchManagerArgument(
      params=model.params,
      states=StatesSlicingWrapper(update_states),
      updates=model.params.new_zeros(model.params.size()),
    )

    iter_pbar = tqdm(range(1, optim_it + 1), 'optim_iteration')
    iter_watch = utils.StopWatch('optim_iteration')
    bp_watch = utils.StopWatch('bp')
    loss_decay = 1.0

    for iteration in iter_pbar:
      iter_watch.touch()
      model_detached = C(model_cls(
        params=batch_arg.params.detach(),
        sb_mode=self.sb_mode))
      loss_detached = model_detached(*inner_data.load())
      loss_detached.backward()
      # import pdb; pdb.set_trace()
      iter_pbar.set_description(f'optim_iteration[loss:{loss_detached}]')
# np.set_printoptions(suppress=True, precision=3, linewidth=200, edgeitems=20)
      if debug_sigint.signal_on:
        debug = iteration == 1 or iteration % 10 == 0
        # debug = True
      else:
        debug = False

      backprop_mask, mask_states, sparsity_loss = self.mask_generator(
        activations=model_detached.activations.grad.detach(),
        debug=debug,
        states=mask_states)

      # if debug_sigstp.signal_on and (iteration == 1 or iteration % 10 == 0):
      #   mask_list.append(backprop_mask)
      #   import pdb; pdb.set_trace()
      # min = 0.001
      # max = 0.01
      unroll_losses += sparsity_loss * 0.01 * loss_decay#* ((max - min) * lambda_ + min)
      # import pdb; pdb.set_trace()

      # model_detached.apply_backprop_mask(backprop_mask.unflat, drop_mode)
      # loss_detached.backward()
      assert model_detached.params.grad is not None

      if mode == 'train':
        model = C(model_cls(
          params=batch_arg.params,
          sb_mode=self.sb_mode))

        loss = model(*outer_data.load())
        # assert loss == loss_detached
        if iteration == 1:
          initial_loss = loss
          loss_decay = 1.0
        else:
          loss_decay = (loss / initial_loss).detach()
        unroll_losses += loss

        # model_detached.apply_backprop_mask(backprop_mask.unflat, drop_mode)

      #model_detached.params.apply_grad_mask(mask, 'soft_drop')
      coord_mask = backprop_mask.expand_as_params(model_detached.params)

      if drop_mode == 'soft_drop':
        # grad = model_detached.params.grad.mul(coord_mask).flat
        # model_detached.params = model_detached.params.mul(params_mask)
        index = None
        # updates, states = self.updater(grad, states, mask=coord_mask)
        # updates = updates * out_mul * coord_mask
        # import pdb; pdb.set_trace()
      elif drop_mode == 'hard_drop':
        index = (coord_mask.flat.squeeze() > 0.5).nonzero().squeeze()
      grad = model_detached.params.grad.flat
      batch_arg.update(BatchManagerArgument(grad=grad, mask=coord_mask))
      ##########################################################################
      indexer = DefaultIndexer(input=grad, index=index, shuffle=False)
      batch_manager = OptimizerBatchManager(
        batch_arg=batch_arg, indexer=indexer)

      for get, set in batch_manager.iterator():
        inp = get.grad().detach()
        #inp = [get.grad().detach(), get.tag().detach()]
        if drop_mode == 'soft_drop':
          updates, new_states = self.updater(inp, get.states())#, mask=get.mask())
          # if iteration == 100:
          #   import pdb; pdb.set_trace()
          #   aaa = get.states() *2
          updates = updates * out_mul * get.mask()
          set.states(get.states() * (1 - get.mask()) + new_states * get.mask())
          #set.states(new_states)
        elif drop_mode == 'hard_drop':
          updates, new_states = self.updater(inp, get.states())
          updates = updates * out_mul
          set.states(new_states)
        else:
          raise RuntimeError('Unknown drop mode!')
        set.params(get.params() + updates)
        # if not mode != 'train':
        # set.updates(updates)
      ##########################################################################
      batch_arg = batch_manager.batch_arg_to_set
      # updates = model.params.new_from_flat(updates)

      if mode == 'train' and iteration % unroll == 0:
        meta_optimizer.zero_grad()
        unroll_losses.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 5)
        meta_optimizer.step()
        unroll_losses = 0

      # if iteration == 1 or iteration % 10 == 0:
      #   import pdb; pdb.set_trace()

      if not mode == 'train' or iteration % unroll == 0:
        mask_states.detach_()
        batch_arg.detach_()
        # model.params = model.params.detach()
        # update_states = update_states.detach()
      walltime += iter_watch.touch('interval')
      result_dict.append(loss=loss_detached)
      if not mode == 'train':
        result_dict.append(
          walltime=walltime,
          # **self.params_tracker(
          #   grad=grad,
          #   update=batch_arg.updates,
          # )
        )
    return result_dict
