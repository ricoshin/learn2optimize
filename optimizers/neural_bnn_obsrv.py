import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_helpers import ParamsIndexTracker, ParamsFlattener
# from nn.maskgen_rnn import MaskGenerator
from nn.maskgen_bnn import FeatureGenerator, StepGenerator, MaskGenerator
from optimizers.optim_helpers import (BatchManagerArgument, DefaultIndexer,
                                      OptimizerBatchManager, OptimizerParams,
                                      OptimizerStates, StatesSlicingWrapper)
from nn.rnn_base import RNNBase
from tqdm import tqdm
from utils import utils
from utils.result import ResultDict
from utils.torchviz import make_dot

C = utils.getCudaManager('default')
debug_sigint = utils.getDebugger('SIGINT')
debug_sigstp = utils.getDebugger('SIGTSTP')


class Optimizer(nn.Module):
  def __init__(self, hidden_sz=32, sb_mode='unified'):
    super().__init__()
    assert sb_mode in ['none', 'normal', 'unified']
    self.hidden_sz = hidden_sz
    self.sb_mode = sb_mode
    self.feature_gen = FeatureGenerator(hidden_sz)
    self.step_gen = StepGenerator(hidden_sz)
    self.mask_gen = MaskGenerator(hidden_sz)
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
    params = model.params

    result_dict = ResultDict()
    unroll_losses = 0
    walltime = 0

    self.feature_gen.new()

    iter_pbar = tqdm(range(1, optim_it + 1), 'optim_iteration')
    iter_watch = utils.StopWatch('optim_iteration')

    for iteration in iter_pbar:
      iter_watch.touch()
      if debug_sigint.signal_on:
        debug = iteration == 1 or iteration % 10 == 0
      else:
        debug = False

      model_detached = C(model_cls(params=params.detach()))
      loss_detached = model_detached(*inner_data.load())
      loss_detached.backward()

      g = model_detached.params.grad.flat.detach()
      w = model_detached.params.flat.detach()

      # step & mask genration
      feature = self.feature_gen(g, w)
      step = self.step_gen(feature)
      step = params.new_from_flat(step)
      mask, kld = self.mask_gen(feature, params.size().unflat())
      mask = ParamsFlattener(mask)
      mask = mask.expand_as(model_detached.params)

      # update
      step = step * mask
      params = params + step

      if mode == 'train':
        model = C(model_cls(params=params))
        optim_loss = model(*outer_data.load())
        if torch.isnan(optim_loss):
          import pdb; pdb.set_trace()
      # import pdb; pdb.set_trace()
        unroll_losses += optim_loss + kld / outer_data.full_size #* 0.00005

      if mode == 'train' and iteration % unroll == 0:
        meta_optimizer.zero_grad()
        unroll_losses.backward()
        # import pdb; pdb.set_trace()
        # nn.utils.clip_grad_norm_(self.parameters(), 0.1)
        meta_optimizer.step()
        unroll_losses = 0

      if not mode == 'train' or iteration % unroll == 0:
        self.mask_gen.detach_lambdas_()
        params = params.detach_()

      # import pdb; pdb.set_trace()

      iter_pbar.set_description(f'optim_iteration'
        f'[optim_loss:{loss_detached.tolist():5.5}')
        # f' sparse_loss:{sparsity_loss.tolist():5.5}]')
      # iter_pbar.set_description(f'optim_iteration[loss:{loss_dense.tolist()}/dist:{dist.tolist()}]')
      # torch.optim.SGD(sparse_params, lr=0.1).step()
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
