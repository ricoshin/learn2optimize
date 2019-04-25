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
from optimize import log_pbar

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
                    out_mul, tf_writer=None, mode='train'):
    assert mode in ['train', 'valid', 'test']
    if mode == 'train':
      self.train()
    else:
      self.eval()

    result_dict = ResultDict()
    unroll_losses = 0
    walltime = 0
    kld_test = torch.tensor(0.)

    params = C(model_cls()).params
    self.feature_gen.new()
    self.step_gen.new()
    iter_pbar = tqdm(range(1, optim_it + 1), 'Inner_loop')
    iter_watch = utils.StopWatch('Inner_loop')

    layer_size = {'layer_0': 500, 'layer_1': 10}  # NOTE: make it smarter

    for iteration in iter_pbar:
      if debug_sigint.signal_on:
        debug_1 = iteration == 1 or iteration % 10 == 0
      else:
        debug_1 = False
      if debug_sigstp.signal_on:
        debug_2 = True
      else:
        debug_2 = False

      cand_params = []
      cand_losses = []
      sparse_r = {}  # sparsity
      best_loss = 9999999
      best_params = None
      n_samples = 5

      iter_watch.touch()
      model_train = C(model_cls(params=params.detach()))
      nll_train = model_train(*data['in_train'].load())
      nll_train.backward()

      g = model_train.params.grad.flat.detach()
      w = model_train.params.flat.detach()

      feature, v_sqrt = self.feature_gen(g)

      size = params.size().unflat()
      kld = self.mask_gen(feature, size)

      for i in range(n_samples):
        # step & mask genration
        mask = self.mask_gen.sample_mask()
        step = self.step_gen(feature, v_sqrt, debug=debug_1)
        step_ = params.new_from_flat(step[0])

        # prunning
        mask = ParamsFlattener(mask)
        mask_layout = mask.expand_as(params)
        step_ = step_ * mask_layout
        params_ = params + step_
        sparse_params = params_.prune(mask > 1e-6)

        if sparse_params.size().unflat()['mat_0'][1] == 0:
          continue
        if debug_2:
          import pdb; pdb.set_trace()
        # cand_loss = model(*outer_data_s)
        sparse_model = C(model_cls(params=sparse_params.detach()))
        loss = sparse_model(*data['in_train'].load())
        try:
          if loss < best_loss:
            best_loss = loss
            best_params = params_
            best_mask = mask
            del loss, params_, mask
        except:
          best_params = params_
          best_mask = mask
          del params_, mask
        del sparse_params, sparse_model, mask_layout

      if best_params is not None:
        params = best_params

      for k, v in layer_size.items():
        r = (best_mask > 1e-6).sum().unflat[k].tolist() / v
        sparse_r[f"sparse_{k.split('_')[1]}"] = r

      if not mode == 'train':
        # meta-test time cost (excluding inner-test time)
        walltime += iter_watch.touch('interval')

      model_test = C(model_cls(params=params))
      nll_test = utils.isnan(model_test(*data['in_test'].load()))
      kld_teset = kld / data['in_test'].full_size
      total_test = nll_test +  kld_test

      if debug_2:
        import pdb; pdb.set_trace()

      if mode == 'train':
        if not iteration % unroll == 0:
          unroll_losses += total_test
        else:
          meta_optimizer.zero_grad()
          unroll_losses.backward()
          nn.utils.clip_grad_value_(self.parameters(), 0.01)
          meta_optimizer.step()
          unroll_losses = 0

      if not mode == 'train' or iteration % unroll == 0:
        # self.mask_gen.detach_lambdas_()
        self.step_gen.lr = self.step_gen.log_lr.detach()
        params = params.detach_()

      if mode == 'train':
        # meta-training time cost
        walltime += iter_watch.touch('interval')

      # result dict
      result = dict(
        nll_train=nll_train.tolist(),
        nll_test=nll_test.tolist(),
        kld_test=kld_test.tolist(),
        total_test=total_test.tolist(),
        walltime=walltime,
        **sparse_r
        )
      result_dict.append(result)
      log_pbar(result, iter_pbar)
      # desc = [f'{k}: {v:5.5}' for k, v in result.items()]
      # iter_pbar.set_description(f"inner_loop [ {' / '.join(desc)} ]")

    return result_dict
