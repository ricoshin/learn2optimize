"""Single mask (Masking part can be ablated)"""

import math
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_helpers import ParamsFlattener, ParamsIndexTracker
# from nn.maskgen_rnn import MaskGenerator
from nn.maskgen_bnn2 import FeatureGenerator, MaskGenerator, StepGenerator
from nn.rnn_base import RNNBase
from optimize import log_pbar
from optimizers.optim_helpers import (BatchManagerArgument, DefaultIndexer,
                                      OptimizerBatchManager, OptimizerParams,
                                      OptimizerStates, StatesSlicingWrapper,
                                      OptimizerBase)
from tqdm import tqdm
from utils import utils
from utils.result import ResultDict
from utils.timer import Walltime, WalltimeChecker
from utils.torchviz import make_dot
from utils import analyzers
from utils.kl_anneal import kl_anneal_function

C = utils.getCudaManager('default')
sigint = utils.getSignalCatcher('SIGINT')
sigstp = utils.getSignalCatcher('SIGTSTP')


class Optimizer(OptimizerBase):
  def __init__(self, hidden_sz=20, drop_rate=0.2, sb_mode='unified'):
    super().__init__()
    assert sb_mode in ['none', 'normal', 'unified']
    self.hidden_sz = hidden_sz
    self.sb_mode = sb_mode
    self.feature_gen = FeatureGenerator(hidden_sz, drop_rate=drop_rate)
    self.step_gen = StepGenerator(hidden_sz)
    self.mask_gen = MaskGenerator(hidden_sz)
    self.params_tracker = ParamsIndexTracker(n_tracks=10)

  def meta_optimize(
    self, cfg, meta_optim, data, model_cls, writer=None, mode='train'):
    assert mode in ['train', 'valid', 'test']
    self.set_mode(mode)

    # mask_mode = ['no_mask', 'structured', 'unstructured'][2]
    # data_mode = ['in_train', 'in_test'][0]

    ############################################################################
    analyze_model = False
    analyze_surface = False
    ############################################################################

    result_dict = ResultDict()
    unroll_losses = 0
    walltime = Walltime()
    test_kld = torch.tensor(0.)

    params = C(model_cls()).params
    self.feature_gen.new()
    self.step_gen.new()
    sparse_r = {}  # sparsity
    iter_pbar = tqdm(range(1, cfg['iter_' + mode] + 1), 'Inner_loop')

    for iter in iter_pbar:
      debug_1 = sigint.is_active(iter == 1 or iter % 10 == 0)
      debug_2 = sigstp.is_active()

      with WalltimeChecker(walltime):
        model_train = C(model_cls(params.detach()))
        data_train = data['in_train'].load()
        train_nll, train_acc = model_train(*data_train)
        train_nll.backward()

        grad = model_train.params.grad.detach()
        # g = model_train.params.grad.flat.detach()
        # w = model_train.params.flat.detach()

        # step & mask genration
        feature, v_sqrt = self.feature_gen(grad.flat.detach())
        # step = self.step_gen(feature, v_sqrt, debug=debug_1)
        # step = params.new_from_flat(step[0])
        size = params.size().unflat()

        if cfg.mask_mode == 'structured':
          mask = self.mask_gen(feature, size, debug=debug_1)
          mask = ParamsFlattener(mask)
          mask_layout = mask.expand_as(params)
          params = params + grad.detach() * mask_layout * (-cfg.inner_lr)
        elif cfg.mask_mode == 'unstructured':
          mask_flat = self.mask_gen.unstructured(feature, size)
          mask = params.new_from_flat(mask_flat)
          params = params + grad.detach() * mask * (-cfg.inner_lr)
          # update = params.new_from_flat(params.flat + grad.flat.detach() * mask * (-cfg.lr))
          # params = params + update
        elif cfg.mask_mode == 'no_mask':
          params = params + grad.detach() * (-cfg.inner_lr)
        else:
          raise Exception('Unknown setting!')

        # import pdb; pdb.set_trace()
        # step_masked = step * mask_layout
        # params = params + step_masked

      with WalltimeChecker(walltime if mode == 'train' else None):
        model_test = C(model_cls(params))
        if cfg.data_mode == 'in_train':
          data_test = data_train
        elif cfg.data_mode == 'in_test':
          data_test = data['in_test'].load()
        test_nll, test_acc = utils.isnan(*model_test(*data_test))

        if debug_2: pdb.set_trace()

        if mode == 'train':
          unroll_losses += test_nll # + test_kld
          if iter % cfg.unroll == 0:
            meta_optim.zero_grad()
            unroll_losses.backward()
            nn.utils.clip_grad_value_(self.parameters(), 0.01)
            meta_optim.step()
            unroll_losses = 0

      with WalltimeChecker(walltime):
        if not mode == 'train' or iter % cfg.unroll == 0:
          params = params.detach_()

      ##########################################################################
      if analyze_model:
        analyzers.model_analyzer(
          self, mode, model_train, params, model_cls, mask.tsize(0), data, iter,
          optim_it, analyze_mask=True, sample_mask=True, draw_loss=False)
      if analyze_surface:
        analyzers.surface_analyzer(
          params, best_mask, step, writer, iter)
      ##########################################################################

      result = dict(
          train_nll=train_nll.tolist(),
          test_nll=test_nll.tolist(),
          train_acc=train_acc.tolist(),
          test_acc=test_acc.tolist(),
          test_kld=test_kld.tolist(),
          walltime=walltime.time,
      )
      if not cfg.mask_mode == 'no_mask':
        result.update(
          **mask.sparsity(overall=True),
          **mask.sparsity(overall=False),
        )
      result_dict.append(result)
      log_pbar(result, iter_pbar)

    return result_dict, params
