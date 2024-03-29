"""Mutiple mask: sparse loss landcapes observation"""

import math
import os
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_helpers import ParamsFlattener, ParamsIndexTracker
# from nn.maskgen_rnn import MaskGenerator
from nn.maskgen_bnn import FeatureGenerator, MaskGenerator, StepGenerator
from nn.rnn_base import RNNBase
from optimize import log_pbar, log_tf_event
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

  def meta_optimize(self, meta_optim, data, model_cls, optim_it, unroll,
    out_mul, k_obsrv=1, no_mask=False, writer=None, mode='train'):
    assert mode in ['train', 'valid', 'test']
    if no_mask is True:
      raise Exception("this module currently does NOT suport no_mask option")
    self.set_mode(mode)

    ############################################################################
    n_samples = k_obsrv
    """MSG: better postfix?"""
    analyze_model = False
    analyze_surface = False
    ############################################################################

    if analyze_surface:
      writer.new_subdirs('best', 'any', 'rand', 'inv', 'dense')

    result_dict = ResultDict()
    unroll_losses = 0
    walltime = Walltime()
    test_kld = torch.tensor(0.)

    params = C(model_cls()).params
    self.feature_gen.new()
    self.step_gen.new()
    iter_pbar = tqdm(range(1, optim_it + 1), 'Inner_loop')
    set_size = {'layer_0': 500, 'layer_1': 10}  # NOTE: make it smarter

    for iter in iter_pbar:
      debug_1 = sigint.is_active(iter == 1 or iter % 10 == 0)
      debug_2 = sigstp.is_active()

      best_loss = 9999999
      best_params = None

      with WalltimeChecker(walltime):
        model_train = C(model_cls(params.detach()))
        train_nll, train_acc = model_train(*data['in_train'].load())
        train_nll.backward()

        g = model_train.params.grad.flat.detach()
        w = model_train.params.flat.detach()

        feature, v_sqrt = self.feature_gen(g)

        size = params.size().unflat()
        kld = self.mask_gen(feature, size)

        losses = []
        lips = []
        valid_mask_patience = 100
        assert n_samples > 0
        """FIX LATER:
        when n_samples == 0 it can behave like no_mask flag is on."""
        for i in range(n_samples):
          # step & mask genration
          for j in range(valid_mask_patience):
            mask = self.mask_gen.sample_mask()
            mask = ParamsFlattener(mask)
            if mask.is_valid_sparsity():
              if j > 0:
                print(f'\n\n[!]Resampled {j + 1} times to get valid mask!')
              break
            if j == valid_mask_patience - 1:
              raise Exception("[!]Could not sample valid mask for "
                              f"{j+1} trials.")

          step_out = self.step_gen(feature, v_sqrt, debug=debug_1)
          step = params.new_from_flat(step_out[0])

          mask_layout = mask.expand_as(params)
          step_sparse = step * mask_layout
          params_sparse = params + step_sparse
          params_pruned = params_sparse.prune(mask > 0.5)

          if params_pruned.size().unflat()['mat_0'][1] == 0:
            continue

          # cand_loss = model(*outer_data_s)
          sparse_model = C(model_cls(params_pruned.detach()))
          loss, _ = sparse_model(*data['in_train'].load())

          if (loss < best_loss) or i == 0:
            best_loss = loss
            best_params = params_sparse
            best_pruned = params_pruned
            best_mask = mask

        if best_params is not None:
          params = best_params

      with WalltimeChecker(walltime if mode == 'train' else None):
        model_test = C(model_cls(params))
        test_nll, test_acc = utils.isnan(*model_test(*data['in_test'].load()))
        test_kld = kld / data['in_test'].full_size / unroll
        ## kl annealing function 'linear' / 'logistic' / None
        test_kld2 = test_kld * kl_anneal_function(
          anneal_function=None, step=iter, k=0.0025, x0=optim_it)
        total_test = test_nll + test_kld2

        if mode == 'train':
          unroll_losses += total_test
          if iter % unroll == 0:
            meta_optim.zero_grad()
            unroll_losses.backward()
            nn.utils.clip_grad_value_(self.parameters(), 0.01)
            meta_optim.step()
            unroll_losses = 0

      with WalltimeChecker(walltime):
        if not mode == 'train' or iter % unroll == 0:
          params = params.detach_()

      ##########################################################################
      """Analyzers"""
      if analyze_model:
        analyzers.model_analyzer(
          self, mode, model_train, params, model_cls, set_size, data, iter,
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
          **best_mask.sparsity(overall=True),
          **best_mask.sparsity(overall=False),
      )
      result_dict.append(result)
      log_pbar(result, iter_pbar)


    return result_dict, params
