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
from utils.eval import (eval_gauss_var, eval_lipschitz, inversed_masked_params,
                        random_masked_params)
from utils.result import ResultDict
from utils.timer import Walltime, WalltimeChecker
from utils.torchviz import make_dot
###
from optimizers.analyze_model import *
###
C = utils.getCudaManager('default')
sigint = utils.getSignalCatcher('SIGINT')
sigstp = utils.getSignalCatcher('SIGTSTP')


class Optimizer(OptimizerBase):
  def __init__(self, hidden_sz=32, sb_mode='unified'):
    super().__init__()
    assert sb_mode in ['none', 'normal', 'unified']
    self.hidden_sz = hidden_sz
    self.sb_mode = sb_mode
    self.feature_gen = FeatureGenerator(hidden_sz)
    self.step_gen = StepGenerator(hidden_sz)
    self.mask_gen = MaskGenerator(hidden_sz)
    self.params_tracker = ParamsIndexTracker(n_tracks=10)

  def meta_optimize(self, meta_optimizer, data, model_cls, optim_it, unroll,
                    out_mul, writer, mode='train'):
    assert mode in ['train', 'valid', 'test']
    self.set_mode(mode)

    lipschitz = False
    if lipschitz:
      writer.new_subdirs('best', 'any', 'rand', 'inv', 'dense')

    result_dict = ResultDict()
    unroll_losses = 0
    walltime = Walltime()
    test_kld = torch.tensor(0.)

    params = C(model_cls()).params
    self.feature_gen.new()
    self.step_gen.new()
    iter_pbar = tqdm(range(1, optim_it + 1), 'Inner_loop')
    # iter_watch = utils.StopWatch('Inner_loop')
    ################
    mask_dict = ResultDict()
    analyze_mask = False
    sample_mask = False
    draw_loss = False
    ################
    set_size = {'layer_0': 500, 'layer_1': 10}  # NOTE: make it smarter

    for iter in iter_pbar:
      debug_1 = sigint.is_active(iter == 1 or iter % 10 == 0)
      debug_2 = sigstp.is_active()

      cand_params = []
      cand_losses = []
      sparse_r = {}  # sparsity
      lips_dense = {}
      lips_any = {}
      lips_best = {}
      lips_rand = {}
      lips_inv = {}
      best_loss = 9999999
      best_params = None
      n_samples = 10

      # iter_watch.touch()

      with WalltimeChecker(walltime):
        model_train = C(model_cls(params=params.detach()))
        train_nll = model_train(*data['in_train'].load())
        train_nll.backward()

        g = model_train.params.grad.flat.detach()
        w = model_train.params.flat.detach()

        feature, v_sqrt = self.feature_gen(g)

        size = params.size().unflat()
        kld = self.mask_gen(feature, size)

        losses = []
        lips = []

        for i in range(n_samples):
          # step & mask genration
          mask = self.mask_gen.sample_mask()
          step_out = self.step_gen(feature, v_sqrt, debug=debug_1)
          step = params.new_from_flat(step_out[0])

          # prunning
          if debug_2: pdb.set_trace()

          mask = ParamsFlattener(mask)
          mask_layout = mask.expand_as(params)
          step_sparse = step * mask_layout
          params_sparse = params + step_sparse
          params_pruned = params_sparse.prune(mask > 0.5)

          if params_pruned.size().unflat()['mat_0'][1] == 0:
            continue

          # cand_loss = model(*outer_data_s)
          sparse_model = C(model_cls(params=params_pruned.detach()))
          loss = sparse_model(*data['in_train'].load())

          # if debug_2:
          #   losses.append(loss.tolist())
          #   lips.append(eval_lipschitz(params_pruned, '0').tolist()[0])

          try:
            if (loss < best_loss) or i == 0:
              best_loss = loss
              best_params = params_sparse
              best_pruned = params_pruned
              best_mask = mask
          except:
            best_params = params_sparse
            best_pruned = params_pruned
            best_mask = mask

        if best_params is not None:
          params = best_params
      # walltime += iter_watch.touch('interval')

      """measures(not included in walltime cost)"""
      for k, v in set_size.items():
        r = (best_mask > 0.5).sum().unflat[k].tolist() / v
        n = k.split('_')[1]
        sparse_r[f"sparse_{n}"] = r

      # _, best_params = inversed_masked_params(params, best_mask, step, set_size, sparse_r)
      _, best_params = random_masked_params(params, step, set_size, sparse_r)

      if lipschitz and iter % 10 == 0:

        params_pruned_inv = inversed_masked_params(
            params, best_mask, step, set_size, sparse_r)
        params_pruned_rand = random_masked_params(
            params, step, set_size, sparse_r)

        step_pruned = step_sparse.prune(mask > 0.5)
        # for k, v in set_size.items():
        #   n = k.split('_')[1]
        #   r = sparse_r[f"sparse_{n}"]
        #   lips_best[f"lips_{n}"] = eval_lipschitz(best_pruned, n).tolist()[0]
        #   lips_any[f"lips_{n}"] = eval_lipschitz(params_pruned, n).tolist()[0]
        #   lips_rand[f"lips_{n}"] = eval_lipschitz(params_pruned_rand, n).tolist()[0]
        #   lips_inv[f"lips_{n}"] = eval_lipschitz(params_pruned_inv, n).tolist()[0]
        #   lips_dense[f"lips_{n}"] = eval_lipschitz(params_sparse, n).tolist()[0]

        # lips_best[f"var"] = eval_gauss_var(model_cls, data, best_pruned).tolist()
        # lips_any[f"var"] = eval_gauss_var(model_cls, data, params_pruned).tolist()
        # lips_rand[f"var"] = eval_gauss_var(model_cls, data, params_pruned_rand).tolist()
        # lips_inv[f"var"] = eval_gauss_var(model_cls, data, params_pruned_inv).tolist()
        # lips_dense[f"var"] = eval_gauss_var(model_cls, data, params_sparse).tolist()

        eval_step_direction(model_cls, data, best_params, best_)

        # import pdb; pdb.set_trace()
        # ppp = eval_lipschitz(best_params * best_mask.expand_as(best_params), n)

      with WalltimeChecker(walltime if mode == 'train' else None):
        model_test = C(model_cls(params=params))
        test_nll = utils.isnan(model_test(*data['in_test'].load()))
        test_kld = kld / data['in_test'].full_size
        total_test = test_nll + test_kld

        if mode == 'train':
          unroll_losses += total_test
          if iter % unroll == 0:
            meta_optimizer.zero_grad()
            unroll_losses.backward()
            nn.utils.clip_grad_value_(self.parameters(), 0.01)
            meta_optimizer.step()
            unroll_losses = 0

      with WalltimeChecker(walltime):
        if not mode == 'train' or iter % unroll == 0:
          params = params.detach_()

      ##############################################################
      text_dir = 'test/analyze_mask'
      result_dir = 'test/drawloss'
      sample_dir = 'test/mask_compare'
      iter_interval = 10
      sample_num = 10000
      if mode == 'test' and analyze_mask:
        analyzing_mask(self.mask_gen, layer_size, mode, iter, iter_interval, text_dir)
      if mode == 'test' and sample_mask:
        mask_result = sampling_mask(self.mask_gen, layer_size, model_train, params, sample_num, mode, iter, iter_interval, sample_dir)
        if mask_result is not None:
          mask_dict.append(mask_result)
      if mode == 'test' and draw_loss:
        plot_loss(model_cls=model_cls, model=model_train, params=params, input_data=data['in_train'].load(), dataset=data['in_train'], 
            feature_gen=self.feature_gen, mask_gen=self.mask_gen, step_gen=self.step_gen, scale_way=None, xmin=-2.0, xmax=0.5, num_x=20, mode=mode, iteration=iter, iter_interval=iter_interval, loss_dir=result_dir)
      ##############################################################
      # import pdb; pdb.set_trace()
      # result dict
      result = dict(
          train_nll=train_nll.tolist(),
          test_nll=test_nll.tolist(),
          test_kld=test_kld.tolist(),
          walltime=walltime.time,
          **sparse_r,
      )
      result_dict.append(result)
      log_pbar(result, iter_pbar)

      # if writer:
      #   log_tf_event(result, tf_writer, iter, 'meta-test/wallclock')
      #   log_tf_event(result, tf_writer, iter, 'meta-test/wallclock')

      if writer and lipschitz and iter % 10 == 0:
        lips_b = dict(
            lips_t=np.prod([l for l in lips_best.values()]),
            **lips_best
        )
        lips_a = dict(
            lips_t=np.prod([l for l in lips_any.values()]),
            **lips_any
        )
        lips_r = dict(
            lips_t=np.prod([l for l in lips_rand.values()]),
            **lips_rand
        )
        lips_i = dict(
            lips_t=np.prod([l for l in lips_inv.values()]),
            **lips_inv
        )
        lips_d = dict(
            lips_t=np.prod([l for l in lips_dense.values()]),
            **lips_dense
        )
        log_tf_event(lips_b, writer_best, iter, 'lipschitz constant')
        log_tf_event(lips_a, writer_any, iter, 'lipschitz constant')
        log_tf_event(lips_r, writer_rand, iter, 'lipschitz constant')
        log_tf_event(lips_i, writer_inv, iter, 'lipschitz constant')
        log_tf_event(lips_d, writer_dense, iter, 'lipschitz constant')
    if sample_mask:
      plot_mask_result(mask_dict, sample_dir)
    return result_dict
