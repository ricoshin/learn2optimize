"""Multiple mask: previous version (back up)"""

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
###
from optimizers.analyze_model import *
###
C = utils.getCudaManager('default')
debug_sigint = utils.getSignalCatcher('SIGINT')
debug_sigstp = utils.getSignalCatcher('SIGTSTP')


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
    elif mode == 'test':
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
    ################
    mask_dict = ResultDict()
    analyze_mask = False
    sample_mask = False
    draw_loss = False
    ################
    iter_pbar = tqdm(range(1, optim_it + 1), 'optim_iteration')
    iter_watch = utils.StopWatch('optim_iteration')
    
    res1 = None
    res2=[]
    res3=[]
    lamb = []
    gamm_g = []
    gamm_l1 = []
    gamm_l2 = []
    iters = []

    for iteration in iter_pbar:
      iter_watch.touch()
      if debug_sigint.signal_on:
        debug_1 = iteration == 1 or iteration % 10 == 0
      else:
        debug_1 = False
      if debug_sigstp.signal_on:
        debug_2 = True
      else:
        debug_2 = False

      model_detached = C(model_cls(params=params.detach()))
      inner_data_s = inner_data.load()
      loss_detached = model_detached(*inner_data_s)
      loss_detached.backward()

      g = model_detached.params.grad.flat.detach()
      w = model_detached.params.flat.detach()

      cand_params = []
      cand_losses = []
      best_loss = 9999999
      best_params = None
      n_samples = 5

      for m in range(1):
        feature = self.feature_gen(g, w, n=n_samples, m_update=(m==0))
        p_size = params.size().unflat()
        mask_gen_out = self.mask_gen(feature, p_size, n=n_samples)
        step_out = self.step_gen(feature, n=n_samples)

        # step & mask genration
        for i in range(n_samples):
          mask, _, kld = mask_gen_out[i]
          step = step_out[i]
          mask = ParamsFlattener(mask)
          mask_layout = mask.expand_as(params)
          step = params.new_from_flat(step)
          # prunning
          step = step * mask_layout
          params_ = params + step
          # cand_params.append(params_.flat)
          sparse_params = params_.prune(mask > 1e-6)
          if sparse_params.size().unflat()['mat_0'][1] == 0:
            continue
          if debug_2:
            import pdb; pdb.set_trace()
          # cand_loss = model(*outer_data_s)
          sparse_model = C(model_cls(params=sparse_params.detach()))
          loss = sparse_model(*inner_data_s)
          try:
            if loss < best_loss:
              best_loss = loss
              best_params = params_
              best_kld = kld
          except:
            best_params = params_
            best_kld = kld

      if best_params is not None:
        params = best_params
        best_kld = 0

      if mode == 'train':
        model = C(model_cls(params=params))
        optim_loss = model(*outer_data.load())
        if torch.isnan(optim_loss):
          import pdb; pdb.set_trace()
        unroll_losses += optim_loss + best_kld / outer_data.full_size

      if mode == 'train' and iteration % unroll == 0:
        meta_optimizer.zero_grad()
        unroll_losses.backward()
        # import pdb; pdb.set_trace()
        nn.utils.clip_grad_value_(self.parameters(), 0.1)
        meta_optimizer.step()
        unroll_losses = 0

      if not mode == 'train' or iteration % unroll == 0:
        # self.mask_gen.detach_lambdas_()
        params = params.detach_()

      # import pdb; pdb.set_trace()
      if params is None:
        import pdb; pdb.set_trace()

      iter_pbar.set_description(f'optim_iteration'
        f'[optim_loss:{loss_detached.tolist():5.5}')
        # f' sparse_loss:{sparsity_loss.tolist():5.5}]')
      # iter_pbar.set_description(f'optim_iteration[loss:{loss_dense.tolist()}/dist:{dist.tolist()}]')
      # torch.optim.SGD(sparse_params, lr=0.1).step()
      walltime += iter_watch.touch('interval')
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
