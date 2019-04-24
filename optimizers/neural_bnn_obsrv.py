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
    self.step_gen.new()
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
    n_sample = 10

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

      model_train = C(model_cls(params=params.detach()))
      nll_train = model_train(*data['train'].load())
      nll_train.backward()

      g = model_train.params.grad.flat.detach()
      w = model_train.params.flat.detach()

      # step & mask genration
      feature, v_sqrt = self.feature_gen(g)
      step = self.step_gen(feature, v_sqrt, debug=debug_1)

      step = params.new_from_flat(step[0])
      # import pdb; pdb.set_trace()
      if debug_2:
        cnt = None
        th = 0.00001
        for i in range(100):
          if cnt is None:
            cnt = ParamsFlattener(
              self.mask_gen(feature, params.size().unflat(), debug_1)[0]) > th
          else:
            cnt += ParamsFlattener(
              elf.mask_gen(feature, params.size().unflat(), debug_1)[0]) > th
        import pdb; pdb.set_trace()

      size = params.size().unflat()
      # mask_out = self.mask_gen(feature, size, debug=debug_1)
      # mask, _, kld = mask_out[0]
      # kld_test = kld / data['test'].full_size #* 0.00005
      # mask = ParamsFlattener(mask)
      # mask_layout = mask.expand_as(params)
      #
      # iters.append(iteration);
      # if res1 is None:
      #   res1 = mask > 0.0001
      # else:
      #   res1 += mask > 0.0001
      # res2.append((mask > 0.0001).sum().unflat['layer_0'].tolist()/500)
      # res3.append((mask > 0.0001).sum().unflat['layer_1'].tolist()/10)
      # lamb.append(round(self.mask_gen.lamb[0].tolist(), 6))
      # gamm_g.append(round(self.mask_gen.gamm_g[0].tolist(), 6))
      # gamm_l1.append(round(self.mask_gen.gamm_l[0][0].tolist(), 6))
      # gamm_l2.append(round(self.mask_gen.gamm_l[1][0].tolist(), 6))

      # mask = mask.expand_as(model_detached.params)

      # update
      step = step #* mask_layout
      params = params + step

      model_test = C(model_cls(params=params))
      loss = utils.isnan(model(*data['test'].load()))

      if mode == 'train':
        unroll_losses += nll_test #+ kld_test
      elif mode == 'valid':
        pass

      if debug_2:
        import pdb; pdb.set_trace()

      if mode == 'train':
        if not iteration % unroll == 0:

          unroll_losses += loss_test +
        else:
          meta_optimizer.zero_grad()
          unroll_losses.backward()
          # import pdb; pdb.set_trace()
          nn.utils.clip_grad_value_(self.parameters(), 0.01)
          meta_optimizer.step()
          unroll_losses = 0

      if not mode == 'train' or iteration % unroll == 0:
        # self.mask_gen.detach_lambdas_()
        self.step_gen.lr = self.step_gen.log_lr.detach()
        params = params.detach_()

      # import pdb; pdb.set_trace()


      iter_pbar.set_description(f'optim_iteration'
        f'[optim_loss:{loss_detached.tolist():5.5}')
        # f' sparse_loss:{sparsity_loss.tolist():5.5}]')
      # iter_pbar.set_description(f'optim_iteration[loss:{loss_dense.tolist()}/dist:{dist.tolist()}]')
      # torch.optim.SGD(sparse_params, lr=0.1).step()
      walltime += iter_watch.touch('interval')
      result_dict.append(
        train_nll=nll_train,
        test_nll=nll_test,
        test_kld=kld_test,
        test_total=(nll_test + kld_test),
        #loss_kld=loss_kld,
        )
      if not mode == 'train':
        result_dict.append(
            walltime=walltime,
            # **self.params_tracker(
            #   grad=grad,
            #   update=batch_arg.updates,
            # )
        )
    return result_dict
