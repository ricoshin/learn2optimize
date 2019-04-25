import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_helpers import ParamsIndexTracker
# from nn.maskgen_rnn import MaskGenerator
from nn.maskgen_topk import MaskGenerator
from nn.rnn_base import RNNBase
from optimize import log_pbar
from optimizers.optim_helpers import (BatchManagerArgument, DefaultIndexer,
                                      OptimizerBatchManager, OptimizerParams,
                                      OptimizerStates, StatesSlicingWrapper)
from tqdm import tqdm
from utils import utils
from utils.result import ResultDict
from utils.torchviz import make_dot

C = utils.getCudaManager('default')
debug_sigint = utils.getDebugger('SIGINT')
debug_sigstp = utils.getDebugger('SIGTSTP')


class Optimizer(nn.Module):
  def __init__(self, hidden_sz=20, preproc_factor=10.0, preproc=False,
               n_layers=1, rnn_cell='gru', sb_mode='unified'):
    super().__init__()
    assert sb_mode in ['none', 'normal', 'unified']
    self.topk_mask_gen = MaskGenerator()

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

    params = C(model_cls()).params

    # lr_sgd = 0.2
    # lr_adam = 0.001
    # sgd = torch.optim.SGD(model.parameters(), lr=lr_sgd)
    # adam = torch.optim.Adam(model.parameters())
    # adagrad = torch.optim.Adagrad(model.parameters())

    iter_pbar = tqdm(range(1, optim_it + 1), 'Inner_loop')
    iter_watch = utils.StopWatch('Inner_loop')

    lr = 0.1
    topk = False
    n_sample = 10
    topk_decay = False
    recompute_grad = False

    self.topk_ratio = 0.5
    # self.topk_ratio = 1.0
    set_size = {'layer_0': 500, 'layer_1': 10}  # NOTE: do it smarter

    for iteration in iter_pbar:
      # # conventional optimizers with API
      # model.zero_grad()
      # loss = model(*inner_data.load())
      # loss.backward()
      # adam.step()
      # loss_detached = loss

      iter_watch.touch()
      model_train = C(model_cls(params=params.detach()))
      train_nll = model_train(*data['in_train'].load())
      train_nll.backward()

      params = model_train.params.detach()
      grad = model_train.params.grad.detach()
      act = model_train.activations.detach()

      # conventional optimizers with manual SGD
      params_good = params + grad * (-lr)
      grad_good = grad

      best_loss_sparse = 999999
      for i in range(n_sample):
        mask_gen = MaskGenerator.topk if topk else MaskGenerator.randk
        mask = mask_gen(grad=grad, act=act, topk=self.topk_ratio)

        if recompute_grad:
          # recompute gradients again using prunned network
          params_sparse = params.prune(mask)
          model_sparse = C(model_cls(params=params_sparse.detach()))
          loss_sparse = model_sparse(*data['in_train'].load())
          loss_sparse.backward()
          grad_sparse = model_sparse.params.grad
        else:
          # use full gradients but sparsified
          params_sparse = params.prune(mask)
          grad_sparse = grad.prune(mask)

        params_sparse_ = params_sparse + grad_sparse * (-lr)  # SGD
        model_sparse = C(model_cls(params=params_sparse_.detach()))
        loss_sparse = model_sparse(*data['in_train'].load())
        if loss_sparse < best_loss_sparse:
          best_loss_sparse = loss_sparse
          best_params_sparse = params_sparse_
          best_grads = grad_sparse
          best_mask = mask

      # below lines are just for evaluation purpose (excluded from time cost)
      walltime += iter_watch.touch('interval')
      # decayu topk_ratio
      if topk_decay:
        self.topk_ratio *= 0.999

      # update!
      params = params.sparse_update(best_mask, best_grads * (-lr))

      # generalization performance
      model_test = C(model_cls(params=params.detach()))
      test_nll = model_test(*data['in_test'].load())

      # result dict
      result = dict(
          train_nll=train_nll.tolist(),
          test_nll=test_nll.tolist(),
          sparse_0=self.topk_ratio,
          sparse_1=self.topk_ratio,
      )
      result_dict.append(result)
      log_pbar(result, iter_pbar)
      # desc = [f'{k}: {v:5.5}' for k, v in result.items()]
      # iter_pbar.set_description(f"inner_loop [ {' / '.join(desc)} ]")

    return result_dict
