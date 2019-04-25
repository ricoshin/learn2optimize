import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_helpers import ParamsIndexTracker
# from nn.maskgen_rnn import MaskGenerator
from nn.maskgen_topk import MaskGenerator
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
                    out_mul, mode):
    assert mode in ['train', 'valid', 'test']
    if mode == 'train':
      self.train()
    else:
      self.eval()

    result_dict = ResultDict()
    unroll_losses = 0
    walltime = 0

    params = C(model_cls()).params

    lr_sgd = 0.2
    lr_adam = 0.001
    sgd = torch.optim.SGD(model.parameters(), lr=lr_sgd)
    adam = torch.optim.Adam(model.parameters())
    adagrad = torch.optim.Adagrad(model.parameters())

    iter_pbar = tqdm(range(1, optim_it + 1), 'Inner_loop')
    iter_watch = utils.StopWatch('Inner_loop')

    loss_decay = 1.0
    dist_list = []
    dense_rank = 0
    sparse_rank = 0
    dense_rank_list = []
    sparse_rank_list = []

    for iteration in iter_pbar:
      # # conventional optimizers with API
      # model.zero_grad()
      # loss = model(*inner_data.load())
      # loss.backward()
      # adam.step()
      # loss_detached = loss

      iter_watch.touch()
      model_dense = C(model_cls(params=params.detach()))
      loss_dense = model_dense(*data['in_train'].load())
      loss_dense.backward()


      # # conventional optimizers with manual SGD
      # params = model_detached.params + model_detached.params.grad * (-0.2)

      # iter_pbar.set_description(f'optim_iteration[loss:{loss_detached}]')
      masks = []
      # sparse_old_loss = []
      # sparse_new_loss = []
      sparse_loss = []
      dense_loss = []
      candidates = []

      lr = 1.0
      n_sample = 10
      recompute_grad = False
      # best_loss_sparse = 999999

      for i in range(n_sample):

        mask = MaskGenerator.randk(
          grad=model_detached.params.grad.detach(),
          topk_ratio=0.1
          )

        if recompute_grad:
          # recompute gradients again using prunned network
          params_sparse = model_dense.params.prune(mask)
          model_sparse = C(model_cls(params=params_sparse.detach()))
          loss_sparse = model_sparse(*data['in_train'].load())
          loss_sparse.backward()
          grads = model_sparse.params.grad
        else:
          # use full gradients but sparsified
          params_sparse = model_dense.params.prune(mask)
          grads = model_detached.params.grad.prune(mask)

        params_sparse = params_sparse + grads*(-lr)  # SGD
        model_sparse = C(model_cls(params=params_sparse.detach()))
        loss_sparse = model_sparse(*data['in_train'].load())
        if loss_sparse < best_loss_sparse:
          best_loss_sparse = loss_sparse
          best_params_sparse = params_sparse
          best_mask = mask
          best_grads = grads

      # params = model_detached.params
      params = model_dense.params.sparse_update(best_mask, best_grads*(-lr))

      model_dense = C(model_cls(params=params.detach()))
      loss_test = model_dense(*data['in_test'].load())
      loss_test.backward()


      iter_pbar.set_description(f'optim_iteration[dense_loss:{loss_dense.tolist():5.5}/sparse_loss:{loss_prunned.tolist():5.5}]')
      # iter_pbar.set_description(f'optim_iteration[loss:{loss_dense.tolist()}/dist:{dist.tolist()}]')
      # torch.optim.SGD(sparse_params, lr=0.1).step()

      result_dict.append(loss=loss_detached)
      if not mode == 'train':
        result_dict.append(
            walltime=walltime,
            # **self.params_tracker(
            #   grad=grad,
            #   update=batch_arg.updates,
            # )
        )
    dist_mean = torch.stack(dist_list, 0).mean()
    import pdb; pdb.set_trace()

    return result_dict
