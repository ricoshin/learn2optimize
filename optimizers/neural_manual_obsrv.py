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
debug_sigint = utils.getSignalCatcher('SIGINT')
debug_sigstp = utils.getSignalCatcher('SIGTSTP')


class Optimizer(nn.Module):
  def __init__(self, hidden_sz=20, preproc_factor=10.0, preproc=False,
               n_layers=1, rnn_cell='gru', sb_mode='unified'):
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
    params = model.params
    lr_sgd = 0.2
    lr_adam = 0.001
    sgd = torch.optim.SGD(model.parameters(), lr=lr_sgd)
    adam = torch.optim.Adam(model.parameters())
    adagrad = torch.optim.Adagrad(model.parameters())
    # print('###', lr_adam, '####')

    iter_pbar = tqdm(range(1, optim_it + 1), 'optim_iteration')
    iter_watch = utils.StopWatch('optim_iteration')
    bp_watch = utils.StopWatch('bp')
    loss_decay = 1.0
    dist_list = []
    dense_rank = 0
    sparse_rank = 0
    dense_rank_list = []
    sparse_rank_list = []

    for iteration in iter_pbar:
      iter_watch.touch()

      # # conventional optimizers with API
      # model.zero_grad()
      # loss = model(*inner_data.load())
      # loss.backward()
      # adam.step()
      # loss_detached = loss

      model_detached = C(model_cls(params=params.detach()))
      loss_detached = model_detached(*inner_data.load())
      loss_detached.backward()

      # # conventional optimizers with manual SGD
      # params = model_detached.params + model_detached.params.grad * (-0.2)

      # iter_pbar.set_description(f'optim_iteration[loss:{loss_detached}]')

      masks = []

      # sparse_old_loss = []
      # sparse_new_loss = []
      sparse_loss = []
      dense_loss = []
      candidates = []

      mode = 2
      n_sample = 10

      if mode == 1:
        lr = 1.0
      elif mode == 2:
        lr = 0.2
      else:
        raise Exception()

      for i in range(n_sample):

        if i == 9:
          mask = MaskGenerator.topk(model_detached.activations.grad.detach())
        else:
          mask = MaskGenerator.randk(
            model_detached.activations.grad.detach())
            
        if model == 1:
          # recompute gradients again using prunned network
          sparse_params = model_detached.params.prune(mask)
          model_prunned = C(model_cls(params=sparse_params.detach()))
          loss_prunned = model_prunned(*inner_data.load())
          loss_prunned.backward()
          grads = model_prunned.params.grad
        elif mode == 2:
          # use full gradients but sparsified
          sparse_params = model_detached.params.prune(mask)
          grads = model_detached.params.grad.prune(mask)
        else:
          raise Exception('unknown model.')

        sparse_params = sparse_params + grads*(-lr)
        model_prunned = C(model_cls(params=sparse_params.detach()))
        loss_prunned = model_prunned(*inner_data.load())

        # params = model_detached.params
        params = model_detached.params.sparse_update(mask, grads*(-lr))
        candidates.append(params)
        model_dense = C(model_cls(params=params))
        loss_dense = model_dense(*inner_data.load())

        sparse_loss.append(loss_prunned)
        dense_loss.append(loss_dense)

      sparse_loss = torch.stack(sparse_loss, 0)
      dense_loss = torch.stack(dense_loss, 0)
      min_id = (sparse_loss.min()==sparse_loss).nonzero().squeeze().tolist()
      params = candidates[min_id]
      sparse_order = sparse_loss.sort()[1].float()
      dense_order = dense_loss.sort()[1].float()
      dist = (sparse_order - dense_order).abs().sum()
      dist_list.append(dist)

      dense_rank += (dense_order == 9).nonzero().squeeze().tolist()
      sparse_rank += (sparse_order == 9).nonzero().squeeze().tolist()

      dense_rank_list.append((dense_order == 9).nonzero().squeeze().tolist())
      sparse_rank_list.append((sparse_order == 9).nonzero().squeeze().tolist())

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
