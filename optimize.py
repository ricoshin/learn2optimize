import copy
import importlib
import os
import sys

import numpy as np
import torch
import torch.optim as optim
from models.mnist import MNISTData, MNISTModel
from models.model_helpers import ParamsIndexTracker
from models.quadratic import QuadraticData, QuadraticModel
from optimizers import (neural_base, neural_estim, neural_estim_sgd,
                        neural_obsrv, neural_sparse, neural_sparse_mask)
from tensorboardX import SummaryWriter
from torch.optim import SGD, Adam, RMSprop
from tqdm import tqdm
from utils import utils
from utils.result import ResultDict

C = utils.getCudaManager('default')
tqdm.monitor_interval = 0


def _get_attr_by_name(name):
  return getattr(sys.modules[__name__], name)


def _get_optim_by_name(name):
  return importlib.import_module("optimizers." + name).Optimizer


def train_neural(name, save_dir, data_cls, model_cls, optim_module, n_epoch=20,
                 n_train=20, n_valid=100, optim_it=100, unroll=20, lr=0.001,
                 preproc=False, out_mul=1.0, tf_write=True):
  data_cls = _get_attr_by_name(data_cls)
  model_cls = _get_attr_by_name(model_cls)
  optim_cls = _get_optim_by_name(optim_module)

  optimizer = C(optim_cls())
  if tf_write and save_dir is not None:
    writer = SummaryWriter(os.path.join(save_dir, name))
  # TODO: handle variable arguments according to different neural optimziers

  # meta_optim = torch.optim.SGD(
    # optimizer.parameters(), lr=0.2, momentum=0.9, weight_decay=1e-6)
  meta_optim = torch.optim.Adam(
    optimizer.parameters(), lr=0.01, weight_decay=1e-5)
  # scheduler = torch.optim.lr_scheduler.MultiStepLR(
  #   meta_optim, milestones=[1, 2, 3, 4], gamma=0.1)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    meta_optim, mode='min', factor=0.5, patience=0, cooldown=0, verbose=1)
  print(f'lr: {lr}')

  # step_lr = 1e-2
  # mask_lr = 1e-2
  # p_feat = optimizer.feature_gen.parameters()
  # p_step = optimizer.step_gen.parameters()
  # p_mask = optimizer.mask_gen.parameters()
  #
  # meta_optim = optim.Adam([
  #     {'params':p_feat, 'lr':step_lr, 'weight_decay':1e-4},
  #     {'params':p_step, 'lr':step_lr, 'weight_decay':1e-4},
  #     {'params':p_mask, 'lr':mask_lr},
  # ])

  # print(f'step_lr: {step_lr} / mask_lr: {mask_lr}')
  data = data_cls()
  best_params = None
  best_valid_loss = 999999
  result_dict = ResultDict()
  epoch_pbar = tqdm(range(n_epoch), 'epoch')

  for i in epoch_pbar:
    train_pbar = tqdm(range(n_train), 'train')
    result_dict = ResultDict()

    for j in train_pbar:
      train_data = data.sample_meta_train()
      result_dict_ = optimizer.meta_optimize(meta_optim, train_data,
        model_cls, optim_it, unroll, out_mul, 'train')
      train_nll = result_dict_['train_nll'].mean()
      test_nll = result_dict_['test_nll'].mean()
      test_kld = result_dict_['test_kld'].mean()
      test_total = result_dict_['test_total'].mean()
      # result_dict.append(test_num=j, loss_ever=iter_loss, **result_dict_)
      train_pbar.set_description(
        f'train[train_nll:{train_nll:10.3f}]'
        f'train[test_nll:{test_nll:10.3f}]'
        f'train[test_kld:{train_kld:10.3f}]'
        f'train[test_total:{test_total:10.3f}]'
        )
      if tf_write and save_dir is not None:
        writer.add_scalar('0_train/0_outer/0_nll', train_nll, j)
        writer.add_scalar('1_test/0_outer/0_nll', test_nll, j)
        writer.add_scalar('1_test/0_outer/1_kld', train_kld, j)
        writer.add_scalar('1_test/0_outer/2_nll', train_nll, j)


    mean_train_loss = result_dict['loss_ever'].mean()
    result_dict = ResultDict()
    valid_pbar = tqdm(range(n_valid), 'valid')

    for j in valid_pbar:
      test_data = data.sample_meta_test()
      result_dict_ = optimizer.meta_optimize(meta_optim, test_data,
        model_cls, optim_it, unroll, out_mul, 'valid')
      iter_loss = result_dict_['loss'].sum()
      result_dict.append(test_num=j, loss_ever=iter_loss, **result_dict_)
      valid_pbar.set_description(f'valid[loss:{iter_loss:10.3f}]')

    mean_valid_loss = result_dict['loss_ever'].mean()
    # scheduler.step(mean_valid_loss)

    if tf_write and save_dir is not None:
      writer.add_scalar('data/mean_valid_loss', mean_valid_loss, i)

    if mean_valid_loss < best_valid_loss:
      best_valid_loss = mean_valid_loss
      optimizer.params.save(name, save_dir)
      best_params = copy.deepcopy(optimizer.params)

    epoch_pbar.set_description(f'epochs[last_valid_loss:{mean_valid_loss:10.3f}'
                               f'/best_valid_loss:{best_valid_loss:10.3f}]')
  return best_params


def test_neural(name, save_dir, learned_params, data_cls, model_cls,
                optim_module, n_test=100, optim_it=100, preproc=False,
                out_mul=1.0):
  data_cls = _get_attr_by_name(data_cls)
  model_cls = _get_attr_by_name(model_cls)
  optim_cls = _get_optim_by_name(optim_module)

  optimizer = C(optim_cls())
  optimizer.params = learned_params
  data = data_cls()
  meta_optim = None
  unroll = 1
  np.random.seed(0)

  result_dict = ResultDict()
  pbar = tqdm(range(n_test), 'test')

  for i in pbar:
    result_dict_ = optimizer.meta_optimize(
        meta_optim, data, model_cls, optim_it, unroll, out_mul, 'test')
    result_dict.append(test_num=i, **result_dict_)
    loss_total = result_dict_['loss'].sum()
    pbar.set_description(f'test[loss:{loss_total:10.3f}]')

  return result_dict.save(name, save_dir)


def test_normal(name, save_dir, data_cls, model_cls, optim_cls, optim_args,
                n_test, optim_it):
  data_cls = _get_attr_by_name(data_cls)
  model_cls = _get_attr_by_name(model_cls)
  optim_cls = _get_attr_by_name(optim_cls)

  tests_pbar = tqdm(range(n_test), 'test')
  result_dict_test = ResultDict()
  params_tracker = ParamsIndexTracker(n_tracks=10)

  for i in tests_pbar:
    data = data_cls().loaders['test']
    model = C(model_cls())
    optimizer = optim_cls(model.parameters(), **optim_args)

    iter_pbar = tqdm(range(optim_it), 'optim_iteration')
    iter_watch = utils.StopWatch('optim_iteration')
    result_dict_iter = ResultDict()
    walltime = 0

    for j in iter_pbar:
      iter_watch.touch()
      loss = model(*data.load())
      optimizer.zero_grad()
      loss.backward()
      # before = model.params.detach('hard').flat
      optimizer.step()
      # update = model.params.detach('hard').flat - before
      # grad = model.params.get_flat().grad
      walltime += iter_watch.touch('interval')
      iter_pbar.set_description(f'optim_iteration[loss:{loss:10.6f}]')
      result_dict_iter.append(
          step_num=j, loss=loss, walltime=walltime,
          # **params_tracker(
          #     grad=grad,
          #     update=update,
          # ),
      )
    result_dict_test.append(test_num=i, **result_dict_iter)
    loss = result_dict_iter['loss'].sum()
    tests_pbar.set_description(f'test[loss:{loss:10.3f}]')  # , time:{time}')

  return result_dict_test.save(name, save_dir)


def find_best_lr(lr_list, data_cls, model_cls, optim_module, optim_args, n_test,
                 optim_it):
  data_cls = _get_attr_by_name(data_cls)
  model_cls = _get_attr_by_name(model_cls)
  optim_cls = _get_attr_by_name(optim_module).Optimizer
  best_loss = 999999
  best_lr = 0.0
  lr_list = [
      1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001
  ]
  for lr in tqdm(lr_list, 'Learning rates'):
    try:
      optim_args['lr'] = lr
      loss = np.mean([np.sum(s) for s in test_normal(
          data_cls, model_cls, optim_cls, optim_args, n_test, optim_it)])
    except RuntimeError:
      print('RuntimeError!')
      pass
    if loss < best_loss:
      best_loss = loss
      best_lr = lr
  return best_loss, best_lr
