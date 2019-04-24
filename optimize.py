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


def log_pbar(result_dict, pbar, keys=None):
  desc = [f'{k}: {v:5.5f}' for k, v in result_dict.items()\
    if keys is None or k not in keys]
  pbar.set_description(f"{pbar.desc.split(' ')[0]} [ {' / '.join(desc)} ]")


def log_tf_event(result_dict, writer, step, tag):
  if writer:
    for i, (k, v) in enumerate(result_dict.items()):
      writer.add_scalar(f'{tag}/{k}', v, step)


def train_neural(
  name, save_dir, data_cls, model_cls, optim_module, n_epoch=20, n_train=20,
  n_valid=100, iter_train=100, iter_valid=100, unroll=20, lr=0.001,
  preproc=False, out_mul=1.0, tf_write=True):
  """Function for meta-training and meta-validation of learned optimizers.
  """

  data_cls = _get_attr_by_name(data_cls)
  model_cls = _get_attr_by_name(model_cls)
  optim_cls = _get_optim_by_name(optim_module)

  optimizer = C(optim_cls())
  if tf_write and save_dir is not None:
    tf_writer = SummaryWriter(os.path.join(save_dir, name))
    # writer = SummaryWriter(os.path.join(save_dir, name))
  else:
    tf_writer = None
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
  best_valid = 999999
  result_outer = ResultDict()
  epoch_pbar = tqdm(range(n_epoch), 'epoch')

  for i in epoch_pbar:
    train_pbar = tqdm(range(n_train), 'outer_train')
    result_outer = ResultDict()

    # Meta-training
    for j in train_pbar:
      train_data = data.sample_meta_train()
      result_inner = optimizer.meta_optimize(meta_optim, train_data,model_cls,
        iter_train, unroll, out_mul, tf_writer, 'train')
      mean = result_inner.mean()
      log_pbar(mean, train_pbar)
      log_tf_event(mean, tf_writer, (i * n_train + j), 'meta_train_outer')

    # Meta-validation
    valid_pbar = tqdm(range(n_valid), 'outer_valid')
    for j in valid_pbar:
      valid_data = data.sample_meta_valid()
      result_inner = optimizer.meta_optimize(meta_optim, valid_data, model_cls,
        iter_valid, unroll, out_mul, tf_writer, 'valid')
      result_outer.append(result_inner)
      mean = result_inner.mean()
      log_pbar(mean, valid_pbar)

    # Log TF-event: averaged valid loss
    mean_over_all = result_outer.mean()
    log_tf_event(mean_over_all, tf_writer, i, 'meta_valid_outer')
    last_valid = mean_over_all['nll_test']
    # Save current snapshot
    if last_valid < best_valid:
      best_valid = last_valid
      optimizer.params.save(name, save_dir)
      best_params = copy.deepcopy(optimizer.params)
    # Update epoch progress bar
    result_epoch = dict(last_valid=last_valid, best_valid=best_valid)
    log_pbar(result_epoch, epoch_pbar)

  return best_params


def test_neural(name, save_dir, learned_params, data_cls, model_cls,
                optim_module, n_test=100, iter_test=100, preproc=False,
                out_mul=1.0, tf_write=True):
  """Function for meta-test of learned optimizers.
  """
  data_cls = _get_attr_by_name(data_cls)
  model_cls = _get_attr_by_name(model_cls)
  optim_cls = _get_optim_by_name(optim_module)

  if tf_write and save_dir is not None:
    tf_writer = SummaryWriter(os.path.join(save_dir, name))
  else:
    tf_writer = None

  optimizer = C(optim_cls())
  optimizer.params = learned_params
  meta_optim = None
  unroll = 1
  np.random.seed(0)

  best_test = 999999
  result_outer = ResultDict()
  tests_pbar = tqdm(range(n_test), 'test')

  # Meta-test
  for j in tests_pbar:
    data = data_cls().sample_meta_test()
    result_inner = optimizer.meta_optimize(meta_optim, data, model_cls,
      iter_test, unroll, out_mul, tf_writer, 'test')
    result_outer.append(result_inner)
    # Update test progress bar
    last_test = result_inner.mean()['nll_test']
    mean_test = result_outer.mean()['nll_test']
    if last_test < best_test:
      best_test = last_test
    result_test = dict(
      last_test=last_test, mean_test=mean_test, best_test=best_test)
    log_pbar(result_test, tests_pbar)

  # TF-events for inner loop (train & test)
  mean_over_j = result_outer.mean(0)
  for i in range(iter_test):
    log_tf_event(mean_over_j.getitem(i), tf_writer, i, 'meta_test_inner')

  return result_outer.save(name, save_dir)


def test_normal(name, save_dir, data_cls, model_cls, optim_cls, optim_args,
                n_test, iter_test, tf_write=True):
  """function for test of static optimizers."""
  data_cls = _get_attr_by_name(data_cls)
  model_cls = _get_attr_by_name(model_cls)
  optim_cls = _get_attr_by_name(optim_cls)

  if tf_write and save_dir is not None:
    tf_writer = SummaryWriter(os.path.join(save_dir, name))
  else:
    tf_writer = None

  tests_pbar = tqdm(range(n_test), 'outer_test')
  result_outer = ResultDict()
  best_test = 999999
  # params_tracker = ParamsIndexTracker(n_tracks=10)

  # Outer loop (n_test)
  for j in tests_pbar:
    data = data_cls().sample_meta_test()
    model = C(model_cls())
    optimizer = optim_cls(model.parameters(), **optim_args)

    iter_pbar = tqdm(range(iter_test), 'Inner_loop')
    watch = utils.StopWatch('Inner_loop')
    result_inner = ResultDict()
    walltime = 0

    # Inner loop (iter_test)
    for k in iter_pbar:
      watch.touch()
      # Inner-training loss
      nll_train = model(*data['in_train'].load())
      optimizer.zero_grad()
      nll_train.backward()
      # before = model.params.detach('hard').flat
      optimizer.step()
      walltime += watch.touch('interval')
      # Inner-test loss
      nll_test = model(*data['in_test'].load())
      # update = model.params.detach('hard').flat - before
      # grad = model.params.get_flat().grad
      result_iter = dict(
        nll_train=nll_train.tolist(),
        nll_test=nll_test.tolist(),
        walltime=walltime,
        )
      log_pbar(result_iter, iter_pbar)
      result_inner.append(result_iter)

    result_outer.append(result_inner)
    # Update test progress bar
    last_test = result_inner.mean()['nll_test']
    mean_test = result_outer.mean()['nll_test']
    if last_test < best_test:
      best_test = last_test
    result_test = dict(
      last_test=last_test, mean_test=mean_test, best_test=best_test)
    log_pbar(result_test, tests_pbar)

  # TF-events for inner loop (train & test)
  mean_over_j = result_outer.mean(0)
  for i in range(iter_test):
    log_tf_event(mean_over_j.getitem(i), tf_writer, i, 'meta_test_inner')

  return result_outer.save(name, save_dir)


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
