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
from tensorboardX import SummaryWriter
from torch.optim import SGD, Adam, RMSprop
from tqdm import tqdm
from utils import utils
from utils.result import ResultDict
from utils.utils import TFWriter
from utils.timer import Walltime, WalltimeChecker

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


def log_tf_event(writer, tag, result_dict, step, walltime=None, w_name='main'):
  try:
    if writer:
      for i, (k, v) in enumerate(result_dict.items()):
        writer[w_name].add_scalar(f'{tag}/{k}', v, step, walltime)
  except:
    import pdb; pdb.set_trace()

def train_neural(
  name, save_dir, data_cls, model_cls, optim_module, n_epoch=20, n_train=20,
  n_valid=100, iter_train=100, iter_valid=100, unroll=20, lr=0.001,
  preproc=False, out_mul=1.0):
  """Function for meta-training and meta-validation of learned optimizers.
  """

  data_cls = _get_attr_by_name(data_cls)
  model_cls = _get_attr_by_name(model_cls)
  optim_cls = _get_optim_by_name(optim_module)

  optimizer = C(optim_cls())
  if len([p for p in optimizer.parameters()]) == 0:
    return None  # no need to be trained if there's no parameter

  writer = TFWriter(save_dir, name) if save_dir else None
  # TODO: handle variable arguments according to different neural optimziers

  """meta-optimizer"""
  meta_optim = 'SGD'
  # meta_optim = 'Adam'
  wd = 1e-5
  print(f'meta optimizer: {meta_optim} / lr: {lr} / wd: {wd}')
  meta_optim = getattr(torch.optim, meta_optim)
  meta_optim = meta_optim(optimizer.parameters(), lr=lr, weight_decay=wd)

  # scheduler = torch.optim.lr_scheduler.MultiStepLR(
  #   meta_optim, milestones=[1, 2, 3, 4], gamma=0.1)
  # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
  #   meta_optim, mode='min', factor=0.5, patience=0, cooldown=0, verbose=1)

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
    result_train = ResultDict()
    result_valid = ResultDict()

    # Meta-training
    for j in train_pbar:
      train_data = data.sample_meta_train()
      result_inner = optimizer.meta_optimize(meta_optim, train_data,model_cls,
        iter_train, unroll, out_mul, writer, 'train')
      mean = result_inner.mean()
      log_pbar(mean, train_pbar)
      if save_dir:
        step = n_train * i + j
        result_train.append(mean, step=step)
        log_tf_event(writer, 'meta_train_outer', mean, step)

    # Meta-validation
    valid_pbar = tqdm(range(n_valid), 'outer_valid')
    result_outer = ResultDict()
    for j in valid_pbar:
      valid_data = data.sample_meta_valid()
      result_inner = optimizer.meta_optimize(meta_optim, valid_data, model_cls,
        iter_valid, unroll, out_mul, writer, 'valid')
      result_outer.append(result_inner)
      log_pbar(result_inner.mean() , valid_pbar)

    # Log TF-event: averaged valid loss
    mean_all = result_outer.mean()
    if save_dir:
      step = n_train * (i + 1)
      result_valid.append(mean_all, step=step)
      log_tf_event(writer, 'meta_valid_outer', mean_all, step)
    # Save current snapshot
    last_valid = mean_all['test_nll']
    if last_valid < best_valid:
      best_valid = last_valid
      optimizer.params.save(name, save_dir)
      best_params = copy.deepcopy(optimizer.params)
    # Update epoch progress bar
    result_epoch = dict(last_valid=last_valid, best_valid=best_valid)
    log_pbar(result_epoch, epoch_pbar)

  if save_dir:
    result_train.save_as_csv(name, save_dir, 'meta_train_outer', True)
    result_valid.save_as_csv(name, save_dir, 'meta_valid_outer', True)
  return best_params


def test_neural(name, save_dir, learned_params, data_cls, model_cls,
                optim_module, n_test=100, iter_test=100, preproc=False,
                out_mul=1.0):
  """Function for meta-test of learned optimizers.
  """
  data_cls = _get_attr_by_name(data_cls)
  model_cls = _get_attr_by_name(model_cls)
  optim_cls = _get_optim_by_name(optim_module)

  writer = TFWriter(save_dir, name) if save_dir else None

  optimizer = C(optim_cls())
  if learned_params is not None:
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
      iter_test, unroll, out_mul, writer, 'test')
    result_outer.append(result_inner)
    # Update test progress bar
    last_test = result_inner.mean()['test_nll']
    mean_test = result_outer.mean()['test_nll']
    if last_test < best_test:
      best_test = last_test
    result_test = dict(
      last_test=last_test, mean_test=mean_test, best_test=best_test)
    log_pbar(result_test, tests_pbar)

  mean_over_j = result_outer.mean(0)
  # TF-events for inner loop (train & test)
  if save_dir:
    for i in range(iter_test):
      mean = mean_over_j.getitem(i)
      walltime = mean_over_j['walltime'][i]
      log_tf_event(writer, 'meta_test_inner', mean, i, walltime)
    result_outer.save(name, save_dir)
    result_outer.save_as_csv(name, save_dir, 'meta_test_inner')
  return result_outer


def test_normal(name, save_dir, data_cls, model_cls, optim_cls, optim_args,
                n_test, iter_test):
  """function for test of static optimizers."""
  data_cls = _get_attr_by_name(data_cls)
  model_cls = _get_attr_by_name(model_cls)
  optim_cls = _get_attr_by_name(optim_cls)

  writer = TFWriter(save_dir, name) if save_dir else None

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
    result_inner = ResultDict()
    walltime = Walltime()

    # Inner loop (iter_test)
    for k in iter_pbar:

      with WalltimeChecker(walltime):
        # Inner-training loss
        train_nll = model(*data['in_train'].load())
        optimizer.zero_grad()
        train_nll.backward()
        # before = model.params.detach('hard').flat
        optimizer.step()
      # Inner-test loss
      test_nll = model(*data['in_test'].load())
      # update = model.params.detach('hard').flat - before
      # grad = model.params.get_flat().grad
      result_iter = dict(
        train_nll=train_nll.tolist(),
        test_nll=test_nll.tolist(),
        walltime=walltime.time,
        )
      log_pbar(result_iter, iter_pbar)
      result_inner.append(result_iter)

    result_outer.append(result_inner)
    # Update test progress bar
    last_test = result_inner.mean()['test_nll']
    mean_test = result_outer.mean()['test_nll']
    if last_test < best_test:
      best_test = last_test
    result_test = dict(
      last_test=last_test, mean_test=mean_test, best_test=best_test)
    log_pbar(result_test, tests_pbar)

  # TF-events for inner loop (train & test)
  mean_over_j = result_outer.mean(0)
  if save_dir:
    for i in range(iter_test):
      mean = mean_over_j.getitem(i)
      walltime = mean_over_j['walltime'][i]
      log_tf_event(writer, 'meta_test_inner', mean, i, walltime)
    result_outer.save(name, save_dir)
    result_outer.save_as_csv(name, save_dir, 'meta_test_inner')
  return result_outer


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
