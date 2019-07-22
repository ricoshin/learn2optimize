import copy
import importlib
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from config import Config
from datasets.loader_imagenet import ImageNetData
from datasets.loader_mnist import MNISTData
from datasets.loader_omniglot import OmniglotData
from models.model import Model
from models.model_helpers import ParamsIndexTracker
from models.quadratic import QuadraticData, QuadraticModel
from tensorboardX import SummaryWriter
from torch.optim import SGD, Adam, RMSprop
from tqdm import tqdm
from utils import utils
from utils.result import ResultDict
from utils.timer import Walltime, WalltimeChecker
from utils.trunc_scheduler import Truncation
from utils.utils import ReduceLROnPlateau, TFWriter, save_figure

C = utils.getCudaManager('default')
tqdm.monitor_interval = 0


def _get_attr_by_name(name):
  return getattr(sys.modules[__name__], name)


def _get_optim_by_name(name):
  return importlib.import_module("optimizers." + name).Optimizer


def get_lr(optim):
  i_groups = enumerate(optim.param_groups)
  return {'meta_lr_' + str(i): g['lr'] for i, g in i_groups}


def log_pbar(result_dict, pbar, nitem_max=7):
  try:
    desc = [f'{k}: {v:3.4f}' for k, v in result_dict.items()][:nitem_max]
    pbar.set_description(f"{pbar.desc.split(' ')[0]} [{' | '.join(desc)}]")
  except:
    import pdb
    pdb.set_trace()


def log_tf_event(writer, tag, result_dict, step, walltime=None, w_name='main'):
  try:
    if writer:
      for i, (k, v) in enumerate(result_dict.items()):
        writer[w_name].add_scalar(f'{tag}/{k}', v, step, walltime)
  except:
    import pdb
    pdb.set_trace()


def train_neural(name, cfg):
  """Function for meta-training and meta-validation of learned optimizers.
  """
  cfg = Config.merge(
    [cfg.problem, cfg.neural_optimizers[name].train_args, cfg.args])
  # NOTE: lr will be overwritten here.

  # Options
  lr_scheduling = False  # learning rate scheduling
  tr_scheduling = False  # Truncation scheduling
  seperate_lr = False
  print(f'data_cls: {cfg.data_cls}')
  print(f'model_cls: {cfg.model_cls}')
  data_cls = _get_attr_by_name(cfg.data_cls)
  model_cls = _get_attr_by_name(cfg.model_cls)
  optim_cls = _get_optim_by_name(cfg.optim_module)
  meta_optim = cfg.meta_optim

  optimizer = C(optim_cls())
  if len([p for p in optimizer.parameters()]) == 0:
    return None  # no need to be trained if there's no parameter
  writer = TFWriter(cfg.save_dir, name) if cfg.save_dir else None
  # TODO: handle variable arguments according to different neural optimziers

  """meta-optimizer"""
  meta_optim = {'sgd': 'SGD', 'adam': 'Adam'}[meta_optim.lower()]

  if cfg.mask_lr == 0.0:
    print(f'meta optimizer: {meta_optim} / lr: {cfg.lr} / wd: {cfg.wd}\n')
    meta_optim = getattr(torch.optim, meta_optim)(
        optimizer.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #   meta_optim, milestones=[1, 2, 3, 4], gamma=0.1)
  else:
    print(f'meta optimizer: {meta_optim} / gen_lr: {cfg.lr} '
          f'/ mask_lr:{cfg.mask_lr} / wd: {cfg.wd}\n')
    p_feat = optimizer.feature_gen.parameters()
    p_step = optimizer.step_gen.parameters()
    p_mask = optimizer.mask_gen.parameters()
    meta_optim = getattr(torch.optim, meta_optim)([
        {'params': p_feat, 'lr': cfg.lr, 'weight_decay': cfg.wd},
        {'params': p_step, 'lr': cfg.lr, 'weight_decay': cfg.wd},
        {'params': p_mask, 'lr': cfg.mask_lr},
    ])

  if lr_scheduling:
    lr_scheduler = ReduceLROnPlateau(
        meta_optim, mode='min', factor=0.5, patience=10, verbose=True)

  # tr_scheduler = Truncation(
  #     cfg.unroll, mode='min', factor=1.5, patience=1, max_len=50, verbose=True)

  data = data_cls()
  best_params = None
  best_valid_loss = 999999
  # best_converge = 999999

  train_result_mean = ResultDict()
  valid_result_mean = ResultDict()

  epoch_pbar = tqdm(range(cfg.n_epoch), 'epoch')
  for i in epoch_pbar:
    train_pbar = tqdm(range(cfg.n_train), 'outer_train')

    # Meta-training
    for j in train_pbar:
      train_data = data.sample_meta_train()
      result, _ = optimizer.meta_optimize(
        cfg, meta_optim, train_data, model_cls, writer, 'train')
      result_mean = result.mean()
      log_pbar(result_mean, train_pbar)
      if cfg.save_dir:
        step = (cfg.n_train * i) + j
        train_result_mean.append(result_mean, step=step)
        log_tf_event(writer, 'meta_train_outer', result_mean, step)

    if cfg.save_dir:
      save_figure(name, cfg.save_dir, writer, result, i, 'train')

    valid_pbar = tqdm(range(cfg.n_valid), 'outer_valid')
    result_all = ResultDict()
    result_final = ResultDict()

    # Meta-validation
    for j in valid_pbar:
      valid_data = data.sample_meta_valid()
      result, params = optimizer.meta_optimize(
        cfg, meta_optim, valid_data, model_cls, writer, 'valid')

      result_all.append(**result)
      result_final.append(final_inner_test(
          C(model_cls(params)), valid_data['in_test'], mode='valid'))
      log_pbar(result.mean(), valid_pbar)

    result_final_mean = result_final.mean()
    result_all_mean = result_all.mean().w_postfix('mean')
    last_valid_loss = result_final_mean['final_loss_mean']
    last_valid_acc = result_final_mean['final_acc_mean']

    # Learning rate scheduling
    if lr_scheduling:  # and last_valid < 0.5:
      lr_scheduler.step(last_valid_loss, i)

    # Truncation scheduling
    if tr_scheduling:
      tr_scheduler.step(last_valid_loss, i)

    # Save TF events and figures
    if cfg.save_dir:
      step = cfg.n_train * (i + 1)
      result_mean = ResultDict(**result_all_mean, **result_final_mean,
                               **get_lr(meta_optim), step=step) #trunc_len=tr_scheduler.len, step=step)
      valid_result_mean.append(result_mean)
      log_tf_event(writer, 'meta_valid_outer', result_mean, step)
      save_figure(name, cfg.save_dir, writer, result_all.mean(0), i, 'valid')

    # Save the best snapshot
    if last_valid_loss < best_valid_loss:
      best_valid_loss = last_valid_loss
      best_valid_acc = last_valid_acc
      best_params = copy.deepcopy(optimizer.params)
      if cfg.save_dir:
        optimizer.params.save(name, cfg.save_dir)

    # Update epoch progress bar
    result_epoch = dict(
        best_valid_loss=best_valid_loss,
        best_valid_acc=best_valid_acc,
        last_valid_loss=last_valid_loss,
        last_valid_acc=last_valid_acc,
    )
    log_pbar(result_epoch, epoch_pbar)

  if cfg.save_dir:
    train_result_mean.save_as_csv(
        name, cfg.save_dir, 'meta_train_outer', True)
    valid_result_mean.save_as_csv(
        name, cfg.save_dir, 'meta_valid_outer', True)

  return best_params


def test_neural(name, cfg, learned_params):
  """Function for meta-test of learned optimizers.
  """
  cfg = Config.merge(
    [cfg.problem, cfg.neural_optimizers[name].test_args, cfg.args])
  data_cls = _get_attr_by_name(cfg.data_cls)
  model_cls = _get_attr_by_name(cfg.model_cls)
  optim_cls = _get_optim_by_name(cfg.optim_module)
  writer = TFWriter(cfg.save_dir, name) if cfg.save_dir else None

  optimizer = C(optim_cls())
  if learned_params is not None:
    optimizer.params = learned_params

  meta_optim = None
  unroll = 1

  best_test_loss = 999999
  result_all = ResultDict()
  result_final = ResultDict()  # test loss & acc for the whole testset
  tests_pbar = tqdm(range(cfg.n_test), 'test')

  # Meta-test
  for j in tests_pbar:
    data = data_cls().sample_meta_test()
    result, params = optimizer.meta_optimize(
        cfg, meta_optim, data, model_cls, writer, 'test')
    result_all.append(result)
    result_final.append(final_inner_test(
        C(model_cls(params)), data['in_test'], mode='test'))
    result_final_mean = result_final.mean()
    last_test_loss = result_final_mean['final_loss_mean']
    last_test_acc = result_final_mean['final_acc_mean']

    if last_test_loss < best_test_loss:
      best_test_loss = last_test_loss
      best_test_acc = last_test_acc

    result_test = dict(
        best_test_loss=best_test_loss,
        best_test_acc=best_test_acc,
        last_test_loss=last_test_loss,
        last_test_acc=last_test_acc,
    )
    log_pbar(result_test, tests_pbar)
    if cfg.save_dir:
      save_figure(name, cfg.save_dir, writer, result, j, 'test')
  result_all_mean = result_all.mean(0)

  # TF-events for inner loop (train & test)
  if cfg.save_dir:
    # test_result_mean = ResultDict()
    for i in range(cfg.iter_test):
      mean_i = result_all_mean.getitem(i)
      walltime = result_all_mean['walltime'][i]
      log_tf_event(writer, 'meta_test_inner', mean_i, i, walltime)
      # test_result_mean.update(**mean_i, step=i, walltime=walltime)
    result_all.save(name, cfg.save_dir)
    result_all.save_as_csv(name, cfg.save_dir, 'meta_test_inner')
    result_final.save_as_csv(
        name, cfg.save_dir, 'meta_test_final', trans_1d=True)

  return result_all


def test_normal(name, cfg):
  """function for test of static optimizers."""

  cfg = Config.merge(
    [cfg.problem, cfg.args, cfg.neural_optimizers[name].test_args])
  data_cls = _get_attr_by_name(cfg.problem.data_cls)
  model_cls = _get_attr_by_name(cfg.problem.model_cls)
  optim_cls = _get_attr_by_name(cfg.problem.optim_cls)

  writer = TFWriter(cfg.save_dir, name) if cfg.save_dir else None
  tests_pbar = tqdm(range(cfg.n_test), 'outer_test')
  result_all = ResultDict()
  result_final = ResultDict()  # test loss & acc for the whole testset
  best_test_loss = 999999
  # params_tracker = ParamsIndexTracker(n_tracks=10)

  # Outer loop (n_test)
  for j in tests_pbar:
    data = data_cls().sample_meta_test()
    model = C(model_cls())
    optimizer = optim_cls(model.parameters(), **cfg.optim_args)

    walltime = Walltime()
    result = ResultDict()
    iter_pbar = tqdm(range(cfg.iter_test), 'inner_train')

    # Inner loop (iter_test)
    for k in iter_pbar:

      with WalltimeChecker(walltime):
        # Inner-training loss
        train_nll, train_acc = model(*data['in_train'].load())
        optimizer.zero_grad()
        train_nll.backward()
        # before = model.params.detach('hard').flat
        optimizer.step()
      # Inner-test loss

      test_nll, test_acc = model(*data['in_test'].load())
      # update = model.params.detach('hard').flat - before
      # grad = model.params.get_flat().grad
      result_ = dict(
          train_nll=train_nll.tolist(),
          test_nll=test_nll.tolist(),
          train_acc=train_acc.tolist(),
          test_acc=test_acc.tolist(),
          walltime=walltime.time,
      )
      log_pbar(result_, iter_pbar)
      result.append(result_)

    if cfg.save_dir:
      save_figure(name, cfg.save_dir, writer, result, j, 'normal')
    result_all.append(result)
    result_final.append(final_inner_test(model, data['in_test'], mode='test'))

    result_final_mean = result_final.mean()
    last_test_loss = result_final_mean['final_loss_mean']
    last_test_acc = result_final_mean['final_acc_mean']
    if last_test_loss < best_test_loss:
      best_test_loss = last_test_loss
      best_test_acc = last_test_acc
    result_test = dict(
        best_test_loss=best_test_loss,
        best_test_acc=best_test_acc,
        last_test_loss=last_test_loss,
        last_test_acc=last_test_acc,
    )
    log_pbar(result_test, tests_pbar)

  # TF-events for inner loop (train & test)
  mean_j = result_all.mean(0)
  if cfg.save_dir:
    for i in range(cfg.iter_test):
      mean = mean_j.getitem(i)
      walltime = mean_j['walltime'][i]
      log_tf_event(writer, 'meta_test_inner', mean, i, walltime)
    result_all.save(name, cfg.save_dir)
    result_all.save_as_csv(name, cfg.save_dir, 'meta_test_inner')
    result_final.save_as_csv(
        name, cfg.save_dir, 'meta_test_final', trans_1d=True)

  return result_all


def final_inner_test(model, data, mode='test', batch_size=2000):
  data_new = data.from_dataset(
      data.dataset, batch_size, rand_with_replace=False)
  loss_mean = []
  acc_mean = []
  sizes = []
  n_batch = range(len(data_new))
  for x, y in tqdm(data_new.iterator, 'Final_inner_{}'.format(mode)):
    loss, acc = model(x, y)
    loss_mean.append(loss)
    acc_mean.append(acc)
    sizes.append(y.size(0))
  # average weighted by the corresponding batch size
  loss_mean = torch.stack(loss_mean, 0)
  acc_mean = torch.stack(acc_mean, 0)
  sizes = C(torch.tensor(sizes)).float()
  loss_mean = (loss_mean * sizes).sum() / sizes.sum()
  acc_mean = (acc_mean * sizes).sum() / sizes.sum()
  return {'final_loss_mean': loss_mean, 'final_acc_mean': acc_mean}


def find_best_lr(name, save_dir, lr_list, data_cls, model_cls, optim_cls,
                 optim_args, n_test, iter_test):
  #data_cls = _get_attr_by_name(data_cls)
  #model_cls = _get_attr_by_name(model_cls)
  #optim_cls = _get_attr_by_name(optim_cls)
  best_loss = 999999
  best_lr = 0.0
  lr_list = [
      1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001
  ]
  for lr in tqdm(lr_list, 'Learning rates'):
    try:
      optim_args['lr'] = lr
      loss = np.mean([np.sum(s) for s in test_normal(name, save_dir,
                                                     data_cls, model_cls, optim_cls, optim_args, n_test, iter_test)])
      print('lr: {} loss: {}'.format(lr, loss))
    except RuntimeError:
      print('RuntimeError!')
      pass
    if loss < best_loss:
      best_loss = loss
      best_lr = lr
  return best_loss, best_lr
