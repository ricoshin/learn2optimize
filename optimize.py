import copy
import importlib
import os
import sys

import numpy as np
import torch
import torch.optim as optim
from models.mnist import MNISTData, MNISTModel
from models.mnist2 import MNISTModel2
from models.model_helpers import ParamsIndexTracker
from models.quadratic import QuadraticData, QuadraticModel
from tensorboardX import SummaryWriter
from torch.optim import SGD, Adam, RMSprop
from tqdm import tqdm
from utils import utils
from utils.result import ResultDict
from utils.utils import TFWriter, ReduceLROnPlateau, save_figure
from utils.timer import Walltime, WalltimeChecker
from utils.trunc_scheduler import Truncation
import matplotlib.pyplot as plt

C = utils.getCudaManager('default')
tqdm.monitor_interval = 0


def _get_attr_by_name(name):
  return getattr(sys.modules[__name__], name)


def _get_optim_by_name(name):
  return importlib.import_module("optimizers." + name).Optimizer


def log_pbar(result_dict, pbar, nitem_max=7):
  desc = [f'{k}: {v:3.4f}' for k, v in result_dict.items()][:nitem_max]
  pbar.set_description(f"{pbar.desc.split(' ')[0]} [{' | '.join(desc)}]")


def log_tf_event(writer, tag, result_dict, step, walltime=None, w_name='main'):
  try:
    if writer:
      for i, (k, v) in enumerate(result_dict.items()):
        writer[w_name].add_scalar(f'{tag}/{k}', v, step, walltime)
  except:
    import pdb; pdb.set_trace()


def train_neural(name, save_dir, data_cls, model_cls, optim_module,
  n_epoch=20, n_train=20, n_valid=100, iter_train=100, iter_valid=100,
  unroll=20, lr=0.001, preproc=False, out_mul=1.0, no_mask=False, k_obsrv=10,
  meta_optim='sgd'):
  """Function for meta-training and meta-validation of learned optimizers.
  """

  # Options
  lr_scheduling = True  # learning rate scheduling
  tr_scheduling = True  # truncation scheduling

  print(f'data_cls: {data_cls}')
  print(f'model_cls: {model_cls}')
  data_cls = _get_attr_by_name(data_cls)
  model_cls = _get_attr_by_name(model_cls)
  optim_cls = _get_optim_by_name(optim_module)

  optimizer = C(optim_cls())
  if len([p for p in optimizer.parameters()]) == 0:
    return None  # no need to be trained if there's no parameter
  writer = TFWriter(save_dir, name) if save_dir else None
  # TODO: handle variable arguments according to different neural optimziers

  """meta-optimizer"""
  meta_optim = {'sgd': 'SGD', 'adam': 'Adam'}[meta_optim.lower()]


  if 'obsrv' not in optim_module:
    wd = 1e-5
    print(f'meta optimizer: {meta_optim} / lr: {lr} / wd: {wd}\n')
    meta_optim = getattr(torch.optim, meta_optim)(
      optimizer.parameters(), lr=lr, weight_decay=wd)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #   meta_optim, milestones=[1, 2, 3, 4], gamma=0.1)
  else:
    wd = 1e-5
    # mask_lr = 0.1  # seperate mask lr
    mask_lr = lr  # same mask lr
    print(f'meta optimizer: {meta_optim} / gen_lr: {lr} '
          f'/ mask_lr:{mask_lr} / wd: {wd}\n')
    p_feat = optimizer.feature_gen.parameters()
    p_step = optimizer.step_gen.parameters()
    p_mask = optimizer.mask_gen.parameters()
    meta_optim = getattr(torch.optim, meta_optim)([
        {'params':p_feat, 'lr':lr, 'weight_decay':wd},
        {'params':p_step, 'lr':lr, 'weight_decay':wd},
        {'params':p_mask, 'lr':mask_lr},
    ])

  if lr_scheduling:
    lr_scheduler = ReduceLROnPlateau(
      meta_optim, mode='min', factor=0.5, patience=1, verbose=True)

  if tr_scheduling:
    tr_scheduler = Truncation(
      unroll, mode='min', factor=1.5, patience=1, verbose=True)

  data = data_cls()
  best_params = None
  best_valid = 999999
  best_converge = 999999
  result_outer = ResultDict()
  epoch_pbar = tqdm(range(n_epoch), 'epoch')

  for i in epoch_pbar:
    train_pbar = tqdm(range(n_train), 'outer_train')
    result_train = ResultDict()
    result_valid = ResultDict()

    # Meta-training
    result_outer = ResultDict()
    for j in train_pbar:
      train_data = data.sample_meta_train()
      result_inner, _ = optimizer.meta_optimize(
        meta_optim, train_data, model_cls, iter_train, unroll,
        out_mul, k_obsrv, no_mask, writer,'train')
      result_outer.append(result_inner)
      mean = result_inner.mean()
      log_pbar(mean, train_pbar)
      tr_scheduler.step(mean['test_nll'])
      if save_dir:
        step = n_train * i + j
        result_train.append(mean, step=step)
        log_tf_event(writer, 'meta_train_outer', mean, step)
    mean_over_train = result_outer.mean(0)
    if save_dir:
      save_figure(name, save_dir, writer, mean_over_train, i, 'train')

    # Meta-validation
    valid_pbar = tqdm(range(n_valid), 'outer_valid')
    result_outer = ResultDict()
    result_final = ResultDict()
    for j in valid_pbar:
      valid_data = data.sample_meta_valid()
      result_inner, params = optimizer.meta_optimize(
        meta_optim, valid_data, model_cls, iter_valid, unroll,
        out_mul, k_obsrv, no_mask, writer, 'valid')
      result_outer.append(result_inner)
      result_mean = result_inner.mean()
      final_model = C(model_cls(params=params))
      result_final.append(final_inner_test(
        final_model, valid_data['in_test'], mode='valid'))
      log_pbar(result_mean ,valid_pbar)
    mean_over_valid = result_outer.mean(0)
    final_mean_over_valid = result_final.mean(0)
    #import pdb; pdb.set_trace()
    if lr_scheduling:
      last_converge = final_mean_over_valid['loss_mean']
      import pdb; pdb.set_trace()
      if last_converge < best_converge:
        best_converge = last_converge
      if last_converge < 0.5:
        lr_scheduler.step(last_converge, i)
    # Log TF-event: averaged valid loss
    mean_all = result_outer.mean()
    if save_dir:
      step = n_train * (i + 1)
      result_valid.append(mean_all, step=step)
      result_valid = result_valid.w_postfix('mean')
      result_valid.append(
        result_inner.getitem(-1).sub('test_nll').w_postfix('final'))
      log_tf_event(writer, 'meta_valid_outer', mean_all, step)
      log_tf_event(writer, 'meta_valid_outer', final_mean_over_valid, step)
      save_figure(name, save_dir, writer, mean_over_valid, i, 'valid')

    # Save current snapshot
    last_valid = final_mean_over_valid['loss_mean']
    if last_valid < best_valid:
      best_valid = last_valid
      optimizer.params.save(name, save_dir)
      best_params = copy.deepcopy(optimizer.params)
    if (i % 10 ==0) and (i>0):
      optimizer.params.save('{}_epoch{}'.format(name, i), save_dir)
    # Update epoch progress bar
    result_epoch = dict(last_valid=last_valid, best_valid=best_valid)
    log_pbar(result_epoch, epoch_pbar)

  if save_dir:
    result_train.save_as_csv(name, save_dir, 'meta_train_outer', True)
    result_valid.save_as_csv(name, save_dir, 'meta_valid_outer', True)
  return best_params


def test_neural(name, save_dir, learned_params, data_cls, model_cls,
                optim_module, n_test=100, iter_test=100, preproc=False,
                out_mul=1.0, no_mask=False, k_obsrv=10):
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
  result_final = ResultDict()  # test loss & acc for the whole testset
  tests_pbar = tqdm(range(n_test), 'test')

  # Meta-test
  for j in tests_pbar:
    data = data_cls().sample_meta_test()
    result_inner, params = optimizer.meta_optimize(meta_optim, data, model_cls,
      iter_test, unroll, out_mul, k_obsrv, no_mask, writer, 'test')
    result_outer.append(result_inner)
    # NOTE: we are not using best_test snapshot for final inner evaluation
    #   since there is no guarentee that it has reached the minimum
    #   w.r.t. the whole test set.
    model = C(model_cls(params=params))
    result_final.append(final_inner_test(model, data['in_test'], mode='test'))
    last_test = result_inner.mean()['test_nll']
    mean_test = result_outer.mean()['test_nll']
    if last_test < best_test:
      best_test = last_test
    result_test = dict(
      last_test=last_test, mean_test=mean_test, best_test=best_test)
    log_pbar(result_test, tests_pbar)
    if save_dir:
      save_figure(name, save_dir, writer, result_inner, j, 'test')
  mean_over_test = result_outer.mean(0)

  # TF-events for inner loop (train & test)
  if save_dir:
    for i in range(iter_test):
      mean = mean_over_test.getitem(i)
      walltime = mean_over_test['walltime'][i]
      log_tf_event(writer, 'meta_test_inner', mean, i, walltime)
    result_outer.save(name, save_dir)
    result_outer.save_as_csv(name, save_dir, 'meta_test_inner')
    result_final.save_as_csv(name, save_dir, 'meta_test_final', trans_1d=True)

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
  result_final = ResultDict()  # test loss & acc for the whole testset
  best_test = 999999
  # params_tracker = ParamsIndexTracker(n_tracks=10)

  # Outer loop (n_test)
  for j in tests_pbar:
    data = data_cls().sample_meta_test()
    model = C(model_cls())
    optimizer = optim_cls(model.parameters(), **optim_args)

    walltime = Walltime()
    result_inner = ResultDict()
    iter_pbar = tqdm(range(iter_test), 'inner_train')

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
      result_iter = dict(
        train_nll=train_nll.tolist(),
        test_nll=test_nll.tolist(),
        train_acc=train_acc.tolist(),
        test_acc=test_acc.tolist(),
        walltime=walltime.time,
        )
      log_pbar(result_iter, iter_pbar)
      result_inner.append(result_iter)
    if save_dir:
      save_figure(name, save_dir, writer, result_inner, j, 'normal')


    result_outer.append(result_inner)
    result_final.append(final_inner_test(model, data['in_test'], mode='test'))
    # NOTE: we are not using best_test snapshot for final inner evaluation
    #   since there is no guarentee that it has reached the minimum
    #   w.r.t. the whole test set.
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
    result_final.save_as_csv(name, save_dir, 'meta_test_final', trans_1d=True)

  return result_outer

def final_inner_test(model, data, mode='test', batch_size=2000):
  data_new = data.from_dataset(
    data.dataset, batch_size, rand_with_replace=False)
  loss_mean = []
  acc_mean = []
  sizes = []
  n_batch = range(len(data_new))
  for x, y in tqdm(data_new.iterator, 'final_inner_{}'.format(mode)):
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
  return {'loss_mean': loss_mean, 'acc_mean': acc_mean}


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
