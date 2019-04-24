import argparse
import logging
import os
from itertools import product
from collections import OrderedDict, namedtuple

from config import Config, getConfig, test_optimizers
import numpy as np
import torch
from utils import utils
from optimize import test_neural, test_normal, train_neural
from utils.result import ResultDict
from optimizers.optim_helpers import OptimizerParams

# optimizer_names = ['LSTM-base', 'LSTM-ours', 'LSTM-ours-sp-loss-0.00001']
optimizer_names = ['LSTM-base', 'LSTM-ours', 'ADAM']
# optimizer_names = ['ADAM']

logger = logging.getLogger('main')

parser = argparse.ArgumentParser(description='Pytorch implementation of L2L')
parser.add_argument('--problem', type=str, default='mnist',
                    choices=['debug', 'quadratic', 'mnist'],
                    help='problem for the optimizee to solve')
parser.add_argument('--title', type=str, default='mnist',
                    choices=['debug', 'quadratic', 'mnist'],
                    help='title of the plot')
parser.add_argument('--result_dir', type=str, default='result')
parser.add_argument('--load_dir', type=str, required=True)
parser.add_argument('--names', type=str, default=optimizer_names)
parser.add_argument('--subdirs', action='store_true',
                    help='load from sub-directories whose name will be used as '
                         'prefix of the optimizer name.')
parser.add_argument('--retest', nargs='*', type=str, default=[],
                    choices=optimizer_names, help='name list of optimizers'
                    'that will be forcibly retested even if loadable.')
parser.add_argument('--retest_all', action='store_true',
                    help='all the optimizers will be forcibly retested.')

def main():
  global optimizer_names
  args = parser.parse_args()
  # set load dir
  load_dir = os.path.join(args.result_dir, args.load_dir)
  if args.load_dir and not os.path.exists(load_dir):
    raise Exception(f'{load_dir} does NOT exist!')
  cfg = Config(getConfig(args.problem))
  cfg.update_from_parsed_args(args)
  problem = cfg.problem.dict
  neural_optimizers = cfg.neural_optimizers.dict
  normal_optimizers = cfg.normal_optimizers.dict
  test_optimizers = cfg.test_optimizers
  if args.retest_all:
    args.retest = test_optimizers

  ##############################################################################
  print('\n\n\nLoading saved optimizers..')
  params = {}
  results = {}

  if args.subdirs:
    # dirs = os.listdir(load_dir)
    dirs = [
      # 'sparsity_0.1',
      # 'sparsity_0.05',
      # 'sparsity_0.01',
      'drop_rate_0.0_no_mask',
      'drop_rate_0.5_no_relative_params_10_obsrv_grad_clip',
      # 'sparsity_0.001',
      # 'sparsity_0.0005',
      'dynamic_scaling_g_only_no_dropout_no_mask_time_encoding',
      # 'sparsity_0.00005',
      # 'sparsity_0.00001',class
      # 'sparsity_0.000005',
      'sparsity_0.000001',
      # 'baseline'
    ]
    dir_name = product(dirs, optimizer_names)
    optimizer_names = [os.path.join(dir, name) for (dir, name) in dir_name]

  import pdb; pdb.set_trace()

  for name in optimizer_names:
    # np.random.seed(0)
    # if not utils.is_loadable_result(load_dir, name, args.force_retest):
    if OptimizerParams.is_loadable(name, load_dir):
      params[name] = OptimizerParams.load(name, load_dir)

    subname = name.split('/')[-1]
    if ResultDict.is_loadable(name, load_dir):
      results[name] = ResultDict.load(name, load_dir)
    else:
      if subname not in args.retest:
        continue
      if subname in normal_optimizers:
        print(f'\n\nOptimizing with static optimizer: {name}')
        kwargs = normal_optimizers[subname]
        result = test_normal(name, load_dir, **problem, **kwargs)
        results[name] = result
      elif subname in neural_optimizers:
        print(f'\n\nOptimizing with learned optimizer: {name}')
        kwargs = neural_optimizers[subname]['test_args']
        result = test_neural(name, load_dir, params[name], **problem, **kwargs)
        results[name] = result

  ##############################################################################
  print('\nPlotting..')
  plotter = utils.Plotter(title=args.title, results=results, out_dir=None)
  # loss w.r.t. step_num
  # plotter.plot('grad_value', 'step_num', ['test_num', 'step_num', 'track_num'])
  plotter.plot('step_num', 'loss', ['test_num', 'step_num'],
               hue='optimizer', logscale=True)
               # mean_group=['optimizer', 'step_num'])
  import pdb; pdb.set_trace()
  plotter.plot('step_num', 'grad', ['test_num', 'step_num', 'track_num'])

  # loss w.r.t. walltime
  # plotter.plot('walltime', 'loss', ['test_num', 'step_num'],
  #              hue='optimizer',
  #              mean_group=['optimizer', 'step_num'])

  # grad_loss w.r.t. step_num (for specified neural opt name)
  # plotter.plot('step_num', 'grad_loss', ['test_num', 'step_num'],
  #              mean_group=['optimizer', 'step_num'],
  #              visible_names=['LSTM-ours'])

  # update w.r.t. step_num (for specified neural opt name)
  plotter.plot('step_num', 'update', ['test_num', 'step_num', 'track_num'],
               visible_names=['LSTM-base', 'LSTM-ours'])

  # grad w.r.t. step_num
  plotter.plot('step_num', 'grad', ['test_num', 'step_num', 'track_num'],
               visible_names=['LSTM-base', 'LSTM-ours'])

  # mu w.r.t. step_num
  plotter.plot('step_num', 'mu', ['test_num', 'step_num', 'track_num'],
               visible_names=['LSTM-ours'])

  # sigma w.r.t. step_num
  plotter.plot('step_num', 'sigma', ['test_num', 'step_num', 'track_num'],
               visible_names=['LSTM-ours'])

  # grad w.r.t. step_num (for specified neural opt name)
  # plotter.plot('step_num', 'grad_pred', ['test_num', 'step_num', 'track_num'],
  #              visible_names=['LSTM-ours'])
  # plotter.plot('step_num', 'grad_value', ['test_num', 'step_num', 'track_num'],
  #              hue='grad_type', visible_names=['LSTM-ours'])
  print('end')
  # utils.loss_plot(args.problem, results, save_dir, 'step_num')
  # utils.loss_plot(args.problem, results, save_dir, 'walltime')
  # utils.loss_plot(args.problem, [results[-1]], save_dir, 'step_num', 'grad_loss')
  #utils.plot(results, args.problem, names=test_optimizers, x='step_num')
  #utils.plot(results, args.problem, names=test_optimizers, x='wallclock_time')
  #utils.plot_update(updates, args.problem, names=list(neural_optimizers.keys()))
  # utils.plot_grid(updates[1], grids[0], args.problem, names='LSTM-ours')

  ##############################################################################


if __name__ == "__main__":
  main()
