import argparse
import logging
import os
from collections import OrderedDict, namedtuple

from config import Config, getConfig, test_optimizers
import numpy as np
import torch
from utils import utils
from optimize import test_neural, test_normal, train_neural
from utils.result import ResultDict
from optimizers.optim_helpers import OptimizerParams


logger = logging.getLogger('main')
C = utils.getCudaManager('default')

parser = argparse.ArgumentParser(description='Pytorch implementation of L2L')
parser.add_argument('--problem', type=str, default='mnist',
                    choices=['debug', 'quadratic', 'mnist'],
                    help='problem for the optimizee to solve')
# parser.add_argument('--optimizer', type=str, default='lstm',
#                     help='type of neural optimizer')
parser.add_argument('--cpu', action='store_true',
                    help='disable CUDA')
# parser.add_argument('--max_epoch', type=int, default=10000, metavar='N',
#                     help='number of epoch (default: 10000)')
parser.add_argument('--result_dir', type=str, default='result')
parser.add_argument('--save_dir', type=str, default='')
#parser.add_argument('--desc', type=str, default='')
parser.add_argument('--load_dir', type=str, default='')
parser.add_argument('--retrain', nargs='*', type=str, default=[],
                    choices=test_optimizers, help='name list of optimizers'
                    'that will be forcibly retrained even if loadable.')
parser.add_argument('--retest', nargs='*', type=str, default=[],
                    choices=test_optimizers, help='name list of optimizers'
                    'that will be forcibly retested even if loadable.')
parser.add_argument('--retest_all', action='store_true',
                    help='all the optimizers will be forcibly retested.')
parser.add_argument('--volatile', action='store_true',
                    help='supress saving fuctions')


def main():
  args = parser.parse_args()
  # set CUDA
  args.cuda = not args.cpu and torch.cuda.is_available()
  C.set_cuda(args.cuda)
  # set load dir
  load_dir = os.path.join(args.result_dir, args.load_dir)
  if args.load_dir and not os.path.exists(load_dir):
    raise Exception(f'{load_dir} does NOT exist!')
  # set save dir: saving functions will be suppressed if save_dir is None
  args.result_dir = None if args.volatile else args.result_dir
  save_dir = utils.prepare_dir(args.problem, args.result_dir, args.save_dir)
  # set problem & config
  print(f'Problem: {args.problem}')
  cfg = Config(getConfig(args.problem))
  cfg.update_from_parsed_args(args)
  cfg.save(save_dir)

  problem = cfg.problem.dict
  neural_optimizers = cfg.neural_optimizers.dict
  normal_optimizers = cfg.normal_optimizers.dict
  test_optimizers = cfg.test_optimizers
  # TODO: force to call this by wrapping it in class
  if args.retest_all:
    args.retest = test_optimizers

  params = {}
  ##############################################################################
  print('\nMeta-training..')
  for name in [opt for opt in test_optimizers if opt in neural_optimizers]:
    if name not in args.retrain and OptimizerParams.is_loadable(name, load_dir):
      params[name] = OptimizerParams.load(name, load_dir).save(name, save_dir)
    else:
      print(f"\nTraining neural optimizer: {name}")
      kwargs = neural_optimizers[name]['train_args']
      print(f"Module name: {kwargs['optim_module']}")
      params[name] = train_neural(name, save_dir, **problem, **kwargs)
  ##############################################################################
  print('\n\n\nMeta-testing..')
  results = {}
  for name in test_optimizers:
    # np.random.seed(0)
    # if not utils.is_loadable_result(load_dir, name, args.force_retest):
    if name not in args.retest and ResultDict.is_loadable(name, load_dir):
      results[name] = ResultDict.load(name, load_dir).save(name, save_dir)
    else:
      if name in normal_optimizers:
        print(f'\nOptimizing with static optimizer: {name}')
        kwargs = normal_optimizers[name]
        result = test_normal(name, save_dir, **problem, **kwargs)
        results[name] = result
      elif name in neural_optimizers:
        print(f'\n\nOptimizing with learned optimizer: {name}')
        print(f"Module name: {kwargs['optim_module']}")
        kwargs = neural_optimizers[name]['test_args']
        result = test_neural(name, save_dir, params[name], **problem, **kwargs)
        results[name] = result

  ##############################################################################
  print('End of program.')


if __name__ == "__main__":
  main()
