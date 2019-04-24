import json
import os
from collections import OrderedDict as odict

import torch

################################################################################

_problems = {
    #'debug': odict(data_cls='QuadraticData', model_cls='QuadraticModel'),
    'debug': odict(data_cls='MNISTData', model_cls='MNISTModel'),
    'quadratic': odict(data_cls='QuadraticData', model_cls='QuadraticModel'),
    'mnist': odict(data_cls='MNISTData', model_cls='MNISTModel'),
}

################################################################################

_neural_optimizers_debug = odict({
    'RNN-base': {
        'train_args': odict(n_epoch=3, n_train=5, n_valid=5, optim_it=100,
                            unroll=20, lr=0.005), # 0.001
    },
    'Proposed': {
        'train_args': odict(n_epoch=10, n_train=5, n_valid=1, optim_it=100,
                            unroll=20, lr=0.009), # 0.003
    },
})

_neural_optimizers_quadratic = odict({
    'RNN-base': {
        'train_args': odict(n_epoch=50, n_train=20, n_valid=20, optim_it=100,
                           unroll=20, lr=0.002),
    },
    'Proposed': {
        'train_args': odict(n_epoch=50, n_train=20, n_valid=20, optim_it=100,
                           unroll=20, lr=0.002),
    },
})

_neural_optimizers_mnist = odict({
    'RNN-base': {
        'train_args': odict(n_epoch=50, n_train=10, n_valid=10, iter_train=300,
                            iter_valid=1000, unroll=20, lr=0.0007),
    },
    'Proposed': {
        'train_args': odict(n_epoch=2, n_train=2, n_valid=2, iter_train=10,
                            iter_valid=10, unroll=20, lr=0.01),
    },
})

_neural_optimizers = odict({
    'debug': _neural_optimizers_debug,
    'quadratic': _neural_optimizers_quadratic,
    'mnist': _neural_optimizers_mnist,
})

_common_neural_args = {
    'RNN-base': odict(optim_module='neural_base', preproc=True, out_mul=0.1),
    'Proposed': odict(optim_module='neural_bnn_obsrv', preproc=True, out_mul=0.1),
}

################################################################################

_normal_optimizers_quadratic = odict({
    'SGD': {
        'optim_cls': 'SGD',
        'optim_args': odict(lr=0.01),
    },
    'RMSprop': {
        'optim_cls': 'RMSprop',
        'optim_args': odict(lr=0.03),
    },
    'NAG': {
        'optim_cls': 'SGD',
        'optim_args': odict(lr=0.01, nesterov=True, momentum=0.9),
    },
    'ADAM': {
        'optim_cls': 'Adam',
        'optim_args': odict(lr=0.1),
    },
})

_normal_optimizers_mnist = odict({
    'SGD': {
        'optim_cls': 'SGD',
        'optim_args': odict(lr=1.0),
    },
    'RMSprop': {
        'optim_cls': 'RMSprop',
        'optim_args': odict(lr=0.01),
    },
    'NAG': {
        'optim_cls': 'SGD',
        'optim_args': odict(lr=1.0, nesterov=True, momentum=0.9),
    },
    'ADAM': {
        'optim_cls': 'Adam',
        'optim_args': odict(lr=0.03),
    },
})

_normal_optimizers = odict({
  'debug': _normal_optimizers_mnist, # TODO
  'quadratic': _normal_optimizers_quadratic,
  'mnist': _normal_optimizers_mnist,
})

_common_test_args_debug = odict(n_test=5, iter_test=200)
_common_test_args = odict(n_test=2, iter_test=10)

################################################################################

# train_optimizers = []
# train_optimizers = ['RNN-base']
train_optimizers = ['Proposed']
# train_optimizers = ['RNN-base','Proposed']
# train_optimizers = ['RNN-base', 'Proposed', 'Proposed']
# test_optimizers = ('SGD', 'RMSprop', 'NAG', 'ADAM', 'RNN-base', 'Proposed')
# test_optimizers = ['ADAM']
# test_optimizers = ['RNN-base']
# test_optimizers = ['Proposed']
test_optimizers = ['Proposed', 'ADAM']
# test_optimizers = ['RNN-base','Proposed', 'ADAM']
# test_optimizers = ('SGD', 'RMSprop', 'NAG', 'ADAM', 'RNN-base', 'Proposed')
# train_optimizers = ()
# test_optimizers = ('SGD')
# test_optimizers = ('SGD', 'RMSprop', 'NAG', 'ADAM')

################################################################################
"""config getters."""

CONFIG = odict()

def _get_problem(problem):
  if 'problem' not in CONFIG:
    CONFIG['problem'] = _problems[problem]
  return CONFIG['problem']

def _get_neural_optimizers(problem):
  if 'neural_optimizers' not in CONFIG:
    neural_optimizers = _neural_optimizers[problem]
    neural_optimizers_ = {}
    for name, optims in neural_optimizers.items():
      if name in train_optimizers:
        neural_optimizers_[name] = optims
    for name, args in _common_neural_args.items():
      if name in train_optimizers:
        neural_optimizers_[name]['train_args'].update(args)
        neural_optimizers_[name]['test_args'] = args
    for name, opt in neural_optimizers_.items():
      if name in train_optimizers:
        if problem == 'debug':
          opt['test_args'].update(_common_test_args_debug)
        else:
          opt['test_args'].update(_common_test_args)
    CONFIG['neural_optimizers'] = neural_optimizers_
  return CONFIG['neural_optimizers']

def _get_normal_optimizers(problem):
  if 'normal_optimizers' not in CONFIG:
    normal_optimizers = _normal_optimizers[problem]
    for name, opt in normal_optimizers.items():
      if problem == 'debug':
        opt.update(_common_test_args_debug)
      else:
        opt.update(_common_test_args)
    CONFIG['normal_optimizers'] = normal_optimizers
  return CONFIG['normal_optimizers']

def _get_test_optimizers():
  if 'test_optimizers' not in CONFIG:
    CONFIG['test_optimizers'] = test_optimizers
  return CONFIG['test_optimizers']

def _sanity_check():
  config_names = [
    'problem', 'neural_optimizers', 'normal_optimizers', 'test_optimizers']
  not_in = [name not in CONFIG for name in config_names]
  missing_names = [name for i, name in enumerate(config_names) if not_in[i]]
  if missing_names:
    raise Exception('Missing config(s): %s' % ', '.join(missing_names))
  for name in CONFIG['test_optimizers']:
    if not (name in CONFIG['normal_optimizers'] or
            name in CONFIG['neural_optimizers']):
      import pdb; pdb.set_trace()
      raise Exception(f"Unknown optimizer name: {name}")

# def _dump_as_json(out_dir):
#   _sanity_check()
#   dump_dict = odict()
#   for name in CONFIG['test_optimizers']:
#     if name in CONFIG['neural_optimizers']:
#       dump_dict[name] = CONFIG['neural_optimizers'][name]
#     else:
#       dump_dict[name] = CONFIG['normal_optimizers'][name]
#   with open(os.path.join(out_dir, 'config.json'), 'w') as f:
#     json.dump(dump_dict, f, indent=2)

def getConfig(problem='debug'):
  global CONFIG
  CONFIG = odict()
  _get_problem(problem)
  _get_neural_optimizers(problem)
  _get_normal_optimizers(problem)
  _get_test_optimizers()
  _sanity_check()
  return CONFIG

################################################################################

class Config(object):
  save_filename = 'config.json'
  def __init__(self, dict_=None):
    assert isinstance(dict_, dict)
    if dict_ is not None:
      self._nested_update(dict_)

  @classmethod
  def load(cls, load_dir):
    with open(os.path.join(load_dir, Config.save_filename), 'r') as f:
      config = cls(json.load(f))
    return config

  @classmethod
  def from_parsed_args(cls, args):
    config = cls(vars(args))
    return config

  def update_from_parsed_args(self, args):
    self._nested_update({'args': vars(args)})

  def items(self):
    return [(k, v) for k, v in self.__dict__.items()]

  def keys(self):
    return [k for k in self.__dict__.keys()]

  def values(self):
    return [v for v in self.__dict__.values()]

  def save(self, save_dir):
    if save_dir is None:
      return
    with open(os.path.join(save_dir, Config.save_filename), 'w') as f:
      json.dump(self.dict, f, indent=2)

  @property
  def dict(self):
    out_dict = odict()
    for k, v in self.__dict__.items():
      if isinstance(v, Config):
        out_dict[k] = v.dict
      else:
        out_dict[k] = v
    return out_dict

  def update(self, dict):
    assert isinstance(dict_, dict)
    self._nested_update(dict)

  def _nested_update(self, dict_):
    assert isinstance(dict_, dict)
    for k, v in dict_.items():
      if isinstance(v, dict):
        dict_[k] = Config(v)
    self.__dict__.update(dict_)

  def _pretty_print(self, dict_, indent_size=2, indent_init=0):
    out_print = "{ "
    indent_init += len(out_print)
    for i, (key, value) in enumerate(dict_.items()):
      indent_key = 0 if i == 0 else indent_init
      str_key = str(key) + ': '
      out_print += " " * indent_key + str_key
      indent_value = indent_key + len(str_key)
      if isinstance(value, Config):
        out_print += self._pretty_print(value, indent_size, indent_value)
      else:
        out_print += str(value) + ',\n'
    out_print += " " * indent_key + '}\n'
    return out_print

  def __repr__(self):
    return f'Config({json.dumps(self.dict, indent=2)})'
