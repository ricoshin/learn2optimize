import signal
import logging
import os
import time
from enum import Enum, unique

import matplotlib
import matplotlib.pyplot as plt
#import mock
import numpy as np
import pandas as pd
import seaborn as sns
import torch

_tensor_managers = {}
_cuda_managers = {}


# def to_gpu(cuda, obj):
#   return obj.cuda() if cuda else obj

_debuggers = {}


def getSignalCatcher(name):
  if name not in _debuggers:
    _debuggers.update({name: Debugger(name)})
  return _debuggers[name]
  # global _debugger
  # if _debugger is None:
  #   _debugger = Debugger()
  # return _debugger


class Debugger(object):
  def __init__(self, name):
    self.name = name
    self._signal = getattr(signal, name)
    self.signal_on = False
    self._set_sigint()

  def _set_sigint(self):
    def _toggle_signal(signal, frame):
      if self.signal_on:
        self.signal_on = False
        print(f'Signal {self.name} Off!')
      else:
        self.signal_on = True
        print(f'Signal {self.name} On!')
    signal.signal(self._signal, _toggle_signal)

  def debug(self):
    if self.signal_on:
      print(f"Signal {self.name}: Enter 'n' to go to the breakpoint!")
      import pdb; pdb.set_trace()


def getCudaManager(name):
  if name not in _cuda_managers:
    _cuda_managers.update({name: CUDAManager(name)})
  return _cuda_managers[name]


def soft_detach(object):
  if isinstance(object, torch.Tensor):
    tensor = self(torch.tensor(object.data, requires_grad=True))
    tensor.retain_grad()
    return tensor
  elif hasattr(object, 'detach'):
    import pdb; pdb.set_trace()
    return object.detach()
  else:
    import pdb; pdb.set_trace()
    raise TypeError(f"Unexpected input datatype {type(object)} for "
      "CUDAManger.detach(). Expected torch.Tensor or incorporate attribute "
      "detach() in itself.")


class CUDAManager(object):
  def __init__(self, cuda=None, name='default'):
    self.cuda = cuda
    self.name = name

  def set_cuda(self, cuda):
    self.cuda = cuda
    print(f"Global cuda manager '{self.name}' is set to {self.cuda}")
    return self



  def __call__(self, obj):
    if self.cuda is None:
      raise Exception("cuda configuration has to be set "
                      "by calling CUDAManager.set_cuda(boolean)")
    return obj.cuda() if self.cuda else obj


class MyFormatter(logging.Formatter):
  info_fmt = "%(message)s"
  else_fmt = '[%(levelname)s] %(message)s'

  def __init__(self, fmt="%(message)s"):
    logging.Formatter.__init__(self, fmt)

  def format(self, record):
    # Save the original format configured by the user
    # when the logger formatter was instantiated
    format_orig = self._fmt
    # Replace the original format with one customized by logging level
    if record.levelno == logging.INFO:
      self._fmt = MyFormatter.info_fmt
    else:
      self._fmt = MyFormatter.else_fmt
    # Call the original formatter class to do the grunt work
    result = logging.Formatter.format(self, record)
    # Restore the original format configured by the user
    self._fmt = format_orig

    return result


def set_logger(cfg):
  # log_fmt = '%(asctime)s %(levelname)s %(message)s'
  # date_fmt = '%d/%m/%Y %H:%M:%S'
  # formatter = logging.Formatter(log_fmt, datefmt=date_fmt)
  formatter = MyFormatter()

  # set log level
  levels = dict(debug=logging.DEBUG,
                info=logging.INFO,
                warning=logging.WARNING,
                error=logging.ERROR,
                critical=logging.CRITICAL)

  log_level = levels.get(cfg.log_level)

  # setup file handler
  file_handler = logging.FileHandler(cfg.log_path)
  file_handler.setFormatter(formatter)
  file_handler.setLevel(log_level)

  # setup stdio handler
  stream_handler = logging.StreamHandler()
  stream_handler.setFormatter(formatter)
  stream_handler.setLevel(log_level)

  # get logger
  logger = logging.getLogger('main')
  logger.setLevel(log_level)

  # add file & stdio handler to logger
  logger.addHandler(file_handler)
  logger.addHandler(stream_handler)


def prepare_dir(problem, result_dir, save_dir, post_digits=3):
  """result_dir is super dir that save_dir will be placed under.
    Unless specified, dir name will be 'problem_mmdd_(serial number).'
  """
  if result_dir is None:
    return None

  if save_dir:
    dir = os.path.join(result_dir, save_dir)
    if not os.path.exists(dir):
      os.makedirs(dir)
      print(f'Make new dir for ouputting params and result: {dir}')
    else:
      print(f'Overwrite existing dir: {dir}')
    return dir

  def _make_unique_dir(dir):
    if os.path.exists(dir):
      if len(os.listdir(dir)) > 0:
        dir = dir.split('_')
        n = int(dir.pop()) + 1
        dir = '_'.join([*dir, str(n).zfill(post_digits)])
        dir = _make_unique_dir(dir)
      else:
        print(f'Reusing already existing but empty dir: {dir}')
    else:
      os.makedirs(dir)
      print(f'Make new dir for outputting params and result: {dir}')
    return dir

  n = 0
  date = time.strftime("%m%d", time.gmtime())
  save_dir = '_'.join([problem, date, str(n).zfill(post_digits)])
  return _make_unique_dir(os.path.join(result_dir, save_dir))


class Plotter(object):
  def __init__(self, title, results, out_dir):
    assert isinstance(results, dict)
    self.title = title
    self.results = results
    self.out_dir = out_dir

  def plot(self, x, y,  dimnames, mean_group=None, hue=None,
           visible_names=None, logscale=False):
    # plotting

    if visible_names is None:
      visible_names = [name for name in self.results.keys()]
    data_frame = pd.DataFrame()
    for name, result in self.results.items():
      if visible_names is not None:
        if name not in visible_names:
          continue
      data_frame = data_frame.append(
        result.data_frame(name, dimnames), sort=True)

    import pdb; pdb.set_trace()

    sns.set(color_codes=True)
    sns.set_style('white')
    # import pdb; pdb.set_trace()
    if mean_group is not None:
      assert isinstance(mean_group, list)
      grouped = data_frame.groupby(mean_group)
      data_frame = grouped.mean().reset_index()

    # if y == 'model_loss':
    #   data_frame['model_loss'] = data_frame['model_loss'].fillna(data_frame['loss'])

    if y == 'grad_value':
      import pdb; pdb.set_trace()
      data_frame = data_frame[data_frame['track_num'] == 0]
      id_vars = ['optimizer', 'step_num', 'track_num']
      vars =  ['grad_real', 'grad_pred']
      data_frame = data_frame[id_vars + vars].melt(
        id_vars=id_vars, var_name='grad_type', value_name='grad_value')
      import pdb; pdb.set_trace()

    if y in ['grad', 'mu', 'sigma']:
      import pdb; pdb.set_trace()

    if y == 'grad_value':
      hue_order = vars
    else:
      hue_order = visible_names if hue is not None else None
      #palette = sns.color_palette(n_colors=len(hue_order)) if hue is not None else None

    # if y in ['update']:
    #   data_frame[y] = data_frame[y].abs()
    ax = sns.lineplot(data=data_frame, x=x, y=y,
                      hue=hue, hue_order=hue_order)
    ax.lines[-1].set_linestyle('-')
    # if hue is not None:
      # ax.legend()
    title = f"{self.title}_{y}_wrt_{x}"
    if logscale:
      plt.yscale('log')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    # if os.environ.get('DISPLAY', '') == '':
    #   plt.savefig('plot.png')
    # else:
    if self.out_dir is not None:
      plt.savefig(f'{self.out_dir}/{title}.png')
      print('Plot saved!')
    plt.show()
    print('Plot displayed!')


def plot_(data, title, names, x='step_num'):
  # plotting
  sns.set(color_codes=True)
  sns.set_style('white')

  data_new = []
  # [6, 2, 5, 200]
  for n_opt, data_opt in enumerate(data):
    for n_test, (t_test, l_test) in enumerate(zip(data_opt[0], data_opt[1])):
      for n_iter, (time, loss) in enumerate(zip(t_test, l_test)):
        data_new.append([names[n_opt], n_test, n_iter, time, loss])

  data_new = np.array(data_new)
  data_frame = pd.DataFrame({
      'optimizer': pd.Series(data_new[:, 0], dtype='str'),
      'test_num': pd.Series(data_new[:, 1], dtype='int'),
      'step_num': pd.Series(data_new[:, 2], dtype='int'),
      'wallclock_time': pd.Series(data_new[:, 3], dtype='float'),
      'loss': pd.Series(data_new[:, 4], dtype='float'),
  })
  # if os.environ.get('DISPLAY', '') == '':
  #   print('No display found. Just try to save figure using Agg backend!')
  #   matplotlib.use('Agg')
  #if x == 'wallclock_time':
  grouped = data_frame.groupby(['optimizer', 'step_num'])
  data_frame = grouped.mean().reset_index()

  palette = sns.color_palette(n_colors=len(names))
  ax = sns.lineplot(data=data_frame, palette=palette, x=x, y='loss',
                    hue='optimizer', hue_order=names)
  ax.lines[-1].set_linestyle('-')
  ax.legend()
  plt.yscale('log')
  plt.xlabel(x)
  plt.ylabel('loss')
  plt.title(title)
  # if os.environ.get('DISPLAY', '') == '':
  #   plt.savefig('plot.png')
  # else:
  plt.show()
  plt.savefig(f'plot_{x}.png')
  print('Plot displayed!')
  print('Plot saved!')


def plot_update(tracks, title, names, x='step_num'):
  # plotting
  sns.set(color_codes=True)
  sns.set_style('white')

  #for nerzip(tracks[0], tracks[1])
  data_new = []
  # [num_neural_opt, n_test, optim_it, n_tracks]
  for n_opt, v_opt in enumerate(tracks):
    for n_test, v_test in enumerate(v_opt):
      for n_iter, v_iter in enumerate(v_test):
        for n_track, v_track in enumerate(v_iter):

          data_new.append([names[n_opt], n_test, n_iter, n_track, v_track])

  # for n_opt, data_opt in enumerate(data):
  #   for n_test, (t_test, l_test) in enumerate(zip(data_opt[0], data_opt[1])):
  #     for n_iter, (time, loss) in enumerate(zip(t_test, l_test)):
  #       data_new.append([names[n_opt], n_test, n_iter, time, loss])
  data_new = np.array(data_new)
  data_frame = pd.DataFrame({
      'optimizer': pd.Series(data_new[:, 0], dtype='str'),
      'test_num': pd.Series(data_new[:, 1], dtype='int'),
      'step_num': pd.Series(data_new[:, 2], dtype='int'),
      'track_num': pd.Series(data_new[:, 3], dtype='int'),
      'update': pd.Series(data_new[:, 4], dtype='float'),
  })
  # if os.environ.get('DISPLAY', '') == '':
  #   print('No display found. Just try to save figure using Agg backend!')
  #   matplotlib.use('Agg')
  #if x == 'wallclock_time':
  #grouped = data_frame.groupby(['optimizer', 'step_num'])
  #data_frame = grouped.mean().reset_index()

  palette = sns.color_palette(n_colors=len(names))
  ax = sns.lineplot(data=data_frame, palette=palette, x=x, y='update',
                    hue='optimizer', hue_order=names)
  ax.lines[-1].set_linestyle('-')
  ax.legend()
  #plt.yscale('log')
  plt.xlabel(x)
  plt.ylabel('update')
  plt.title(title)
  # if os.environ.get('DISPLAY', '') == '':
  #   plt.savefig('plot.png')
  # else:
  plt.show()
  plt.savefig(f'plot_update_{x}.png')
  print('Plot displayed!')
  print('Plot saved!')



def plot_grid(update, grid, title, names, x='step_num'):
  # plotting
  sns.set(color_codes=True)
  sns.set_style('white')

  #update: [n_test, optim_it, n_track]
  #grid: [n_test, optim_it, n_track]

  #for nerzip(tracks[0], tracks[1])
  data_new = []
  # [num_neural_opt, n_test, optim_it, n_tracks]
  len_n_test = len(update)
  len_optim_it = len(update[0])
  len_n_track = len(update[0][0])
  mag = lambda x: np.sqrt(np.dot(x, x))
  for n_test in range(len_n_test):
    for n_optim in range(len_optim_it):
      for n_track in range(len_n_track):
        update_ = mag(update[n_test][n_optim][n_track])
        grid_ = mag(grid[n_test][n_optim][n_track])
        data_new.append([n_test, n_optim, n_track, 'update', update_])
        data_new.append([n_test, n_optim, n_track, 'grid', grid_])
        data_new.append([n_test, n_optim, n_track, 'ratio', update_/grid_])

  data_new = np.array(data_new)
  data_frame = pd.DataFrame({
      'test_num': pd.Series(data_new[:, 0], dtype='int'),
      'step_num': pd.Series(data_new[:, 1], dtype='int'),
      'track_num': pd.Series(data_new[:, 2], dtype='int'),
      'update/grid': pd.Series(data_new[:, 3], dtype='str'),
      'magnitude': pd.Series(data_new[:, 4], dtype='float'),
  })

  palette = sns.color_palette(n_colors=3)
  #import pdb; pdb.set_trace()
  df = data_frame.loc[data_frame['update/grid'].isin(['update','grid'])]
  ax = sns.lineplot(data=df, x=x, y='magnitude', legend=False,
                    hue='update/grid', hue_order=['update', 'grid'])

  #ax.lines[-1].set_linestyle('-')
  ax2 = ax.twinx()
  import pdb; pdb.set_trace()
  df = data_frame.loc[data_frame['update/grid'].isin(['ratio'])]
  sns.lineplot(data=df, ax=ax2, color='r', legend=False,
               x=x, y='magnitude')
  ax2.legend()
  #plt.yscale('log')
  plt.xlabel(x)
  plt.ylabel('magnitude')
  plt.title(title)
  # if os.environ.get('DISPLAY', '') == '':
  #   plt.savefig('plot.png')
  # else:
  plt.show()
  plt.savefig(f'plot_{x}_grid.png')
  print('Plot displayed!')
  print('Plot saved!')


@unique
class WatchMode(Enum):
  NONE = 0
  GO_STOP = 1
  TOUCH = 2


class StopWatch(object):

  def __init__(self, name='default'):
    self.name = name
    self.initial = None
    self.cumulative = [0]
    self.interval = []

  @property
  def average(self):
    if len(interval) == 0:
      return 0
    return self.interval / len(self.interval)

  def __enter__(self):
    return self.go()

  def __exit__(self, type, value, trace_back):
    self.stop()

  def go(self):
    if self.initial is not None:
      raise Exception(f'StopWatch[{self.name}] already started!')
    self.initial = time.time()
    return self

  def stop(self, mode=None, verbose=False):
    if self.initial is None:
      raise Exception(f'start StopWatch[{self.name}] first!')

    cumulative = time.time() - self.initial
    self.cumulative.append(cumulative)
    self.interval.append(self.cumulative[-1] - self.cumulative[-2])

    if mode == 'cumulative':
      out = self.cumulative[-1]
    elif mode == 'interval':
      out = self.interval[-1]
    else:
      out = None

    if verbose:
      print('\n' + self.format(out, mode) + '\n')

    return out

  def touch(self, mode='cumulative', verbose=False):
    if self.initial is None:
      self.go()
      return 0.0
    return self.stop(mode, verbose)


  def format(self, sec, mode, hms=False):
    outstr = f'StopWatch[{self.name}/{mode}] {sec}'
    if hms:
      outstr += time.strftime("(%Hhrs %Mmins %Ssecs)", time.gmtime(seconds))
    return outstr
