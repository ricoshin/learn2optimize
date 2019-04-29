import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_helpers import ParamsFlattener, ParamsIndexTracker
# from nn.maskgen_rnn import MaskGenerator
from nn.maskgen_bnn import FeatureGenerator, MaskGenerator, StepGenerator
from nn.rnn_base import RNNBase
from optimize import log_pbar, log_tf_event
from optimizers.optim_helpers import (BatchManagerArgument, DefaultIndexer,
                                      OptimizerBatchManager, OptimizerParams,
                                      OptimizerStates, StatesSlicingWrapper)
from tqdm import tqdm
from utils import utils
from utils.lipschitz_estim import (eval_lipschitz, random_masked_params,
                                   inversed_masked_params)
from utils.result import ResultDict
from utils.torchviz import make_dot

C = utils.getCudaManager('default')
debug_sigint = utils.getDebugger('SIGINT')
debug_sigstp = utils.getDebugger('SIGTSTP')


class Optimizer(nn.Module):
  def __init__(self, hidden_sz=32, sb_mode='unified'):
    super().__init__()
    assert sb_mode in ['none', 'normal', 'unified']
    self.hidden_sz = hidden_sz
    self.sb_mode = sb_mode
    self.feature_gen = FeatureGenerator(hidden_sz)
    self.step_gen = StepGenerator(hidden_sz)
    self.mask_gen = MaskGenerator(hidden_sz)
    self.params_tracker = ParamsIndexTracker(n_tracks=10)

  @property
  def params(self):
    return OptimizerParams.from_optimizer(self)

  @params.setter
  def params(self, optimizer_params):
    optimizer_params.to_optimizer(self)

  def meta_optimize(self, meta_optimizer, data, model_cls, optim_it, unroll,
                    out_mul, tf_writer=None, mode='train'):
    assert mode in ['train', 'valid', 'test']
    if mode == 'train':
      self.train()
    else:
      self.eval()

    from tensorboardX import SummaryWriter
    save_dir = 'result/0428/lips_9'
    writer_best = SummaryWriter(os.path.join(save_dir, 'best'))
    writer_any = SummaryWriter(os.path.join(save_dir, 'any'))
    writer_rand = SummaryWriter(os.path.join(save_dir, 'rand'))
    writer_inv = SummaryWriter(os.path.join(save_dir,  'inv'))
    writer_dense = SummaryWriter(os.path.join(save_dir,  'dense'))

    result_dict = ResultDict()
    unroll_losses = 0
    walltime = 0
    test_kld = torch.tensor(0.)

    params = C(model_cls()).params
    self.feature_gen.new()
    self.step_gen.new()
    iter_pbar = tqdm(range(1, optim_it + 1), 'Inner_loop')
    iter_watch = utils.StopWatch('Inner_loop')

    set_size = {'layer_0': 500, 'layer_1': 10}  # NOTE: make it smarter

    for iteration in iter_pbar:
      if debug_sigint.signal_on:
        debug_1 = iteration == 1 or iteration % 10 == 0
      else:
        debug_1 = False
      if debug_sigstp.signal_on:
        debug_2 = True
      else:
        debug_2 = False

      cand_params = []
      cand_losses = []
      sparse_r = {}  # sparsity
      lips_dense = {}
      lips_any = {}
      lips_best = {}
      lips_rand = {}
      lips_inv = {}
      best_loss = 9999999
      best_params = None
      n_samples = 5

      iter_watch.touch()
      model_train = C(model_cls(params=params.detach()))
      train_nll = model_train(*data['in_train'].load())
      train_nll.backward()

      g = model_train.params.grad.flat.detach()
      w = model_train.params.flat.detach()

      feature, v_sqrt = self.feature_gen(g)

      size = params.size().unflat()
      kld = self.mask_gen(feature, size)

      for i in range(n_samples):
        # step & mask genration
        mask = self.mask_gen.sample_mask()
        step_out = self.step_gen(feature, v_sqrt, debug=debug_1)
        step = params.new_from_flat(step_out[0])

        # prunning
        mask = ParamsFlattener(mask)
        mask_layout = mask.expand_as(params)
        step_sparse = step * mask_layout
        params_sparse = params + step_sparse
        params_pruned = params_sparse.prune(mask > 1e-8)


        if params_pruned.size().unflat()['mat_0'][1] == 0:
          continue
        if debug_2:
          import pdb
          pdb.set_trace()
        # cand_loss = model(*outer_data_s)
        sparse_model = C(model_cls(params=params_pruned.detach()))
        loss = sparse_model(*data['in_train'].load())
        import pdb; pdb.set_trace()
        try:
          if (loss < best_loss) or i == 0:
            best_loss = loss
            best_params = params_sparse
            best_pruned = params_pruned
            best_mask = mask
        except:
          best_params = params_sparse
          best_pruned = params_pruned
          best_mask = mask

      # measures
      for k, v in set_size.items():
        r = (best_mask > 1e-8).sum().unflat[k].tolist() / v
        n = k.split('_')[1]
        sparse_r[f"sparse_{n}"] = r

      if iteration % 10 == 0:

        params_pruned_inv = inversed_masked_params(params, best_mask, step, set_size, sparse_r)
        params_pruned_rand = random_masked_params(params, step, set_size, sparse_r)

        for k, v in set_size.items():
          n = k.split('_')[1]
          r = sparse_r[f"sparse_{n}"]
          lips_best[f"lips_{n}"] = eval_lipschitz(best_pruned, n).tolist()[0]
          lips_any[f"lips_{n}"] = eval_lipschitz(params_pruned, n).tolist()[0]
          lips_rand[f"lips_{n}"] = eval_lipschitz(params_pruned_rand, n).tolist()[0]
          lips_inv[f"lips_{n}"] = eval_lipschitz(params_pruned_inv, n).tolist()[0]
          lips_dense[f"lips_{n}"] = eval_lipschitz(params_sparse, n).tolist()[0]

      if not mode == 'train':
        # meta-test time cost (excluding inner-test time)
        walltime += iter_watch.touch('interval')

      if best_params is not None:
        params = best_params

      model_test = C(model_cls(params=params))
      test_nll = utils.isnan(model_test(*data['in_test'].load()))
      test_kld = kld / data['in_test'].full_size
      total_test = test_nll + test_kld

      if debug_2:
        import pdb
        pdb.set_trace()

      if mode == 'train':
        if not iteration % unroll == 0:
          unroll_losses += total_test
        else:
          meta_optimizer.zero_grad()
          unroll_losses.backward()
          nn.utils.clip_grad_value_(self.parameters(), 0.01)
          meta_optimizer.step()
          unroll_losses = 0

      if not mode == 'train' or iteration % unroll == 0:
        # self.mask_gen.detach_lambdas_()
        self.step_gen.lr = self.step_gen.log_lr.detach()
        params = params.detach_()

      if mode == 'train':
        # meta-training time cost
        walltime += iter_watch.touch('interval')


      # import pdb; pdb.set_trace()
      # result dict
      result = dict(
          train_nll=train_nll.tolist(),
          test_nll=test_nll.tolist(),
          test_kld=test_kld.tolist(),
          walltime=walltime,
          **sparse_r,
      )

      result_dict.append(result)
      log_pbar(result, iter_pbar)

      if iteration % 10 == 0:
        lips_b = dict(
          lips_t=np.prod([l for l in lips_best.values()]),
          **lips_best
        )
        lips_a = dict(
          lips_t=np.prod([l for l in lips_any.values()]),
          **lips_any
        )
        lips_r = dict(
          lips_t=np.prod([l for l in lips_rand.values()]),
          **lips_rand
        )
        lips_i = dict(
          lips_t=np.prod([l for l in lips_inv.values()]),
          **lips_inv
        )
        lips_d = dict(
          lips_t=np.prod([l for l in lips_dense.values()]),
          **lips_dense
        )

        if tf_writer :
          log_tf_event(result, tf_writer, iteration, 'meta-test')
          log_tf_event(lips_b, writer_best, iteration, 'lipschitz constant')
          log_tf_event(lips_a, writer_any, iteration, 'lipschitz constant')
          log_tf_event(lips_r, writer_rand, iteration, 'lipschitz constant')
          log_tf_event(lips_i, writer_inv, iteration, 'lipschitz constant')
          log_tf_event(lips_d, writer_dense, iteration, 'lipschitz constant')

      # desc = [f'{k}: {v:5.5}' for k, v in result.items()]
      # iter_pbar.set_description(f"inner_loop [ {' / '.join(desc)} ]")
    return result_dict
