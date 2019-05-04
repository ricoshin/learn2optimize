import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_helpers import ParamsIndexTracker, ParamsFlattener
# from nn.maskgen_rnn import MaskGenerator
from nn.maskgen_bnn import FeatureGenerator, StepGenerator, MaskGenerator
from optimizers.optim_helpers import (BatchManagerArgument, DefaultIndexer,
                                      OptimizerBatchManager, OptimizerParams,
                                      OptimizerStates, StatesSlicingWrapper)
from nn.rnn_base import RNNBase
from tqdm import tqdm
from utils import utils
from utils.result import ResultDict
from utils.torchviz import make_dot
from optimize import log_pbar
###
#import sys
#sys.path.append('/home/user/Personal/Baek/CS671/loss-landscape')
#from plot_surface import *
import matplotlib; matplotlib.use('agg')
from nn.maskgen_topk import MaskGenerator as MaskGenerator2
import os
import seaborn as sns
###
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
    self.loss = 0
  @property
  def params(self):
    return OptimizerParams.from_optimizer(self)

  @params.setter
  def params(self, optimizer_params):
    optimizer_params.to_optimizer(self)

  def meta_optimize(self, meta_optimizer, data, model_cls, optim_it, unroll,
                    out_mul, tf_writer=None, mode='train'):
    assert mode in ['train', 'valid', 'test', 'draw']
    if mode == 'train':
      self.train()
    else:
      self.eval()

    result_dict = ResultDict()
    unroll_losses = 0
    walltime = 0
    test_kld = torch.tensor(0.)

    params = C(model_cls()).params
    self.feature_gen.new()
    self.step_gen.new()
    iter_pbar = tqdm(range(1, optim_it + 1), 'Inner_loop')
    iter_watch = utils.StopWatch('Inner_loop')

    layer_size = {'layer_0': 500, 'layer_1': 10}  # NOTE: make it smarter
    ##############################################################
    mask_dict = ResultDict()
    ##############################################################
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
      best_loss = 9999999
      best_params = None
      n_samples = 10

      ##############################################################
      plotting_mask = False
      topk = True
      draw_loss = True
      ##############################################################
      iter_watch.touch()
      model_train = C(model_cls(params=params.detach()))
      input_data = data['in_train'].load()
      train_nll = model_train(*input_data)
      train_nll.backward()

      g = model_train.params.grad.flat.detach()
      w = model_train.params.flat.detach()

      feature, v_sqrt = self.feature_gen(g)

      size = params.size().unflat()
      kld = self.mask_gen(feature, size)
      ##############################################################
      result_dir = 'result/mask_compare'
      if not os.path.exists(result_dir):
          os.makedirs(result_dir)
      sample_num = 100
      if mode == 'test' and plotting_mask:
        for i in range(sample_num):
          mask = self.mask_gen.sample_mask()
          mask = ParamsFlattener(mask)
          mask_flat = mask.flat
          mask.unflat
          if i == 0:
            mask_cat = mask_flat.unsqueeze(dim=0)
            mask_sum = mask
          else:
            mask_cat = torch.cat((mask_cat, mask_flat.unsqueeze(dim=0)), dim=0)
            mask_sum += mask
        mask_mean, mask_var = mask_cat.mean(dim=0), mask_cat.var(dim=0)
        mask_sum /= sample_num
        mask = mask>0.5

        grad = model_train.params.grad.detach()
        act = model_train.activations.detach()
        for k, v in layer_size.items():
          r = (mask > 0.5).sum().unflat[k].tolist() / v
          sparse_r[f"sparse_{k.split('_')[1]}"] = r

        mask_gen = MaskGenerator2.topk if topk else MaskGenerator2.randk
        layer_0_topk = mask_gen(grad=grad, act=act, topk=sparse_r['sparse_0'])._unflat['layer_0'].view(-1)
        layer_0_topk = layer_0_topk>0.5
        layer_1_topk = mask_gen(grad=grad, act=act, topk=sparse_r['sparse_1'])._unflat['layer_1'].view(-1)
        layer_1_topk = layer_1_topk>0.5
        #import pdb; pdb.set_trace()
        layer_0_prefer_topk = mask_gen(grad=mask_sum.expand_as(params), act=act, topk=sparse_r['sparse_0'])._unflat['layer_0'].view(-1)
        layer_1_prefer_topk = mask_gen(grad=mask_sum.expand_as(params), act=act, topk=sparse_r['sparse_1'])._unflat['layer_1'].view(-1)
        layer_0_prefer_topk = layer_0_prefer_topk>0.5
        layer_1_prefer_topk = layer_1_prefer_topk>0.5
        layer_0 = torch.cat([grad.unflat['mat_0'],
        grad.unflat['bias_0'].unsqueeze(0)], dim=0).abs().sum(0)
        layer_1 = torch.cat([grad.unflat['mat_1'],
        grad.unflat['bias_1'].unsqueeze(0)], dim=0).abs().sum(0)
        layer_0_abs = ParamsFlattener({'layer_0': layer_0})
        layer_1_abs = ParamsFlattener({'layer_1': layer_1})
        #import pdb; pdb.set_trace()
        mask_result = self.plot_masks(mask, layer_0_topk, layer_1_topk, mask_sum, layer_0_prefer_topk, layer_1_prefer_topk, layer_0, layer_1, result_dir, iteration, sparse_r)
        mask_dict.append(mask_result)
        
      ##############################################################
      #import pdb; pdb.set_trace()
      for i in range(n_samples):
        # step & mask genration
        mask = self.mask_gen.sample_mask()
        mask = ParamsFlattener(mask)
        step = self.step_gen(feature, v_sqrt, debug=debug_1)
        step_ = params.new_from_flat(step[0])

        # prunning
        
        mask_layout = mask.expand_as(params)
        step_ = step_ * mask_layout
        params_ = params + step_
        sparse_params = params_.prune(mask > 1e-6)

        if sparse_params.size().unflat()['mat_0'][1] == 0:
          continue
        if debug_2:
          import pdb; pdb.set_trace()
        # cand_loss = model(*outer_data_s)
        sparse_model = C(model_cls(params=sparse_params.detach()))
        loss = sparse_model(*data['in_train'].load())
        try:
          if loss < best_loss:
            best_loss = loss
            best_params = params_
            best_mask = mask
            del loss, params_, mask
        except:
          best_params = params_
          best_mask = mask
          del params_, mask
        del sparse_params, sparse_model, mask_layout

      if best_params is not None:
        params = best_params

      for k, v in layer_size.items():
        r = (best_mask > 0.5).sum().unflat[k].tolist() / v
        sparse_r[f"sparse_{k.split('_')[1]}"] = r

      if not mode == 'train':
        # meta-test time cost (excluding inner-test time)
        walltime += iter_watch.touch('interval')

      model_test = C(model_cls(params=params))
      test_nll = utils.isnan(model_test(*data['in_test'].load()))
      kld_teset = kld / data['in_test'].full_size
      total_test = test_nll +  test_kld

      if debug_2:
        import pdb; pdb.set_trace()

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
      #iter_pbar = tqdm(range(1, 5 + 1), 'Draw_loop')
      ##############################################################
      if draw_loss and ((iteration-1) % 20 == 0) and (iteration >1):
        num_loss = 20
        X = np.linspace(-2.0, 0.5, num_loss)
        Y = np.linspace(-2.0, 0.5, num_loss)

        model_train = C(model_cls(params=params.detach()))
        #step_data = data['in_train'].load()
        step_data = input_data
        train_nll = model_train(*step_data)
        train_nll.backward()

        g = model_train.params.grad.flat.detach()
        w = model_train.params.flat.detach()

        feature, v_sqrt = self.feature_gen(g)

        size = params.size().unflat()
        kld = self.mask_gen(feature, size)
        # step & mask genration
        mask = self.mask_gen.sample_mask()
        mask = ParamsFlattener(mask)
        mask_layout = mask.expand_as(params)
        step_X = self.step_gen(feature, v_sqrt, debug=debug_1)
        step_X = params.new_from_flat(step_X[0]) * mask_layout
        step_X = step_X.flat.view(-1)
        step_Y = model_train.params.grad.flat.view(-1) 
        step_X_ = params.new_from_flat(-1.0 * step_X)
        step_X2_ = params.new_from_flat(-10.0 * step_X)
        step_Y_ = params.new_from_flat(1.0 * step_Y)
        step_Y2_ = params.new_from_flat(0.1 * step_Y)
        #import pdb; pdb.set_trace()
        #step_Y = step_Y * step_X.abs().sum() / step_Y.abs().sum()
        L2_X = (step_X * step_X).sum()
        L2_Y = (step_Y * step_Y).sum()

        layer_settings = [['mat_0', 'bias_0', 'mat_1', 'bias_1'], ['mat_0', 'bias_0'], ['mat_1', 'bias_1']]
        result_dirs = ['result/savefig_all', 'result/savefig_layer0', 'result/savefig_layer1']
        for layer_set, result_dir in zip(layer_settings, result_dirs):
          if not os.path.exists(result_dir):
            os.makedirs(result_dir)
          step_X_ = params.new_from_flat(-1.0 * step_X)
          step_Y_ = params.new_from_flat(1.0 * step_Y)
          #step_X2_ = params.new_from_flat(-10.0 * step_X)
          #step_Y2_ = params.new_from_flat(0.1 * step_Y)
          abs_X = step_X_.abs().sum()
          abs_Y = step_Y_.abs().sum()
          scale_X, scale_Y = 0, 0
          for layer in layer_set:
            scale_X += abs_X.unflat[layer].item()
            scale_Y += abs_Y.unflat[layer].item()
          scale_g = scale_X/scale_Y
          #import pdb; pdb.set_trace()
          #step_Y_ = step_Y_ * (step_X_.abs().sum() /  step_Y_.abs().sum())
          
          Z_X = get_1D_Loss(X, step_X_, 1.0, step_X.size(), layer_set, data['in_train'], model_cls, params)
          Z_Y = get_1D_Loss(Y, step_Y_, scale_g, step_Y.size(), layer_set,data['in_train'], model_cls, params)
          #Z_X2 = get_1D_Loss(X, step_X2_, scale, step_X.size(), layer_set, data['in_train'], model_cls, params)
          #Z_Y2 = get_1D_Loss(Y, step_Y2_, scale,step_Y.size(), layer_set,data['in_train'], model_cls, params)
          plot_2d(X, Z_X, os.path.join(result_dir, 'iter_{}_STEPxMASK_1dLoss.png'.format(iteration)))
          plot_2d(Y, Z_Y, os.path.join(result_dir,'iter_{}_1.0xGradient_1dLoss.png'.format(iteration)))
          #plot_2d(Y, Z_X2, os.path.join(result_dir,'iter_{}_10xStepxMASK_1dLoss.png'.format(iteration)))
          #plot_2d(Y, Z_Y2, os.path.join(result_dir,'iter_{}_0.1xGradient_1dLoss.png'.format(iteration)))
          
          

      # result dict
      ##############################################################
      result = dict(
        train_nll=train_nll.tolist(),
        test_nll=test_nll.tolist(),
        test_kld=test_kld.tolist(),
        walltime=walltime,
        **sparse_r
      )
      result_dict.append(result)
      #import pdb; pdb.set_trace()
      log_pbar(result, iter_pbar)
      # desc = [f'{k}: {v:5.5}' for k, v in result.items()]
      # iter_pbar.set_description(f"inner_loop [ {' / '.join(desc)} ]")

      ### Draw Loss landscape for whole dataset / batch ###
  
    if mode == 'test' and plotting_mask:
      result_dir = 'result/savefig2'
      if not os.path.exists(result_dir):
        os.makedirs(result_dir)
      plot_2d(mask_dict['sparse_0'], os.path.join(result_dir,'sparse_0.png'), 'sparse ratio of layer 0')
      plot_2d(mask_dict['sparse_1'], os.path.join(result_dir,'sparse_1.png'), 'sparse ratio of layer 1')
      plot_2d(mask_dict['overlap_mask_ratio_0'], os.path.join(result_dir,'overlap_mask_ratio_0.png'), 'overlap ratio between topk/mask of layer 0')
      plot_2d(mask_dict['overlap_mask_ratio_1'], os.path.join(result_dir,'overlap_mask_ratio_1.png'), 'overlap ratio between topk/mask of layer 1')
      plot_2d(mask_dict['overlap_prefer_ratio_0'], os.path.join(result_dir,'overlap_prefer_ratio_0.png'), 'overlap ratio between topk/prefer of layer 0')
      plot_2d(mask_dict['overlap_prefer_ratio_1'], os.path.join(result_dir,'overlap_prefer_ratio_1.png'), 'overlap ratio between topk/prefer of layer 1')
      import pdb; pdb.set_trace()
    return result_dict
  
  def plot_masks(self, mask, layer_0_topk, layer_1_topk, mask_mean, layer_0_prefer_topk, layer_1_prefer_topk, layer_0, layer_1, result_dir, iteration, sparse_r):
    plot_mask(np.array(mask._unflat['layer_0'].detach().cpu()), os.path.join(result_dir,'iter{}_mask_layer0.png'.format(iteration)), 'sparse ratio - layer 0 = {}'.format(sparse_r['sparse_0']))
    plot_mask(np.array(mask._unflat['layer_1'].detach().cpu()), os.path.join(result_dir,'iter{}_mask_layer1.png'.format(iteration)), 'sparse ratio - layer 1 = {}'.format(sparse_r['sparse_1']))

    #import pdb; pdb.set_trace()
    plot_mask(np.array(layer_0_topk.detach().cpu()), os.path.join(result_dir,'iter{}_topk_layer0.png'.format(iteration)), 'topk - layer 0 ({})'.format(sparse_r['sparse_0']))
    plot_mask(np.array(layer_1_topk.detach().cpu()), os.path.join(result_dir,'iter{}_topk_layer1.png'.format(iteration)), 'topk - layer 1 ({})'.format(sparse_r['sparse_1']))

    plot_mask(np.array(layer_0_prefer_topk.detach().cpu()), os.path.join(result_dir,'iter{}_prefer_topk_layer0.png'.format(iteration)), 'prefer topk - layer 0 ({})'.format(sparse_r['sparse_0']))
    plot_mask(np.array(layer_1_prefer_topk.detach().cpu()), os.path.join(result_dir,'iter{}_prefer_topk_layer1.png'.format(iteration)), 'prefer topk - layer 1 ({})'.format(sparse_r['sparse_1']))

    plot_mask(np.array(mask_mean._unflat['layer_0'].detach().cpu()), os.path.join(result_dir, 'iter{}_mask_mean_layer0.png'.format(iteration)), 'prefer - layer 0 ({})'.format(sparse_r['sparse_0']))
    plot_mask(np.array(mask_mean._unflat['layer_1'].detach().cpu()), os.path.join(result_dir, 'iter{}_mask_mean_layer1.png'.format(iteration)), 'prefer - layer 1 ({})'.format(sparse_r['sparse_1']))

    plot_mask(np.array(layer_0.detach().cpu()), os.path.join(result_dir,'iter{}_grad_layer0.png'.format(iteration)), 'grad - layer 0 ({})'.format(sparse_r['sparse_0']))
    plot_mask(np.array(layer_1.detach().cpu()), os.path.join(result_dir,'iter{}_grad_layer1.png'.format(iteration)), 'grad - layer 1 ({})'.format(sparse_r['sparse_1']))

    #import pdb; pdb.set_trace()
    overlap_mask_0 = layer_0_topk * mask._unflat['layer_0']
    overlap_mask_1 = layer_1_topk * mask._unflat['layer_1']

    overlap_mask_ratio_0 = overlap_mask_0.sum().float()/(len(overlap_mask_0) * sparse_r['sparse_0'])
    overlap_mask_ratio_1 = overlap_mask_1.sum().float()/(len(overlap_mask_1) * sparse_r['sparse_1'])

    plot_mask(np.array(overlap_mask_0.detach().cpu()), os.path.join(result_dir,'iter{}_overlap_mask_layer0.png'.format(iteration)), 
                '{}/{}(overlap/topk) overlap ratio({:.2f})'.format(overlap_mask_0.sum(),int(sparse_r['sparse_0'] * len(overlap_mask_0)), overlap_mask_ratio_0))
    plot_mask(np.array(overlap_mask_1.detach().cpu()), os.path.join(result_dir,'iter{}_overlap_mask_layer1.png'.format(iteration)), 
                '{}/{}(overlap/topk) overlap ratio({:.2f})'.format(overlap_mask_1.sum(),int(sparse_r['sparse_1'] * len(overlap_mask_1)), overlap_mask_ratio_1))

    #import pdb; pdb.set_trace()
    overlap_prefer_0 = layer_0_topk * layer_0_prefer_topk
    overlap_prefer_1 = layer_1_topk * layer_1_prefer_topk

    overlap_prefer_ratio_0 = overlap_prefer_0.sum().float()/(len(overlap_prefer_0) * sparse_r['sparse_0'])
    overlap_prefer_ratio_1 = overlap_prefer_1.sum().float()/(len(overlap_prefer_1) * sparse_r['sparse_1']) 
    #import pdb; pdb.set_trace()
    plot_mask(np.array(overlap_prefer_0.detach().cpu()), os.path.join(result_dir,'iter{}_overlap_prefer_layer0.png'.format(iteration)), 
                '{}/{}(overlap/topk) overlap ratio({:.2f})'.format(overlap_prefer_0.sum(),int(sparse_r['sparse_0'] * len(overlap_prefer_0)), overlap_prefer_ratio_0))
    plot_mask(np.array(overlap_prefer_1.detach().cpu()), os.path.join(result_dir,'iter{}_overlap_prefer_layer1.png'.format(iteration)), 
                '{}/{}(overlap/topk) overlap ratio({:.2f})'.format(overlap_prefer_1.sum(),int(sparse_r['sparse_1'] * len(overlap_prefer_1)), overlap_prefer_ratio_1))
    mask_result = dict(
          sparse_0 = sparse_r['sparse_0'],
          sparse_1 = sparse_r['sparse_1'],
          overlap_mask_ratio_0 = overlap_mask_ratio_0.tolist(),
          overlap_mask_ratio_1 = overlap_mask_ratio_1.tolist(),
          overlap_prefer_ratio_0 = overlap_prefer_ratio_0.tolist(),
          overlap_prefer_ratio_1 = overlap_prefer_ratio_1.tolist()
        )
    return mask_result

def get_1D_Loss(X, step_X_, scale, flat_size, layers, dataset, model_cls, params):
  Z = np.zeros(len(X))
  temp = np.zeros(len(X))
  for axis_x in tqdm(range(len(X))):
    #import pdb; pdb.set_trace()
    direction = params.new_from_flat(torch.zeros(flat_size).cuda())
    for layer in layers:
      direction.unflat[layer] = step_X_.unflat[layer]
    X_scale = params.new_from_flat(scale * X[axis_x] * torch.ones(flat_size).cuda())
    step_X__ =  X_scale * direction
    
    params_z = params + step_X__
    model_train = C(model_cls(params=params_z.detach()))
    for index in range(len(dataset)):
      input_data = dataset.load()
      temp[axis_x] = model_train(*input_data).item()/input_data[0].size(0)
      if index==0:
        Z[axis_x] = temp[axis_x]
      else:
        Z[axis_x] += temp[axis_x]
  return Z    
def plot_3D(X=None, Y=None, Z=None, savefig='test.png'):
  """
  ======================
  3D surface (color map)
  ======================

  Demonstrates plotting a 3D surface colored with the coolwarm color map.
  The surface is made opaque by using antialiased=False.

  Also demonstrates using the LinearLocator and custom formatting for the
  z axis tick labels.
  """
  from mpl_toolkits.mplot3d import Axes3D
  import matplotlib.pyplot as plt
  from matplotlib import cm
  from matplotlib.ticker import LinearLocator, FormatStrFormatter
  import numpy as np


  fig = plt.figure()
  ax = fig.gca(projection='3d')

  # Make data.
  if Z is None:
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    #R = np.sqrt(X**2 + Y**2)
    #Z = np.sin(R)
    Z = np.sqrt(X**2 + Y**2)
  # Plot the surface.
  surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

  # Customize the z axis.
  ax.set_zlim(Z.reshape(-1, 1).min(), Z.reshape(-1, 1).max())
  ax.zaxis.set_major_locator(LinearLocator(10))
  ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

  # Add a color bar which maps values to colors.
  fig.colorbar(surf, shrink=0.5, aspect=5)
  plt.xlabel('Step Generator')
  plt.ylabel('Gradient')
  plt.savefig(savefig)
  plt.close()
  #plt.show()

def plot_contour(X=None, Y=None, Z=None, savefig='test.png'):
  from mpl_toolkits.mplot3d import Axes3D
  import matplotlib.pyplot as plt
  from matplotlib import cm
  from matplotlib.ticker import LinearLocator, FormatStrFormatter
  import numpy as np
  """
  fig = plt.figure()
  plt.title("Contour plots")
  plt.contour(X, Y, Z, alpha=.75, cmap='jet')
  """
  fig, ax = plt.subplots()
  CS = ax.contour(X, Y, Z)
  ax.clabel(CS, inline=1, fontsize=10)
  #ax.set_title('Simplest default with labels')
  plt.xlabel('Step Generator')
  plt.ylabel('Gradient')
  plt.savefig(savefig)
  plt.close()

def plot_mask(mask, savefig='mask.png', title=None):
  import matplotlib.pyplot as plt
  x = np.linspace(1,len(mask),len(mask))
  y = mask
  extent = [x[0]-(x[1]-x[0])/2., x[-1]+(x[1]-x[0])/2.,0,1]
  #y = mask/mask.mean()
  
  fig, (ax,ax2) = plt.subplots(nrows=2, sharex=True)

  ax.imshow(y[np.newaxis,:], cmap="plasma", aspect="auto", extent=extent)
  ax.set_yticks([])
  ax.set_xlim(extent[0], extent[1])
  plt.title(title)

  ax2.plot(x,y)
  #plt.tight_layout()
  
  """
  fig = plt.figure()
  plt.imshow(y[np.newaxis,:], cmap="plasma", aspect="auto", extent=extent)
  #plt.set_yticks([])
  #plt.set_xlim(extent[0], extent[1])
  """
  plt.title(title)
  plt.savefig(savefig)
  #plt.show()
  
def plot_2d(x, y, savefig='mask.png', title=None):
  import matplotlib.pyplot as plt

  fig = plt.figure()
  plt.title(title)
  plt.plot(x,y)
  plt.savefig(savefig)


  #plt.show()
def filter_norm(direction):
  return direction