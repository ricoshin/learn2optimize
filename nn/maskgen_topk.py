import numpy as np
import random
import torch
import torch.nn as nn
from models.model_helpers import ParamsFlattener
from utils import utils


C = utils.getCudaManager('default')

class MaskGenerator(object):
  def __init__(self, *args, **kwargs):
    # dummy init for compatibility
    pass

  @classmethod
  def topk(cls, grad, act, topk, mode='ratio'):
    # NOTE: act is used to match the size (fix later)
    # dirty work...
    assert mode in ['ratio', 'number']
    set_size = act.tsize(1)
    if mode == 'ratio':
      topk_n = cls.compute_topk_n_with_ratio(topk, set_size)
    else:
      topk_n = cls.compute_topk_n_with_number(topk, set_size)
    # abs_sum = grad.abs().sum(0, keepdim=True)
    layer_0 = torch.cat([grad.unflat['mat_0'],
      grad.unflat['bias_0'].unsqueeze(0)], dim=0).abs().sum(0)
    layer_1 = torch.cat([grad.unflat['mat_1'],
      grad.unflat['bias_1'].unsqueeze(0)], dim=0).abs().sum(0)
    abs_sum = ParamsFlattener({'layer_0': layer_0, 'layer_1': layer_1})
    ids = abs_sum.topk_id(topk_n, sorted=False)
    ids = {k: v.view(1, -1) for k, v in ids.unflat.items()}
    # ids = abs_sum.topk_id(topk_n, sorted=False)
    mask = ParamsFlattener(
      {k: C(torch.zeros(1, v)) for k, v in set_size.items()})
    mask = mask.scatter_float_(1, ids, 1)
    return mask

  @classmethod
  def randk(cls, grad, act, topk, mode='ratio'):
    # NOTE: act is used to match the size (fix later)
    assert mode in ['ratio', 'number']
    set_size = act.tsize(1)
    if mode == 'ratio':
      topk_n = cls.compute_topk_n_with_ratio(topk, set_size)
    else:
      topk_n = cls.compute_topk_n_with_number(topk, set_size)
    sizes = set_size
    rand_id = lambda k, v: random.sample(range(v), topk_n[k])
    sizes = {k: rand_id(k, v) for k, v in sizes.items()}
    ids = {k: torch.tensor(v).cuda().view(1, -1) for k, v in sizes.items()}
    # import pdb; pdb.set_trace()
    abs_sum = grad.abs().sum(0, keepdim=True) # fix later!
    mask = ParamsFlattener(
      {k: C(torch.zeros(1, v)) for k, v in set_size.items()})
    mask = mask.scatter_float_(1, ids, 1)
    return mask

  def compute_topk_n_with_ratio(r, set_size):
    return {k: int(v*r) for k, v in set_size.items()}

  def compute_topk_n_with_number(n, set_size, exclude_last=False):
    topk_n = {}
    for i, (name, size) in enumerate(set_size.items()):
      max_n = (exclude_last and i == len(sizes) - 1) or n > size
      topk_n[name] = size if max_n else n
    return topk_n
