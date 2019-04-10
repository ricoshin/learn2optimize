import numpy as np
import random
import torch
import torch.nn as nn


class MaskGenerator(object):
  def __init__(self, *args, **kwargs):
    # dummy init for compatibility
    pass

  @classmethod
  def topk(cls, grad, *args, **kwargs):
    # topk_n = cls.compute_topk_n_with_ratio(grad, 0.1)
    topk_n = cls.compute_topk_n_with_number(grad, 20)
    abs_sum = grad.abs().sum(0, keepdim=True)
    ids = abs_sum.topk_id(topk_n, sorted=False)
    mask = abs_sum.new_zeros(abs_sum.size()).scatter_float_(1, ids, 1)
    return mask

  @classmethod
  def randk(cls, grad, *args, **kwargs):
    # topk_n = cls.compute_topk_n_with_ratio(grad, 0.1)
    topk_n = cls.compute_topk_n_with_number(grad, 20)
    sizes = grad.tsize(1)
    rand_id = lambda k, v: random.sample(range(v), topk_n[k])
    sizes = {k: rand_id(k, v) for k, v in sizes.items()}
    ids = {k: torch.tensor(v).cuda().view(1, -1) for k, v in sizes.items()}
    # import pdb; pdb.set_trace()
    abs_sum = grad.abs().sum(0, keepdim=True) # fix later!
    mask = abs_sum.new_zeros(abs_sum.size()).scatter_float_(1, ids, 1)
    return mask

  def compute_topk_n_with_ratio(grad, ratio):
    return {k: int(v*ratio) for k, v in grad.tsize(1).items()}

  def compute_topk_n_with_number(grad, n, exclude_last=False):
    topk_n = {}
    sizes = grad.tsize(1)
    for i, (name, size) in enumerate(grad.tsize(1).items()):
      max_n = (exclude_last and i == len(sizes) - 1) or n > size
      topk_n[name] = size if max_n else n
    return topk_n
