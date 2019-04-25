import numpy as np
import random
import torch
import torch.nn as nn


class MaskGenerator(object):
  def __init__(self, *args, **kwargs):
    # dummy init for compatibility
    pass

  @classmethod
  def topk(cls, grad, topk_ratio, topk_num):
    assert all([isinstance(k, bool) for k in [topk_ratio, topk_num]])
    assert (topk_ratio is True) != (topk_num is True)
    # has to be set either of the two, not both
    if topk_ratio:
      topk_n = cls.compute_topk_n_with_ratio(grad, topk_ratio)
    else:
      topk_n = cls.compute_topk_n_with_number(grad, topk_num)
    abs_sum = grad.abs().sum(0, keepdim=True)
    ids = abs_sum.topk_id(topk_n, sorted=False)
    mask = abs_sum.new_zeros(abs_sum.size()).scatter_float_(1, ids, 1)
    return mask

  @classmethod
  def randk(cls, grad):
    assert all([isinstance(k, bool) for k in [topk_ratio, topk_num]])
    assert (topk_ratio is True) != (topk_num is True)
    # has to be set either of the two, not both
    if topk_ratio:
      topk_n = cls.compute_topk_n_with_ratio(grad, topk_ratio)
    else:
      topk_n = cls.compute_topk_n_with_number(grad, topk_num)
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
