import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from utils import utils

C = utils.getCudaManager('default')


class RelaxedBetaBernoulli(nn.Module):
  def __init__(self, alpha=1e-4, a_uc_init=-1.0, thres=1e-3, kl_scale=1.0):
    super(RelaxedBetaBernoulli, self).__init__()
    self.alpha = alpha
    self.thres = thres
    self.kl_scale = kl_scale
    # self.a_uc = nn.Parameter(a_uc_init * torch.ones(num_gates))
    # self.b_uc = nn.Parameter(0.5413 * torch.ones(num_gates))

  # def get_params(self, x):
  #   """Convert 2-d logits into distribution parameters a, b."""
  #   assert x.size(1) == 2
  #   a = F.softplus(x[:, 0].clamp(min=-10.))
  #   b = F.softplus(x[:, 1].clamp(min=-10., max=50.))
  #   return a, b

  def sample_pi(self, a, b):
    """Sample pi from Beta(Kumaraswamy) distribution with parameter a and b."""
    assert a.size() == b.size()
    u = C(torch.rand(a.size(0)).clamp(1e-8, 1 - 1e-8))
    return (1 - u.pow_(1. / b)).pow_(1. / a)

  def get_Epi(self, a, b):
    """Compute expectation of pi analytically."""
    assert a.size() == b.size()
    Epi = b * torch.exp(torch.lgamma(1 + 1. / a) + torch.lgamma(b)
                        - torch.lgamma(1 + 1. / a + b))
    return Epi

  def compute_kld(self, a, b):
    kld = (1 - self.alpha / a) * (-0.577215664901532
                                  - torch.digamma(b) - 1. / b) \
        + torch.log(a * b + 1e-10) - math.log(self.alpha) \
        - (b - 1) / b
    kld = kld.sum()
    return kld

  def forward(self, a, b):
    """
    Args:
      a, b (tensor): beta distribution parameters.

    Returns:
      z (tensor): Relaxed Beta-Bernoulli binary masks.
      pi (tensor): Bernoulli distribution parameters
    """
    # a, b = self.get_params(x)
    pi = self.sample_pi(a, b)
    temp = C(torch.Tensor([0.1]))
    z = RelaxedBernoulli(temp, probs=pi).rsample()
    return z, pi

  # def get_mask(self, Epi=None):
  #   Epi = self.get_Epi() if Epi is None else Epi
  #   return (Epi >= self.thres).float()

  # def get_num_active(self):
  #   return self.get_mask().sum().int().item()
