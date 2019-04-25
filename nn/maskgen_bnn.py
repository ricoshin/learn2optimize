import numpy as np
import torch
import torch.nn.functional as F
from models.model_helpers import ParamsFlattener
from nn.relaxed_bb import RelaxedBetaBernoulli
from torch import nn
from collections import OrderedDict as odict

try:
  from utils import utils
except ModuleNotFoundError:
  # For execution as a __main__ file
  import sys
  sys.path.append('../')
  from utils import utils


C = utils.getCudaManager('default')


class FeatureGenerator(nn.Module):
  """This model will generate coordinate-wise features considering taking
  their uncertainty into account. MCdropout aids model to sample mutiple number
  of most promising update behavior without additive cost.Each coordinate will
  be passed as a set of intances along the batch dimension.
    Inputs: grad, weight, grad with k decaying weights
    Outputs: feature for each coordinates
  """

  def __init__(self, hidden_sz=32, batch_std_norm=False, n_timecode=3,
               b1=[0.0, 0.5, 0.9, 0.99, 0.999],
               b2=[0.5, 0.9, 0.99, 0.999, 0.9999],
               drop_rate=0.0):
    super().__init__()
    assert all([isinstance(b, (list, tuple)) for b in (b1, b2)])
    assert len(b1) == len(b2)
    self.input_sz = len(b1) + n_timecode # grad, weight, momentums
    self.hidden_sz = hidden_sz
    self.batch_std_norm = batch_std_norm
    self.b1 = C(torch.tensor(b1))
    self.b2 = C(torch.tensor(b2))
    # self.mb1 = 1 - self.b1
    # self.mb2 = 1 - self.b2
    self.m = None  # 1st momentum vector
    self.v = None  # 2nd momentum vector
    self.b1_t = 1  # for bais-correction of 1st momentum
    self.b2_t = 1  # for bais-correction of 2nd momentum
    self.drop_rate = drop_rate
    self.linear = nn.Linear(self.input_sz, self.hidden_sz)
    self.nonlinear = nn.ReLU(inplace=True)
    self.eps = 1e-9
    self.t = 0
    self.t_scales = np.linspace(1, np.log(1000) / np.log(10), n_timecode)
    self.t_encoder = lambda t: [np.tanh(3*t / 10**s - 1) for s in self.t_scales]


  def new(self):
    # reset momentum
    self.m = None
    self.v = None
    self.b1_t = 1
    self.b2_t = 1
    self.t = 0
    return self

  def forward(self, g, p=None, n=1, step=True):
    """
    Args:
      g(tensor): current gradient (n_params x 1)
      w(tensor): current weight (n_params x 1)
      p(float): for setting dropout probability in runtime.
        By default, use self.drop_rate.
      n(int): number of instances of MC dropout.

    Returns:
      x(tensor): representation for step & mask generation
                 (n_params x hidden_sz)
    """
    g_sq = g**2

    if self.m is None:
      # Initialize momentum matrix
      #   momentum matrix: m[B, len(decay_values)]
      #   should keep track of momentums for each B dimension
      m = g.repeat(1, len(self.b1))
      v = g_sq.repeat(1, len(self.b2))
    else:
      # Dimension will be matched by broadcasting
      m = self.b1 * self.m + (1 - self.b1) * g
      v = self.b2 * self.v + (1 - self.b2) * g_sq

    # bias-correction
    b1_t = self.b1_t * self.b1
    b2_t = self.b2_t * self.b2
    m_ = m.div(1 - b1_t)
    v_ = v.div(1 - b2_t)

    # Scaling
    # import pdb; pdb.set_trace()
    v_sqrt = v_.sqrt()#.detach()
    scaled_g = g.div(v_sqrt + self.eps) # AdaGrad
    scaled_m = m_.div(v_sqrt + self.eps)  # Adam

    # Time encoding
    self.t += 1
    t = C(torch.tensor(self.t_encoder(self.t)))
    t = t.repeat(scaled_g.size(0), 1)

    # regularization(as in Metz et al.)
    if self.batch_std_norm:
      g = g.div(g.var(0, keepdim=True))
      # g_sq = g_sq.div(g_sq.var(0, keepdim=True))
      w = w.div(w.var(0, keepdim=True))

    # import pdb; pdb.set_trace()
    # scaled_g = scaled_g.div(scaled_g.var(0, keepdim=True))
    x = torch.cat([scaled_m, t], 1)#.detach()
    # x = torch.cat([g, g_sq], dim=1)
    # x = torch.cat([g, g_sq], dim=1)
    if step:
      self.m = m
      self.v = v
      self.b1_t = b1_t
      self.b2_t = b2_t
    """Submodule inputs:
       x[n_params, 0]: current gradient
       x[n_params, 1]: current weight
       x[n_params, 2:]: momentums (for each decaying values)
    """
    # import pdb; pdb.set_trace()
    assert x.size(1) == self.input_sz
    x = self.nonlinear(self.linear(x))
    assert x.size(1) == self.hidden_sz

    x = torch.cat([x] * n, dim=0)   # coordinate-wise feature
    x = F.dropout(x, p=self.drop_rate if p is None else p)
    return x, v_sqrt


class StepGenerator(nn.Module):
  def __init__(self, hidden_sz=32, out_temp=1e-2, drop_rate=0.0):
    super().__init__()
    self.output_sz = 1  # log learning rate, update direction
    self.hidden_sz = hidden_sz
    self.out_temp = out_temp
    self.linear = nn.Linear(self.hidden_sz, self.output_sz)
    # self.lr = C(torch.tensor(1.).log())
    self.drop_rate = drop_rate
    self.tanh = nn.Tanh()
    # self.nonliear = nn.Tanh()
    # NOTE: last nonliearity can be critical (avoid ReLU + exp > 1)
    self.new()

  def new(self):
    self.log_lr = C(torch.tensor(1.).log())

  def forward(self, x, v_sqrt, n=1, p=None, debug=False):
    """
    Args:
      x (tensor): representation for step & mask generation
                  (n_params x hidden_sz)

    Returns:
      step (tensor): update vector in the parameter space. (n_params x 1)
    """
    assert x.size(1) == self.hidden_sz

    x = torch.cat([x] * n, dim=0)   # coordinate-wise feature
    x = F.dropout(x, p=self.drop_rate if p is None else p)

    x = self.linear(x)
    """Submodule outputs:
      y[n_params, 0]: per parameter log learning rate
      y[n_params, 1]: unnormalized update direction
    """
    assert x.size(1) == self.output_sz
    out_1 = x[:, 0]  # * self.out_temp
    step = out_1 * v_sqrt[:,-1].repeat(n)
    # out_2 = x[:, 1]  # * self.out_temp # NOTE: normalizing?
    # # out_3 = x[:, 2]
    # # out_3 = out_3.div(out_3.norm(p=2).detach())
    # # import pdb; pdb.set_trace()
    # self.log_lr = self.log_lr.detach() + out_1 * self.out_temp
    # direction = out_2.div(out_2.norm(p=2).detach())
    # step = torch.exp(self.log_lr) * direction
    # step = self.tanh(step) * 0.01
    if debug:
      import pdb; pdb.set_trace()
    step = torch.clamp(step, max=0.01, min=-0.01)

    return torch.chunk(step, n)

  def forward_with_mask(self, x):
    pass


class MaskGenerator(nn.Module):
  def __init__(self, hidden_sz=512):
    super().__init__()
    self.hidden_sz = hidden_sz
    self.gamm_g = None
    self.gamm_l = None

    self.nonlinear = nn.Tanh()
    self.eyes = None
    self.ones = None
    self.zeros = None
    self.avg_layer = nn.Linear(hidden_sz, 3)  # lamb, gamm_g, gamm_l
    self.out_layer = nn.Linear(hidden_sz, 2)  # a, b
    self.beta_bern = RelaxedBetaBernoulli()
    self.sigmoid = nn.Sigmoid()
    self.sigmoid_temp = 1e1
    self.gamm_scale = 1e-3
    self.lamb_scale = 1
    self.p_lamb = nn.Parameter(torch.ones(1) * self.lamb_scale)
    self.p_gamm = nn.Parameter(torch.ones(1) * self.gamm_scale)

  def detach_lambdas_(self):
    if self.gamm_g is not None:
      self.gamm_g = self.gamm_g.detach()
    if self.gamm_l is not None:
      self.gamm_l = [l.detach() for l in self.gamm_l]
    if self.lamb is not None:
      self.lamb = self.lamb.detach()

  def build_block_diag(self, tensors):
    """Build block diagonal matrix by aligning input tensors along the digonal
    elements and filling the zero blocks for the rest of area.

    Args: list (list of square matrices(torch.tensor))

    Returns: tensor (block diagonal matrix)
    """
    # left and right marginal zero matrices
    #   (to avoid duplicate allocation onto GPU)
    if self.zeros is None:
      blocks = []
      offset = 0
      size = [s.size(0) for s in tensors]
      total_size = sum(size)
      self.zeros = []
      # NOTE: exception handling for different tnesors size
      for i in range(len(tensors)):
        blocks_sub = []
        cur = tensors[i].size(0)
        if i == 0:
          blocks_sub.append(None)
        else:
          blocks_sub.append(C(torch.zeros(cur, offset)))
        offset += tensors[i].size(0)
        if i == len(tensors) - 1:
          blocks_sub.append(None)
        else:
          blocks_sub.append(C(torch.zeros(cur, total_size - offset)))
        self.zeros.append(blocks_sub)
    # concatenate the tensors with left and right block zero matices
    blocks = []
    for i in range(len(tensors)):
      blocks_sub = []
      zeros = self.zeros[i]
      if zeros[0] is not None:
        blocks_sub.append(zeros[0])
      blocks_sub.append(tensors[i])
      if zeros[1] is not None:
        blocks_sub.append(zeros[1])
      blocks.append(torch.cat(blocks_sub, dim=1))
    return torch.cat(blocks, dim=0)

  def forward(self, x, size, n=1, debug=False):
    """
    Args:
      x (tensor): representation for step & mask generation (batch x hidden_sz)
      size (dict): A dict mapping names of weight matrices to their
        torch.Size(). For example:

      {'mat_0': torch.Size([784, 500]), 'bias_0': torch.Size([500]),
       'mat_1': torch.Size([500, 10]), 'bias_1': torch.Size([10])}
    """
    assert isinstance(size, dict)
    assert x.size(1) == self.hidden_sz

    """Set-based feature averaging (Permutation invariant)"""
    sizes = [s for s in size.values()]
    # [torch.Size([784, 500]), torch.Size([500]),
    #  torch.Size([500, 10]), torch.Size([10])]
    split_sizes = [np.prod(s) for s in size.values()] * n
    # [392000, 500, 5000, 10]
    # split_size = [sum(offsets[:i]) for i in range(len(offsets))]
    x_set = torch.split(x, split_sizes, dim=0)
    # mean = torch.cat([s.mean(0).expand_as(s) for s in sections], dim=0)
    x_set = [x.view(*s, self.hidden_sz) for x, s in zip(x_set, sizes * n)]
    x_set_sizes = [x_set[i].size() for i in range(len(x_set))]
    # [torch.Size([784, 500, 32]), torch.Size([500, 32]),
    #  torch.Size([500, 10, 32]), torch.Size([10, 32])]

    # unsqueeze bias to match dim
    max_dim = max([len(s) for s in x_set_sizes])  # +1: hidden dimension added
    assert all([(max_dim - len(s)) <= 1 for s in x_set_sizes])
    x_set = [x.unsqueeze_(0) if len(x.size()) < max_dim else x for x in x_set]
    # [torch.Size([784, 500, 32]), torch.Size([1, 500, 32]),
    #  torch.Size([500, 10, 32]), torch.Size([1, 10, 32])]

    # concatenate bias hidden with weight hidden in the same layer
    # bias will be dropped with its corresponding weight column
    #   (a) group the parameters layerwisely
    keys = [n for n in size.keys()]
    names = []
    for i in range(n):
      names += [k + '_' + str(i) for k in keys]
    # names = [str(i) + '_' + n for n in names] *
    # ['mat_0', 'bias_0', 'mat_1', 'bias_1']
    set = odict()
    for name, x in zip(names, x_set):
      id = '_'.join(name.split('_')[1:])
      if id in set:
        set[id].append(x)
      else:
        set[id] = [x]
    #   (b) compute mean over the set
    set = {k: torch.cat(v, dim=0).mean(0) for k, v in set.items()}
    x_set = torch.cat([s for s in set.values()], dim=0)
    # non-linearity
    x_set = self.nonlinear(x_set)
    # x_set: [total n of set(channels) x hidden_sz]


    # """Reflecting relative importance (Permutation equivariant)"""
    # # to avoid allocate new tensor to GPU at every iteration
    # if self.eyes is None or not self.eyes.size(0) == (x_set.size(0) * 2):
    #   self.eyes = C(torch.eye(x_set.size(0)))
    #   self.ones = C(torch.ones(x_set.size(0), x_set.size(0)))
    #   # self.zeros = [torch.zeros(s[0]) for s in x_set_sizes]
    #
    # # Identity term w_i: [(total n of set)^2]
    # w_i = self.eyes
    # # Relativity term w_r: [(total n of set)^2]
    # if self.gamm_g is None or self.gamm_l is None:
    #   # .normal_(0, 1e-3))
    #   ones = C(torch.ones([x_set.size(0)] * 2))
    #   self.gamm_g = self.gamm_l = ones  * self.gamm_scale
    #   self.lamb = ones * self.lamb_scale
    #
    # # gamm_gl = (self.gamm_g + self.gamm_l)
    # out = self.avg_layer(x_set)
    #
    # lamb = out[:, 0].unsqueeze(-1)
    # self.lamb = lamb.mean(0) * self.lamb_scale
    # lamb = self.lamb.expand_as(lamb)
    # # self.lamb = lamb.div(lamb.size(0))
    # lamb = lamb * self.eyes
    #
    # # out[:, 1] -> globally averaged lamba_g
    # gamm_g = out[:, 1].unsqueeze(-1)
    # self.gamm_g = gamm_g.mean(0) * self.gamm_scale
    # gamm_g = self.gamm_g.expand_as(gamm_g)
    # # self.gamm_g = gamm_g.div(gamm_g.size(0))
    # gamm_g = gamm_g * self.ones
    #
    # # out[:, 2] -> locally averaged gamm_l
    # gamm_l = out[:, 2].unsqueeze(-1)
    # gamm_l = torch.split(gamm_l, [s[0] for s in x_set_sizes], dim=0)
    # self.gamm_l = [l.mean(0) * self.gamm_scale for l in gamm_l]
    # sizes = [(l.size(0),) * 2 for l in gamm_l]
    # # # gamm_l = [l.mean(0).expand(l.size(0)) for l in gamm_l]
    # # gamm_l = [l.div(l.size(0)).expand(l.size(0)) for l in gamm_l]
    # # gamm_l = torch.cat(gamm_l, dim=0)
    # gamm_l = [l.mean(0).expand(s) for l, s in zip(self.gamm_l, sizes)]
    # # gamm_l = [l.div(l.size(0)).expand([l.size(0)] * 2) for l in gamm_l]
    # gamm_l = self.build_block_diag(gamm_l)
    #
    # if debug:
    #   import pdb; pdb.set_trace()
    # # Equivariant weight multiplication
    # # inp = x_set.t().matmul(self.eyes + self.gamm_g + self.gamm_l)
    # inp = x_set.t().matmul(self.eyes + self.p_gamm/510 * self.ones)
    # # term_2 = x_set.t().matmul(gamm_gl * self.ones)
    # # inp = term_1 + term_2
    # # NOTE: make it faster?
    # # last layers for mask generation
    # out = self.out_layer(inp.t())

    out = self.out_layer(x_set)
    self.a = F.softplus(out[:, 0].clamp(min=-10.))
    self.b = F.softplus(out[:, 1].clamp(min=-10., max=50.))
    self.n = [k for k in set.keys()]
    keys = [k for k in set.keys() if k.split('_')[1] == '0']
    self.s = [set[k].size(0) for k in keys]
    kld = self.beta_bern.compute_kld(self.a, self.b)
  #
    return kld
  #
  #
  def sample_mask(self):
    mask, pi = self.beta_bern(self.a, self.b)
    m = torch.split(mask, self.s, dim=0)
    return {'layer_' + n.split('_')[0]: m for n, m in zip(self.n, m)}




    # mask = torch.chunk(mask, n)
    # pi = torch.chunk(pi, n)
    # kld = []
    #
    #
    # out_list = []
    # keys = [k for k in set.keys() if k.split('_')[1] == '0']
    # split_sizes = [set[k].size(0) for k in keys]
    #
    # for m, p, a, b in zip(mask, pi, torch.chunk(a, n), torch.chunk(b, n)):
    #   m = torch.split(m, split_sizes, dim=0)
    #   p = torch.split(p, split_sizes, dim=0)
    #   name = [k for k in set.keys()]
    #   out_list.append((
    #     {'layer_' + n.split('_')[0]: m for n, m in zip(name, m)},
    #     {'layer_' + n.split('_')[0]: p for n, p in zip(name, p)},
    #     self.beta_bern.compute_kld(a, b),
    #   ))
    #
    # #
    # # mask = torch.split(mask, [s[0] for s in x_set_sizes], dim=0)
    # # name = [n for n in layerwise.keys()]
    # # mask = {'layer_' + n: m for n, m in zip(name, mask)}
    #
    # return out_list
