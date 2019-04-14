import torch
from torch import nn
import numpy as np
from models.model_helpers import ParamsFlattener
from nn.relaxed_bb import RelaxedBetaBernoulli
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
  def __init__(self, hidden_sz=32, n_layers=1, batch_std_norm=True,
    decay_values=[0.5, 0.9, 0.99, 0.999, 0.9999], dropout_rate=0.5):
    super().__init__()
    assert n_layers > 0 # has to have at least single layer
    assert isinstance(decay_values, (list, tuple))

    self.input_sz = 2 + len(decay_values) # grad, weight, momentums
    self.hidden_sz = hidden_sz
    self.batch_std_norm = batch_std_norm
    self.decay_values = C(torch.tensor(decay_values))
    self.momentum = None
    self.momentum_sq = None

    self.layers = nn.ModuleList()
    for i in range(n_layers):
      input_sz = self.input_sz if i == 0 else hidden_sz
      self.layers.append(nn.Linear(input_sz, hidden_sz))
      self.layers.append(nn.ReLU())
      self.layers.append(nn.Dropout(p=dropout_rate))

  def new(self):
    # reset momentum
    self.momentum = None
    return self

  def forward(self, g, w):
    """
    Args:
      g(tensor): current gradient (n_params x 1)
      w(tensor): current weight (n_params x 1)

    Returns:
      x(tensor): representation for step & mask generation
                 (n_params x hidden_sz)
    """
    if self.momentum is None:
      # Initialize momentum matrix
      #   momentum matrix: m[B, len(decay_values)]
      #   should keep track of momentums for each B dimension
      self.momentum = g.repeat(1, len(self.decay_values))
      self.momentum_sq = (g**2).repeat(1, len(self.decay_values))
    else:
      # Dimension will be matched by broadcasting
      r = self.decay_values
      self.momentum = r * self.momentum + (1 - r) * g
      self.momentum_sq = r * self.momentum_sq + (1 - r) * (g**2)

    # # regularization(as in Metz et al.)
    # g_sq = g**2
    if self.batch_std_norm:
      g = g.div(g.var(0, keepdim=True))
      # g_sq = g_sq.div(g_sq.var(0, keepdim=True))
      w = w.div(w.var(0, keepdim=True))
    #
    x = torch.cat([g, w, self.momentum], 1).detach()
    # x = torch.cat([g, g_sq], dim=1)
    # x = torch.cat([g, g_sq], dim=1)
    """Submodule inputs:
       x[n_params, 0]: current gradient
       x[n_params, 1]: current weight
       x[n_params, 2:]: momentums (for each decaying values)
    """
    assert x.size(1) == self.input_sz
    for layer in self.layers:
      x = layer(x)
    assert x.size(1) == self.hidden_sz

    return x  # coordinate-wise feature


class StepGenerator(nn.Module):
  def __init__(self, hidden_sz=32, n_layers=1, out_temp=1e-5):
    super().__init__()
    assert n_layers > 0 # has to have at least single layer
    self.hidden_sz = hidden_sz
    self.output_sz = 2 # log learning rate, update direction
    self.out_temp = out_temp
    self.layers = nn.ModuleList()
    for i in range(n_layers):
      output_sz = self.output_sz if (i == n_layers - 1) else hidden_sz
      self.layers.append(nn.Linear(hidden_sz, output_sz))
      if i < n_layers - 1:
        self.layers.append(nn.Tanh())
        # NOTE: last nonliearity can be critical (avoid ReLU + exp > 1)

  def forward(self, x, debug=False):
    """
    Args:
      x (tensor): representation for step & mask generation
                  (n_params x hidden_sz)

    Returns:
      step (tensor): update vector in the parameter space. (n_params x 1)
    """
    assert x.size(1) == self.hidden_sz

    for layer in self.layers:
      x = layer(x)
    """Submodule outputs:
      y[n_params, 0]: per parameter log learning rate
      y[n_params, 1]: unnormalized update direction
    """
    assert x.size(1) == self.output_sz
    out_1 = x[:, 0] #* self.out_temp
    out_2 = x[:, 1] #* self.out_temp # NOTE: normalizing?
    # out_3 = x[:, 2]
    # out_3 = out_3.div(out_3.norm(p=2).detach())
    step = torch.exp(out_1 * self.out_temp) * out_2 * self.out_temp
    if debug:
      import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()
    step = torch.clamp(step, max=0.01, min=-0.01)
    return step

  def forward_with_mask(self, x):
    pass


class MaskGenerator(nn.Module):
  def __init__(self, hidden_sz=512, n_layers=1):
    super().__init__()
    assert n_layers > 0
    self.hidden_sz = hidden_sz
    self.lambda_g = None
    self.lambda_l = None

    self.nonlinear = nn.Tanh()
    self.eyes = None
    self.ones = None
    self.out_layers = nn.ModuleList()
    for i in range(n_layers):
      output_sz = 4 if (i == n_layers - 1) else hidden_sz
      # a, b, lambda_g, lambda_l
      self.out_layers.append(nn.Linear(hidden_sz, output_sz))
    self.beta_bern = RelaxedBetaBernoulli()
    self.sigmoid = nn.Sigmoid()
    self.sigmoid_temp = 1e1
    self.lambda_scale = 1e-3
    self.lambda_init = (-1) * 1e-4

  def detach_lambdas_(self):
    if self.lambda_g is not None:
      self.lambda_g = self.lambda_g.detach()
    if self.lambda_l is not None:
      self.lambda_l = self.lambda_l.detach()

  def forward(self, x, size):
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
    split_sizes = [np.prod(s) for s in size.values()]
    # split_size = [sum(offsets[:i]) for i in range(len(offsets))]
    x_set = torch.split(x, split_sizes, dim=0)
    # mean = torch.cat([s.mean(0).expand_as(s) for s in sections], dim=0)
    x_set = [x.view(*s, self.hidden_sz) for x, s in zip(x_set, sizes)]
    x_set_sizes = [x_set[i].size() for i in range(len(x_set))]

    # unsqueeze bias to match dim
    max_dim = max([len(s) for s in x_set_sizes]) # +1: hidden dimension added
    assert all([(max_dim - len(s)) <= 1 for s in x_set_sizes])
    x_set = [x.unsqueeze_(0) if len(x.size()) < max_dim else x for x in x_set]

    # concatenate bias hidden with weight hidden in the same layer
    # bias will be dropped with its corresponding weight column
    #   (a) group the parameters layerwise
    names = [n for n in size.keys()]
    layerwise = {}
    for n, x in zip(names, x_set):
      n_set = n.split('_')[1]
      if n_set in layerwise:
        layerwise[n_set].append(x)
      else:
        layerwise[n_set] = [x]
    #   (b) compute mean over the set
    x_set = [torch.cat(xs, dim=0).mean(0) for xs in layerwise.values()]
    x_set_sizes = [x_set[i].size() for i in range(len(x_set))]

    # non-linearity
    x_set = self.nonlinear(torch.cat(x_set, dim=0))
    # x_set: [total n of set(channels) x hidden_sz]


    """Reflecting relative importance (Permutation equivariant)"""
    # to avoid allocate new tensor to GPU at every iteration
    if self.eyes is None or self.eyes.size(0) == (x_set.size(0) * 2):
      self.eyes = torch.eye(x_set.size(0)).cuda()
      self.ones = torch.ones(x_set.size(0), x_set.size(0)).cuda()

    # Identity term w_i: [(total n of set)^2]
    w_i = self.eyes
    # Relativity term w_r: [(total n of set)^2]
    if self.lambda_g is None or self.lambda_l is None:
      self.lambda_g = C(torch.ones(x_set.size(0), 1)) * self.lambda_init
      self.lambda_l = self.lambda_g
    lambda_gl = (self.lambda_g + self.lambda_l)

    # Equivariant weight multiplication
    x_set = x_set.t().matmul(self.eyes) \
            + x_set.t().matmul(lambda_gl * self.ones).detach()
            # NOTE: make it faster?

    # last layers for mask generation
    for layer in self.out_layers:
      x_set = layer(x_set.t())

    # out[:, 0] -> mask
    mask, kld = self.beta_bern(x_set[:, :2])
    # mask_logit = x_set[:, 0]
    # mask = self.sigmoid(mask_logit * self.sigmoid_temp)
    # kld = mask.norm(p=1)
    mask = torch.split(mask, [s[0] for s in x_set_sizes], dim=0)
    name = [n for n in layerwise.keys()]
    mask = {'layer_' + n: m for n, m in zip(name, mask)}

    # out[:, 1] -> globally averaged lamba_g
    lambda_g = x_set[:, 2].unsqueeze(-1)
    lambda_g = lambda_g.mean(0).expand_as(lambda_g)
    self.lambda_g = lambda_g * self.lambda_scale

    # out[:, 2] -> locally averaged lambda_l
    lambda_l = x_set[:, 3].unsqueeze(-1)
    lambda_l = torch.split(lambda_l, [s[0] for s in x_set_sizes], dim=0)
    lambda_l = torch.cat([l.mean(0).expand_as(l) for l in lambda_l], dim=0)
    self.labmda_l = lambda_l * self.lambda_scale
    # import pdb; pdb.set_trace()

    return mask, kld
