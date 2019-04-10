import torch
import torch.nn as nn
from nn.rnn_base import RNNBase


class MaskGenerator(RNNBase):
  def __init__(
          self, input_sz=1, hidden_sz=20, preproc_factor=10.0, preproc=True,
          n_layers=1, rnn_cell='gru', mask_type='hybrid'):
    assert mask_type in ['normal', 'unified', 'hybrid']
    if mask_type in ['normal', 'hybrid']:
      out_sz = 2  # for beta & gamma
    elif mask_type in ['unified']:
      out_sz = 1  # for directly computed p
    super().__init__(
        input_sz=input_sz, out_sz=out_sz, hidden_sz=hidden_sz,
        preproc_factor=preproc_factor, preproc=preproc,
        n_layers=n_layers, rnn_cell=rnn_cell)
    self.mask_type = mask_type
    self.sigmoid = nn.Sigmoid()
    self.relu = nn.ReLU()
    self.sparsity_loss = nn.L1Loss(reduction='sum')

  def forward(self, activations, states, debug=False):
    mean = activations.abs().mean(0)
    std = activations.std(0)
    # local (layerwise) mean & std
    mean_l = activations.abs().mean().expand_as(mean)
    std_l = activations.std().expand_as(std)
    # global mean & std
    mean_g = mean.new_ones(mean.size()) * activations.flat.abs().mean()
    std_g = std.new_ones(std.size()) * activations.flat.std()

    x = [mean.flat.detach(), mean_l.flat.detach(), mean_g.flat.detach(),
         std.flat.detach(), std_l.flat.detach(), std_g.flat.detach()]

    # x = [x_ * 10 for x_ in x]

    output, states = super().forward(x, states)
    # mask = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
    #   temperature=0.1, probs=probs).rsample()
    # mask = mean.new_from_flat(mask)
    if self.mask_type in ['unified']:
      p_logit = output[:, 0]
      # l_logit = output[:, 1]

    elif self.mask_type in ['normal', 'hybrid']:  # NOTE: include l_logit!
      gamma = mean.new_from_flat(output[:, 0])
      beta = mean.new_from_flat(output[:, 1])
      mean = activations.mean(0)
      p_logit = gamma * ((activations - mean) / (std + 1e-12)) + beta
      if self.mask_type == 'hybrid':
        p_logit = p_logit.mean(0)
      p_logit = p_logit.flat

    probs = self.sigmoid(p_logit)
    # probs = torch.clamp(p_logit, min=0, max=1)
    # lambda_ = self.sigmoid(l_logit).mean()
    # mask = probs
    mask = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
        temperature=1, probs=probs).rsample()
    # sparsity_loss = mask.norm(p=2) + probs.norm(p=1)
    sparsity_loss = probs.norm(p=1)
    if debug:
      print(probs)
      print(sparsity_loss)
      # min = 0.001
      # max = 0.01
      # print('out_lambda', lambda_)
      # print('applied_lambda', ((max - min) * lambda_ + min))
      import pdb
      pdb.set_trace()

    if self.mask_type in ['normal']:
      mask = activations.new_from_flat(mask)
    elif self.mask_type in ['unified', 'hybrid']:
      mask = mean.new_from_flat(mask)

    return mask, states, sparsity_loss  # , lambda_
