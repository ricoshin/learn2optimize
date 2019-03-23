import torch
from torch.autograd import Function

"""https://github.com/lancopku/meProp/blob/master/src/pytorch/functions.py"""


class SBLinear(Function):
  '''
  linear function with sparse backpropagation.
  '''

  def __init__(self, name):
    super(SBLinear, self).__init__()
    self.name = name
    self.backprop_mode = 'no_drop'

  def forward(self, x, w, b):
    '''
    forward propagation
    x should be of size [minibatch, input feature]
    w should be of size [input feature, output feature]
    b should be of size [output feature]
    '''
    self.save_for_backward(x, w, b)
    y = x.new(x.size(0), w.size(1))
    y.addmm_(0, 1, x, w)
    self.add_buffer = x.new(x.size(0)).fill_(1)
    y.addr_(self.add_buffer, b)
    return y

  def mask_backprop(self, mask, mode):
    assert mode in ['no_drop', 'soft_drop', 'hard_drop']
    self.backprop_mode = mode
    self.dropmask = mask

  def backward(self, dy):
    x, w, b = self.saved_tensors
    dx = dw = db = None

    if self.backprop_mode in ['no_drop', 'soft_drop']:
      dy = dy * self.dropmask if self.backprop_mode == 'soft_drop' else dy
      if self.needs_input_grad[0]:
        dx = torch.mm(dy, w.t())
      if self.needs_input_grad[1]:
        dw = torch.mm(x.t(), dy)
    elif self.backprop_mode == 'hard_drop':
      retain_mask = self.dropmask > 0.5
      sparse_values = dy.masked_select(retain_mask).view(-1)
      sparse_indices = retain_mask.nonzero().t()
      pdy = torch.cuda.FloatTensor(sparse_indices, sparse_values, dy.size())
      if self.needs_input_grad[0]:
        dx = torch.sparse.mm(pdy, w.t())
      if self.needs_input_grad[1]:
        dw = torch.sparse.mm(pdy.t(), x).t()
    else:
      raise Exception(f"Unknown backprop_mode: {self.backprop_mode}")

    if self.needs_input_grad[2]:
      db = torch.mv(dy.t(), self.add_buffer)

    return dx, dw, db


class UnifiedSBLinear(Function):
  '''
  linear function with sparse backpropagation that shares the same sparsity
  pattern across minibatch
  '''

  def __init__(self, name):
    super(UnifiedSBLinear, self).__init__()
    self.name = name
    self.dropmask = None
    self.backprop_mode = 'no_drop'

  def forward(self, x, w, b):
    '''
    forward propagation
    x should be of size [minibatch, input feature]
    w should be of size [input feature, output feature]
    b should be of size [output feature]
    '''
    self.save_for_backward(x, w, b)
    y = x.new(x.size(0), w.size(1))
    y.addmm_(0, 1, x, w)
    self.add_buffer = x.new(x.size(0)).fill_(1)
    y.addr_(self.add_buffer, b)
    return y

  def mask_backprop(self, mask, mode):
    assert mode in ['no_drop', 'soft_drop', 'hard_drop']
    self.backprop_mode = mode
    self.dropmask = mask

  def backward(self, dy):
    x, w, b = self.saved_tensors
    dx = dw = db = None

    if self.backprop_mode in ['no_drop', 'soft_drop']:
      dy = dy * self.dropmask if self.backprop_mode == 'soft_drop' else dy
      if self.needs_input_grad[0]:
        dx = torch.mm(dy, w.t())
      if self.needs_input_grad[1]:
        dw = torch.mm(x.t(), dy)

    elif self.backprop_mode == 'hard_drop':
      retain_index = (self.dropmask > 0.5).nonzero().squeeze(-1)
      pdy = dy.index_select(-1, retain_index)
      # compute the gradients of x, w, and b, using the smaller dy matrix
      if self.needs_input_grad[0]:
        if retain_index.size(0) == 0:
          pass
        else:
          dx = torch.mm(pdy, w.index_select(-1, retain_index).t_())
      if self.needs_input_grad[1]:
        if retain_index.size(0) == 0:
          pass
        else:
          dw = w.new(w.size()).zero_().index_copy_(
            -1, retain_index, torch.mm(x.t(), pdy))
    else:
      raise Exception(f"Unknown backprop_mode: {self.backprop_mode}")

    if self.needs_input_grad[2]:
      db = torch.mv(dy.t(), self.add_buffer)

    return dx, dw, db
