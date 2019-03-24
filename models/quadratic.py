import copy
import torch
import torch.nn as nn
from utils import utils
from models.model_helpers import ParamsFlattener

C = utils.getCudaManager('defalt')


class QuadraticData:
  def __init__(self, **kwargs):
    batch_size = 10
    self.W = C(torch.randn(batch_size,10))
    self.y = C(torch.randn(batch_size))

  def sample(self):
    return self.W, self.y

  # def get_loss(self, theta):
    # return torch.sum((self.W.matmul(theta) - self.y)**2)



class QuadraticModel(nn.Module):
  def __init__(self, params=None):
    super().__init__()
    if params is not None:
      if not isinstance(params, ParamsFlattener):
        raise TypeError("params argumennts has to be "
                        "an instance of ParamsFlattener!")
      self.params = params
    else:
      theta = torch.zeros(10)
      self.params = ParamsFlattener({'theta': theta})

    #self.params.register_params_to_module(self)
  #
  # def cuda(self):
  #   self.params = self.params.cuda()
  #   return self
  #
  # def parameters(self):
  #   for param in self.params.unflat.values():
  #     yield param

  def forward(self, inp, out):
    theta = self.params.unflat['theta']
    return torch.sum((inp.matmul(theta) - out)**2)
    #return target.get_loss(self.params.unflat['theta'])

  # def all_named_parameters(self):
  #   return [('theta', self.params.dict['theta'])]
