import numpy as np
import torch.nn as nn
from utils import utils

T = utils.getTensorManager('default')


class QuadraticModel(nn.Module):
  pass


class MNISTModel(nn.Module):
  def __init__(self):
    super(MNISTModel, self).__init__()
    self.linear1 = nn.Linear(28 * 28, 32)
    self.linear2 = nn.Linear(32, 10)

  def forward(self, inputs):
    x = inputs.view(-1, 28 * 28)
    x = F.relu(self.linear1(x))
    x = self.linear2(x)
    return F.log_softmax(x)


class GradientPreprocessor(nn.Module):
  def __init__(self, p=10.0, eps=1e-6):
    super(GradientPreprocessor, self).__init__()
    self.eps = eps
    self.p = p

  def forward(self, input):
    indicator = (x.abs() > math.exp(-self.p)).float()
    x1 = (x.abs() + self.eps).log() / self.p * indicator - (1 - indicator)
    x2 = x.sign() * indicator + math.exp(self.p) * input * (1 - indicator)

    return torch.cat((x1, x2), 1)


class LSTMOptimizer(nn.Module):
  def __init__(self):
    super(LSTMOptimizer, self).__init__()
    self.hidden_size = 20

    self.preprocess = GradientPreprocessor()
    self.lstm1 = nn.LSTMCell(2, self.hidden_size)
    self.lstm2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
    self.output = nn.Linear(self.hidden_size, 1)

  def forward(slef, input, hx, cx):
    input = T(self.preprocess(input).detach())
    hx0, cx0 = self.lstm1(input, (hx[0], cx[0]))
    hx1, cx1 = self.lstm2(hx0, (hx[1], cx[1]))
    return self.output(hx1), (hx0, hx1), (cx0, cx1)
