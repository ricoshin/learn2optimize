import torch
import torch.nn as nn
from optimizers.optim_helpers import (GradientPreprocessor, LSTMStates,
                                      OptimizerStates)
from torch.nn import GRUCell, LSTMCell
from utils import utils


C = utils.getCudaManager('default')

class RNNBase(nn.Module):
  def __init__(self, input_sz=1, out_sz=1, hidden_sz=20, preproc_factor=10.0,
               preproc=True, n_layers=1, rnn_cell='gru'):
    super().__init__()
    assert rnn_cell in ['lstm', 'gru']
    assert isinstance(preproc, (list, tuple, bool))
    self.rnn_cell = rnn_cell
    rnn_cell = LSTMCell if rnn_cell == 'lstm' else GRUCell
    self.hidden_sz = hidden_sz
    self.n_layers = n_layers
    self.rnn = []
    if isinstance(preproc, (list, tuple)) and any(preproc):
      assert len(preproc) == input_sz
      self.process = GradientPreprocessor()
      self.preproc = preproc
      input_sz += sum(preproc)
    elif isinstance(preproc, bool) and preproc:
      self.process = GradientPreprocessor()
      self.preproc = [preproc] * input_sz
      if preproc:
        input_sz *= 2
    module = rnn_cell(input_sz, hidden_sz)  # mean, std
    self.add_module('rnn_0', module)
    self.rnn.append(module)
    for i in range(self.n_layers - 1):
      module = rnn_cell(hidden_sz, hidden_sz)
      self.add_module(f'rnn_{i+1}', module)
      self.rnn.append(module)
    self.output = nn.Linear(hidden_sz, out_sz)

  def forward(self, x, states):
    assert isinstance(x, (tuple, list, torch.Tensor))
    if isinstance(x, (list, tuple)):
      assert len(x) == len(self.preproc)
      x_p = []
      for bool, inp in zip(self.preproc, x):
        inp = C(self.process(inp)) if bool else C(inp)
        x_p.append(inp)
      x = torch.cat(x_p, dim=1)
    elif isinstance(x, torch.Tensor) and self.preproc:
      assert len(self.preproc) == 1
      x = C(self.process(x))

    out_states = []
    if self.rnn_cell == 'lstm':
      for i, rnn in enumerate(self.rnn):
        out_states.append(LSTMStates(*rnn(x, (states[i].h, states[i].c))))
        x = out_states[-1].h
    elif self.rnn_cell == 'gru':
      for i, rnn in enumerate(self.rnn):
        out_states.append(rnn(x, states[i]))
        x = out_states[-1]
    else:
      raise RuntimeError(f'Unknown rnn_cell type: {self.rnn_cell}')
    return self.output(x), OptimizerStates(out_states)
