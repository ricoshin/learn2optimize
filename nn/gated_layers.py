import torch.nn as nn


class Gate(nn.Module):
  def __init__(self, num_gates, **kwargs):
    super(Gate, self).__init__()
    self.num_gates = num_gates

  def get_weight(self, x):
    raise NotImplementedError

  def get_mask(self):
    raise NotImplementedError

  def get_reg(self, base):
    raise NotImplementedError

  def get_num_active(self):
    return self.get_mask().sum().int().item()

  def forward(self, x):
    z = self.get_weight(x)
    if len(x.size()) == 4:
      z = z.view(-1, self.num_gates, 1, 1)
    else:
      z = z.view(-1, self.num_gates)
    return x * z


class GatedLayer(nn.Module):
  def __init__(self, num_gates):
    super(GatedLayer, self).__init__()
    self.num_gates = num_gates
    self.base = None
    self.gate = None
    self.dgate = None

  def build_gate(self, gate_fn, **kwargs):
    self.gate = gate_fn(self.num_gates, **kwargs)

  def build_gate_dep(self, dgate_fn, **kwargs):
    self.dgate = dgate_fn(self.num_gates, **kwargs)

  def apply_gate(self, x):
    if self.gate is None:
      return x
    else:
      if self.dgate is None:
        return self.gate(x)
      else:
        z = self.gate.get_weight(x)
        return self.dgate(x, z)

  def get_mask(self):
    return None if self.gate is None \
        else self.gate.get_mask()

  def get_reg(self):
    return None if self.gate is None \
        else self.gate.get_reg(self.base)

  def get_reg_dep(self):
    return None if self.dgate is None \
        else self.dgate.get_reg(self.base)

  def get_num_active(self):
    return None if self.gate is None \
        else self.gate.get_num_active()


class GatedLinear(GatedLayer):
  def __init__(self, in_features, out_features, bias=True):
    super(GatedLinear, self).__init__(in_features)
    self.base = nn.Linear(in_features, out_features, bias=bias)

  def forward(self, x):
    return self.base(self.apply_gate(x))


class GatedConv2d(GatedLayer):
  def __init__(self, in_channels, out_channels, kernel_size,
               stride=1, padding=0, dilation=1, groups=1, bias=True):
    super(GatedConv2d, self).__init__(out_channels)
    self.base = nn.Conv2d(in_channels, out_channels, kernel_size,
                          stride=stride, padding=padding, dilation=dilation,
                          groups=groups, bias=bias)

  def forward(self, x):
    return self.apply_gate(self.base(x))
