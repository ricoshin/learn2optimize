import torch
from torch import nn
try:
  from utils import utils
except ModuleNotFoundError:
  # For execution as a __main__ file
  import sys
  sys.path.append('../')
  from utils import utils


C = utils.getCudaManager('default')

class MCdropBNN(nn.Module):
  """This model is a Bayesian MLP optimizer based on simplified version
  of the one proposed in 'Understanding and correcting pathologies in the
  training of learned optimizers.(Metz et al.)', where it only incorporates
  a few significant input features, such as momentums at different decaying
  values. And Posterior estimation is performed by test time MCdropout.
  """
  def __init__(self, hidden_sz=32, n_layers=1, out_temp=1e-3,
    decay_values=[0.5, 0.9, 0.99, 0.999, 0.9999], batch_std_norm=True,
    dropout_rate=0.5):
    super().__init__()
    assert n_layers > 0
    assert isinstance(decay_values, (list, tuple))
    self.input_sz = 2 + len(decay_values) # grad, weight, momentums
    self.output_sz = 2 # log learning rate, update direction
    self.hidden_sz = hidden_sz
    self.out_temp = out_temp
    self.decay_values = C(torch.tensor(decay_values))
    self.momentum = None
    # self.momentum = [None for _ in range(decay_values)]
    self.batch_std_norm = batch_std_norm
    self.layers = nn.ModuleList()
    for i in range(n_layers + 1):
      input_sz = self.input_sz if i == 0 else hidden_sz
      output_sz = self.output_sz if i == n_layers else hidden_sz
      self.layers.append(nn.Linear(input_sz, output_sz))
      if not i == n_layers: # accept for the last layer
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(p=dropout_rate))

  def new(self):
    # reset momentum
    self.momentum = None
    return self

  def forward(self, x):
    """What you should feed to this function:
       x[B, 0]: current gradient
       x[B, 1]: current weight
    """
    assert x.size(1) == 2
    batch_sz = x.size(0)
    g = x[:, 0].unsqueeze(-1) # gradient

    if self.momentum is None:
      # Initialize momentum matrix
      #   momentum matrix: m[B, len(decay_values)]
      #   should keep track of momentums for each B dimension
      self.momentum = g.repeat(1, len(self.decay_values))
    else:
      # Dimension will be matched by broadcasting
      r = self.decay_values
      self.momentum = r * self.momentum + (1 - r) * g

    x = torch.cat([x, self.momentum], 1)

    # regularization(as in Metz et al.)
    if self.batch_std_norm:
      x = x.div(x.std(0, keepdim=True))

    x = x.detach() # this has to be a leaf node(stop gradient)

    """Actual inputs model expects:
       x[B, 0]: current gradient
       x[B, 1]: current weight
       x[B, 2:]: momentums (for each decaying values)
    """
    assert x.size(1) == self.input_sz
    for layer in self.layers:
      x = layer(x)

    """Outputs:
      y[B, 0]: per parameter log learning rate
      y[B, 1]: unnormalized update direction
    """
    assert x.size(1) == self.output_sz
    out_1 = x[:, 0] * self.out_temp
    out_2 = x[:, 1] * self.out_temp

    return torch.exp(out_1) + out_2


  def measure_uncertainty(self, x, n_samples=1000):
    sum = 0
    sq_sum = 0

    for i in range(n_samples):
      out = self.forward(x)
      sum += out
      sq_sum += out*out

    mean = sum / n_samples
    mean_sq = sq_sum / n_samples
    var = (mean_sq - mean**2) * (n_samples) / (n_samples -1)
    std = var.sqrt()

    return mean, std


def target_func(x):
  assert isinstance(x, torch.Tensor)
  assert x.size(1) == 2
  x_1 = x[:, 0]
  x_2 = x[:, 1]
  sin = torch.sin
  y = 0.4*x_1 + 0.02*sin(200*x_2) + 0.02*sin(70*x_1) + 0.3*sin(10*x_2)
  noise = torch.zeros(y.size()).normal_(0.02).cuda()
  return y + noise


def sample_data(batch_sz, input_sz):
  x = torch.cat([
    # torch.zeros(batch_sz//2, input_sz).uniform_(-1, 0),
    # torch.zeros(batch_sz//2, input_sz).uniform_(1, 2),
    torch.zeros(batch_sz//2, input_sz).uniform_(-2, -1),
    torch.zeros(batch_sz//2, input_sz).uniform_(0, 1),
  ], 0).cuda()
  return x, target_func(x)


def sample_data_unseen(batch_sz, input_sz):
  x = torch.cat([
    # torch.zeros(batch_sz//2, input_sz).uniform_(-2, -1),
    # torch.zeros(batch_sz//2, input_sz).uniform_(0, 1),
    torch.zeros(batch_sz//2, input_sz).uniform_(-1, 0),
    torch.zeros(batch_sz//2, input_sz).uniform_(1, 2),
  ], 0).cuda()
  return x, target_func(x)


def test():
  print('Test for MCdropBNN module.')
  batch_size = 128
  input_size = 2
  training_step = 1000
  uncentainty_test_step = 10

  # Bayesian MLP network
  bnn = MCdropBNN().cuda()
  print(bnn, '\n')

  # minimize mse loss w.r.t an arbitrary function
  print('[!] Training BNN')
  optim = torch.optim.SGD(bnn.parameters(), lr=0.1, weight_decay=1e-4)
  for _ in range(training_step):
    x, y_label = sample_data(batch_size, input_size)
    y_pred = bnn(x)
    loss = ((y_label - y_pred)**2).mean()
    loss.backward()
    optim.step()
  print('[!] Done!\n')

  print(f'[!] Input size: {x.size()}')
  print(f'[!] Output size: {y_pred.size()}')
  print(f'[!] Output values: {y_pred}\n')

  # bnn.eval()  # uncomment this to test deterministic network

  print('[!] Measuring uncertainty of seen & unseen data\n')
  # test uncertainty for seen and unseen data distribution
  for _ in range(uncentainty_test_step):
    x_seen, _ = sample_data(batch_size//2, input_size)
    x_unseen, _ = sample_data_unseen(batch_size//2, input_size)
    x = torch.cat([x_seen, x_unseen], 0)
    y = bnn.new()(x)
    mu, sigma = bnn.measure_uncertainty(x, 1000)
    # print(f'[!] sample(seen): {y[:batch_size//2]}')
    # print(f'[!] sample(unseen): {y[batch_size//2:]}')
    print(f'[!] mu(seen): {mu[:batch_size//2].mean()}')
    print(f'[!] mu(unseen): {mu[batch_size//2:].mean()}')
    print(f'[!] sigma(seen): {sigma[:batch_size//2].mean()}')
    print(f'[!] sigma(unseen): {sigma[batch_size//2:].mean()}')
    ratio = sigma[batch_size//2:].mean() / sigma[:batch_size//2].mean()
    print(f'[!] uncertainty(sigma) of unseen data is x{ratio} larger.\n')

  print('End of test.')


if __name__ == '__main__':
  test()
