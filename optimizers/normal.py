import numpy as np
from tqdm import tqdm
from utils import utils
from torch.optim import SGD, Adam, RMSprop

C = utils.getCudaManager('default')
tqdm.monitor_interval = 0


def step_and_get_update(optim, params):
  params_before = params.detach().copy().flat
  optim.step()
  params_after = params.detach().copy().flat
  return params_after - params_before


def test(data_cls, model_cls, optim_cls, optim_args, n_test, optim_it):
  results = []
  for _ in tqdm(range(n_test), 'tests'):
    target = data_cls(training=False)
    optimizee = C(model_cls())
    optimizer = optim_cls(optimizee.parameters(), **optim_args)
    total_loss = []
    for _ in tqdm(range(optim_it), 'optim_interation'):
      loss = optimizee(target)

      total_loss.append(loss.data.cpu().numpy())

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    results.append(total_loss)
  return np.array(results)


def find_best_lr(lr_list, data_cls, model_cls, optim_cls, optim_args, n_test,
                 optim_it):
  best_loss = 999999
  best_lr = 0.0
  lr_list = [
    1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001
  ]
  for lr in tqdm(lr_list, 'Learning rates'):
    try:
      optim_args['lr'] = lr
      loss = np.mean([np.sum(s) for s in test(
        data_cls, model_cls, optim_cls, optim_args, n_test, optim_it)])
    except RuntimeError:
      print('RuntimeError!')
      pass
    if loss < best_loss:
      best_loss = loss
      best_lr = lr
  return best_loss, best_lr
