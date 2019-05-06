from utils.analyze_model import analyzing_mask, plot_loss, sampling_mask
from utils.analyze_surface import (eval_gauss_var, eval_lipschitz,
                                   inversed_masked_params,
                                   random_masked_params)
from utils.result import ResultDict


def model_analyzer(optim, mode, model, params, model_cls, data, iter,
                   analyze_mask, sample_mask, draw_loss):

  text_dir = 'test/analyze_mask'
  result_dir = 'test/drawloss'
  sample_dir = 'test/mask_compare'
  mask_dict = ResultDict()
  iter_interval = 10
  sample_num = 10000

  if mode == 'test' and analyze_mask:
    analyzing_mask(
        optim.mask_gen, layer_size, mode, iter, iter_interval, text_dir)

  if mode == 'test' and sample_mask:
    mask_result = sampling_mask(
        optim.mask_gen, layer_size, model_train, params, sample_num, mode,
        iter, iter_interval, sample_dir)
    if mask_result is not None:
      mask_dict.append(mask_result)

  if mode == 'test' and draw_loss:
    plot_loss(
        model_cls=model_cls, model=model_train, params=params,
        input_data=data['in_train'].load(), dataset=data['in_train'],
        feature_gen=optim.feature_gen, mask_gen=optim.mask_gen,
        step_gen=optim.step_gen, scale_way=None, xmin=-2.0, xmax=0.5, num_x=20,
        mode=mode, iteration=iter, iter_interval=iter_interval,
        loss_dir=result_dir)

  if sample_mask:
    plot_mask_result(mask_dict, sample_dir)

  return


def surface_analyzer(params, best_mask, step, set_size, sparse_r, writer, iter):
  lips_dense = {}
  lips_any = {}
  lips_best = {}
  lips_rand = {}
  lips_inv = {}

  params_pruned_inv = inversed_masked_params(
    params, best_mask, step, set_size, sparse_r)
  params_pruned_rand = random_masked_params(
    params, step, set_size, sparse_r)

  for k, v in set_size.items():
    n = k.split('_')[1]
    r = sparse_r[f"sparse_{n}"]
    lips_best[f"lips_{n}"] = eval_lipschitz(best_pruned, n).tolist()[0]
    lips_any[f"lips_{n}"] = eval_lipschitz(params_pruned, n).tolist()[0]
    lips_rand[f"lips_{n}"] = eval_lipschitz(params_pruned_rand, n).tolist()[0]
    lips_inv[f"lips_{n}"] = eval_lipschitz(params_pruned_inv, n).tolist()[0]
    lips_dense[f"lips_{n}"] = eval_lipschitz(params_sparse, n).tolist()[0]

  log_tf_event(result, 'meta-test', iter)
  log_tf_event(lips_b, 'lipschitz constant', iter)
  log_tf_event(lips_a, 'lipschitz constant', iter)
  log_tf_event(lips_r, 'lipschitz constant', iter)
  log_tf_event(lips_i, 'lipschitz constant', iter)
  log_tf_event(lips_d, 'lipschitz constant', iter)