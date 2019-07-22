import copy
import numpy as np
import torch
import torch.nn as nn
from utils import utils
from utils.result import ResultList, ResultDict
from collections import OrderedDict

C = utils.getCudaManager('default')


class ParamsIndexTracker(object):
  def __init__(self, n_tracks=30):
    self._track_num = n_tracks
    self._track_id = None
    self.n_params = None

  @property
  def track_num(self):
    assert self._track_id is not None
    return ResultList(list(range(self._track_num)))

  @property
  def track_id(self):
    assert self._track_id is not None
    return ResultList(self._track_id.tolist())

  def result_dict(self):
    if not len(self._result_dict) > 0:
      raise RuntimeError("No parameter index is being tracked!")
    self._result_dict.append(
      track_num=ResultList(list(range(self._track_num))),
      track_id=ResultList(self._track_id.tolist()),
    )
    return self._result_dict

  def set_n_params(self, n_params):
    self.n_params = n_params
    return self

  def __call__(self, *args, **kwargs):
    dict_ = dict(*args, **kwargs)
    for name, array_like in dict_.items():
      assert isinstance(name, str)
      if isinstance(array_like, torch.Tensor):
        array_like = array_like.data.tolist()
      n_params = len(array_like)
      if self._track_id is None:
        # tracker automatically recognizes n_params if not specified beforehand
        #   and adjusts preset _track_num to be smaller than that.
        self.n_params = n_params
        if self._track_num > n_params:
          self._track_num = n_params
        self._track_id = np.random.randint(0, self.n_params, self._track_num)
      else:
        # n_params has to be consistent with the one of the first input.
        assert self.n_params is not None
        assert self._track_num is not None
        if not n_params == self.n_params:
          import pdb; pdb.set_trace()
        assert n_params == self.n_params
      tracked = np.take(array_like, self._track_id)
      tracked = [tracked[i] for i in range(self._track_num)]
      dict_[name] = ResultList(tracked)
    dict_.update({'track_num': self.track_num, 'track_id': self.track_id})
    return ResultDict(dict_)


class ParamsSize(object):
  def __init__(self, size_dict, is_flat):
    size_dict_v = [v for v in size_dict.values()][0]
    if (not isinstance(size_dict, dict) or
        not isinstance(size_dict_v, torch.Size)):
      raise Exception("Expected dict with torch.Size instances as values!")
    self._is_flat = is_flat
    self._unflat = size_dict
    self.n_params = 0
    for size in self._unflat.values():
      self.n_params += int(np.prod(size))

  def __repr__(self):
    if self._is_flat:
      return f'ParamsSize({str(self.flat())}, is_flat=True)'
    else:
      return f'ParamsSize({str(self.unflat())}, is_flat=False)'

  # def __get__(self, instance, owner):
  #   if self._is_flat:
  #     return getattr(instance, 'n_params')
  #   else:
  #     return getattr(instance, 'n_params')

  def unflat(self, dim=None, match=None):
    if dim is None:
      return self._unflat
    else:
      assert isinstance(dim, int) and dim > 0
      out = {}
      for k, v in self._unflat:
        if match is not None:
          assert isinstance(match, str)
          if match in k:
            assert len(v) > dim
            out[k] = v[dim]
      return out

  def flat(self):
    return torch.Size([self.n_params, 1])

  def bound(self):
    n_params = 0
    bound = []
    for size in self._unflat.values():
      n_params += int(np.prod(size))
      bound.append(n_params)
    return bound

  def bound_layerwise(self):
    n_params = 0
    bound = []
    for size_list in self._group_by_uint_postfix(self._unflat).values():
      n_params += int(np.sum([np.prod(size) for size in size_list]))
      bound.append(n_params)
    return bound

  def _group_by_uint_postfix(self, size_dict):
    group = OrderedDict()
    invalid_postfix = []
    for name, size_ in size_dict.items():
      postfix = -1
      if '_' in name:
        postfix = name.split('_')[-1]
        if postfix.isdigit() and '.' not in postfix and int(postfix) > 0:
          postfix = int(postfix)
      if not postfix == -1:
        if postfix not in group.keys():
          group[postfix] = [size_]
        else:
          group[postfix].append(size_)
      else:
        invalid_postfix.append(postfix)

    if invalid_postfix and not group:
      return {'-1': invalid_postfix}
    elif group and not invalid_postfix:
      return group
    else:
      raise Exception("Unmatched parameter name postfix. "
                      "Unable to group layerwise!")

  def __eq__(self, other):
    compared = []
    for name in self.unflat.keys():
      compared.append(self.unflat[name] == other.unflat[name])
    if all(compared) and self.flat == other.flat:
      return True
    else:
      return False


class ParamsFlattener(object):
  """This class helps to safely retain gradients during indexing from flattened
  parameters, so that entire optimizee parameters can be conveniently divided
  into mini-batches that still preserves 'grad' attribute.
  Detailed control of setting & getting elements of parameters in batchwise
  manner will be delegated to help.OptimizerBatchManager.

  """
  # NOTE: Once flattened, unflattended version is removed that has been
  #   registered during initialization, and even though we unflatten back
  #   they are different ones that were being tracked by normal optimizers.

  def __init__(self, param_dict):
    assert isinstance(param_dict, dict)
    self._unflat = param_dict
    self._flat = None
    self._shapes = {}
    self._n_params = 0
    self._is_flat = False
    self._grad_cut_hook_handle = None
    self._grad_set_aside = None
    self.retain_shape = False
    # self._flat = None
    # self._flat_grads = None
    # count n_params & remember shapes by name
    for name, param in param_dict.items():
      if param is None:
        self._n_params += 0
        self._shape[name] = None
      else:
        self._n_params += int(np.prod(param.size()))
        self._shapes[name] = param.size()

  # def __get__(self, instance, owner):
  #   if self._is_flat:
  #     return getattr(instance, '_flat')
  #   else:
  #     return getattr(instance, '_unflat')

  @property
  def grad(self):
    if self._is_flat:
      if self._flat.grad is None:
        return None
      else:
        assert self._flat.size() == self._flat.grad.size()
        params = self.new_from_flat(self._flat.grad)
        params._maybe_flat()
        return params
    else:
      if list(self._unflat.values())[0].grad is None:
        ## NOTE: assuming the unflat values require grad or not altogether.
        return None
      else:
        grads = self._get_grads(self._unflat)
        return ParamsFlattener(grads)

  def __len__(self):
    return self._n_params

  def __repr__(self):
    if self._is_flat:
      return f'ParamsFlattener({str(self._flat)}, is_flat=True)'
    else:
      return f'ParamsFlattener({str(self._unflat)}, is_flat=False)'

  def _get_grads(self, params):
    grads = {}
    for k, v in params.items():
      grads[k] = v.grad
    return grads

  def __getitem__(self, key):
    tensor = self.flat[key]
    # if self.flat.grad is not None:
    #   tensor.grad = self.flat.grad[key]
    return tensor

  def __setitem__(self, key, item):
    self.flat[key] = item
    if item.grad is not None:
      self.flat.grad[key] = item.grad

  def _check_if_retain_shape(self):
    if self.retain_shape:
      raise RuntimeError("Can't change shape(flattend/unflatten) when "
                         "ModelParams.retain_shape=True!")

  def _flatten(self, unflat):
    params = [p for p in unflat.values()]
    grads = [p.grad for p in params if p.grad is not None]
    # params = [p for p in unflat.values() if p is not None]
    if params:
      flat = torch.cat([p.contiguous().view(-1, 1) for p in params], 0)
      flat.retain_grad()
    else:
      flat = None
    if len(grads) == len(params):
      flat.grad = torch.cat([g.view(-1, 1) for g in grads], 0)
    return flat

  def _unflatten(self, flat, shapes):
    start = 0
    unflat = {}
    grad_backup = flat.grad
    for name, shape in shapes.items():
      end = start + np.prod(shape)
      unflat[name] = flat[start:end].view(*shape)
      if grad_backup is not None:
        unflat[name].grad = grad_backup[start:end].view(*shape)
      unflat[name].retain_grad()  # NOTE
      start = end
    return unflat

  def _maybe_flat(self):
    self._check_if_retain_shape()
    if self._is_flat:
      assert self._unflat is None
    else:
      self._flat = self._flatten(self._unflat)
      self._unflat = None
      self._is_flat = True

  def _maybe_unflat(self):
    self._check_if_retain_shape()
    if not self._is_flat:
      assert self._flat is None
    else:
      self._unflat = self._unflatten(self._flat, self._shapes)
      self._flat = None
      self._is_flat = False

  def shape_immutable_(self):
    self.retain_shape = True
    return self

  def shape_mutable_(self):
    self.retain_shape = False
    return self

  def new_from_flat(self, flat, reflat=False):
    unflat = self._unflatten(flat, self._shapes)
    params = ParamsFlattener(unflat)
    if reflat:
      params._maybe_flat()
    return params

  def new_from_unflat(self, dim=None, new_size=None):
    assert (dim is not None) == (new_size is not None)
    import pdb; pdb.set_trace()

  def size(self):
    return ParamsSize(self._shapes, self._is_flat)

  def update(self, other):
    assert self.size() == other.size()
    if self._is_flat:
      self._flat = other._flat
    else:
      self._unflat = other._unflat

  def new_zeros(self, size):
    if not isinstance(size, ParamsSize):
      raise TypeError("Expected ParamSize as size argument!")
    if self._is_flat:
      device = self._flat.device
    else:
      device = [v for v in self._unflat.values()][0].device
    sizes = size.unflat().items()
    zeros = {name: torch.zeros(size, device=device) for name, size in sizes}
    return ParamsFlattener(zeros)

  def new_ones(self, size):
    if not isinstance(size, ParamsSize):
      raise TypeError("Expected ParamSize as size argument!")
    if self._is_flat:
      device = self._flat.device
    else:
      device = [v for v in self._unflat.values()][0].device
    sizes = size.unflat().items()
    zeros = {name: torch.ones(size, device=device) for name, size in sizes}
    return ParamsFlattener(zeros)

  def add_flat_(self, flat):
    self.flat = self.flat + flat
    return self

  @property
  def is_flat(self):
    return self._is_flat

  # NOTE: Fix later!1
  def get_flat(self):
    if self._is_flat:
      return self._flat
    else:
      return self._flatten(self._unflat)

  @property
  def flat(self):
    self._maybe_flat()
    return self._flat

  @property
  def unflat(self):
    self._maybe_unflat()
    return self._unflat

  @flat.setter
  def flat(self, value):
    self._maybe_flat()
    assert self._flat.size() == value.size()
    self._flat = value

  @unflat.setter
  def unflat(self, value):
    assert isinstance(value, dict)
    self._maybe_unflat()
    assert len(self._unflat) == len(value)
    self._unflat = value

  def _detach(self, tensor, mode='soft'):
    assert isinstance(tensor, torch.Tensor)
    if mode == 'soft':
      tensor = tensor.clone().detach().requires_grad_(True)
      #tensor = torch.tensor(tensor.data, requires_grad=True)
      tensor.retain_grad()
    elif mode == 'hard':
      tensor = tensor.detach()
    else:
      raise Exception('Unknown detach mode!')
    return tensor

  def detach(self, mode='soft'):
    #detach_ = lambda x: torch.tensor(x.data, requires_grad=True)
    unflat = {}
    for k, v in self.unflat.items():
      unflat[k] = self._detach(v, mode)
    return ParamsFlattener(unflat)

  def detach_(self, mode='soft'):
    #detach_ = lambda x: torch.tensor(x.data, requires_grad=True)
    assert mode in ['soft', 'hard']
    if self._is_flat:
      self._flat = self._detach(self._flat, mode)
    else:
      for k, v in self._unflat.items():
        self._unflat[k] = self._detach(v, mode)
    return self

  def copy(self):
    return copy.deepcopy(self)

  def _default_key_trans_func(self, key):
    # mat_0  ->  linear_0  (params ->  mask)
    prefix, i = key.split('_')
    return '_'.join(['layer', i])
    # if prefix in ['mat']:
    #   return '_'.join(['layer', i])
    # else:
    #   return None

  def apply_func_(self, func, other, *args, **kwargs):
    assert isinstance(other, (ParamsFlattener, dict))
    if isinstance(other, ParamsFlattener):
      other = other.unflat # as dict
    was_flat = self._flat
    self._maybe_unflat()
    out_dict = func(other, *args, **kwargs)
    if was_flat:
      self._maybe_flat()
    return ParamsFlattener(out_dict)

  def __neg__(self):
    return self.mul(-1)

  def __add__(self, other):
    return self.add(other)

  def __sub__(self, other):
    return self.add(-other)

  def __mul__(self, other):
    return self.mul(other)

  def __truediv__(self, other):
    return self.div(other)

  def __gt__(self, other):
    return self.apply_op_with_other('gt', other)

  def __lt__(self, other):
    return self.apply_op_with_other('lt', other)

  def __ge__(self, other):
    return self.apply_op_with_other('ge', other)

  def __le__(self, other):
    return self.apply_op_with_other('le', other)

  def __eq__(self, other):
    return self.apply_op_with_other('eq', other)

  def add(self, other):
    return self.apply_op_with_other('add', other)

  def mul(self, other):
    return self.apply_op_with_other('mul', other)

  def div(self, other):
    return self.apply_op_with_other('div', other)

  def abs(self, *args, **kwargs):
    func = lambda t: torch.abs(t, *args, **kwargs)
    return self.apply_func(func, inplace=False)

  def mean(self, *args, **kwargs):
    func = lambda t: torch.mean(t, *args, **kwargs)
    return self.apply_func(func, inplace=False)

  def std(self, *args, **kwargs):
    func = lambda t: torch.std(t, *args, **kwargs)
    return self.apply_func(func, inplace=False)

  def to(self, *args, **kwargs):
    func = lambda t: getattr(t, 'to')(*args, **kwargs)
    return self.apply_func(func, inplace=False)

  def float(self):
    return self.to(torch.float32)

  def long(self):
    return self.to(torch.int64)

  def byte(self):
    return self.to(torch.unit8)

  def sum(self, *args, **kwargs):
    func = lambda t: torch.sum(t, *args, **kwargs)
    return self.apply_func(func, inplace=False)

  def topk_id(self, other, *args, **kwargs):
    func = lambda inp, k: torch.topk(inp, k, *args, **kwargs)[1]
    return self.apply_func(func, other, inplace=False)

  def tsize(self, *args, **kwargs):
    func = lambda t: t.size(*args, **kwargs)
    return self.apply_func(func, inplace=False, as_dict=True)

  def get_by_substr(self, string):
    return ParamsFlattener(
      {k: v for k, v in self.unflat.items() if string in k})

  def channels(self):
    """Returns sizes of channel dimension as a list in sequence of stacked
    layers. Will be used for module initialization in sparse model."""
    return [v for v in self.get_by_substr('mat').tsize(0).values()][:-1]

  def sparsity(self, threshold=0.5, overall=False, new_prefix='sp',
    as_dict=True):
    if overall:
      on = sum([v for v in (self > threshold).sum().float().unflat.values()])
      # all = sum([v for v in self.tsize(0).values()])
      sparsity = {new_prefix + '_' + 'overall': on / len(self)}
      if not as_dict:
        sparsity = ParamsFlattener(sparsity)
    else:
      sparsity = (self > threshold).sum().float() / self.tsize(0)
      convert_key = lambda k: new_prefix + k.split('_')[1]
      sparsity = sparsity.apply_func_to_keys(convert_key)
      if as_dict:
        sparsity = sparsity.unflat
    return sparsity

  def is_valid_sparsity(self, threshold=0.5):
    return all([v > 0 for v in self.sparsity(overall=False).values()])

  def scatter_float_(self, dim, index, float_):
    func = lambda inp, id: inp.scatter_(dim, id, float_)
    return self.apply_func(func, index, inplace=False)

  def apply_op_with_other(self, op_name, other):
    assert isinstance(other, (ParamsFlattener, dict, int, float, torch.Tensor))
    if isinstance(other, (int, float, torch.Tensor)):
      other_as_dict = {}
      for k, v in self.unflat.items():
        other_as_dict[k] = other
    elif isinstance(other, ParamsFlattener):
      other_as_dict = other.unflat
    elif isinstance(other, dict):
      other_as_dict = other
    else:
      raise Exception(f'Unsupported type for other argument: {type(other)}')
    func = lambda t1, t2: getattr(torch, op_name)(t1, t2)
    return self.apply_func(func, other_as_dict, inplace=False)

  def apply_func_to_keys(self, func):
    assert callable(func)
    return ParamsFlattener({func(k): v for k, v in self.unflat.items()})

  def apply_func(self, func, other=None, inplace=False, as_dict=False):
    if other is not None:
      assert isinstance(other, (ParamsFlattener, dict))
      if isinstance(other, ParamsFlattener):
        other = other.unflat # as dict
      assert self.unflat.keys() == other.keys()

    if not inplace:
      out_dict = {}
    for k, v in self.unflat.items():
      if other is None:
        if inplace:
          self.unflat[k] = func(v)
        else:
          out_dict[k] = func(v)
      else:
        if inplace:
          self.unflat[k] = func(v, other[k])
        else:
          out_dict[k] = func(v, other[k])
    if inplace:
      return None
    else:
      if as_dict:
        return out_dict
      else:
        return ParamsFlattener(out_dict)

  def expand_as(self, other, key_trans_func=None):
    if key_trans_func is None:
      key_trans_func = self._default_key_trans_func
      assert callable(key_trans_func)
    def func(other):
      key_translated = [key_trans_func(k) for k in other.keys()]
      out_dict = {}
      for i, (k_other, v_other) in enumerate(other.items()):
        k_self = key_translated[i]
        size_diff = len(v_other.size()) - len(self.unflat[k_self].size())
        if k_self: # mat (copy the mask pattern along the column axis)
          if size_diff > 0:
            v_self = self.unflat[k_self].view(-1, *((1,) * size_diff))
          elif size_diff == 0:
            v_self = self.unflat[k_self]
          else:
            import pdb; pdb.set_trace()
            # raise Exception()
          out_dict[k_other] = v_self.expand_as(v_other)
        else: # bias (no need to drop - fill ones)
          # NOTE: only when key of bias is None
          out_dict[k_other] = v_other.new_ones(v_other.size())
        # else:
          # out_dict[k_other] = self.unflat[k_self].squeeze()
      return out_dict
    return self.apply_func_(func, other)

  def masked_select(self, mask, key_trans_func=None):
    if key_trans_func is None:
      key_trans_func = self._default_key_trans_func
      assert callable(key_trans_func)
    def func(mask):
      key_translated = [key_trans_func(k) for k in self.unflat.keys()]
      out_dict = {}
      for i, (k_self, v_self) in enumerate(self.unflat.items()):
        k_mask = key_translated[i]
        v_mask = mask[k_mask].byte()
        import pdb; pdb.set_trace()
        out_dict[k_mask] = self.unflat[k_self].masked_select(v_mask)
      return out_dict

  def sparse_update(self, mask, delta, key_trans_func=None):
    assert isinstance(mask, ParamsFlattener)
    assert isinstance(delta, ParamsFlattener)
    if key_trans_func is None:
      key_trans_func = self._default_key_trans_func
      assert callable(key_trans_func)
    last_layer = int([k for k in mask.unflat.keys()][-1].split('_')[1])

    def func(mask, delta):
      m_keys = [key_trans_func(k) for k in self.unflat.keys()]  # mask
      p_keys = [k for k in self.unflat.keys()]  # params
      out = {}
      for m_key, p_key in zip(m_keys, p_keys):
        p_name, p_layer = p_key.split('_')
        p = self.unflat[p_key]
        d = delta.unflat[p_key]
        if int(p_layer) < last_layer:
          m = mask[m_key].squeeze().nonzero().squeeze()
          d_size = (d.size(0), p.size(1)) if p_name == 'mat' else p.size()
          d_ = p.new(*d_size).zero_().index_copy_(-1, m, d)
        else:  # do not prune the last units
          d_ = d
          # when mat_x, dim=1 / when bias_x, dim=0
        if p_name == 'mat' and int(p_layer) > 0:
          prev_m_key = key_trans_func(p_name + '_' + str(int(p_layer) - 1))
          m = mask[prev_m_key].squeeze().nonzero().squeeze()
          d_ = p.new(p.size()).zero_().index_copy_(0, m, d_)
        out[p_key] = p + d_
      return out

    return self.apply_func_(func, mask, delta=delta)

  def prune(self, mask, key_trans_func=None):
    assert isinstance(mask, ParamsFlattener)
    if key_trans_func is None:
      key_trans_func = self._default_key_trans_func
      assert callable(key_trans_func)
    last_layer = int([k for k in mask.unflat.keys()][-1].split('_')[1])

    def func(mask):
      m_keys = [key_trans_func(k) for k in self.unflat.keys()]  # mask
      p_keys = [k for k in self.unflat.keys()]  # params
      # v_self = [v for v in self.unflat.values()]
      out = {}
      for m_key, p_key in zip(m_keys, p_keys):# range(len(self.unflat)):
        p_name, p_layer = p_key.split('_')
        p = self.unflat[p_key]
        # m = mask[m_key].squeeze().nonzero().squeeze()
        # out[p_key] = p.index_select(-1, m)
        if int(p_layer) < last_layer:
          m = mask[m_key].squeeze().nonzero().squeeze()
          out[p_key] = p.index_select(0, m)
          # import pdb; pdb.set_trace()
        else:  # do not prune the last units
          out[p_key] = p
          # when mat_x, dim=1 / when bias_x, dim=0
        if p_name == 'mat' and int(p_layer) > 0:
          prev_m_key = key_trans_func(p_name + '_' + str(int(p_layer) - 1))
          m = mask[prev_m_key].squeeze().nonzero().squeeze()
          if p_key in out.keys():
            out[p_key] = out[p_key].index_select(1, m)
          else:
            out[p_key] = p.index_select(1, m)
      return out

    return self.apply_func_(func, mask)

  def register_grad_cut_hook_(self):
    if self._grad_cut_hook_handle is not None:
      raise RuntimeError("grad cut hook already exists!")
    if self._is_flat:
      raise RuntimeError("grad cut hook can registered on unflattened params!")
    self._grad_cut_hook_handle = []
    self._grad_set_aside = {}
    for k, v in self._unflat.items():
      def grad_cut(grad):
        self._grad_set_aside[k] = grad
        return grad.new_zeros(grad.size())
      self._grad_cut_hook_handle.append(v.register_hook(grad_cut))
    return self.shape_immutable_()

  def remove_grad_cut_hook_(self):
    if self._grad_cut_hook_handle is None or self._grad_set_aside is None:
      raise RuntimeError("grad cut hook does not exist!")
    assert isinstance(self._grad_cut_hook_handle, list)
    assert len(self._grad_set_aside) > 0
    assert not self._is_flat
    for handle in self._grad_cut_hook_handle:
      handle.remove()
    for k, v in self._grad_set_aside.items():
      self._unflat[k].grad = v
    return self.shape_mutable_()

  def register_parameter_to(self, module):
    """Function for registering self.unflat as parameters.

    Expected to be called when optimizee instance is initialized so that
    parameters can be accessed by other pytorch APIs right after the
    construction of optimizee.

    Registration should be conducted with unflattened parameters which is
    considered as standard form when you do the optimization with normal
    optimizers.

    """
    for name, value in self.unflat.items():
      self.unflat[name] = nn.Parameter(value)
      module.register_parameter(name, self.unflat[name])

  def _registered_params(self, params):
    """Wrap torch.Tensor in the params dictionary with torch.nn.parameters,
    and register them in the module. Parameters will be converted to tensors
    when they are flattened and can't be tracked by module.parameters() or
    module.cuda(). To resolve this issue, parameters should be re-registered
    after being unflattened back.

    Args:
      params (dict): Dictionary containing model parameter(tensors).

    Returns:
      out_params (dict): Dictionary containing model parameters wrapped with
        nn.Parameters that have been already registered in the module.

    """
    out_params = {}
    for name, value in params.items():
      out_params[name] = nn.Parameter(value)
      self.register_parameter(name, out_params[name])
    return out_params
