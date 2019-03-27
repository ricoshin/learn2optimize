import torch
from torch import nn

# configurations
batch_size = 128
input_size = 28*28
hidden_size = 20
n_samples = 10
droprate = 0.1
w_scale = 1e-1
dW_scale = 1e-3

# single layer with non-linearity
U_1 = torch.zeros(batch_size, input_size).cuda()
X_1 = torch.zeros(batch_size, input_size).normal_().cuda()
W_1 = torch.zeros(input_size, hidden_size).normal_().mul(w_scale).cuda()
dW_1 = [torch.zeros(
  W_1.size()).normal_().mul(dW_scale).cuda() for _ in range(n_samples)]
# X_2 = torch.zeros(batch_size, input_size).normal_().cuda()
W_2 = torch.zeros(hidden_size, hidden_size).normal_().mul(w_scale).cuda()
dW_2 = [torch.zeros(
  W_2.size()).normal_().mul(dW_scale).cuda() for _ in range(n_samples)]
act = torch.nn.ReLU()

# drop mask for previous and current layer
drop_masks_1 = [torch.bernoulli(
  torch.ones(hidden_size) * droprate).cuda() for _ in range(n_samples)]
drop_masks_2 = [torch.bernoulli(
  torch.ones(hidden_size) * droprate).cuda() for _ in range(n_samples)]

# generate sparse update
dW_1 = [d.mul(m) for d, m in zip(dW_1, drop_masks_1)]
dW_2 = [d.mul(m) for d, m in zip(dW_2, drop_masks_2)]

# just for warming up GPU
print('Warming up')
X_2_warmup = [[X_1.matmul(W_1 + dW) for dW in dW_1] for _ in range(1000)]
print('Done!')

# Single forward pass
print('\nSingle forward pass')
Z_2_origin = X_1.matmul(W_1)
X_2_origin = act(Z_2_origin)
Z_3_origin = X_2_origin.matmul(W_2)
X_3_origin = act(Z_3_origin)
print('Done!')

# Simply adding different candidate updates
#   for serial forward propagations
print('\nUpdatewise multiple forward pass')
Z_2_baseline = []
X_2_baseline = []
Z_3_baseline = []
X_3_baseline = []
for i in range(n_samples):
  Z_2 = X_1.matmul(W_1 + dW_1[i])
  X_2 = act(Z_2)
  Z_3 = X_2.matmul(W_2 + dW_2[i])
  X_3 = act(Z_3)
  Z_2_baseline.append(Z_2)
  X_2_baseline.append(X_2)
  Z_3_baseline.append(Z_3)
  X_3_baseline.append(X_3)
print('Done!')

# Simulate sparse update
#   x = x + u = x + [df(x) ∘ (x * dW + (W + dW) * u)]
#   (where ∘ : Elementwise product, * : matrix multiplication)
#   Let's denote the first term (x * dW) as (a) and
#   the second term as (W + dW) * u as (b).
print('\nSparsified forward pass')
# To perallelize (x * dW) opration,
#   convert the bigger sparse matrix into smaller dense matrix,
#   then horizontally stack them to single matrix
dW_1_dense = []
dW_2_dense = []
sparse_id_1 = []
sparse_id_2 = []
n_sparse_dim_1 = []
n_sparse_dim_2 = []

for i in range(n_samples):
  sparse_id_1.append(drop_masks_1[i].nonzero().squeeze(-1))
  sparse_id_2.append(drop_masks_2[i].nonzero().squeeze(-1))
  n_sparse_dim_1.append(drop_masks_1[i].sum().long())
  n_sparse_dim_2.append(drop_masks_2[i].sum().long())
  dW_1_dense.append(dW_1[i].index_select(-1, sparse_id_1[-1]))
  dW_2_dense.append(dW_2[i].index_select(-1, sparse_id_2[-1]))

#   Once we concatenate dense smaller matrices for parallel computation,
#     we keep this dense form utill dimension recovery is finally required.
dW_1_dense_b = torch.cat(dW_1_dense, 1)
dW_2_dense_b = torch.cat(dW_2_dense, 1)

# 1st layer

#   term (a)
A_2_dense_b = X_1.matmul(dW_1_dense_b)

#   (a) + (b)
AB_2_dense_b = A_2_dense_b # + B_2_dense_b

#   u
Z_2_dense_b = torch.cat(
  [Z_2_origin.index_select(1, sparse_id_1[i]) for i in range(n_samples)], 1)
df_Z_2_dense_b = (Z_2_dense_b >= 0).float()
U_2_dense_b = df_Z_2_dense_b * AB_2_dense_b

# 2nd layer

#   term (a)
A_3_dense_b = X_2.matmul(dW_2_dense_b)

#   term (b)
B_3_dense_b = []
offset = 0
for i in range(n_samples):
  if not (len(sparse_id_1[i]) and len(sparse_id_2[i])):
    # both previous and current u layer can be dropped(no connections)
    B_3_dense_b.append(torch.zeros(
      X_2.size(0), len(sparse_id_2[i])).cuda())
    continue
  dW = dW_2_dense[i].index_select(0, sparse_id_1[i])
  W = W_2.index_select(0, sparse_id_1[i]).index_select(1, sparse_id_2[i])
  U_2_dense = U_2_dense_b[:, offset:(offset + n_sparse_dim_1[i])]
  offset += n_sparse_dim_1[i]
  B_3_dense_b.append(U_2_dense.matmul(W + dW))

B_3_dense_b = torch.cat(B_3_dense_b, 1)

#   (a) + (b)
AB_3_dense_b = A_3_dense_b + B_3_dense_b

# u
Z_3_dense_b = torch.cat(
  [Z_3_origin.index_select(1, sparse_id_2[i]) for i in range(n_samples)], 1)
df_Z_3_dense_b = (Z_3_dense_b >= 0).float()
U_3_dense_b = df_Z_3_dense_b * AB_3_dense_b

# Recover sparse dimension
U_2 = []
U_2_proposed = []
offset = 0
sparse_cum_id_1 = [sparse_id_1[i] + i * hidden_size for i in range(n_samples)]
sparse_cum_id_2 = [sparse_id_2[i] + i * hidden_size for i in range(n_samples)]
sparse_cum_id_1 = torch.cat(sparse_cum_id_1, 0).squeeze()
sparse_cum_id_2 = torch.cat(sparse_cum_id_2, 0).squeeze()
U_2 = torch.zeros(X_1.size(0), W_1.size(1) * n_samples).cuda()
U_2.index_copy_(1, sparse_cum_id_1, U_2_dense_b)
U_3 = torch.zeros(X_2.size(0), W_2.size(1) * n_samples).cuda()
U_3.index_copy_(1, sparse_cum_id_2, U_3_dense_b)
U_2 = U_2.split(W_1.size(1), 1)
U_3 = U_3.split(W_2.size(1), 1)
X_2_proposed = [X_2_origin + U_2[i] for i in range(n_samples)]
X_3_proposed = [X_3_origin + U_3[i] for i in range(n_samples)]

import pdb; pdb.set_trace()

# Compute error for 1st layer
squared_update = 0
squared_update_err = 0
for i in range(n_samples):
  # compute mean square of update for retained coordinates
  update = X_2_baseline[i] - X_2_origin
  squared_update += update.mul(update).sum()
  # compute mean square of update error for retained coordinates
  update_err = X_2_baseline[i] - X_2_proposed[i]
  squared_update_err += update_err.mul(update_err).sum()

n_nonzero = drop_masks_1[0].sum() * X_2_origin.size(0)
rms_update_1 = squared_update.div(n_nonzero * n_samples).sqrt()
rms_update_err_1 = squared_update_err.div(n_nonzero * n_samples).sqrt()
error_ratio_1 = rms_update_err_1 / rms_update_1

# Compute error for 1st layer
squared_update = 0
squared_update_err = 0
for i in range(n_samples):
  # compute mean square of update for retained coordinates
  update = X_3_baseline[i] - X_3_origin
  squared_update += update.mul(update).sum()
  # compute mean square of update error for retained coordinates
  update_err = X_3_baseline[i] - X_3_proposed[i]
  squared_update_err += update_err.mul(update_err).sum()

n_nonzero = drop_masks_2[0].sum() * X_3_origin.size(0)
rms_update_2 = squared_update.div(n_nonzero * n_samples).sqrt()
rms_update_err_2 = squared_update_err.div(n_nonzero * n_samples).sqrt()
error_ratio_2 = rms_update_err_2 / rms_update_2

import pdb; pdb.set_trace()
print("EOF")
