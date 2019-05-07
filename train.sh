## Need to change train_optimizers & test_optimzers in config  ##

# meta_model -> rnn / ours
# optim -> sgd / adam
# lr -> 0.1 ...
# not_masking -> for single mlp, if add this option, neural_bnn_obsrv optimizer do not masking
# multi_obsrv -> Change train/test optim_module with neural_bnn_obsrv3 optimizer
# k_obsrv -> the number of samples for sparse observation 
##
## RNN-base
CUDA_VISIBLE_DEVICES=0 python main.py --meta_model rnn --optim sgd --lr 1.0 --save_dir RNN_base_sgd_1.0
CUDA_VISIBLE_DEVICES=0 python main.py --meta_model rnn --optim sgd --lr 0.1 --save_dir RNN_base_sgd_0.1
CUDA_VISIBLE_DEVICES=0 python main.py --meta_model rnn --optim adam --lr 1.0 --save_dir RNN_base_adam_1.0
CUDA_VISIBLE_DEVICES=0 python main.py --meta_model rnn --optim adam --lr 0.1 --save_dir RNN_base_adam_0.1
## MLP
CUDA_VISIBLE_DEVICES=0 python main.py --meta_model ours --not_masking --optim sgd --lr 1.0 --save_dir MLP_single_sgd_1.0
CUDA_VISIBLE_DEVICES=0 python main.py --meta_model ours --not_masking --optim sgd --lr 0.1 --save_dir MLP_single_sgd_0.1
CUDA_VISIBLE_DEVICES=0 python main.py --meta_model ours --not_masking --optim adam --lr 1.0 --save_dir MLP_single_adam_0.1

## MLP k obsrv
CUDA_VISIBLE_DEVICES=0 python main.py --meta_model ours --multi_obsrv --k_obsrv 5 --optim adam --lr 1e-1 --save_dir MLP_obsrv10_adam_0.1