## Need to change train_optimizers & test_optimzers in config  ##
## Need to change drop rate of step generator ##


# meta_model -> rnn / ours
# optim -> sgd / adam
# lr -> 0.1 ...
# not_masking -> for single mlp, if add this option, neural_bnn_obsrv optimizer do not masking
# multi_obsrv -> Change train/test optim_module with neural_bnn_obsrv3 optimizer
# k_obsrv -> the number of samples for sparse observation 
# save_image -> save the inner plots in save_dir folder

## RNN-base SGD 1.0 unstable / ADAM 1e-4 converge slow
#CUDA_VISIBLE_DEVICES=0 python main.py --meta_model rnn --optim sgd --lr 1.0 --save_dir RNN_base_sgd_1.0
#CUDA_VISIBLE_DEVICES=0 python main.py --meta_model rnn --optim sgd --lr 1.0 --save_dir RNN_base_sgd_1.0_2
#CUDA_VISIBLE_DEVICES=1 python main.py --meta_model rnn --optim sgd --lr 0.1 --save_dir RNN_base_sgd_0.1
#CUDA_VISIBLE_DEVICES=1 python main.py --meta_model rnn --optim sgd --lr 0.1 --save_dir RNN_base_sgd_0.1_2
#CUDA_VISIBLE_DEVICES=2 python main.py --meta_model rnn --optim adam --lr 1e-3 --save_dir RNN_base_adam_1e-3
#CUDA_VISIBLE_DEVICES=2 python main.py --meta_model rnn --optim adam --lr 1.0 --save_dir RNN_base_adam_1.0_2
#CUDA_VISIBLE_DEVICES=3 python main.py --meta_model rnn --optim adam --lr 1e-4 --save_dir RNN_base_adam_1e-4
#CUDA_VISIBLE_DEVICES=3 python main.py --meta_model rnn --optim adam --lr 0.1 --save_dir RNN_base_adam_0.1_2
## MLP Not masking SGD 1.0 / ADAM 1e-3 unstable
CUDA_VISIBLE_DEVICES=0 python main.py --meta_model ours --not_masking --optim sgd --lr 1.0 --save_dir MLP_single_nomask_sgd_1.0
#CUDA_VISIBLE_DEVICES=0 python main.py --meta_model ours --not_masking --optim sgd --lr 1.0 --save_dir MLP_single_nomask_sgd_1.0_2
CUDA_VISIBLE_DEVICES=0 python main.py --meta_model ours --not_masking --optim sgd --lr 0.1 --save_dir MLP_single_nomask_sgd_0.1
#CUDA_VISIBLE_DEVICES=0 python main.py --meta_model ours --not_masking --optim sgd --lr 0.1 --save_dir MLP_single_nomask_sgd_0.1_2
CUDA_VISIBLE_DEVICES=0 python main.py --meta_model ours --not_masking --optim adam --lr 1e-3 --save_dir MLP_single_nomask_adam_1e-3
#CUDA_VISIBLE_DEVICES=0 python main.py --meta_model ours --not_masking --optim adam --lr 1e-3 --save_dir MLP_single_nomask_adam_1e-3_2
CUDA_VISIBLE_DEVICES=1 python main.py --meta_model ours --not_masking --optim adam --lr 1e-4 --save_dir MLP_single_nomask_adam_1e-4
#CUDA_VISIBLE_DEVICES=1 python main.py --meta_model ours --not_masking --optim adam --lr 1e-4 --save_dir MLP_single_nomask_adam_1e-4_2
## MLP masking SGD 1.0 / ADAM 1e-3 unstable
CUDA_VISIBLE_DEVICES=0 python main.py --meta_model ours --optim sgd --lr 1.0 --save_dir MLP_single_mask_sgd_1.0
#CUDA_VISIBLE_DEVICES=0 python main.py --meta_model ours --optim sgd --lr 1.0 --save_dir MLP_single_mask_sgd_1.0_2
CUDA_VISIBLE_DEVICES=1 python main.py --meta_model ours --optim sgd --lr 0.1 --save_dir MLP_single_mask_sgd_0.1
#CUDA_VISIBLE_DEVICES=1 python main.py --meta_model ours --optim sgd --lr 0.1 --save_dir MLP_single_mask_sgd_0.1_2
CUDA_VISIBLE_DEVICES=1 python main.py --meta_model ours --optim adam --lr 1e-3 --save_dir MLP_single_mask_adam_1e-3
#CUDA_VISIBLE_DEVICES=1 python main.py --meta_model ours --optim adam --lr 1e-3 --save_dir MLP_single_mask_adam_1e-3_2
CUDA_VISIBLE_DEVICES=1 python main.py --meta_model ours --optim adam --lr 1e-4 --save_dir MLP_single_mask_adam_1e-4
#CUDA_VISIBLE_DEVICES=1 python main.py --meta_model ours --optim adam --lr 1e-4 --save_dir MLP_single_mask_adam_1e-4_2


## MLP k obsrv
#CUDA_VISIBLE_DEVICES=0 python main.py --meta_model ours --multi_obsrv --k_obsrv 5 --optim adam --lr 1e-1 --save_dir MLP_obsrv10_adam_0.1