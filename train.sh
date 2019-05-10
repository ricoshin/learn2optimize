CUDA_VISIBLE_DEVICES=0 python main.py --volatile

CUDA_VISIBLE_DEVICES=0 python main.py --train_optim obsrv_single --test_optim obsrv_single --meta_optim adam --lr 1e-3 --save_dir MPL_single_mask_adam_1e-3_0510

CUDA_VISIBLE_DEVICES=0 python main.py --train_optim obsrv_single --test_optim obsrv_single --meta_optim adam --lr 1e-3 --save_dir MPL_single_mask_adam_1e-3_masklr_1e-4


CUDA_VISIBLE_DEVICES=1 python main.py --train_optim obsrv_single --test_optim obsrv_single --meta_optim adam --lr 1e-3 --save_dir MPL_single_mask_adam_1e-3_kl_linear

###
CUDA_VISIBLE_DEVICES=1 python main.py --train_optim obsrv_single --test_optim obsrv_single --meta_optim adam --lr 1e-3 --save_dir MPL_single_mask_adam_1e-3_kl_logistic

CUDA_VISIBLE_DEVICES=0 python main.py --train_optim neural_base --test_optim neural_base --meta_optim sgd --lr 0.1 --save_dir RNN_sgd_0.1_0510