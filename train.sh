CUDA_VISIBLE_DEVICES=0 python main.py --volatile
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
