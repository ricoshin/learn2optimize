CUDA_VISIBLE_DEVICES=0 python main.py --volatile

CUDA_VISIBLE_DEVICES=0 python main.py --train_optim obsrv_single --test_optim obsrv_single --meta_optim adam --lr 1e-3 --save_dir MPL_single_mask_adam_1e-3_0510

CUDA_VISIBLE_DEVICES=0 python main.py --train_optim obsrv_single --test_optim obsrv_single --meta_optim adam --lr 1e-3 --save_dir MPL_single_mask_adam_1e-3_masklr_1e-4


CUDA_VISIBLE_DEVICES=1 python main.py --train_optim obsrv_single --test_optim obsrv_single --meta_optim adam --lr 1e-3 --save_dir MPL_single_mask_adam_1e-3_kl_linear

CUDA_VISIBLE_DEVICES=1 python main.py --train_optim obsrv_single --test_optim obsrv_single --meta_optim adam --lr 1e-3 --save_dir MPL_single_mask_adam_1e-3_kl_logistic

CUDA_VISIBLE_DEVCIES=0 python main.py --train_optim obsrv_single --test_optim obsrv_single --load_dir CNN_MPL_no_mask_adam_1e-3_0513_divde_unroll --no_mask --retest_all --save_dir onlystep_0516

CUDA_VISIBLE_DEVCIES=0 python main.py --train_optim obsrv_single --test_optim obsrv_single --load_dir CNN_MPL_no_mask_adam_1e-3_0513_divde_unroll --retest_all --save_dir init_mask_step_0516

CUDA_VISIBLE_DEVCIES=1 python main.py --train_optim obsrv_single --test_optim obsrv_single --load_dir CNN_MPL_no_mask_adam_1e-3_0513_divde_unroll --no_mask --retest_all --save_dir topk_step_0516

CUDA_VISIBLE_DEVICES=0 python main.py --train_optim obsrv_single --test_optim obsrv_single --meta_optim adam --lr 1e-3 --save_dir CNN_MPL_single_mask_adam_1e-3_0516_meta_kl_linear

CUDA_VISIBLE_DEVICES=1 python main.py --train_optim obsrv_single --test_optim obsrv_single --meta_optim adam --lr 1e-3 --save_dir CNN_MPL_single_mask_adam_1e-3_kl_hard_15

CUDA_VISIBLE_DEVICES=1 python main.py --train_optim obsrv_single --test_optim obsrv_single --meta_optim adam --lr 1e-3 --save_dir CNN_MPL_single_mask_adam_1e-3_kl_hard_30

CUDA_VISIBLE_DEVICES=1 python main.py --train_optim obsrv_single --test_optim obsrv_single --load_dir CNN_MPL_single_mask_adam_1e-2_0514 --save_dir CNN_MPL_single_mask_adam_1e-2_0514_hardmask --retest_all

CUDA_VISIBLE_DEVICES=0 python main.py --train_optim obsrv_single --test_optim obsrv_single --meta_optim adam --lr 1e-3 --save_dir CNN_MPL_single_mask_adam_1e-3_temp_0.1

CUDA_VISIBLE_DEVICES=0 python main.py --train_optim obsrv_single --test_optim obsrv_single --meta_optim adam --lr 1e-3 --save_dir CNN_MPL_single_mask_adam_1e-3_temp_0.1

CUDA_VISIBLE_DEVICES=0 python main.py --train_optim obsrv_single --test_optim obsrv_single --meta_optim adam --lr 1e-3 --save_dir CNN_MPL_single_mask_adam_1e-3_temp_0.001_invloss