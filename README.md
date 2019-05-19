# learned_optimizer
Pytorch implementation of learned optimizer (research in progress)

## Dependencies:

+ Python 3.3+.
+ PyTorch 1.0+.
+ Tensorpack(https://github.com/tensorpack/tensorpack)
```
pip install --upgrade git+https://github.com/tensorpack/tensorpack.git
# or add `--user` to install to user's local directories
```

## How to run:

Example(which can be changed):
```
python main.py --problem=imagenet --train_optim=obsrv_single --save_dir=./savepath
```
