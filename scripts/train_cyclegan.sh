set -ex
# python train.py --dataroot ./datasets/4fold_1 --name 4fold_1 --model cycle_gan --pool_size 50 --no_dropout
python train.py --dataroot ./datasets/4fold_2 --name 4fold_2 --model cycle_gan --pool_size 50 --no_dropout

