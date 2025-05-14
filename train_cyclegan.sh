set -ex
# python train.py --dataroot ./datasets/4fold_1 --name 4fold_1 --model cycle_gan --pool_size 50 --no_dropout
# python train.py --dataroot ./datasets/4fold_2 --name 4fold_2 --model cycle_gan --pool_size 50 --no_dropout
# python train.py --dataroot ./datasets/4fold_3 --name 4fold_3 --model cycle_gan --pool_size 50 --no_dropout
# python train.py --dataroot ./datasets/4fold_4 --name 4fold_4 --model cycle_gan --pool_size 50 --no_dropout
python train.py --dataroot ./datasets/241004_allbody_fullset --name 2505014_LSeSim_0.5patch_241004_allbody_fullset --model cycle_LSeSim --pool_size 50 --no_dropout --dataset_mode dicom --original_size 365 --crop_patch_size 128 --make_patch --input_nc 5 --output_nc 5 --load_size 256 --learned_attn --lambda_spatial 10000000.0 --lambda_perceptual 0.1 --lambda_style 0.1
