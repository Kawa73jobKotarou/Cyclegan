set -ex
# python test.py --dataroot ./datasets/4fold_1 --name 4fold_1 --model cycle_gan --phase test --no_dropout
# python test.py --dataroot ./datasets/4fold_3 --name 4fold_3 --model cycle_gan --phase test --no_dropout
# python test.py --dataroot ./datasets/4fold_4 --name 4fold_4 --model cycle_gan --phase test --no_dropout
python test.py --dataroot ./datasets/241004_allbody_fullset --name 2505020_LSeSim_no-perceptual_0.5patch_241004_allbody_fullset --model cycle_LSeSim --phase test --no_dropout --load_size 256 --dataset_mode dicom --input_nc 5 --output_nc 5
