set -ex
# python test.py --dataroot ./datasets/4fold_1 --name 4fold_1 --model cycle_gan --phase test --no_dropout
# python test.py --dataroot ./datasets/4fold_3 --name 4fold_3 --model cycle_gan --phase test --no_dropout
# python test.py --dataroot ./datasets/4fold_4 --name 4fold_4 --model cycle_gan --phase test --no_dropout
# python test.py --dataroot ./datasets/241004_allbody_fullset --name 250604_patch196_CTpixelloss_LSeSim_0.5patch_241004_allbody_fullset --model cycle_LSeSim --phase test --no_dropout --load_size 256 --dataset_mode dicom --input_nc 5 --output_nc 5
# python test.py --dataroot ./datasets/241004_allbody_fullset --name 250624_no-patch_BoneGene_0.5patch_241004_allbody_fullset --model cycleSesim_extention --phase test --no_dropout --load_size 256 --dataset_mode dicom_extention --input_nc 5 --output_nc 5
# python test.py --dataroot ./datasets/241004_allbody_fullset --name 250704_nopatch_Bonemask_0.5patch_241004_allbody_fullset --model cycle_LSeSim --dataset_mode dicom_extention --phase test --no_dropout --load_size 256 --input_nc 5 --output_nc 5
python test.py --dataroot ./datasets/241004_allbody_fullset --name 250708_nopatch_swin_unet_241004_allbody_fullset --netG swin_unet --dataset_mode dicom --model cycle_LSeSim --phase test --no_dropout --load_size 256 --input_nc 5 --output_nc 5

