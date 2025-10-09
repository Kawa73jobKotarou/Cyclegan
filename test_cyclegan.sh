set -ex
# python test.py --dataroot ./datasets/2509_add_train_before --name 251007_pix2pix_2508_add_train --model pix2pix --phase test --no_dropout --load_size 256 --dataset_mode dicom_extention --input_nc 5 --output_nc 5
python test.py --results_dir ./results/MRCAT --dataroot ./datasets/MRCAT --name 251007_pix2pix_2508_add_train --model pix2pix --phase test --no_dropout --load_size 256 --dataset_mode dicom_extention --input_nc 5 --output_nc 5
# python test.py --dataroot ./datasets/241004_allbody_fullset --name 250604_patch196_CTpixelloss_LSeSim_0.5patch_241004_allbody_fullset --model cycle_LSeSim --phase test --no_dropout --load_size 256 --dataset_mode dicom --input_nc 5 --output_nc 5
# python test.py --dataroot ./datasets/2508_add_train --name 250802_no-patch_BoneGene_0.5patch_2508_add_train --model cycleSesim_extention --phase test --no_dropout --load_size 256 --dataset_mode dicom_extention --input_nc 5 --output_nc 5
# python test.py --dataroot ./datasets/241004_allbody_fullset --name 250704_nopatch_Bonemask_0.5patch_241004_allbody_fullset --model cycle_LSeSim --dataset_mode dicom_extention --phase test --no_dropout --load_size 256 --input_nc 5 --output_nc 5
# python test.py --dataroot ./datasets/241004_allbody_fullset --name 250708_nopatch_swin_unet_241004_allbody_fullset --netG swin_unet --dataset_mode dicom --model cycle_LSeSim --phase test --no_dropout --load_size 256 --input_nc 5 --output_nc 5
# python test.py --results_dir ./results/only_before --dataroot ./datasets/2509_add_train_before --name 250927_unpair_no-patch_BoneGene_0.5patch_2508_add_train_before --model cycleSesim_extention --phase test --no_dropout --load_size 256 --dataset_mode dicom_extention --input_nc 5 --output_nc 5
# python test.py --results_dir ./results/MRCAT --dataroot ./datasets/MRCAT --name 250912_unpair_no-patch_BoneGene_0.5patch_2508_add_train --model cycleSesim_extention --phase test --no_dropout --load_size 256 --dataset_mode dicom_extention --input_nc 5 --output_nc 5

