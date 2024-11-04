CUDA_VISIBLE_DEVICES=1 /opt/data/private/rstao/code/DeepfakeDetection/pytorch18/bin/python3 train5_2_domains.py --name horse_chair_2_classes_2_layers_ours \
--dataroot /root/rstao/datasets \
--dataset1_classes horse \
--dataset2_classes chair \
--classes chair,horse --batch_size 32 --delr_freq 4 --lr 0.0002 --niter 40 --lnum 64 --delr 0.9 \
--pth random_sobel --train_split progan_train --val_split progan_val --mode_method ours