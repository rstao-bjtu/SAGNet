GPU=0
directory="checkpoints/clip_chair_horse_2_domains_diff_ours__2024_01_06_18_33_20__lnum_64__random_sobel/"


# 切换到目录
cd "$directory"

# 获取.pth文件列表，从第24个pth文件开始
files=($(find . -maxdepth 1 -type f -name "model_epoch_*.pth" | sort -V | tail -n +0))

cd "/opt/data/private/rstao/code/DeepfakeDetection/CNNDetection_UniversalFakeDetect_Adv"
# 遍历文件
for file in "${files[@]}"; do
    echo "Processing file: $file"
    CUDA_VISIBLE_DEVICES=$GPU /opt/data/private/rstao/code/DeepfakeDetection/pytorch18/bin/python3 /opt/data/private/rstao/code/DeepfakeDetection/CNNDetection_UniversalFakeDetect_Adv/eval_test8gan.py \
    --dataroot /root/rstao/datasets/CNN_synth_testset --pth_dataroot $file
    # 在这里可以执行你想要的操作，比如打印文件内容或者执行其他命令
done

# CUDA_VISIBLE_DEVICES=$GPU /opt/data/private/rstao/code/DeepfakeDetection/pytorch18/bin/python3 eval_test8gan.py --dataroot /root/rstao/datasets/CNN_synth_testset --pth_path $directory --pth_file model_epoch_24.pth