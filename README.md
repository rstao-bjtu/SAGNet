# SAGNet: Decoupling Semantic-Agnostic Artifacts from Limited Training Data for Robust Generalization in Deepfake Detection

**Classification environment:** 
We recommend installing the required packages by running the command:
```sh
pip install -r requirements.txt
```

## Getting the data
Download dataset from [CNNDetection](https://github.com/peterwang512/CNNDetection).

## Training the model 
```sh
sh ./train5_2_domains_ours.sh
```

## Testing the detector
Download all pretrained weight files from<https://drive.google.com/drive/folders/17-MAyCpMqyn4b_DFP2LekrmIgRovwoix?usp=share_link>.
```sh
cd CNNDetection
CUDA_VISIBLE_DEVICES=0 python eval_test8gan.py --model_path {Model-Path}  --dataroot {Grad-Test-Path} --batch_size {BS}
```

```bash
CUDA_VISIBLE_DEVICES=0 ./eval_test8gan.py --dataroot ./datasets/CNN_synth_testset --pth_dataroot ./checkpoints/xxx.pth
```

## Acknowledgments

This repository borrows partially from the [CNNDetection](https://github.com/peterwang512/CNNDetection), [stylegan](https://github.com/NVlabs/stylegan), and [genforce](https://github.com/genforce/genforce/).
