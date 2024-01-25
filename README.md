Now, we are open-sourcing the inference code, and the trained model weight files are located in the "checkpoints" folder. The inference command is provided below (please adjust the paths for weight files and data according to individual preferences):
```bash
CUDA_VISIBLE_DEVICES=0 ./eval_test8gan.py --dataroot ./datasets/CNN_synth_testset --pth_dataroot ./checkpoints/xxx.pth
```
