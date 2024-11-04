# SAGNet: Decoupling Semantic-Agnostic Artifacts from Limited Training Data for Robust Generalization in Deepfake Detection

## Environment setup
**Img2grad environment:** 
We suggest transforming the image into a gradient using the tensorflow environment in docker image `nvcr.io/nvidia/tensorflow:21.02-tf1-py3` from [nvidia](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow/tags).

```bash
CUDA_VISIBLE_DEVICES=0 ./eval_test8gan.py --dataroot ./datasets/CNN_synth_testset --pth_dataroot ./checkpoints/xxx.pth
```