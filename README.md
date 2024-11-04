# SAGNet: Decoupling Semantic-Agnostic Artifacts from Limited Training Data for Robust Generalization in Deepfake Detection

**Classification environment:** 
We recommend installing the required packages by running the command:
```sh
pip install -r requirements.txt
```

## Getting the data
Download dataset from [CNNDetection](https://github.com/peterwang512/CNNDetection).

## Training the model
Download all pretrained weight files from<https://drive.google.com/drive/folders/17-MAyCpMqyn4b_DFP2LekrmIgRovwoix?usp=share_link>.
```sh
cd CNNDetection
sh ./train5_2_domains_ours.sh
```

## Testing the detector
```sh
cd CNNDetection
sh ./test.sh
```

## Acknowledgments

This repository borrows partially from the [CNNDetection](https://github.com/peterwang512/CNNDetection), [stylegan](https://github.com/NVlabs/stylegan), and [genforce](https://github.com/genforce/genforce/).
