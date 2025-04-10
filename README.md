# EgoSpeak: Learning When to Speak for Egocentric Conversational Agents in the Wild

This is the official codebase for our paper [EgoSpeak: Learning When to Speak for Egocentric Conversational Agents in the Wild](https://arxiv.org/abs/2502.14892), accepted to NAACL 2025 Findings.

## Overview

EgoSpeak is a framework for predicting speech initiation in egocentric perspective.  
This repository contains two main components:

1. **LSTR (Long-Short Term Transformer)**: A transformer-based model
2. **MiniROAD (Minimal RNN Framework for Online Action Detection)**: A lightweight RNN-based alternative

## Environment Setup

```bash
# Create a conda/micromamba environment
micromamba create -n egospeak python=3.10 -y
pip install -r requirements.txt

# Install LSTR package
cd LSTR
pip install -e .
```

## Datasets

This work uses multiple datasets for turn-taking prediction in conversations:

```bash
# Create dataset directory
mkdir dataset
cd dataset

# Download datasets
wget https://huggingface.co/datasets/kjunh/EgoSpeak/resolve/main/EasyCom.tar.gz
wget https://huggingface.co/datasets/kjunh/EgoSpeak/resolve/main/Ego4D.tar.gz

# Set up symlinks
cd ../LSTR/data
ln -s ../../dataset/EasyCom EasyCom
ln -s ../../dataset/Ego4D Ego4D
```

Alternatively, download datasets via Hugging Face:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download kjunh/EgoSpeak EasyCom.tar.gz Ego4D.tar.gz --repo-type dataset
```

If you want to download the YTConv dataset:

```bash
huggingface-cli download kjunh/EgoSpeak YTConv.tar.gz --repo-type dataset
```

## Dataset Structure

Each dataset follows a similar structure:
```
EasyCom/
├── audio_wav2vec2/ # Audio features extracted with wav2vec2
├── rgb_kinetics_resnet50/ # Visual features extracted with ResNet50
└── target_perframe/ # Frame-level annotations
```


## Pre-trained Models

We provide pre-trained checkpoints for both models:

### LSTR Checkpoints

```bash
cd LSTR
mkdir checkpoints
cd checkpoints
wget https://huggingface.co/kjunh/EgoSpeak/resolve/main/lstr_easycom_AV.pth
```

### MiniROAD & Mamba Checkpoints

```bash
cd MiniROAD
mkdir checkpoints
cd checkpoints
wget https://huggingface.co/kjunh/EgoSpeak/resolve/main/miniroad_ego4dshuffle_AV.pth
wget https://huggingface.co/kjunh/EgoSpeak/resolve/main/mamba_easycom_AV.pth
```

## Usage

### 1. LSTR

Training:
```bash
bash run_LSTR_train.sh
```

Inference:
```bash
bash run_LSTR_inference.sh
```

To customize modalities:
```bash
bash run_LSTR_train.sh configs/EasyCom/LSTR/lstr_aa_easycom.yaml INPUT.MODALITY "visual+motion+object"
```

### 2. MiniROAD

Training:
```bash
bash run_miniroad_train.sh
```

Inference:
```bash
bash run_miniroad_inference_cfg.sh
```

To customize modalities or enable wandb logging:
```bash
bash run_miniroad_train.sh configs/miniroad_aa_easycom.yaml --no_flow --no_rgb --wandb
```

## Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@article{kim2025egospeak,
  title={EgoSpeak: Learning When to Speak for Egocentric Conversational Agents in the Wild},
  author={Kim, Junhyeok and Kim, Min Soo and Chung, Jiwan and Cho, Jungbin and Kim, Jisoo and Kim, Sungwoong and Sim, Gyeongbo and Yu, Youngjae},
  journal={arXiv preprint arXiv:2502.14892},
  year={2025}
}
```

## Acknowledgements

This codebase builds upon:
- [LSTR](https://github.com/amazon-research/long-short-term-transformer)
- [MiniROAD](https://github.com/joungbin/MiniROAD)

We thank the authors of these repositories for making their code available.

## License

This project is licensed under the Apache-2.0 License.