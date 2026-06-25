# CT-UMamba

## 👀 Introduction

**DFMamba** is a novel semantic segmentation network for high-resolution remote sensing images, built upon and improved from the UNetMamba architecture. It introduces enhanced feature fusion mechanisms and attention modules for better segmentation performance.

## 📂 Folder Structure

Prepare the following folders to organize this repo:
```none
DFMamba/
├── configs                    # Configuration files for different datasets
│   ├── vaihingen/
│   └── potsdam/
├── dfmamba/                   # Main model package
│   ├── models/                # Model definitions
│   ├── datasets/              # Dataset loaders
│   ├── losses/                # Loss functions
│   ├── mamba_ssm/             # Mamba state-space model modules
│   └── kernels/               # Custom CUDA kernels
├── tools/                     # Utility scripts
├── pretrain_weights/          # Pretrained backbone weights
├── model_weights/             # Trained model checkpoints
├── data/                      # Dataset directory
│   ├── vaihingen/
│   └── potsdam/
├── train.py                   # Training script
├── vaihingen_test.py          # Vaihingen testing script
├── potsdam_test.py            # Potsdam testing script
└── requirements.txt           # Python dependencies
```

## 🛠 Installation

```bash
# Create conda environment
conda create -n dfmamba python=3.8
conda activate dfmamba

# Install dependencies
pip install -r requirements.txt
```

💁 **Tips**: If you're having difficulty installing `causal_conv1d` or `mamba_ssm`, please refer to:
- [causal_conv1d releases](https://github.com/Dao-AILab/causal-conv1d/releases)
- [mamba_ssm releases](https://github.com/state-spaces/mamba/releases)

Download the appropriate wheel files for your CUDA version and install them manually.

## 🧩 Pretrained Weights

Download pretrained backbone weights and place them in `pretrain_weights/`:
- `rest_lite.pth` - ResT-Lite backbone
- `convnext_tiny-983f1562.pth` - ConvNeXt-Tiny backbone

**Download Link (Quark Cloud):** https://pan.quark.cn/s/b43c5a7ce3b8  
**Extraction Code:** `HnCK`

## 💿 Data Preprocessing

Download the datasets from official sources and preprocess them.

### 1️⃣ Vaihingen
[Vaihingen Official](https://www.isprs.org/education/benchmarks/UrbanSemLab/Default.aspx)

```bash
# Generate training set
python tools/vaihingen_patch_split.py \
    --img-dir "data/vaihingen/train_images" --mask-dir "data/vaihingen/train_masks" \
    --output-img-dir "data/vaihingen/train_1024/images" --output-mask-dir "data/vaihingen/train_1024/masks" \
    --mode "train" --split-size 1024 --stride 1024

# Generate validation set
python tools/vaihingen_patch_split.py \
    --img-dir "data/vaihingen/val_images" --mask-dir "data/vaihingen/val_masks_eroded" \
    --output-img-dir "data/vaihingen/val_1024/images" --output-mask-dir "data/vaihingen/val_1024/masks" \
    --mode "val" --split-size 1024 --stride 1024 --eroded

# Generate test set
python tools/vaihingen_patch_split.py \
    --img-dir "data/vaihingen/test_images" --mask-dir "data/vaihingen/test_masks_eroded" \
    --output-img-dir "data/vaihingen/test_1024/images" --output-mask-dir "data/vaihingen/test_1024/masks" \
    --mode "val" --split-size 1024 --stride 1024 --eroded
```

### 2️⃣ Potsdam
[Potsdam Official](https://www.isprs.org/education/benchmarks/UrbanSemLab/Default.aspx)

```bash
python tools/potsdam_patch_split.py \
    --img-dir "data/potsdam/train_images" --mask-dir "data/potsdam/train_masks" \
    --output-img-dir "data/potsdam/train_1024/images" --output-mask-dir "data/potsdam/train_1024/masks" \
    --mode "train" --split-size 1024 --stride 1024
```

## 🏋 Training

Use different config files to train on different datasets:

```bash

# Train on Vaihingen
python train.py -c configs/vaihingen/dfmamba_config.py

# Train on Potsdam
python train.py -c configs/potsdam/dfmamba_config.py
```

## 🎯 Testing

Arguments:
- `-c`: Path to the config file
- `-o`: Output path for predictions
- `-t`: Test time augmentation (TTA), options: `None`, `'lr'` (flip), `'d4'` (multiscale)
- `--rgb`: Output masks in RGB format

### Vaihingen
```bash
python vaihingen_test.py -c configs/vaihingen/dfmamba_config.py -o fig_results/vaihingen/dfmamba_test
python vaihingen_test.py -c configs/vaihingen/dfmamba_config.py -o fig_results/vaihingen/dfmamba_test -t 'lr'
python vaihingen_test.py -c configs/vaihingen/dfmamba_config.py -o fig_results/vaihingen/dfmamba_rgb --rgb
```

### Potsdam
```bash
python potsdam_test.py -c configs/potsdam/dfmamba_config.py -o fig_results/potsdam/dfmamba_test
python potsdam_test.py -c configs/potsdam/dfmamba_config.py -o fig_results/potsdam/dfmamba_test -t 'lr'
```

## ❤ Acknowledgement

This project is built upon and inspired by:
- [UNetMamba](https://github.com/EnzeZhu2001/UNetMamba)
- [GeoSeg](https://github.com/WangLibo1995/GeoSeg)
- [mamba](https://github.com/state-spaces/mamba)
- [VMamba](https://github.com/MzeroMiko/VMamba)
- [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d)
- [LoveDA](https://github.com/Junjue-Wang/LoveDA)

## 📄 License

This project is released under the MIT License.
