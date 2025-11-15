# 🚀 Quick Start Guide

## Project Overview
Indonesian Food Classification using Vision Transformers (ViT, Swin, DeiT)

## Setup Instructions

### 1. Clone Repository
```bash
git clone git@github.com:kevinnaufaldany/VisionTransformer-Comparison.git
cd VisionTransformer-Comparison
```

**Note**: Requires Git LFS to be installed. Install from: https://github.com/git-lfs/git-lfs

### 2. Install Python Dependencies
```bash
conda create -n transformer python=3.10
conda activate transformer
pip install torch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
pip install timm torchinfo pandas scikit-learn matplotlib seaborn tqdm
```

### 3. Dataset Setup
Place your dataset in the `dataset/` folder with a `dataset.csv` file:
```csv
image_path,label
image1.jpg,bakso
image2.jpg,gado_gado
...
```

### 4. Run Training

#### Train ViT
```bash
python train_vit.py
```

#### Train Swin Transformer
```bash
python train_swin.py
```

#### Train DeiT
```bash
python train_deit.py
```

### 5. Test Models
Open and run the notebook:
```bash
jupyter notebook test_models_clean.ipynb
```

This will generate:
- `jawaban_testViT.csv`
- `jawaban_testSwin.csv`
- `jawaban_testDeiT.csv`
- Visualization plots

## Project Structure
```
.
├── README.md                      # Project overview
├── GIT_LFS_SETUP.md              # Git LFS documentation
├── TESTING_RESULTS_SUMMARY.md    # Test results
├── train_vit.py                  # ViT training
├── train_swin.py                 # Swin training
├── train_deit.py                 # DeiT training
├── model.py                      # Model utilities
├── datareader.py                 # Dataset loader
├── utils.py                      # Training utilities
├── test_models_clean.ipynb       # Testing notebook
├── results/
│   ├── ViT/
│   │   ├── weights/best_vit_epoch10.pth
│   │   ├── plots/
│   │   └── history_train.json
│   ├── Swin/
│   │   ├── weights/best_swin_epoch3.pth
│   │   ├── plots/
│   │   └── history_train.json
│   └── DeiT/
│       ├── weights/best_deit_epoch5.pth
│       ├── plots/
│       └── history_train.json
├── dataset/                      # Training images
├── test/                         # Test images
├── jawaban_testViT.csv          # ViT predictions
├── jawaban_testSwin.csv         # Swin predictions
├── jawaban_testDeiT.csv         # DeiT predictions
└── .gitattributes               # Git LFS tracking
```

## Key Results

| Model | Avg Confidence | Performance |
|-------|---|---|
| **ViT** | 82.79% | Good |
| **Swin** | 99.27% | **Excellent** ⭐ |
| **DeiT** | 97.74% | Excellent |

## Common Commands

### Training
```bash
# Train with progress bar and GPU
python train_vit.py    # ~5-10 min on RTX 3050

# Check GPU usage
nvidia-smi
```

### Testing
```bash
# Run test notebook
jupyter notebook test_models_clean.ipynb

# Or convert to Python script
jupyter nbconvert --to script test_models_clean.ipynb
python test_models_clean.py
```

### Git Operations
```bash
# Check status
git status

# Push changes (with LFS)
git add .
git commit -m "Update training results"
git push origin main

# Verify LFS files
git lfs ls-files
```

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size in training scripts
batch_size=32  # From 64
```

### Missing Dataset
```bash
# Dataset should be placed in dataset/ folder
# Download and extract your images
# Create dataset.csv with image paths and labels
```

### Git LFS Not Working
```bash
# Install Git LFS
git lfs install

# Re-track files
git lfs track "*.pth" "*.csv" "*.png"
```

## Resources
- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)
- [Swin Transformer Paper](https://arxiv.org/abs/2103.14030)
- [DeiT Paper](https://arxiv.org/abs/2012.12556)
- [TIMM Library](https://github.com/rwightman/pytorch-image-models)

## Author
Kevin Naufal Dany

## License
MIT

---
**Last Updated**: November 16, 2025  
**Status**: ✅ Ready for deployment
