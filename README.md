# VisionTransformer-Comparison

Comparison of Vision Transformer, Swin Transformer, and DeiT models for Indonesian food classification.

## Overview

This project implements and compares three state-of-the-art vision transformer architectures for classifying Indonesian food dishes. The models are trained on a custom dataset of 1108 images across 5 food classes.

## Dataset

- **Classes**: Bakso, Gado-gado, Nasi Goreng, Rendang, Soto Ayam (5 classes)
- **Total Images**: 1108 images
- **Train/Validation Split**: 80/20 (886 train, 222 validation)
- **Location**: `dataset/` folder

## Models

### Vision Transformer (ViT)
- Base model: `vit_base_patch16_224`
- Training epochs: 10
- Best checkpoint: `results/ViT/weights/best_vit_epoch10.pth`

### Swin Transformer
- Model: `swin_base_patch4_window7_224`
- Training epochs: 3
- Best checkpoint: `results/Swin/weights/best_swin_epoch3.pth`

### DeiT
- Model: `deit_base_patch16_224`
- Training epochs: 5
- Best checkpoint: `results/DeiT/weights/best_deit_epoch5.pth`

## Project Structure

```
.
├── dataset/                    # Training dataset (1108 images)
├── test/                       # Test images (10 images)
├── results/
│   ├── ViT/
│   │   ├── weights/           # Best model checkpoint
│   │   ├── plots/             # Training visualizations
│   │   └── history_train.json # Training history
│   ├── Swin/
│   │   ├── weights/
│   │   ├── plots/
│   │   └── history_train.json
│   └── DeiT/
│       ├── weights/
│       ├── plots/
│       └── history_train.json
├── train_vit.py               # ViT training script
├── train_swin.py              # Swin training script
├── train_deit.py              # DeiT training script
├── test_models_clean.ipynb    # Model evaluation notebook
├── model.py                   # Model utilities
├── datareader.py              # Dataset loader
├── train_utils.py             # Training utilities
├── evaluation.py              # Evaluation functions
├── metrics_utils.py           # Metrics utilities
└── requirements.txt           # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kevinnaufaldany/VisionTransformer-Comparison.git
cd VisionTransformer-Comparison
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training Models

Train Vision Transformer:
```bash
python train_vit.py
```

Train Swin Transformer:
```bash
python train_swin.py
```

Train DeiT:
```bash
python train_deit.py
```

### Model Evaluation

Open and run the evaluation notebook:
```bash
jupyter notebook test_models_clean.ipynb
```

This will:
- Load all three trained models
- Run inference on test images
- Generate prediction CSV files
- Create visualization plots

## Results

Model predictions and evaluation metrics are saved in:
- `jawaban_testViT.csv` - ViT predictions
- `jawaban_testSwin.csv` - Swin predictions
- `jawaban_testDeiT.csv` - DeiT predictions

Visualization plots for each model:
- `results/{Model}/plots/01_training_curves.png` - Loss and accuracy curves
- `results/{Model}/plots/02_confusion_matrix.png` - Confusion matrix
- `results/{Model}/plots/03_per_class_metrics.png` - Per-class performance metrics
- `results/{Model}/plots/{Model}_predictions.png` - Sample predictions

## Hardware Requirements

- GPU: Minimum 4GB VRAM (tested on RTX 3050 Laptop)
- RAM: 8GB minimum
- Storage: 2GB for dataset and model weights

## Dependencies

See `requirements.txt` for complete list of dependencies:
- PyTorch 2.0.1
- torchvision 0.15.2
- timm 0.9.8
- numpy, pandas, matplotlib, seaborn
- scikit-learn, opencv-python, Pillow
- tqdm, psutil

## Full Documentation

For complete documentation and detailed analysis, please refer to the full documentation:
[Full Documentation](https://drive.google.com/file/d/1K7PWaS8ZgM7H0eOUyXtVyIWdlmIT1HHY/view?usp=sharing)

## Author

Kevin Naufaldany

## License

MIT
