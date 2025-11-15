# Git & Git LFS Setup Documentation

## Overview
This repository uses **Git LFS (Large File Storage)** to efficiently manage large binary files like model checkpoints, images, and CSV files.

## What is Git LFS?
Git LFS replaces large files with small pointer files in the Git repository, while the actual large files are stored on a separate LFS server. This keeps the repository size manageable and improves clone/push/pull performance.

## Files Tracked with Git LFS

### Model Checkpoints (*.pth) - ~1.0 GB
```
results/ViT/weights/best_vit_epoch10.pth
results/Swin/weights/best_swin_epoch3.pth
results/DeiT/weights/best_deit_epoch5.pth
```

### Predictions & Visualizations
```
jawaban_testViT.csv
jawaban_testSwin.csv
jawaban_testDeiT.csv
class_distribution.png
confidence_distribution.png
selected_predictions_detail.png
dataset.csv
results/*/history_train.json
```

## Git LFS Configuration

### .gitattributes
```
*.pth filter=lfs diff=lfs merge=lfs -text
*.csv filter=lfs diff=lfs merge=lfs -text
*.png filter=lfs diff=lfs merge=lfs -text
*.json filter=lfs diff=lfs merge=lfs -text
```

### LFS Tracked Files Summary
- **13 LFS objects** tracked
- **~1.0 GB** total size
- **100%** of large files properly tracked

## Installation & Setup

### Prerequisites
```bash
# Install Git LFS
# Windows: https://github.com/git-lfs/git-lfs/releases
# macOS: brew install git-lfs
# Linux: sudo apt install git-lfs
```

### Clone Repository (with LFS)
```bash
# Method 1: Clone with LFS (recommended)
git clone git@github.com:kevinnaufaldany/VisionTransformer-Comparison.git
cd VisionTransformer-Comparison
git lfs install

# Method 2: Clone without LFS (files as pointers)
git clone --no-checkout git@github.com:kevinnaufaldany/VisionTransformer-Comparison.git
cd VisionTransformer-Comparison
git lfs install
git checkout
```

## Repository Statistics

```
Total Repository Size with LFS:
- Source Files (Python, Notebooks, Markdown): ~63 KB
- LFS Objects (Models, Images, CSV, JSON): ~1.0 GB
- Total: ~1.0 GB
```

## Common Git LFS Commands

### Check LFS Status
```bash
git lfs ls-files                    # List all LFS tracked files
git lfs migrate info --everything   # Show detailed LFS statistics
du -sh .git/lfs/objects            # Show LFS cache size
```

### Add New LFS Tracked Files
```bash
git lfs track "*.pth"              # Track pattern
git add .gitattributes
git add your_file.pth
git commit -m "Add model checkpoint"
```

### Verify LFS Objects
```bash
git lfs fsck                        # Check LFS object integrity
git lfs prune                       # Remove unreferenced LFS objects
```

## Repository Workflow

### Initial Setup (Already Done)
```bash
git init
git lfs install
git lfs track "*.pth" "*.csv" "*.png" "*.json"
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin git@github.com:kevinnaufaldany/VisionTransformer-Comparison.git
```

### For Future Commits
```bash
# When adding model checkpoints
git lfs track "path/to/*.pth"
git add .gitattributes
git add path/to/model.pth
git commit -m "Add new model checkpoint"
git push origin main
```

### Push to GitHub
```bash
# Ensure LFS pointer files are pushed
git push -u origin main

# Check push was successful
git lfs ls-files
```

## Troubleshooting

### Files Showing as Text Instead of LFS
```bash
# Check current LFS tracking
cat .gitattributes

# Re-track and migrate
git lfs migrate import --include="*.pth"
git push origin --force
```

### Large Clones/Pulls
```bash
# Clone with shallow history (faster)
git clone --depth 1 git@github.com:kevinnaufaldany/VisionTransformer-Comparison.git

# Pull LFS objects only when needed
git lfs pull origin main --include="*.pth"
```

### Bandwidth Optimization
```bash
# LFS bandwidth monitoring
git lfs env  # Show LFS configuration

# Fetch only specific files
git lfs pull origin main --include="results/ViT/weights"
```

## Files NOT Tracked in Git

### .gitignore Excludes
```
__pycache__/          # Python cache
*.pyc                 # Compiled Python
venv/                 # Virtual environments
.vscode/              # IDE settings
dataset/              # Original dataset (too large)
test/                 # Test images (for local use)
results/*/plots/      # Temporary plots
```

**Reason**: Keep repository clean and fast. These can be regenerated.

## Best Practices

✅ **DO**
- Use LFS for files > 100 MB
- Commit `.gitattributes` before large files
- Keep model checkpoints in `results/*/weights/`
- Use meaningful commit messages

❌ **DON'T**
- Manually edit `.gitattributes`
- Mix Git and non-Git LFS operations
- Commit dataset folder (too large, >100 MB)
- Store temporary outputs

## Performance Metrics

| Aspect | Value |
|--------|-------|
| Repository Clone Time | ~2-5 min (depends on internet) |
| LFS Overhead | ~5% (pointer files) |
| Storage Saved | ~99.5% (vs storing all files locally) |
| Average File Size | Model: 350 MB, Images: 100 KB |

---

**Status**: ✅ Git LFS configured and ready for production  
**Repository**: VisionTransformer-Comparison  
**Branch**: main  
**Date**: November 16, 2025
