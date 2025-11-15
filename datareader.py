import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import torch
import random
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, Normalize, Compose, Resize, RandomHorizontalFlip, RandomResizedCrop
from PIL import Image
from sklearn.model_selection import train_test_split
import warnings

# Suppress PIL warnings
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

# Set random seed for reproducibility
RANDOM_SEED = 2025
random.seed(RANDOM_SEED)  # random seed-nya Python
np.random.seed(RANDOM_SEED)  # random seed-nya Numpy
torch.manual_seed(RANDOM_SEED)  # random seed-nya PyTorch


class MakananIndoDataset(Dataset):
    """
    Custom Dataset untuk Indonesian Food Classification
    """
    # ImageNet normalization values
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    def __init__(self,
                 df,
                 img_dir='dataset',
                 img_size=(224, 224),
                 transform=None,
                 label2idx=None,
                 infer=False):
        
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.img_size = img_size
        self.transform = transform
        self.infer = infer
        
        if not infer:
            if label2idx is None:
                self.labels = sorted(self.df['label'].unique())
                self.label2idx = {label: idx for idx, label in enumerate(self.labels)}
            else:
                self.label2idx = label2idx
                self.labels = list(label2idx.keys())
        else:
            self.labels = None
            self.label2idx = None
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        filename = self.df.iloc[idx]['filename']
        
        # Handle filename
        if pd.isna(filename) or isinstance(filename, float):
            filename = f"unknown_{idx}.jpg"
        filename = str(filename).strip()
        
        img_path = os.path.join(self.img_dir, filename)
        
        # Load image
        try:
            image = Image.open(img_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', self.img_size, color=(128, 128, 128))
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        # Get label
        if not self.infer:
            label = self.df.iloc[idx]['label']
            label = self.label2idx[label]
            return image, label, img_path
        else:
            return image, img_path


def create_train_val_split(csv_file, img_dir='dataset', train_ratio=0.8, random_state=RANDOM_SEED):
    """
    Split the combined data into train and validation sets with stratified sampling
    """
    df = pd.read_csv(csv_file)
    
    # Create label2idx mapping
    labels = sorted(df['label'].unique())
    label2idx = {label: idx for idx, label in enumerate(labels)}
    num_classes = len(labels)
    
    print(f"Classes: {labels}")
    print(f"Number of classes: {num_classes}")
    
    # Stratified split
    train_df, val_df = train_test_split(
        df,
        test_size=1 - train_ratio,
        random_state=random_state,
        stratify=df['label']
    )
    
    return train_df, val_df, label2idx, num_classes


def get_food_dataset(batch_size=16, img_size=224, data_dir='dataset', csv_file='dataset.csv'):
    """
    Create train and validation dataloaders untuk custom food dataset
    """
    
    # Split data
    train_df, val_df, label2idx, num_classes = create_train_val_split(
        csv_file=csv_file,
        img_dir=data_dir,
        train_ratio=0.8,
        random_state=RANDOM_SEED
    )
    
    # Define transforms
    transform_train = Compose([
        Resize((img_size, img_size)),
        RandomHorizontalFlip(p=0.5),
        ToTensor(),
        Normalize(mean=MakananIndoDataset.IMAGENET_MEAN,
                  std=MakananIndoDataset.IMAGENET_STD),
    ])
    
    transform_val = Compose([
        Resize((img_size, img_size)),
        ToTensor(),
        Normalize(mean=MakananIndoDataset.IMAGENET_MEAN,
                  std=MakananIndoDataset.IMAGENET_STD),
    ])
    
    # Create datasets
    train_dataset = MakananIndoDataset(
        df=train_df,
        img_dir=data_dir,
        img_size=(img_size, img_size),
        transform=transform_train,
        label2idx=label2idx,
        infer=False
    )
    
    val_dataset = MakananIndoDataset(
        df=val_df,
        img_dir=data_dir,
        img_size=(img_size, img_size),
        transform=transform_val,
        label2idx=label2idx,
        infer=False
    )
    
    # Create dataloaders
    trainloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # WAJIB untuk Windows
        pin_memory=True
    )
    
    valloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"\nTrain data: {len(train_dataset)}")
    print(f"Val data: {len(val_dataset)}")
    print(f"Total: {len(train_dataset) + len(val_dataset)}")
    
    return trainloader, valloader, label2idx, num_classes


if __name__ == "__main__":
    # Test dataset loading
    print("=" * 60)
    print("Testing Food Dataset")
    print("=" * 60)
    
    # Get dataloaders
    trainloader, valloader, label2idx, num_classes = get_food_dataset(
        batch_size=4,
        img_size=224,
        data_dir='dataset',
        csv_file='dataset.csv'
    )
    
    print(f"\nLabel to Index mapping: {label2idx}")
    print(f"Number of classes: {num_classes}")
    
    # Get one batch for testing
    print("\n" + "=" * 60)
    print("Sample Train Batch")
    print("=" * 60)
    
    for images, labels, filepaths in trainloader:
        print(f"Images shape: {images.shape}")
        print(f"Labels: {labels}")
        for filepath in filepaths:
            print(f"File: {filepath}")
        
        # Visualize first 4 images from batch
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes = axes.flatten()
        
        for i in range(min(4, len(images))):
            img = images[i].clone()
            
            # Denormalize
            for j in range(3):
                img[j] = img[j] * MakananIndoDataset.IMAGENET_STD[j] + MakananIndoDataset.IMAGENET_MEAN[j]
            
            # Convert to displayable format
            img = img.permute(1, 2, 0)
            img = torch.clamp(img, 0, 1)
            
            # Get label name
            label_name = [k for k, v in label2idx.items() if v == labels[i].item()][0]
            
            axes[i].imshow(img)
            axes[i].set_title(f"Label: {label_name}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('sample_batch.png')
        print("\nSample batch visualization saved as 'sample_batch.png'")
        break
    
    # Get samples from both train and val
    print("\n" + "=" * 60)
    print("Detailed Dataset Samples")
    print("=" * 60)
    
    train_dataset = trainloader.dataset
    val_dataset = valloader.dataset
    
    # Sample 5 random images from each
    train_indices = random.sample(range(len(train_dataset)), min(5, len(train_dataset)))
    val_indices = random.sample(range(len(val_dataset)), min(5, len(val_dataset)))
    
    print("\nTrain Dataset Samples:")
    for i, idx in enumerate(train_indices):
        image, label, filepath = train_dataset[idx]
        label_name = [k for k, v in label2idx.items() if v == label][0]
        print(f"Train data ke-{i} (index {idx})")
        print(f"  Image shape: {image.shape}")
        print(f"  Label index: {label}, Label name: {label_name}")
        print(f"  File path: {filepath}")
        print("-" * 40)
    
    print("\nValidation Dataset Samples:")
    for i, idx in enumerate(val_indices):
        image, label, filepath = val_dataset[idx]
        label_name = [k for k, v in label2idx.items() if v == label][0]
        print(f"Val data ke-{i} (index {idx})")
        print(f"  Image shape: {image.shape}")
        print(f"  Label index: {label}, Label name: {label_name}")
        print(f"  File path: {filepath}")
        print("-" * 40)
    
    # Visualize samples from both train and val
    print("\nCreating visualization for train and val samples...")
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    # Plot train images
    for i, idx in enumerate(train_indices):
        image, label, filepath = train_dataset[idx]
        label_name = [k for k, v in label2idx.items() if v == label][0]
        
        # Denormalize
        img_display = image.clone()
        for j in range(3):
            img_display[j] = img_display[j] * MakananIndoDataset.IMAGENET_STD[j] + MakananIndoDataset.IMAGENET_MEAN[j]
        
        img_display = img_display.permute(1, 2, 0)
        img_display = torch.clamp(img_display, 0, 1)
        
        axes[0, i].imshow(img_display)
        axes[0, i].set_title(f"Train: {label_name}")
        axes[0, i].axis('off')
    
    # Plot val images
    for i, idx in enumerate(val_indices):
        image, label, filepath = val_dataset[idx]
        label_name = [k for k, v in label2idx.items() if v == label][0]
        
        # Denormalize
        img_display = image.clone()
        for j in range(3):
            img_display[j] = img_display[j] * MakananIndoDataset.IMAGENET_STD[j] + MakananIndoDataset.IMAGENET_MEAN[j]
        
        img_display = img_display.permute(1, 2, 0)
        img_display = torch.clamp(img_display, 0, 1)
        
        axes[1, i].imshow(img_display)
        axes[1, i].set_title(f"Val: {label_name}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('train_val_samples.png')
    print("Train and validation samples visualization saved as 'train_val_samples.png'")
