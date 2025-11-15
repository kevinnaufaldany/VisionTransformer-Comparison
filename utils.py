# utils.py - Combined utilities for training, metrics, and evaluation
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import time
import os
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import seaborn as sns
import numpy as np
from tqdm import tqdm
import warnings
import json
import psutil
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)

def check_set_gpu(override=None):
    if override == None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            print(f"Using MPS: {torch.backends.mps.is_available()}")
        else:
            device = torch.device('cpu')
            print(f"Using CPU: {torch.device('cpu')}")
    else:
        device = torch.device(override)
    return device


class ResultsLogger:
    """Comprehensive results logger untuk tracking semua metrics training"""
    
    def __init__(self, model_name, num_classes, save_dir='results'):
        self.model_name = model_name
        self.num_classes = num_classes
        self.save_dir = save_dir
        self.experiment_dir = os.path.join(save_dir, f"{model_name}")
        
        # Create directories
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, 'weights'), exist_ok=True)
        
        # Initialize results dictionary
        self.results = {
            'model_name': model_name,
            'num_classes': num_classes,
            'parameters': {},
            'performance_metrics': {},
            'inference_metrics': {},
            'hardware': {},
            'best_model_path': None
        }
        # Training history per epoch
        self.history = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
        self.best_val_acc = 0
        self.best_epoch = 0
    
    def save_model_checkpoint(self, model, epoch, val_acc):
        """Save model checkpoint jika accuracy meningkat - hanya simpan 1 best model"""
        if val_acc > self.best_val_acc:
            # Delete old best model jika ada
            if self.results['best_model_path'] and os.path.exists(self.results['best_model_path']):
                try:
                    os.remove(self.results['best_model_path'])
                except:
                    pass
            
            self.best_val_acc = val_acc
            self.best_epoch = epoch
            
            model_path = os.path.join(
                self.experiment_dir, 'weights', 
                f"best_{self.model_name.lower()}_epoch{epoch+1}.pth"
            )
            torch.save(model.state_dict(), model_path)
            self.results['best_model_path'] = model_path
            print(f"✓ Best model saved: {model_path} (Acc: {val_acc:.2f}%)")
            return model_path
        return None
    
    def log_parameters(self, model, trainable_params, total_params, non_trainable_params):
        """Log model parameters"""
        model_size_mb = sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024)
        
        self.results['parameters'] = {
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params),
            'non_trainable_parameters': int(non_trainable_params),
            'model_size_mb': round(model_size_mb, 2)
        }
        
        print("\n" + "="*60)
        print("PARAMETER INFORMATION")
        print("="*60)
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Non-Trainable Parameters: {non_trainable_params:,}")
        print(f"Model Size: {model_size_mb:.2f} MB")
        print("="*60 + "\n")
    
    def log_training_metrics(self, train_losses, val_losses, val_accs):
        """Log training metrics dan save plots"""
        # Populate training history
        self.history['epochs'] = list(range(1, len(train_losses) + 1))
        self.history['train_loss'] = [float(x) for x in train_losses]
        self.history['val_loss'] = [float(x) for x in val_losses]
        self.history['val_accuracy'] = [float(x) for x in val_accs]
        
        # Plot training curves
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        epochs = list(range(1, len(train_losses) + 1))
        axes[0].plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, val_losses, 'r-s', label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[1].plot(epochs, val_accs, 'g-^', linewidth=2, markersize=8)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].set_title('Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 105])
        
        plt.tight_layout()
        plot_path = os.path.join(self.experiment_dir, 'plots', '01_training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to {plot_path}")
    
    def log_performance_metrics(self, y_true, y_pred, class_names=None):
        """Log performance metrics"""
        # Accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        # Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro'
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )
        
        self.results['performance_metrics'] = {
            'overall_accuracy': round(float(accuracy), 4),
            'macro_precision': round(float(precision), 4),
            'macro_recall': round(float(recall), 4),
            'macro_f1': round(float(f1), 4),
            'per_class_metrics': {}
        }
        
        # Per-class metrics
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(self.num_classes)]
        
        for i, class_name in enumerate(class_names):
            self.results['performance_metrics']['per_class_metrics'][class_name] = {
                'precision': round(float(precision_per_class[i]), 4),
                'recall': round(float(recall_per_class[i]), 4),
                'f1_score': round(float(f1_per_class[i]), 4),
                'support': int(support[i])
            }
        
        # Print metrics
        print("\n" + "="*60)
        print("PERFORMANCE METRICS")
        print("="*60)
        print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"\nMacro-averaged Metrics:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"\nPer-Class Metrics:")
        print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 66)
        for i, class_name in enumerate(class_names):
            print(f"{class_name:<20} {precision_per_class[i]:<12.4f} {recall_per_class[i]:<12.4f} {f1_per_class[i]:<12.4f} {support[i]:<10}")
        print("="*60 + "\n")
        
        # Plot confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Count'}, ax=ax)
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        cm_path = os.path.join(self.experiment_dir, 'plots', '02_confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to {cm_path}")
        
        # Plot per-class metrics
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        x_pos = np.arange(len(class_names))
        
        # Precision
        axes[0].bar(x_pos, precision_per_class, color='skyblue', edgecolor='navy', alpha=0.7)
        axes[0].set_xlabel('Class', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Precision', fontsize=11, fontweight='bold')
        axes[0].set_title('Precision per Class', fontsize=12, fontweight='bold')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(class_names, rotation=45, ha='right')
        axes[0].set_ylim([0, 1.1])
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Recall
        axes[1].bar(x_pos, recall_per_class, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
        axes[1].set_xlabel('Class', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Recall', fontsize=11, fontweight='bold')
        axes[1].set_title('Recall per Class', fontsize=12, fontweight='bold')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(class_names, rotation=45, ha='right')
        axes[1].set_ylim([0, 1.1])
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # F1-Score
        axes[2].bar(x_pos, f1_per_class, color='salmon', edgecolor='darkred', alpha=0.7)
        axes[2].set_xlabel('Class', fontsize=11, fontweight='bold')
        axes[2].set_ylabel('F1-Score', fontsize=11, fontweight='bold')
        axes[2].set_title('F1-Score per Class', fontsize=12, fontweight='bold')
        axes[2].set_xticks(x_pos)
        axes[2].set_xticklabels(class_names, rotation=45, ha='right')
        axes[2].set_ylim([0, 1.1])
        axes[2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        metrics_path = os.path.join(self.experiment_dir, 'plots', '03_per_class_metrics.png')
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Per-class metrics plot saved to {metrics_path}")
    
    def log_inference_metrics(self, avg_time_ms, throughput, device_name):
        """Log inference metrics"""
        self.results['inference_metrics'] = {
            'avg_time_per_image_ms': round(avg_time_ms, 2),
            'throughput_images_per_sec': round(throughput, 2),
            'device': device_name
        }
        
        print("\n" + "="*60)
        print("INFERENCE METRICS")
        print("="*60)
        print(f"Average Time per Image: {avg_time_ms:.2f} ms")
        print(f"Throughput: {throughput:.2f} images/second")
        print(f"Device: {device_name}")
        print("="*60 + "\n")
    
    def log_hardware(self):
        """Log hardware information"""
        try:
            device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
        except:
            device_name = "CPU"
            gpu_memory = 0
        
        cpu_count = psutil.cpu_count()
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        
        self.results['hardware'] = {
            'device': device_name,
            'gpu_memory_gb': round(gpu_memory, 2),
            'cpu_cores': cpu_count,
            'ram_gb': round(total_memory_gb, 2),
            'cuda_available': torch.cuda.is_available()
        }
        
        print("\n" + "="*60)
        print("HARDWARE INFORMATION")
        print("="*60)
        print(f"Device: {device_name}")
        if gpu_memory > 0:
            print(f"GPU Memory: {gpu_memory:.2f} GB")
        print(f"CPU Cores: {cpu_count}")
        print(f"RAM: {total_memory_gb:.2f} GB")
        print("="*60 + "\n")
    
    def save_history(self):
        """Save training history to JSON file"""
        history_path = os.path.join(self.experiment_dir, 'history_train.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\nTraining history saved to {history_path}")
        print(f"Experiment directory: {self.experiment_dir}")
    
    
    def print_summary(self):
        """Print final summary"""
        print("\n" + "="*80)
        print("EXPERIMENT COMPLETE")
        print("="*80)
        print(f"Model: {self.results['model_name']}")
        print(f"Experiment directory: {self.experiment_dir}")
        if self.history['epochs']:
            print(f"Total Epochs Trained: {len(self.history['epochs'])}")
            print(f"Best Validation Accuracy: {self.best_val_acc:.4f}")
            print(f"Best Epoch: {self.best_epoch}")
        if self.results['best_model_path']:
            print(f"Best Model: {self.results['best_model_path']}")
        print("="*80 + "\n")


def train_model(model, trainloader, testloader, device, model_name="Model",
                epochs=10, lr=3e-4, weight_decay=1e-4, logger=None):
    """Train model dengan optional logger untuk save best weights"""

    # SPEED UP GPU
    torch.backends.cudnn.benchmark = True

    # ------- MODEL KE GPU WAJIB DULU ------
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    # ------- MIXED PRECISION (WAJIB untuk VITransformer) -------
    scaler = torch.amp.GradScaler('cuda')

    train_losses = []
    val_losses = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

        for batch in pbar:
            # Handle both 2 values (CIFAR10) dan 3 values (custom dataset)
            if len(batch) == 3:
                x, y, _ = batch  # Ignore filepath
            else:
                x, y = batch

            # harus pindah ke GPU
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()

            # --------------- AMP AUTocast ----------------
            with torch.amp.autocast('cuda'):
                out = model(x)
                loss = criterion(out, y)

            # SCALE loss
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': train_loss / (pbar.n + 1)})

        train_loss /= len(trainloader)
        train_losses.append(train_loss)

        # --------------- VALIDATION ----------------
        model.eval()
        correct = 0
        val_loss = 0

        pbar_val = tqdm(testloader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
        with torch.no_grad():
            for batch in pbar_val:
                # Handle both 2 values (CIFAR10) dan 3 values (custom dataset)
                if len(batch) == 3:
                    x, y, _ = batch  # Ignore filepath
                else:
                    x, y = batch
                    
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with torch.amp.autocast('cuda'):
                    out = model(x)
                    loss = criterion(out, y)

                val_loss += loss.item()
                _, pred = torch.max(out, 1)
                correct += (pred == y).sum().item()

                pbar_val.set_postfix({'val_loss': val_loss / (pbar_val.n + 1)})

        val_loss /= len(testloader)
        val_losses.append(val_loss)
        acc = correct / len(testloader.dataset) * 100
        val_accs.append(acc)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} "
              f"| Val Loss: {val_loss:.4f} | Acc: {acc:.2f}%")
        
        # Save best model checkpoint jika ada logger
        if logger is not None:
            logger.save_model_checkpoint(model, epoch, acc)

    return train_losses, val_losses, val_accs, model

# ============ Metrics Functions ============
@torch.no_grad()
def evaluate_metrics(model, dataloader, device):
    model.eval()
    preds = []
    trues = []

    pbar = tqdm(dataloader, desc="Evaluating Metrics")
    for batch in pbar:
        # Handle both 2 values (CIFAR10) dan 3 values (custom dataset)
        if len(batch) == 3:
            x, y, _ = batch  # Ignore filepath
        else:
            x, y = batch
            
        x, y = x.to(device), y.to(device)
        out = model(x)
        _, pred = torch.max(out, 1)

        preds.extend(pred.cpu().numpy())
        trues.extend(y.cpu().numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(
        trues, preds, average="macro"
    )

    return precision, recall, f1, trues, preds


def plot_confusion(trues, preds, classes=100):
    cm = confusion_matrix(trues, preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig('confusion_matrix.png')
    plt.close()
    print("Confusion matrix saved as 'confusion_matrix.png'")


# ============ Evaluation Functions ============
@torch.no_grad()
def measure_inference_time(model, dataloader, device):
    """
    Measure inference time dengan best practices:
    - Warm-up terlebih dahulu
    - Minimal 100 gambar untuk akurasi statistik
    - Hitung rata-rata dan standar deviasi
    - GPU sync untuk timing akurat
    - Mode evaluasi dan no_grad
    """
    model.eval()
    
    # Warm-up dengan dummy input
    print("Warming up model...")
    dummy = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        for _ in tqdm(range(10), desc="Warmup"):
            model(dummy)
    
    # Synchronize GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    total_time = 0
    total_images = 0
    batch_times = []
    
    print("Measuring inference time (minimal 100 images)...")
    pbar = tqdm(dataloader, desc="Measuring Inference Time")
    
    with torch.no_grad():
        for batch in pbar:
            # Handle both 2 values (CIFAR10) dan 3 values (custom dataset)
            if len(batch) == 3:
                x, _, _ = batch  # Ignore label dan filepath
            else:
                x, _ = batch
            
            x = x.to(device)
            batch_size = x.size(0)
            
            # Synchronize sebelum timing untuk GPU
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start = time.time()
            model(x)
            
            # Synchronize setelah timing untuk GPU
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end = time.time()
            
            batch_time = (end - start)
            batch_times.append(batch_time)
            total_time += batch_time
            total_images += batch_size
            
            pbar.set_postfix({
                'Images': total_images,
                'Avg ms/img': f"{(total_time / total_images) * 1000:.3f}"
            })
            
            # Stop jika sudah cukup untuk statistik akurat
            if total_images >= 100:
                if total_images > 222:  # Limit untuk dataset kecil
                    break
    
    # Calculate statistics
    avg_ms = (total_time / total_images) * 1000
    throughput = total_images / total_time
    
    # Standard deviation
    batch_times_per_img = [t / (x.size(0) if torch.cuda.is_available() else 1) * 1000 
                           for t in batch_times]
    std_ms = np.std(batch_times_per_img) if batch_times_per_img else 0
    
    print(f"\n{'='*60}")
    print(f"Inference Time Measurement (n={total_images} images):")
    print(f"  Average: {avg_ms:.3f} ms/image")
    print(f"  Std Dev: {std_ms:.3f} ms/image")
    print(f"  Throughput: {throughput:.2f} images/sec")
    print(f"{'='*60}\n")
    
    return avg_ms, throughput
