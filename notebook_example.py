#!/usr/bin/env python3


import os
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import Dataset, ConcatDataset, DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from skopt import gp_minimize
from skopt.callbacks import CheckpointSaver # Improved saving
import time # To track time
import wandb

import kagglehub
from tqdm import tqdm

# ==========================================
# 1. Data Preprocessing & Loading
# ==========================================
# Define transforms required for pre-trained EfficientNet-B0
# We normalize using ImageNet statistics because we use pre-trained weights
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if os.path.exists("./dataset/Training"):
    dataset_path = "./dataset"
else:
    dataset_path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset", output_dir="./dataset")

training_dataset = ImageFolder(os.path.join(dataset_path, "Training"), transform=transform)
testing_dataset = ImageFolder(os.path.join(dataset_path, "Testing"), transform=transform)
dataset = ConcatDataset([training_dataset, testing_dataset])

# ==========================================
# 2. Model Definition
# ==========================================
class BrainTumorModel(nn.Module):
    def __init__(self, num_classes=4, dropout=.1):
        super(BrainTumorModel, self).__init__()
        # Load pre-trained EfficientNet-B0
        self.base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        # Remove the original classifier
        self.base_model.classifier = nn.Identity()
        
        # Add a custom classifier with variable dropout for optimization
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(1280, 128),  # EfficientNet-B0 outputs 1280 features
            nn.ReLU(),
            nn.Dropout(0.25),      # Fixed dropout for stability
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.base_model(x)
        x = self.classifier(x)
        return x

criterion = nn.CrossEntropyLoss()

# ==========================================
# 3. Training Params & Hardware Tuning
# ==========================================
CALLS = 10     # Total BO trials
EPOCHS = 3     # Epochs per trial
BATCH_SIZE = 32 # Optimized for stability with high worker count on 32GB RAM

# Global counter for customization
current_call = 0

def train_model(params):
    """
    Objective function for Bayesian Optimization.
    Trains the model with specific hyperparameters and returns validation loss.
    """
    global current_call
    current_call += 1
    
    dropout = params[0]
    
    # Initialize WandB for this specific trial
    run = wandb.init(
        entity="2121jmmn-danmarks-tekniske-universitet-dtu",
        project="brain-tumor-bo-optimization",
        name=f"trial_{current_call}",
        config={
            "dropout": dropout,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "optimizer": "Adamax",
            "k_fold": current_call 
        }
    )

    print(f"\n-------------------- Round {current_call}/{CALLS} ----------------------------")
    print(f"Testing with dropout: {dropout:.4f}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Optional: Print device once to be sure
    print(f"Using device: {device}") 

    model = BrainTumorModel(num_classes=4, dropout=dropout).to(device)
    optimizer = optim.Adamax(model.parameters(), lr=0.001)

    # Split dataset for this trial
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    # Fixed seed ensures identical train/val split every trial
    # This isolates dropout as the only variable, making BO's effect clearly visible
    generator = torch.Generator().manual_seed(SEED)
    train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=generator)

    # OPTIMIZATION FOR RYZEN 7 PRO 7840U (CPU Mode)
    # Your CPU has 16 threads. Leaving 2-4 for Windows/Chrome is safe.
    # Setting workers to 8-12 allows data preparation to happen in parallel.
    workers = 14 
    
    # persistent_workers=True keeps the RAM allocated between epochs, speeding up training
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, 
                            num_workers=workers, persistent_workers=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, 
                            num_workers=workers, persistent_workers=True)

    # ------------------------------------------------------------------------------------------------------

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        # Progress bar for the batches in the current epoch
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # Use tqdm's own elapsed + remaining for a more accurate epoch total estimate
            elapsed = pbar.format_dict.get('elapsed', 0)
            remaining = (pbar.format_dict.get('total', 1) - pbar.format_dict.get('n', 0)) * pbar.format_dict.get('elapsed', 0) / max(pbar.format_dict.get('n', 1), 1)
            epoch_total = elapsed + remaining
            et_min, et_sec = divmod(int(epoch_total), 60)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'epoch_est': f'{et_min:02d}:{et_sec:02d}'
            })

        # Calculate average epoch loss
        avg_train_loss = running_loss / len(train_loader)
        
        # Log epoch metrics to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
        })

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs) # Get raw logits
            loss = criterion(outputs, labels) # Calculate loss
            val_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    avg_val_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total
    
    # Log final results for this trial
    wandb.log({
        "val_loss": avg_val_loss,
        "val_accuracy": accuracy
    })
    
    print(f"Trial {current_call} finished. Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    # Finish the WandB run so the next trial starts fresh
    run.finish()
    
    return avg_val_loss # Return metric to minimize

SEED = 123

if __name__ == '__main__':
    # Checkpoint saver: saves the optimization state after every call
    # This allows resuming if the script crashes
    checkpoint_callback = CheckpointSaver("./result_checkpoint.pkl")
    
    print(f"Starting optimization with {CALLS} calls...")
    start_time = time.time()

    # Run Bayesian Optimization
    res = gp_minimize(train_model,
                [(0.0, 0.5)],       # Search space for dropout
                acq_func = "EI",    # Expected Improvement
                n_calls=CALLS,
                n_random_starts=2,
                noise="gaussian",
                random_state=SEED,
                callback=[checkpoint_callback])

    end_time = time.time()
    print(f"Optimization finished in {(end_time - start_time)/60:.2f} minutes.")
    print(f"Best Dropout: {res.x[0]}, Best Loss: {res.fun}")