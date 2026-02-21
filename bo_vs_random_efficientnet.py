"""
BO vs Random Search (Dropout tuning) for Brain Tumor MRI dataset (Kaggle)
========================================================================

This script is designed to be GitHub-friendly for group work:
- No hardcoded personal paths
- No W&B entity / org references
- No data leakage: train/val split is made ONLY from Training/
- Testing/ is held out for a final evaluation

What it does
------------
1) Ensures the dataset exists locally in ./dataset:
   - Expects ./dataset/Training and ./dataset/Testing
   - If missing, tries to download via Kaggle CLI (kaggle.json required)
2) Builds EfficientNet-B0 (pretrained on ImageNet) + a small custom classifier head
3) Tunes ONE hyperparameter: dropout (range [0.0, 0.5])
4) Compares:
   - Random Search: N trials with random dropout
   - Bayesian Optimization: N trials with gp_minimize (Expected Improvement)
5) Uses a "fast demo" setting: per trial, sample TRAIN_SUBSAMPLE images from the training pool
   and train for EPOCHS epochs with BATCH_SIZE.
   - If TRAIN_SUBSAMPLE=2000 and BATCH_SIZE=100, you get exactly 20 batches per epoch (drop_last=True).
6) Saves plots in ./plots:
   - bo_vs_random_best_so_far.png  (the key report figure)
   - bo_evaluations_scatter.png
   - random_evaluations_scatter.png
   - dropout_over_time_bo.png

How to run
----------
From an activated environment (conda/venv):

    pip install torch torchvision scikit-optimize tqdm matplotlib kaggle

Optional (for W&B logging):
    pip install wandb

Dataset setup (one-time per machine):
- Put kaggle.json here:
    Windows: C:\\Users\\<you>\\.kaggle\\kaggle.json
    Mac/Linux: ~/.kaggle/kaggle.json
- Accept dataset rules on Kaggle if prompted.

Then run:
    python bo_vs_random_efficientnet.py

W&B (optional)
--------------
By default, W&B is disabled to avoid login prompts.
Enable it by setting:
    WANDB_MODE=online

Examples:
    PowerShell:
        $env:WANDB_MODE="online"
        python bo_vs_random_efficientnet.py

    bash/zsh:
        WANDB_MODE=online python bo_vs_random_efficientnet.py
"""

import os
import time
import shutil
import subprocess
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from skopt import gp_minimize
from skopt.callbacks import CheckpointSaver

from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------------------------
# Configuration (edit freely)
# ---------------------------

SEED = 123

# Compare BO vs Random Search
CALLS_BO = 10
CALLS_RANDOM = 10

# Training speed/quality tradeoff
EPOCHS = 8                   # recommended: 5–10 for smoother BO curves
BATCH_SIZE = 100             # with 2000 subsample, gives 20 batches per epoch
TRAIN_SUBSAMPLE = 2000       # number of training images sampled per trial

# Dropout search space
DROPOUT_MIN = 0.0
DROPOUT_MAX = 0.5

# Dataset (Kaggle) config
DATASET_SLUG = "masoudnickparvar/brain-tumor-mri-dataset"
DATASET_DIR = "./dataset"

# W&B config (optional)
WANDB_MODE = os.getenv("WANDB_MODE", "disabled")  # "disabled" by default
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "brain-tumor-bo-vs-random")

# ---------------------------
# Reproducibility
# ---------------------------

torch.manual_seed(SEED)
np.random.seed(SEED)


# ==========================================
# 1) Dataset: local -> Kaggle CLI fallback
# ==========================================

def ensure_dataset():
    """
    Ensures dataset exists at:
      ./dataset/Training
      ./dataset/Testing

    If missing, tries to download via Kaggle CLI:
      kaggle datasets download -d <slug> -p ./dataset --unzip
    """
    train_dir = os.path.join(DATASET_DIR, "Training")
    test_dir = os.path.join(DATASET_DIR, "Testing")

    if os.path.isdir(train_dir) and os.path.isdir(test_dir):
        return DATASET_DIR

    kaggle_exe = shutil.which("kaggle")
    if kaggle_exe is None:
        raise RuntimeError(
            "Dataset not found locally and Kaggle CLI not available.\n\n"
            "Fix:\n"
            "  1) pip install kaggle\n"
            "  2) Put kaggle.json here:\n"
            "     Windows: C:\\Users\\<you>\\.kaggle\\kaggle.json\n"
            "     Mac/Linux: ~/.kaggle/kaggle.json\n"
            "  3) Accept dataset rules on Kaggle if prompted\n"
            "  4) Re-run this script.\n"
        )

    os.makedirs(DATASET_DIR, exist_ok=True)
    cmd = [
        kaggle_exe, "datasets", "download",
        "-d", DATASET_SLUG,
        "-p", DATASET_DIR,
        "--unzip"
    ]

    print("Dataset missing. Downloading via Kaggle CLI...")
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    if not (os.path.isdir(train_dir) and os.path.isdir(test_dir)):
        raise RuntimeError(
            "Download finished but expected folders were not found.\n"
            f"Expected:\n  {train_dir}\n  {test_dir}\n"
            "Check ./dataset contents."
        )

    return DATASET_DIR


# ==========================================
# 2) Data preprocessing & no-leakage split
# ==========================================

# EfficientNet-B0 expects ImageNet normalization when using pretrained weights.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset_root = ensure_dataset()
print("Dataset root:", dataset_root)

training_dataset = ImageFolder(os.path.join(dataset_root, "Training"), transform=transform)
test_dataset = ImageFolder(os.path.join(dataset_root, "Testing"), transform=transform)

# IMPORTANT: No leakage:
# We create train/val split ONLY from Training.
val_ratio = 0.2
val_len = int(len(training_dataset) * val_ratio)
train_len = len(training_dataset) - val_len

train_pool, val_subset = random_split(
    training_dataset,
    [train_len, val_len],
    generator=torch.Generator().manual_seed(SEED)
)

print("Training total:", len(training_dataset))
print("Train pool:", len(train_pool))
print("Val subset:", len(val_subset))
print("Test set:", len(test_dataset))


# ==========================================
# 3) Model definition
# ==========================================

class BrainTumorModel(nn.Module):
    """
    EfficientNet-B0 backbone (pretrained on ImageNet)
    + small custom classifier head for 4 classes.
    """
    def __init__(self, num_classes=4, dropout=0.1):
        super().__init__()
        self.base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        # Remove original head
        self.base_model.classifier = nn.Identity()

        # New head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),      # <-- tuned by BO/Random
            nn.Linear(1280, 128),
            nn.ReLU(),
            nn.Dropout(0.25),         # fixed dropout
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.classifier(x)
        return x


criterion = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Windows stability tip: too many workers can crash; adjust as needed.
WORKERS = 4
PIN_MEMORY = (device.type == "cuda")


# ==========================================
# 4) Helpers: subsampling and evaluation
# ==========================================

def make_train_subset_for_trial(trial_id: int) -> Subset:
    """
    Samples TRAIN_SUBSAMPLE indices from train_pool deterministically per trial_id.
    This keeps experiments reproducible and fair across runs.

    With TRAIN_SUBSAMPLE=2000 and BATCH_SIZE=100 and drop_last=True:
      -> exactly 20 batches per epoch.
    """
    n = len(train_pool)
    k = min(TRAIN_SUBSAMPLE, n)

    g = torch.Generator().manual_seed(SEED + 1000 + trial_id)
    idx = torch.randperm(n, generator=g)[:k].tolist()
    return Subset(train_pool, idx)


def evaluate(model: nn.Module, loader: DataLoader):
    """
    Returns (avg_loss, accuracy_percent) on a dataloader.
    """
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss_sum += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = loss_sum / max(1, len(loader))
    acc = 100.0 * correct / max(1, total)
    return float(avg_loss), float(acc)


# ==========================================
# 5) Single trial runner
# ==========================================

def run_trial(dropout: float, trial_id: int, log_to_wandb: bool = False):
    """
    Trains a fresh model with the given dropout and returns validation metrics.

    We train on a subsample of train_pool for speed.
    Validation is always on the same fixed val_subset.
    """
    run = None
    if log_to_wandb and WANDB_MODE != "disabled":
        import wandb
        run = wandb.init(
            project=WANDB_PROJECT,
            name=f"trial_{trial_id}",
            mode=WANDB_MODE,
            config={
                "dropout": dropout,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "optimizer": "Adamax",
                "trial": trial_id,
                "train_subsample": TRAIN_SUBSAMPLE,
                "seed": SEED,
            },
            reinit=True
        )

    try:
        model = BrainTumorModel(num_classes=4, dropout=dropout).to(device)
        optimizer = optim.Adamax(model.parameters(), lr=1e-3)

        trial_train_subset = make_train_subset_for_trial(trial_id)

        train_loader = DataLoader(
            trial_train_subset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=(WORKERS > 0),
            drop_last=True  # ensures exact number of batches if divisible
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=(WORKERS > 0),
        )

        # Training loop
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0

            pbar = tqdm(train_loader, desc=f"Trial {trial_id} Epoch {epoch+1}/{EPOCHS}", leave=False)
            for inputs, labels in pbar:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            avg_train_loss = running_loss / max(1, len(train_loader))

            if run:
                import wandb
                wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss})

        # Validation
        val_loss, val_acc = evaluate(model, val_loader)

        if run:
            import wandb
            wandb.log({"val_loss": val_loss, "val_accuracy": val_acc})

        return val_loss, val_acc

    finally:
        if run:
            run.finish()


# ==========================================
# 6) Random Search baseline
# ==========================================

def run_random_search(n_calls: int):
    rng = np.random.default_rng(SEED)
    xs, ys, accs, best_so_far = [], [], [], []
    best = float("inf")

    for i in range(1, n_calls + 1):
        dropout = float(rng.uniform(DROPOUT_MIN, DROPOUT_MAX))
        val_loss, val_acc = run_trial(dropout, trial_id=10_000 + i, log_to_wandb=False)

        xs.append(dropout)
        ys.append(val_loss)
        accs.append(val_acc)

        best = min(best, val_loss)
        best_so_far.append(best)

        print(f"[Random {i}/{n_calls}] dropout={dropout:.3f} val_loss={val_loss:.4f} acc={val_acc:.2f}%")

    return xs, ys, accs, best_so_far


# ==========================================
# 7) Bayesian Optimization objective wrapper
# ==========================================

_bo_trial_counter = 0

def bo_objective(params):
    """
    Objective for gp_minimize:
    - params[0] = dropout
    - return validation loss (to minimize)
    """
    global _bo_trial_counter
    _bo_trial_counter += 1
    dropout = float(params[0])

    # Log BO trials to W&B (optional, controlled via WANDB_MODE)
    val_loss, val_acc = run_trial(dropout, trial_id=_bo_trial_counter, log_to_wandb=True)

    print(f"[BO {_bo_trial_counter}/{CALLS_BO}] dropout={dropout:.3f} val_loss={val_loss:.4f} acc={val_acc:.2f}%")
    return val_loss


# ==========================================
# 8) Plotting: show BO beats random
# ==========================================

def plot_bo_vs_random(rand_best, rand_x, rand_y, bo_res, out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)

    # BO best-so-far curve
    bo_best = []
    best = float("inf")
    for v in bo_res.func_vals:
        best = min(best, float(v))
        bo_best.append(best)

    # Plot best-so-far comparison (key plot for report)
    plt.figure()
    plt.plot(range(1, len(rand_best) + 1), rand_best, label="Random search (best-so-far)")
    plt.plot(range(1, len(bo_best) + 1), bo_best, label="Bayesian Optimization (best-so-far)")
    plt.xlabel("Trial")
    plt.ylabel("Best validation loss so far")
    plt.title("BO vs Random Search (dropout tuning)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bo_vs_random_best_so_far.png"), dpi=200)
    plt.close()

    # Scatter plots of evaluations
    bo_x = [float(x[0]) for x in bo_res.x_iters]
    bo_y = [float(y) for y in bo_res.func_vals]

    plt.figure()
    plt.scatter(bo_x, bo_y)
    plt.xlabel("Dropout")
    plt.ylabel("Validation loss")
    plt.title("BO evaluations: dropout vs val loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bo_evaluations_scatter.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.scatter(rand_x, rand_y)
    plt.xlabel("Dropout")
    plt.ylabel("Validation loss")
    plt.title("Random evaluations: dropout vs val loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "random_evaluations_scatter.png"), dpi=200)
    plt.close()

    # Dropout selection over time (BO)
    plt.figure()
    plt.plot(range(1, len(bo_x) + 1), bo_x)
    plt.xlabel("Trial")
    plt.ylabel("Dropout chosen")
    plt.title("BO sampling strategy over time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dropout_over_time_bo.png"), dpi=200)
    plt.close()

    print(f"Saved plots to ./{out_dir}/")


# ==========================================
# 9) Final test evaluation (optional but recommended)
# ==========================================

def train_final_model(best_dropout: float):
    """
    Trains a final model (fresh) on the FULL train_pool + val_subset (i.e., all Training/)
    using best_dropout, then evaluates on Testing/.

    This is a common "proper" workflow:
    - Use Training split for tuning
    - Once hyperparameters are selected, retrain on all Training
    - Evaluate once on Testing (held-out)
    """
    # Combine train_pool and val_subset into "all training"
    # Note: both are subsets of training_dataset, so indices refer correctly.
    # We'll just create one big Subset by concatenating indices.
    train_pool_indices = getattr(train_pool, "indices", None)
    val_subset_indices = getattr(val_subset, "indices", None)

    if train_pool_indices is None or val_subset_indices is None:
        # Fallback: if random_split changes type, just skip combining.
        # This is unlikely in current PyTorch, but keeps script robust.
        full_training = training_dataset
    else:
        all_indices = list(train_pool_indices) + list(val_subset_indices)
        full_training = Subset(training_dataset, all_indices)

    train_loader = DataLoader(
        full_training,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(WORKERS > 0)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(WORKERS > 0)
    )

    model = BrainTumorModel(num_classes=4, dropout=best_dropout).to(device)
    optimizer = optim.Adamax(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Final Train Epoch {epoch+1}/{EPOCHS}", leave=False)
        for inputs, labels in pbar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    test_loss, test_acc = evaluate(model, test_loader)
    return test_loss, test_acc


# ==========================================
# 10) Main
# ==========================================

if __name__ == "__main__":
    print("\n==============================")
    print("Random Search baseline")
    print("==============================")
    start = time.time()
    rand_x, rand_y, rand_acc, rand_best = run_random_search(CALLS_RANDOM)
    print(f"Random search finished in {(time.time() - start)/60:.2f} min")

    print("\n==============================")
    print("Bayesian Optimization")
    print("==============================")
    checkpoint_callback = CheckpointSaver("./result_checkpoint.pkl")

    start = time.time()
    bo_res = gp_minimize(
        bo_objective,
        [(DROPOUT_MIN, DROPOUT_MAX)],
        acq_func="EI",
        n_calls=CALLS_BO,
        n_random_starts=2,
        noise=0.1**2,
        random_state=SEED,
        callback=[checkpoint_callback]
    )
    print(f"BO finished in {(time.time() - start)/60:.2f} min")

    best_dropout = float(bo_res.x[0])
    best_val_loss = float(bo_res.fun)
    print(f"\nBest dropout (BO): {best_dropout:.4f}")
    print(f"Best validation loss (BO): {best_val_loss:.4f}")

    print("\n==============================")
    print("Saving plots")
    print("==============================")
    plot_bo_vs_random(rand_best, rand_x, rand_y, bo_res, out_dir="plots")

    print("\n==============================")
    print("Final test evaluation (hold-out Testing/)")
    print("==============================")
    # Optional but strongly recommended for reporting:
    # retrain with best dropout on all Training/ and evaluate once on Testing/.
    test_loss, test_acc = train_final_model(best_dropout)
    print(f"FINAL TEST -> loss: {test_loss:.4f} | acc: {test_acc:.2f}%")

    print("\nDone.")
