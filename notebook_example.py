#!/usr/bin/env python3


import os
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets import ImageFolder
from torchvision.models import efficientnet_b0
from skopt import gp_minimize
import kagglehub

dataset_path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset", output_dir="./dataset")
training_dataset = ImageFolder(os.path.join(dataset_path, "Training"))
testing_dataset = ImageFolder(os.path.join(dataset_path, "Testing"))
dataset = ConcatDataset([training_dataset, testing_dataset])

class BrainTumorModel(nn.Module):
    def __init__(self, num_classes=4, dropout=.1):
        super(BrainTumorModel, self).__init__()
        self.base_model = efficientnet_b0(pretrained=True)
        self.base_model.classifier = nn.Identity()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(1280, 128),  # EfficientNet-B0 has 1280 features
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.base_model(x)
        x = self.classifier(x)
        return x

model = BrainTumorModel(num_classes=4)
optimizer = optim.Adamax(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def train_model(dropout):
    m = BrainTumorModel(num_classes=4, dropout=dropout)
    # todo: train loop
    # return error
    pass

SEED = "123"

gp_minimize(train_model,
            [0, 1],
            acq_func = "EI",
            n_calls=CALLS,
            n_random_starts=2,
            noise=.1**2,
            random_state=SEED)
