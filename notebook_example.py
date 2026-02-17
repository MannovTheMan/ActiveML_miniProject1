#!/usr/bin/env python3


import os
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets import ImageFolder
import kagglehub

dataset_path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset", output_dir="./dataset")
training_dataset = ImageFolder(os.path.join(dataset_path, "Training"))
testing_dataset = ImageFolder(os.path.join(dataset_path, "Testing"))
dataset = ConcatDataset([training_dataset, testing_dataset])

print(f"Unified dataset created with {len(dataset)} samples")
print(f"Classes: {training_dataset.classes}")

img_shape=(299,299,3)
base_model = tf.keras.applications.Xception(include_top= False, weights= "imagenet",
                            input_shape= img_shape, pooling= 'max')

# for layer in base_model.layers:
#     layer.trainable = False
    
model = Sequential([
    base_model,
    Flatten(),
    Dropout(rate= 0.3),
    Dense(128, activation= 'relu'),
    Dropout(rate= 0.25),
    Dense(4, activation= 'softmax')
])

model.compile(Adamax(learning_rate= 0.001),
              loss= 'categorical_crossentropy',
              metrics= ['accuracy',
                        Precision(),
                        Recall()])

model.summary()
