import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import cv2
import numpy as np
import matplotlib.pyplot as plt

import os
import neptune.new as neptune
from tqdm import tqdm

from transformer import VisionTransformer


def neptune_monitoring(config):
    PARAMS = {}
    for key, val in config.__dict__.items():
        if key not in ["__module__", "__dict__", "__weakref__", "__doc__"]:
            PARAMS[key] = val
    return PARAMS


def train_Engine(n_epochs,
                 train_data,
                 val_data,
                 model,
                 optimizer,
                 loss_fn,
                 device,
                 monitoring=True):
    
    train_accuracy = 0
    val_accuracy = 0
    best_accuracy = 0
    for epoch in range(1, n_epochs + 1):
        total = 0
        with tqdm(train_data, unit="iteration") as train_epoch:
            train_epoch.set_description(f"Epoch {epoch}")
            for i, (data, target) in enumerate(train_epoch):
                total_samples = len(train_data.dataset)
                #device
                model = model.to(device)
                x = data.to(device)
                y = target.to(device)
                optimizer.zero_grad()

                logits, attn_weights = model(x)
                proba = F.log_softmax(logits, dim=1)
                loss = F.nll_loss(proba, y, reduction='sum')
                loss.backward()           
                optimizer.step()
        
                _, pred = torch.max(logits, dim=1) #
                train_accuracy += torch.sum(pred==y).item()
                total += target.size(0)
                accuracy_=(100 *  train_accuracy/ total)
                train_epoch.set_postfix(loss=loss.item(), accuracy=accuracy_)
                
                if monitoring:
                    run['Training_loss'].log(loss.item())
                    run['Training_acc'].log(accuracy_)

                if accuracy_ > best_accuracy:
                    best_accuracy = accuracy_
                    best_model = model
                    torch.save(best_model, f'/metadata/model.pth')
                
        
        total_samples = len(val_data.dataset)
        correct_samples = 0
        total_ = 0
        model.eval()
        with torch.no_grad():
            with tqdm(val_data, unit="iteration") as val_epoch:
                val_epoch.set_description(f"Epoch {epoch}")
                for i, (data, target) in enumerate(val_epoch):
                    
                    model = model.to(device)
                    x = data.to(device)
                    y = target.to(device)
                    
                    logits,attn_weights = model(x)
                    proba = F.log_softmax(logits, dim=1)
                    val_loss = F.nll_loss(proba, y, reduction='sum')
                    
                    _, pred = torch.max(logits, dim=1)#
                    val_accuracy += torch.sum(pred==y).item()
                    total_ += target.size(0)
                    val_accuracy_ = (100 *  val_accuracy/ total_)
                    val_epoch.set_postfix(loss=val_loss.item(), accuracy=val_accuracy_)

                    if monitoring:
                        run['Val_accuracy '].log(val_accuracy_)
                        run['Val_loss'].log(loss.item())
    


if __name__ == "__main__":
    from preprocessing import Dataset
    from config import Config

    config = Config()
    params = neptune_monitoring(Config)
    run = neptune.init(
        project="nielspace/ViT-bird-classification",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkYjRhYzI0Ny0zZjBmLTQ3YjYtOTY0Yi05ZTQ4ODM3YzE0YWEifQ==",
    )
    run["parameters"] = params

    model = VisionTransformer(
        img_size=config.IMG_SIZE,
        num_classes=config.NUM_CLASSES,
        hidden_size=config.HIDDEN_SIZE,
        in_channels=config.IN_CHANNELS,
        num_layers=config.NUM_LAYERS,
        num_attention_heads=config.NUM_ATTENTION_HEADS,
        linear_dim=config.LINEAR_DIM,
        dropout_rate=config.DROPOUT_RATE,
        attention_dropout_rate=config.ATTENTION_DROPOUT_RATE,
        eps=config.EPS,
        std_norm=config.STD_NORM,
    )

    train_data, val_data, test_data = Dataset(
        config.BATCH_SIZE, config.IMG_SIZE, config.DATASET_SAMPLE
    )  # neptune.save_checkpoint(

    optimizer = optim.Adam(model.parameters(), lr=0.003)
    train_Engine(
        n_epochs=config.N_EPOCHS,
        train_data=train_data,
        val_data=val_data,
        model=model,
        optimizer=optimizer,
        loss_fn="nll_loss",
        device=config.DEVICE[1],
        monitoring=True,
    )

"""
This script implements the training engine for a Vision Transformer (ViT) model and integrates it with Neptune for monitoring training and validation metrics. Below is an explanation of the various components:

Key Components of the Script
1. Imports
Core Libraries: Includes torch, torchvision, numpy, cv2 for computer vision tasks, and matplotlib.pyplot for visualization.
Neptune: Used for experiment tracking and monitoring.
VisionTransformer: A custom Vision Transformer model is imported from transformer.py.
Dataset: A custom data preprocessing and loading utility from preprocessing.py.
2. Neptune Monitoring
The neptune_monitoring function:

Extracts parameters from a config object and organizes them into a dictionary for logging to Neptune.
3. Training Engine (train_Engine)
This function handles the training and validation of the Vision Transformer model.

Parameters:
n_epochs: Number of training epochs.
train_data and val_data: Data loaders for training and validation datasets.
model: The Vision Transformer to train.
optimizer: Optimizer for updating model weights.
loss_fn: Loss function (in this case, negative log likelihood).
device: Specifies whether training should run on GPU or CPU.
monitoring: Enables or disables logging to Neptune.
Training Loop:
Uses tqdm for a progress bar to monitor iterations.
For each batch in the train_data:
Forward Pass: Computes predictions (logits) and attention weights.
Loss Calculation: Applies the negative log-likelihood loss (nll_loss).
Backward Pass and Optimization: Updates model parameters.
Accuracy Calculation: Compares predictions with ground truth.
Logs loss and accuracy to Neptune if monitoring is enabled.
Saves the best model (with highest accuracy) to a file.
Validation Loop:
Similar to training, but without gradient calculations (torch.no_grad()).
Computes validation accuracy and logs metrics to Neptune.
4. Main Script
The main block initializes the Vision Transformer, loads the dataset, and begins training.

Steps:
Configuration:

Config defines model hyperparameters, dataset details, and training settings.
Example attributes: IMG_SIZE, BATCH_SIZE, N_EPOCHS, etc.
Initialize Neptune:

Connects to a Neptune project using the provided API token.
Logs configuration parameters to the Neptune run.
Model Definition:

Instantiates the VisionTransformer with configuration parameters.
Data Preparation:

Uses the Dataset class to load training, validation, and test datasets.
Optimizer:

Uses Adam optimizer with a learning rate of 0.003.
Training:

Calls train_Engine with the defined parameters to start training.
Additional Notes
Attention Weights:

The script also extracts attention weights during the forward pass of the Vision Transformer, though these are not explicitly visualized.
Model Saving:

Saves the model with the best validation accuracy to /metadata/model.pth.
Neptune API Token:

The provided API token is used for connecting to Neptune. Ensure this is kept secure to avoid misuse.
Custom Components:

The Dataset and Config classes need to be implemented in preprocessing.py and config.py, respectively. They are essential for loading data and defining hyperparameters.
Device Usage:

The device parameter from the configuration determines whether the model runs on GPU or CPU. Ensure CUDA is available if using a GPU.
Example Output
When running the script:

Progress bars from tqdm will display the current epoch, iteration, loss, and accuracy.
Metrics (e.g., training/validation loss and accuracy) will be logged to Neptune for visualization and tracking.
This script provides a structured approach for training a Vision Transformer while enabling robust experiment monitoring with Neptune."""
