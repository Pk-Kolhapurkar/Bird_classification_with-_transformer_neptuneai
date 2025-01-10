import torch
from torch import nn
from torch.nn import CrossEntropyLoss, Dropout, Linear, Conv2d, LayerNorm
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self, hidden_size, linear_dim, dropout_rate, std_norm):
        super(Mlp, self).__init__()
        self.fc1 = Linear(hidden_size, linear_dim)
        self.fc2 = Linear(linear_dim, hidden_size)
        self.act_fn = torch.nn.functional.gelu
        self.dropout = Dropout(dropout_rate)
        self.std_norm = std_norm
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=self.std_norm)
        nn.init.normal_(self.fc2.bias, std=self.std_norm)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


"""
What does this code do?
This code implements a Multi-Layer Perceptron (MLP) module, commonly used in Vision Transformers (ViT) as a feed-forward network. It processes the refined outputs from the attention mechanism and helps the model learn deeper features.

Analogy: The Chef and the Blender
Imagine:

You’re making a smoothie.
The blender has two stages:
First, the chef adds all ingredients (e.g., fruits and milk) and blends them (step 1).
Next, the chef pours the blended mix into another jar for further processing and adjusts its taste (step 2).
The MLP here is like a two-stage blender:

First layer: Mixes raw input features into richer intermediate features.
Second layer: Processes these intermediate features back to their original size but with more meaningful information.
Step-by-Step Breakdown
1. Input and Dimensions
Input: x (a tensor with shape [batch_size, sequence_length, hidden_size]).

Example:
batch_size = 1 (1 image).
sequence_length = 197 (196 patches + 1 classification token).
hidden_size = 768 (features per patch).
Goal: Pass this input through two linear layers (fc1 and fc2) with:

First layer: Expands features to a larger size (linear_dim).
Second layer: Compresses features back to their original size.
2. Layer Definitions
fc1:

Linear layer that increases feature size from hidden_size → linear_dim.
Example: Think of adding extra fruits to make the smoothie richer.
act_fn (GELU):

Activation function that adds non-linearity (like ReLU but smoother).
Example: Think of this as stirring the mix for better consistency.
dropout:

Regularization technique to prevent overfitting by randomly deactivating some neurons.
Example: Imagine spilling a bit of the smoothie to ensure it doesn't over-concentrate.
fc2:

Linear layer that compresses features from linear_dim → hidden_size (back to the original size).
Example: This step pours the smoothie back into the serving glass.
3. Weight Initialization
Custom initialization ensures the weights and biases start with good values:

xavier_uniform_: Sets weights to values suitable for deeper networks.
normal_: Adds randomness to biases (standard deviation: std_norm).
Example: The chef calibrates the blender so it works efficiently from the start.

4. Forward Pass:
Input: The input tensor x (raw features).

Step-by-step flow:

Pass x through fc1 (expand features).
Apply GELU activation for non-linearity.
Apply dropout for regularization.
Pass through fc2 (compress back to original size).
Apply dropout again for regularization.
Output: A tensor of the same shape as the input but with refined features.

Simple Example
Inputs:
Input tensor: x (shape [1, 197, 768]).
1: Batch size (1 image).
197: Tokens (196 patches + 1 classification token).
768: Hidden size (features of each token).
linear_dim = 3072: Expansion size.
Flow:
fc1:
Shape: [1, 197, 768] → [1, 197, 3072] (expansion).
`act_fn (GELU):
Apply non-linearity (no shape change).
dropout:
Regularization (no shape change).
fc2:
Shape: [1, 197, 3072] → [1, 197, 768] (compression).
dropout:
Regularization (no shape change).
Output Explanation
The output has the same shape as the input:

Shape: [1, 197, 768]
Meaning: Each token's features are now richer and more meaningful after the two-step blending process.
Step-by-Step Analogy
Imagine making a smoothie:

fc1: Add more ingredients to the mix (expand).
GELU: Stir the mix for smoother consistency.
dropout: Spill a little to avoid over-concentration.
fc2: Pour back into the original glass (compress).
dropout: Adjust to ensure balance.
"""
