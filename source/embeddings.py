import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm

from config import Config

config = Config()


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings."""

    def __init__(self, img_size: int, hidden_size: int, in_channels: int):

        super(Embeddings, self).__init__()
        img_size = _pair(img_size)

        patch_size = _pair(img_size)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.patch_embeddings = Conv2d(
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.position_embeddings = nn.Parameter(
            torch.zeros(1, n_patches + 1, hidden_size)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

        self.dropout = Dropout(0.1)

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


if __name__ == "__main__":
    x = torch.randn(1, config.IN_CHANNELS * config.IMG_SIZE * config.IMG_SIZE)
    x = x.reshape(1, config.IN_CHANNELS, config.IMG_SIZE, config.IMG_SIZE)

    embeddings = Embeddings(
        img_size=(config.IMG_SIZE, config.IMG_SIZE),
        hidden_size=config.HIDDEN_SIZE,
        in_channels=config.IN_CHANNELS,
    )
    print(embeddings(x))

"""
What does this code do?
This code creates embeddings for an image so that the image can be processed by a Vision Transformer (ViT) model. Think of it like chopping an image into smaller pieces (patches), turning them into numbers (embeddings), and preparing them for further processing. Here's how it works:

Example: Think of a Puzzle Game
Imagine you have a big image of a bird that is 224x224 pixels (like a puzzle). You break this image into small square pieces (patches), let's say 16x16 pixel pieces. Now:

Breaking the Image into Patches:

The image is divided into small patches (like dividing a big puzzle into small pieces).
If the image is 224x224 and the patches are 16x16, you will have 
14
×
14
=
196
14×14=196 patches.
Turning Each Patch into Numbers:

Each patch is passed through a small function (a convolution layer) that turns the patch into a vector of numbers of size hidden_size (e.g., 768 numbers). This is like summarizing each puzzle piece into important information.
Adding a Special Token (The "Big Picture" Piece):

A special "classification token" is added to represent the overall image. Think of this as a "summary card" that collects information from all the pieces.
Adding Positional Information:

To help the model understand where each patch came from in the image, we add positional embeddings (like labeling each puzzle piece with its position).
Final Preparation:

Everything (patches + special token) is combined into a single sequence and passed through a dropout layer to make it more robust for learning.
Step-by-Step with Numbers:
Input:
Suppose the image is of shape [1, 3, 224, 224]:
1: Batch size (1 image).
3: RGB channels (Red, Green, Blue).
224, 224: Width and height of the image.
Process:
Patches:

The image is divided into 16x16 patches. This results in 196 patches (14 rows x 14 columns).
Patch Embeddings:

Each 16x16 patch is turned into a vector of size hidden_size (e.g., 768). So, now you have a table of size [1, 196, 768] (196 patches, each with 768 features).
Add Classification Token:

A special token is added. Now the size becomes [1, 197, 768] (196 patches + 1 special token).
Add Positional Embeddings:

Positional information is added, so the model knows where each patch belongs in the image.
Output:

Final size: [1, 197, 768].
Simple Analogy:
Think of this process as preparing a jigsaw puzzle for an AI model:

Step 1: Break the image into small pieces (patches).
Step 2: Analyze each piece (patch embeddings).
Step 3: Add a label to each piece showing its position (positional embeddings).
Step 4: Add a "summary card" to represent the entire puzzle (classification token).
Step 5: Package everything together and give it to the AI for further processing.
Output Example:
If you print the embeddings for one image:

Shape: [1, 197, 768].
It’s a tensor (a big table of numbers) that represents the input image in a way the AI can understand."""

"""
1. Imports:
Core PyTorch modules (nn, Conv2d, Dropout, etc.) are imported.
_pair is used to handle cases where input sizes or parameters may need to be specified as tuples (e.g., img_size or patch_size).
The Config object from config.py holds hyperparameters for the model.
2. Embeddings Class:
The Embeddings class is defined to compute patch embeddings and add positional embeddings.

Inputs:

img_size: The size of the input image (e.g., 224x224).
hidden_size: The size of the embedding dimension.
in_channels: Number of input channels (e.g., 3 for RGB images).
Attributes:

patch_embeddings:
A convolutional layer that extracts embeddings for each patch in the image.
The kernel size and stride are equal to the patch size, ensuring that each patch is treated as a single "unit."
position_embeddings:
A learnable parameter that encodes positional information for the patches.
Its shape is (1, n_patches + 1, hidden_size):
n_patches: Number of patches in the image.
+1: Accounts for the additional classification token (cls_token).
cls_token:
A learnable token appended to the input embeddings. This token will eventually capture global information for classification tasks.
dropout:
A dropout layer to regularize the embeddings and prevent overfitting.
3. Forward Pass (forward):
Extract Patch Embeddings:

The input tensor x (of shape [B, C, H, W], where B = batch size, C = channels, H = height, W = width) is passed through the patch_embeddings convolution.
Output shape: [B, hidden_size, n_patches_height, n_patches_width].
Flatten and Transpose:

The output is flattened to [B, hidden_size, n_patches] and then transposed to [B, n_patches, hidden_size].
Add Classification Token:

The cls_token is expanded to match the batch size and concatenated to the patch embeddings along the sequence dimension.
Add Positional Embeddings:

The positional embeddings are added to the combined embeddings to encode positional information.
Apply Dropout:

Dropout is applied to the final embeddings.
4. Testing the Module:
The script tests the Embeddings class by:

Creating a random input tensor x representing a single image.
Shape: [1, in_channels, img_size, img_size].
Initializing an Embeddings object using the hyperparameters from the Config class.
Passing the input tensor x through the Embeddings module and printing the output.
Key Features:
Patch Embeddings: Convolution is used to embed patches of the image into a higher-dimensional space.
Positional Embeddings: Learnable positional encodings are added to retain spatial information.
Classification Token (cls_token): Appended to the embeddings to represent the overall image during classification.
Dropout: Helps regularize the embeddings.
Output Shape:
For an input of shape [B, C, H, W]:

The output embeddings will have a shape [B, n_patches + 1, hidden_size].
This module is a critical building block for Vision Transformers and sets up the image data for processing by subsequent transformer layers."""
