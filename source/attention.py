import torch
from torch.nn import CrossEntropyLoss, Dropout, Linear, Conv2d, LayerNorm
import torch.nn.functional as F
from torch import nn

import math

from config import Config

config = Config()


class Attention(nn.Module):
    def __init__(self, num_attention_heads, hidden_size, attention_dropout_rate):

        super(Attention, self).__init__()

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = Linear(hidden_size, self.all_head_size)
        self.key = Linear(hidden_size, self.all_head_size)
        self.value = Linear(hidden_size, self.all_head_size)

        self.out = Linear(hidden_size, self.all_head_size)
        self.attn_dropout = Dropout(attention_dropout_rate)
        self.proj_dropout = Dropout(attention_dropout_rate)

        self.softmax = torch.nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


if __name__ == "__main__":
    from embeddings import Embeddings

    x = torch.randn(1, config.IN_CHANNELS * config.IMG_SIZE * config.IMG_SIZE)
    x = x.reshape(1, config.IN_CHANNELS, config.IMG_SIZE, config.IMG_SIZE)

    embeddings = Embeddings(
        img_size=(config.IMG_SIZE, config.IMG_SIZE),
        hidden_size=config.HIDDEN_SIZE,
        in_channels=config.IN_CHANNELS,
    )

    att = Attention(
        num_attention_heads=config.NUM_ATTENTION_HEADS,
        hidden_size=config.HIDDEN_SIZE,
        attention_dropout_rate=config.ATTENTION_DROPOUT_RATE,
    )

    print(att(embeddings(x)))



"""
What does this code do?
This code implements the Attention mechanism used in Vision Transformers (ViT). It allows the model to focus on important parts of an input image by learning relationships between different parts (patches). Think of it as a spotlight that highlights key areas of an image.

Example: Think of a Teacher Reading a Book
Imagine a teacher reading a book to a class:

The book has many sentences (like patches of an image).
The teacher focuses on certain words in each sentence that are important (attention mechanism).
The teacher uses these important words to understand the meaning of the book (context representation).
Similarly, attention highlights important patches of the image to extract useful information.

Step-by-Step Breakdown
1. Input Explanation:
The input hidden_states represents the embeddings from the image.
Example input shape: [1, 197, 768]:
1: Batch size (1 image).
197: Number of tokens (196 patches + 1 classification token).
768: Hidden size (features of each patch).
2. Query, Key, Value (Q, K, V):
Imagine the model asking questions like:

Query (Q): "Which patches are important?"
Key (K): "What information does each patch contain?"
Value (V): "What details should we extract from each patch?"
This is done using three Linear layers:

query = Linear(hidden_size, all_head_size)
key = Linear(hidden_size, all_head_size)
value = Linear(hidden_size, all_head_size)
3. Split into Attention Heads:
The attention mechanism uses multiple attention heads to focus on different aspects of the image. For example:

One head might focus on colors.
Another head might focus on shapes.
Each head processes part of the image independently:

transpose_for_scores: Splits the features into num_attention_heads and reshapes for easier computation.
4. Attention Scores:
The attention mechanism calculates how much each patch should focus on every other patch:

This is done by multiplying query and key:
Attention Scores
=
Query
×
Key
𝑇
Attention Scores=Query×Key 
T
 
Then divide by the square root of the patch size for scaling (normalization).
Example: If the teacher reads a sentence, attention scores determine which words are important to the sentence's meaning.

5. Softmax for Probabilities:
Attention scores are passed through a Softmax function to get probabilities (e.g., "how important is this patch compared to others?").
6. Compute the Context Layer:
Multiply attention probabilities with the value to extract relevant details for each patch:
Context Layer
=
Attention Probs
×
Value
Context Layer=Attention Probs×Value
Example: The teacher highlights the important words and summarizes the meaning of the sentence.
7. Combine the Information:
The context layer is reshaped back to its original format.
A final Linear layer combines information from all attention heads.
Output Explanation
Attention Output:

The refined representation of the image after focusing on the important patches.
Shape: [1, 197, 768] (same as input).
Weights:

The attention weights (probabilities) that show how much each patch focuses on every other patch.
Step-by-Step Example:
Suppose:

An image has 196 patches + 1 classification token.
hidden_size = 768, num_attention_heads = 12.
Input Shape: [1, 197, 768].
Query, Key, Value Layers: Transform input to [1, 12, 197, 64] (12 heads, each with 64 features).
Attention Scores: [1, 12, 197, 197] (how each patch relates to every other patch).
Context Layer: [1, 197, 768] (refined representation).
Output Example:
If you run the code, you’ll see something like this:

python
Copy code
(tensor([[[...]]], grad_fn=<AddBackward0>), tensor([[[...]]], grad_fn=<SoftmaxBackward>))
The first tensor is the attention output (refined image representation).
The second tensor is the weights (importance of each patch).
Simple Analogy:
The attention mechanism is like a teacher reading a book:

Queries: "What should I focus on?"
Keys: "What does each part of the book say?"
Values: "What details should I summarize?"
Output: The teacher focuses on key parts and summarizes the book effectively."""
