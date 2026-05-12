import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Your code here
    d_k = K.size(-1)

    # QK^T
    scores = torch.matmul(Q, K.transpose(-2, -1))

    # Scale
    scores = scores / math.sqrt(d_k)

    # Softmax
    attention_weights = F.softmax(scores, dim=-1)

    # Weighted sum
    output = torch.matmul(attention_weights, V)

    return output