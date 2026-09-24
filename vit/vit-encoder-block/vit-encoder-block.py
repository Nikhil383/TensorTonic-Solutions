import numpy as np

def vit_encoder_block(x: np.ndarray, num_heads: int,
                      Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray,
                      Wo: np.ndarray, W1: np.ndarray, W2: np.ndarray) -> np.ndarray:
    """
    Returns the float64 output of one pre-normalized ViT encoder block.
    """
    def layer_norm(values):
        mean = np.mean(values, axis=-1, keepdims=True)
        variance = np.mean((values - mean) ** 2, axis=-1, keepdims=True)
        return (values - mean) / np.sqrt(variance + 1e-6)

    def softmax(values):
        shifted = values - np.max(values, axis=-1, keepdims=True)
        exp_values = np.exp(shifted)
        return exp_values / np.sum(exp_values, axis=-1, keepdims=True)

    batch, tokens, width = x.shape
    head_width = width // num_heads
    normalized = layer_norm(x)
    q = (normalized @ Wq).reshape(batch, tokens, num_heads, head_width).transpose(0, 2, 1, 3)
    k = (normalized @ Wk).reshape(batch, tokens, num_heads, head_width).transpose(0, 2, 1, 3)
    v = (normalized @ Wv).reshape(batch, tokens, num_heads, head_width).transpose(0, 2, 1, 3)
    scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(head_width)
    attended = (softmax(scores) @ v).transpose(0, 2, 1, 3).reshape(batch, tokens, width)
    after_attention = x + attended @ Wo
    normalized = layer_norm(after_attention)
    hidden = normalized @ W1
    gelu = 0.5 * hidden * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (hidden + 0.044715 * hidden ** 3)))
    return after_attention + gelu @ W2