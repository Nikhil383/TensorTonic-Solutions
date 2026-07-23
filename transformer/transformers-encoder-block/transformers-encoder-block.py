import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    x = x.astype(np.float64)
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    x = x.astype(np.float64)
    gamma = gamma.astype(np.float64)
    beta = beta.astype(np.float64)

    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)

    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta


def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """

    # Convert everything to float
    Q = Q.astype(np.float64)
    K = K.astype(np.float64)
    V = V.astype(np.float64)
    W_q = W_q.astype(np.float64)
    W_k = W_k.astype(np.float64)
    W_v = W_v.astype(np.float64)
    W_o = W_o.astype(np.float64)

    batch_size, seq_len, d_model = Q.shape
    head_dim = d_model // num_heads

    # Linear projections
    Q_proj = np.matmul(Q, W_q)
    K_proj = np.matmul(K, W_k)
    V_proj = np.matmul(V, W_v)

    # Split into heads
    Q_proj = Q_proj.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    K_proj = K_proj.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    V_proj = V_proj.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)

    # Scaled Dot-Product Attention
    scores = np.matmul(Q_proj, K_proj.transpose(0, 1, 3, 2))
    scores = scores / np.sqrt(head_dim)

    attention = softmax(scores, axis=-1)

    context = np.matmul(attention, V_proj)

    # Concatenate heads
    context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)

    # Final projection
    output = np.matmul(context, W_o)

    return output


def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """

    x = x.astype(np.float64)
    W1 = W1.astype(np.float64)
    W2 = W2.astype(np.float64)
    b1 = b1.astype(np.float64)
    b2 = b2.astype(np.float64)

    hidden = np.maximum(0, np.matmul(x, W1) + b1)
    output = np.matmul(hidden, W2) + b2

    return output


def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """

    x = x.astype(np.float64)

    # Multi-Head Attention
    attn_output = multi_head_attention(
        x, x, x,
        W_q, W_k, W_v,
        W_o,
        num_heads
    )

    # Residual + LayerNorm
    x = layer_norm(x + attn_output, gamma1, beta1)

    # Feed Forward
    ffn_output = feed_forward(x, W1, b1, W2, b2)

    # Residual + LayerNorm
    output = layer_norm(x + ffn_output, gamma2, beta2)

    return output