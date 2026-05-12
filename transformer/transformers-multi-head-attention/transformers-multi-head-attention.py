import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # Your code here
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // num_heads

    # Linear projections
    Q_proj = Q @ W_q
    K_proj = K @ W_k
    V_proj = V @ W_v

    # Split into heads
    Q_proj = Q_proj.reshape(batch_size, seq_len, num_heads, d_k)
    K_proj = K_proj.reshape(batch_size, seq_len, num_heads, d_k)
    V_proj = V_proj.reshape(batch_size, seq_len, num_heads, d_k)

    # Transpose for attention
    Q_proj = Q_proj.transpose(0, 2, 1, 3)
    K_proj = K_proj.transpose(0, 2, 1, 3)
    V_proj = V_proj.transpose(0, 2, 1, 3)

    # Scaled dot-product attention
    scores = np.matmul(Q_proj, K_proj.transpose(0, 1, 3, 2))
    scores = scores / np.sqrt(d_k)

    attention_weights = softmax(scores, axis=-1)

    attention_output = np.matmul(attention_weights, V_proj)

    # Concatenate heads
    attention_output = attention_output.transpose(0, 2, 1, 3)
    attention_output = attention_output.reshape(batch_size, seq_len, d_model)

    # Final linear projection
    output = attention_output @ W_o

    return output