import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    """
    Compute InfoNCE Loss for contrastive learning.
    """
    # Convert lists to NumPy arrays
    Z1 = np.array(Z1, dtype=float)
    Z2 = np.array(Z2, dtype=float)

    # Similarity matrix
    S = np.dot(Z1, Z2.T) / temperature

    # Numerical stability
    S = S - np.max(S, axis=1, keepdims=True)

    exp_S = np.exp(S)

    # Positive pairs
    positive = np.diag(exp_S)

    # Denominator
    denominator = np.sum(exp_S, axis=1)

    # InfoNCE loss
    loss = -np.mean(np.log(positive / denominator))

    return loss