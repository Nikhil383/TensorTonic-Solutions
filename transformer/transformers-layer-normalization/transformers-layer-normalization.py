import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns: Normalized array of same shape as x
    """
    # Your code here
    # Compute mean and variance along the last dimension
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    
    # Normalize
    x_norm = (x - mean) / np.sqrt(variance + eps)
    
    # Scale and shift
    output = gamma * x_norm + beta
    
    return output