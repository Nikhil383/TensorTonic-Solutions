import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    # Write code here
    X=np.array(X)
    # Check if input is 2D
    if X.ndim != 2:
        return None

    n_samples = X.shape[0]

    # Need at least 2 samples
    if n_samples < 2:
        return None

    # Compute mean (per feature)
    mean = np.mean(X, axis=0)

    # Center the data
    X_centered = X - mean

    # Compute covariance matrix
    cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)

    return cov_matrix