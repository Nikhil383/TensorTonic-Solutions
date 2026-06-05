import numpy as np

def pearson_correlation(X):
    """
    Compute Pearson correlation matrix from dataset X.
    """
    # Write code here
    try:
        X = np.asarray(X, dtype=float)

        n_samples, n_features = X.shape

        # Center the data
        X_centered = X - np.mean(X, axis=0)

        # Covariance matrix
        cov = np.dot(X_centered.T, X_centered) / (n_samples - 1)

        # Standard deviations
        std = np.sqrt(np.diag(cov))

        # Correlation matrix
        corr = cov / np.outer(std, std)

        return corr

    except:
        return None