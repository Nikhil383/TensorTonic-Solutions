import numpy as np

def ridge_regression(X, y, lam):
    """
    Compute ridge regression weights using the closed-form solution.

    w = (X^T X + lambda * I)^(-1) X^T y

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Design matrix.
    y : ndarray of shape (n_samples,)
        Target values.
    lam : float
        Regularization strength (lambda).

    Returns
    -------
    w : ndarray of shape (n_features,)
        Learned weight vector.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n_features = X.shape[1]

    identity = np.eye(n_features)
    A = X.T @ X + lam * identity
    b = X.T @ y

    w = np.linalg.solve(A, b)
    return w