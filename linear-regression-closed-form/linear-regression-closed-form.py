import numpy as np

def linear_regression_closed_form(X, y):
    """
    Compute the optimal weight vector using the normal equation.
    """
    # Write code here
    X=np.array(X,dtype=float)
    y=np.array(y,dtype=float)
    w = np.linalg.inv(X.T @ X) @ (X.T @ y)

    return w.tolist()