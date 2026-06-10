import numpy as np

def gini_impurity(y_left, y_right):
    """
    Compute weighted Gini impurity for a binary split.
    """
    # Write code here
    y_left=np.array(y_left)
    y_right=np.array(y_right)
    n_l=len(y_left)
    n_r=len(y_right)
    N=n_l+n_r
    if N == 0:
        return 0.0
    def gini(y):
        if len(y) == 0:
            return 0.0

        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1.0 - np.sum(probs ** 2)

    gini_l = gini(y_left)
    gini_r = gini(y_right)

    return (n_l / N) * gini_l + (n_r / N) * gini_r