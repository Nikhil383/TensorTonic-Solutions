import numpy as np

def _entropy(y):
    """
    Helper: Compute Shannon entropy (base 2) for labels y.
    """
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    vals, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0

def information_gain(y, split_mask):
    """
    Compute Information Gain of a binary split on labels y.
    Use the _entropy() helper above.
    """
    # Write code here
    y = np.asarray(y)
    split_mask = np.asarray(split_mask, dtype=bool)

    left_y = y[split_mask]
    right_y = y[~split_mask]

    parent_entropy = _entropy(y)

    n = len(y)
    left_weight = len(left_y) / n
    right_weight = len(right_y) / n

    child_entropy = (
        left_weight * _entropy(left_y) +
        right_weight * _entropy(right_y)
    )

    return parent_entropy - child_entropy
