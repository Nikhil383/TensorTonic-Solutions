import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    x = np.array(x, dtype=float)

    if rng is None:
        rng = np.random

    # keep probability
    keep_prob = 1 - p

    # generate mask (True = keep, False = drop)
    mask = rng.random(x.shape) < keep_prob

    # scale factor for kept units
    scale = 1.0 / keep_prob if keep_prob > 0 else 0.0

    # dropout pattern
    pattern = mask.astype(float) * scale

    # apply dropout
    out = x * pattern

    return out, pattern
    