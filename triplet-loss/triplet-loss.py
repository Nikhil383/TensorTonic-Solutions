import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Compute Triplet Loss for embedding ranking.
    """
    # Write code here
    anchor=np.array(anchor,dtype=float)
    positive=np.array(positive,dtype=float)
    negative=np.array(negative,dtype=float)
    if anchor.ndim == 1:
        anchor = anchor.reshape(1, -1)
        positive = positive.reshape(1, -1)
        negative = negative.reshape(1, -1)
    d_ap = np.sum((anchor - positive) ** 2, axis=1)
    d_an = np.sum((anchor - negative) ** 2, axis=1)
    losses = np.maximum(0, d_ap - d_an + margin)

    return float(np.mean(losses))