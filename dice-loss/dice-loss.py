import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss for segmentation.
    """
    # Write code here
    p=np.array(p,dtype=float)
    y=np.array(y,dtype=float)
    flatten_p=p.flatten()
    flatten_y=y.flatten()
    intersection=np.sum(flatten_p*flatten_y)
    sum_p=np.sum(p)
    sum_y=np.sum(y)
    dice=(2*intersection + eps) / (sum_p + sum_y + eps)
    dice_loss=1-dice
    return dice_loss