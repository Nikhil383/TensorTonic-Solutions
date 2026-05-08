import math

def log_loss(y_true, y_pred, eps=1e-15):
    """
    Compute per-sample log loss.
    """
    # Write code here
    losses = []
    
    for y, p in zip(y_true, y_pred):
        # Clip probabilities to avoid log(0)
        p = max(eps, min(1 - eps, p))
        
        # Binary cross-entropy / log loss formula
        loss = -(y * math.log(p) + (1 - y) * math.log(1 - p))
        
        losses.append(loss)
    
    return losses