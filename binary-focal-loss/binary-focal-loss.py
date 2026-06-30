import math
def binary_focal_loss(predictions, targets, alpha, gamma):
    """
    Compute the mean binary focal loss.
    """
    total_loss = 0
    eps = 1e-15  # Prevent log(0)

    for p, y in zip(predictions, targets):
        # Clip prediction
        p = min(max(p, eps), 1 - eps)

        # Compute p_t
        if y == 1:
            p_t = p
        else:
            p_t = 1 - p

        # Compute focal loss
        loss = -alpha * ((1 - p_t) ** gamma) * math.log(p_t)
        total_loss += loss

    return total_loss / len(predictions)
    