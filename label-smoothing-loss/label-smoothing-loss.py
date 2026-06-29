def label_smoothing_loss(predictions, target, epsilon):
    """
    Compute cross-entropy loss with label smoothing.
    """
    # Write code here
    k=len(predictions)
    loss=0.0
    for i in range(k):
        if i == target:
            q_i = (1 - epsilon) + (epsilon / k)
        else:
            q_i = epsilon / k

        loss -= q_i * math.log(predictions[i])

    return loss
        