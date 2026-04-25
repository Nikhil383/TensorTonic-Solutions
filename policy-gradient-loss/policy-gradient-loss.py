def policy_gradient_loss(log_probs, rewards, gamma):
    """
    Compute REINFORCE policy gradient loss with mean-return baseline.
    """
    # Write code here
    # Validate input
    if not isinstance(log_probs, (list, tuple)) or not isinstance(rewards, (list, tuple)):
        return None
    if len(log_probs) != len(rewards) or len(rewards) < 1:
        return None

    T = len(rewards)

    # Step 1: Compute returns G backward
    G = [0.0] * T
    G[-1] = rewards[-1]
    for t in range(T - 2, -1, -1):
        G[t] = rewards[t] + gamma * G[t + 1]

    # Step 2: Mean baseline
    mean_G = sum(G) / T

    # Step 3 & 4: Compute advantages and loss
    loss = 0.0
    for lp, g in zip(log_probs, G):
        advantage = g - mean_G
        loss -= lp * advantage

    # Normalize by T
    loss /= T

    return loss