def gae(rewards, values, gamma, lam):
    """
    Compute Generalized Advantage Estimation.
    """
    # Write code here
    advantages = []
    gae = 0

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)

    return advantages