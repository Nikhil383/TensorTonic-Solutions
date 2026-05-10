def discount_returns(rewards, gamma):
    """
    Compute the discounted return at every timestep.
    """
    returns = [0] * len(rewards)
    G = 0

    # Traverse backwards
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        returns[t] = G

    return returns