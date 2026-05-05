import numpy as np

def compute_advantage(states, rewards, V, gamma):
    """
    Returns: A (NumPy array of advantages)
    """
    # Write code here
    states=np.array(states)
    rewards = np.array(rewards)
    V = np.array(V)

    T = len(rewards)
    G = np.zeros(T, dtype=np.float32)

    # Step 1: Compute returns backward
    G[-1] = rewards[-1]
    for t in reversed(range(T - 1)):
        G[t] = rewards[t] + gamma * G[t + 1]

    # Step 2: Compute advantage
    A = G - V

    return A
