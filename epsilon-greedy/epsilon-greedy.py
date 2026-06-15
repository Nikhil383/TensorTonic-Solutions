import numpy as np

def epsilon_greedy(q_values, epsilon, rng=None):
    """
    Returns: action index (int)
    """
    # Write code here
    q_values=np.array(q_values,dtype=float)
    if rng is None:
        rng = np.random.default_rng()

    n_actions = len(q_values)

    # Exploration
    if rng.random() < epsilon:
        return rng.integers(n_actions)

    # Exploitation
    return np.argmax(q_values)
    
