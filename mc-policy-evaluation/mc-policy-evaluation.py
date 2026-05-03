import numpy as np

def mc_policy_evaluation(episodes, gamma, n_states):
    """
    Returns: V (NumPy array of shape (n_states,))
    """
    # Write code here
    V = np.zeros(n_states)
    returns_sum = np.zeros(n_states)
    returns_count = np.zeros(n_states)
    
    for episode in episodes:
        G = 0
        visited = set()
        
        # Traverse backward to compute returns
        for t in reversed(range(len(episode))):
            s, r = episode[t]
            G = r + gamma * G
            
            # First-visit condition (check forward occurrence)
            if s not in [ep[0] for ep in episode[:t]]:
                returns_sum[s] += G
                returns_count[s] += 1
    
    # Final averaging
    for s in range(n_states):
        if returns_count[s] > 0:
            V[s] = returns_sum[s] / returns_count[s]
    
    return V
