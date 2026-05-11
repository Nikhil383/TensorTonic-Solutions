import numpy as np

def replay_buffer_sample(buffer, batch_size, seed):
    """
    Sample a batch of transitions from the replay buffer.
    """
    rng = np.random.RandomState(seed)
    
    indices = rng.choice(len(buffer), size=batch_size, replace=False)
    
    sampled = []
    for i in indices:
        sampled.append(buffer[i])
    
    return sampled