import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    # Your code here
    # Initialize vector with zeros
    bow = np.zeros(len(vocab), dtype=int)
    
    # Count occurrences of each vocab word
    for i, word in enumerate(vocab):
        bow[i] = tokens.count(word)
    
    return bow
    