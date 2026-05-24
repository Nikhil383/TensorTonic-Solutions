def frequency_encoding(values):
    """
    Replace each value with its frequency proportion.
    """
    counts = {}
    
    # Count occurrences
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    
    total = len(values)
    
    # Replace with frequency proportion
    return [counts[v] / total for v in values]
    