def top_k_recommendations(scores, rated_indices, k):
    """
    Return indices of top-k unrated items by predicted score.
    """
    # Create (score, index) pairs for unrated items
    candidates = [
        (scores[i], i)
        for i in range(len(scores))
        if i not in rated_indices
    ]

    # Sort by score in descending order
    candidates.sort(key=lambda x: -x[0])

    # Return top-k indices
    return [index for score, index in candidates[:k]]