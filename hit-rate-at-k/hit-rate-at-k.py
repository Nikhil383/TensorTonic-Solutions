def hit_rate_at_k(recommendations, ground_truth, k):
    """
    Compute the hit rate at K.
    """
    # Write code here
    hits = 0
    total_users = len(recommendations)

    for recs, truth in zip(recommendations, ground_truth):
        top_k = set(recs[:k])
        relevant = set(truth)

        if top_k.intersection(relevant):
            hits += 1

    return hits / total_users if total_users else 0.0