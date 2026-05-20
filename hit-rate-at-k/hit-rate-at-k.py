def hit_rate_at_k(recommendations, ground_truth, k):
    """
    Compute the hit rate at K.
    """

    hits = 0
    total_users = len(recommendations)

    for i in range(total_users):

        # Take top-k recommendations for current user
        top_k = recommendations[i][:k]

        # Check if any ground truth item exists in top-k
        found = False

        for item in ground_truth[i]:
            if item in top_k:
                found = True
                break

        if found:
            hits += 1

    return (hits / total_users)