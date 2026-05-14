def perplexity(prob_distributions, actual_tokens):
    """
    Compute the perplexity of a token sequence given predicted distributions.
    """
    # Write code here
    log_probs = []

    for dist, token in zip(prob_distributions, actual_tokens):
        p_i = dist[token]
        log_probs.append(np.log(p_i))

    avg_log_prob = np.mean(log_probs)

    return np.exp(-avg_log_prob)