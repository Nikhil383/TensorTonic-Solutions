from collections import defaultdict

def bigram_probabilities(tokens):
    """
    Returns:
      counts: dict mapping (w1, w2) -> integer count
      probs: dict mapping (w1, w2) -> float P(w2 | w1) with add-1 smoothing
    """
    counts = defaultdict(int)
    unigram_counts = defaultdict(int)

    # Count bigrams and first-word occurrences
    for i in range(len(tokens) - 1):
        w1, w2 = tokens[i], tokens[i + 1]
        counts[(w1, w2)] += 1
        unigram_counts[w1] += 1

    vocab = sorted(set(tokens))
    V = len(vocab)

    probs = {}

    # Compute probabilities for ALL possible bigrams
    for w1 in vocab:
        for w2 in vocab:
            c = counts[(w1, w2)]  # 0 if unseen
            probs[(w1, w2)] = (c + 1) / (unigram_counts[w1] + V)

    return dict(counts), probs