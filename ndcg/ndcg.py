import math

def ndcg(relevance_scores, k):
    """
    Compute NDCG@k.
    """
    # Write code here
    # DCG
    dcg = 0
    for i, rel in enumerate(relevance_scores[:k]):
        gain = (2 ** rel) - 1
        dcg += gain / math.log2(i + 2)

    # IDCG (sorted relevance scores)
    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg = 0
    for i, rel in enumerate(ideal_scores[:k]):
        gain = (2 ** rel) - 1
        idcg += gain / math.log2(i + 2)

    # Avoid division by zero
    if idcg == 0:
        return 0

    return dcg / idcg