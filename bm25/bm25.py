import numpy as np
from collections import Counter
import math

def bm25_score(query_tokens, docs, k1=1.2, b=0.75):

    if not docs:
        return np.array([])

    N = len(docs)
    doc_lengths = [len(doc) for doc in docs]
    avgdl = sum(doc_lengths) / N

    if avgdl == 0:
        avgdl = 1

    df = Counter()
    for term in query_tokens:
        df[term] = sum(1 for doc in docs if term in doc)

    scores = []

    for doc, dl in zip(docs, doc_lengths):
        tf = Counter(doc)
        score = 0.0

        for term in query_tokens:
            if term not in tf:
                continue

            idf = math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1)

            numerator = tf[term] * (k1 + 1)
            denominator = tf[term] + k1 * (1 - b + b * dl / avgdl)

            score += idf * numerator / denominator

        scores.append(score)

    return np.array(scores)