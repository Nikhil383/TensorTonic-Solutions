import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    """
    Build TF-IDF matrix from a list of text documents.
    Returns tuple of (tfidf_matrix, vocabulary).
    """
    # Write code here
    # Handle empty corpus
    if not documents:
        return np.array([]).reshape(0, 0), []

    # Tokenize documents
    tokenized_docs = [doc.lower().split() for doc in documents]

    # Build vocabulary
    vocabulary = sorted(set(
        word
        for doc in tokenized_docs
        for word in doc
    ))

    vocab_size = len(vocabulary)
    n_docs = len(documents)

    # Handle case where all documents are empty
    if vocab_size == 0:
        return np.zeros((n_docs, 0)), []

    # Word -> column index mapping
    word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}

    # Document Frequency (DF)
    df = Counter()
    for doc in tokenized_docs:
        for word in set(doc):
            df[word] += 1

    # Inverse Document Frequency (IDF)
    idf = {
        word: math.log(n_docs / df[word])
        for word in vocabulary
    }

    # TF-IDF matrix
    tfidf_matrix = np.zeros((n_docs, vocab_size))

    for doc_idx, doc in enumerate(tokenized_docs):
        if not doc:  # Empty document
            continue

        term_counts = Counter(doc)
        total_terms = len(doc)

        for word, count in term_counts.items():
            tf = count / total_terms
            tfidf_matrix[doc_idx, word_to_idx[word]] = tf * idf[word]

    return tfidf_matrix, vocabulary