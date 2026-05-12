import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here
    # Position indices: shape (seq_length, 1)
    position = np.arange(seq_length).reshape(-1, 1)

    # Compute scaling term for even dimensions
    div_term = np.exp(
        np.arange(0, d_model, 2) *
        (-np.log(10000.0) / d_model)
    )

    # Initialize encoding matrix
    pe = np.zeros((seq_length, d_model))

    # Apply sine to even indices
    pe[:, 0::2] = np.sin(position * div_term)

    # Apply cosine to odd indices
    pe[:, 1::2] = np.cos(position * div_term)

    return pe
    