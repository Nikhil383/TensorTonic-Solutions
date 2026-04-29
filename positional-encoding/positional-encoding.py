import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    # positions: (seq_len, 1)
    positions = np.arange(seq_len)[:, np.newaxis]

    # number of sin/cos pairs
    half_dim = (d_model + 1) // 2  # ceil(d_model/2)

    # frequencies: (1, half_dim)
    i = np.arange(half_dim)[np.newaxis, :]
    div_term = base ** (2 * i / d_model)

    # angles: (seq_len, half_dim)
    angles = positions / div_term

    # initialize
    pe = np.zeros((seq_len, d_model))

    # fill even indices (sin)
    pe[:, 0::2] = np.sin(angles[:, :pe[:, 0::2].shape[1]])

    # fill odd indices (cos)
    if d_model > 1:
        pe[:, 1::2] = np.cos(angles[:, :pe[:, 1::2].shape[1]])

    return pe
    