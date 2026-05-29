import numpy as np

def patch_embed(image: np.ndarray, patch_size: int, embed_dim: int, W_proj: np.ndarray = None) -> np.ndarray:
    """
    Convert image to patch embeddings.
    W_proj: projection matrix of shape (patch_dim, embed_dim). If None, initialize randomly.
    """
    # YOUR CODE HERE
    B, H, W, C = image.shape

    assert H % patch_size == 0
    assert W % patch_size == 0

    N = (H // patch_size) * (W // patch_size)
    patch_dim = patch_size * patch_size * C

    # Separate patch grid from pixels inside patches
    patches = image.reshape(
        B,
        H // patch_size,
        patch_size,
        W // patch_size,
        patch_size,
        C
    )

    # Bring patch-grid dimensions together
    patches = patches.transpose(0, 1, 3, 2, 4, 5)

    # Flatten each patch
    patches = patches.reshape(B, N, patch_dim)

    # Projection matrix
    if W_proj is None:
        W_proj = np.random.randn(patch_dim, embed_dim)

    # Linear projection
    embeddings = patches @ W_proj

    return embeddings