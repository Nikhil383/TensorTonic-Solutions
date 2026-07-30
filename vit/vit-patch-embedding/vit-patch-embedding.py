import numpy as np

def patch_embed(image: np.ndarray, patch_size: int, embed_dim: int, W_proj: np.ndarray = None) -> np.ndarray:
    """
    Convert image(s) to patch embeddings.

    Parameters
    ----------
    image : np.ndarray
        Shape (H, W, C) for a single image or
        Shape (B, H, W, C) for a batch of images.
    patch_size : int
        Size of each square patch.
    embed_dim : int
        Dimension of output embedding.
    W_proj : np.ndarray, optional
        Projection matrix of shape (patch_dim, embed_dim).
        If None, initialized randomly.

    Returns
    -------
    np.ndarray
        Shape (num_patches, embed_dim) for a single image or
        Shape (B, num_patches, embed_dim) for a batch.
    """

    # ---------- Single Image ----------
    if image.ndim == 3:
        H, W, C = image.shape

        assert H % patch_size == 0, "Height must be divisible by patch_size"
        assert W % patch_size == 0, "Width must be divisible by patch_size"

        patch_dim = patch_size * patch_size * C

        if W_proj is None:
            W_proj = np.random.randn(patch_dim, embed_dim)

        embeddings = []

        for i in range(0, H, patch_size):
            for j in range(0, W, patch_size):
                patch = image[i:i+patch_size, j:j+patch_size, :]
                patch = patch.reshape(-1)           # Flatten
                embedding = patch @ W_proj          # Linear projection
                embeddings.append(embedding)

        return np.array(embeddings)

    # ---------- Batch of Images ----------
    elif image.ndim == 4:
        B, H, W, C = image.shape

        assert H % patch_size == 0, "Height must be divisible by patch_size"
        assert W % patch_size == 0, "Width must be divisible by patch_size"

        patch_dim = patch_size * patch_size * C

        if W_proj is None:
            W_proj = np.random.randn(patch_dim, embed_dim)

        batch_embeddings = []

        for b in range(B):
            embeddings = []

            for i in range(0, H, patch_size):
                for j in range(0, W, patch_size):
                    patch = image[b, i:i+patch_size, j:j+patch_size, :]
                    patch = patch.reshape(-1)
                    embedding = patch @ W_proj
                    embeddings.append(embedding)

            batch_embeddings.append(embeddings)

        return np.array(batch_embeddings)

    else:
        raise ValueError(
            "Input image must have shape (H, W, C) or (B, H, W, C)"
        )