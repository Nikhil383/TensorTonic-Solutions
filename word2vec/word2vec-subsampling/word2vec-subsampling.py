import torch

def subsample_keep_probs(counts: torch.Tensor, t: float = 1e-5) -> torch.Tensor:
    """
    Returns torch.Tensor of shape (vocab_size,) with the keep-probability for each word.
    """
    # Compute word frequencies
    freqs = counts.float() / counts.sum()
    keep_probs = torch.sqrt(t / freqs)
    keep_probs = torch.clamp(keep_probs, max=1.0)
    return keep_probs