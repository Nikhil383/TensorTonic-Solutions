import torch

def skipgram_pairs(token_ids: torch.Tensor, window: int) -> torch.Tensor:
    """
    Returns the ordered center-context pairs as an int64 tensor.
    """
    pairs = []
    for center_index, center_token in enumerate(token_ids.tolist()):
        start = max(0, center_index - window)
        stop = min(len(token_ids), center_index + window + 1)
        for context_index in range(start, stop):
            if context_index != center_index:
                pairs.append([center_token, int(token_ids[context_index])])
    return torch.tensor(pairs, dtype=torch.int64).reshape(-1, 2)