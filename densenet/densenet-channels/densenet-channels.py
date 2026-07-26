import math
import torch

def densenet_channel_counts(stem_channels: int, growth_rate: int, block_layers, compression: float) -> torch.Tensor:
    """
    Returns a 1D int64 torch.Tensor of channel counts at each stage.
    """
    # YOUR CODE HERE
    channels = stem_channels
    counts = [channels]

    for i, num_layers in enumerate(block_layers):
        # Dense block
        channels += num_layers * growth_rate
        counts.append(channels)

        # Transition layer (except after last block)
        if i != len(block_layers)-1:
            channels = math.floor(channels * compression)
            counts.append(channels)

    return torch.tensor(counts, dtype=torch.int64)
