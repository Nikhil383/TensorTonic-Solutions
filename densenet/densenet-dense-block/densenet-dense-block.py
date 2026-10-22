import torch
import torch.nn.functional as F

def dense_block(x: torch.Tensor, layers: list, growth_rate: int,
                eps: float = 1e-5) -> torch.Tensor:
    """
    Returns the float64 concatenation of the block input and all layer outputs.
    """
    features = [x]
    for layer in layers:
        current = torch.cat(features, dim=1)
        channels = current.shape[1]
        gamma = layer["bn_gamma"].view(1, channels, 1, 1)
        beta = layer["bn_beta"].view(1, channels, 1, 1)
        mean = layer["bn_mean"].view(1, channels, 1, 1)
        variance = layer["bn_var"].view(1, channels, 1, 1)
        normalized = gamma * (current - mean) / torch.sqrt(variance + eps) + beta
        added = F.conv2d(F.relu(normalized), layer["conv_weight"], bias=None, padding=1)
        features.append(added.reshape(added.shape[0], growth_rate, added.shape[2], added.shape[3]))
    return torch.cat(features, dim=1)